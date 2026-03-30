from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from solver_skill import solve_payload


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_ROOT = ROOT_DIR / "normalized_instances"


@dataclass
class IngestResult:
    payload: dict
    detected_format: str
    source_path: str
    saved_directory: str
    payload_json_path: str
    canonical_path: str
    original_copy_path: str
    summary: dict

    def to_dict(self) -> dict:
        return {
            "payload": self.payload,
            "detected_format": self.detected_format,
            "source_path": self.source_path,
            "saved_directory": self.saved_directory,
            "payload_json_path": self.payload_json_path,
            "canonical_path": self.canonical_path,
            "original_copy_path": self.original_copy_path,
            "summary": self.summary,
        }


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return slug or "instance"


def _as_number(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    stripped = str(value).strip()
    if not stripped:
        raise ValueError("Expected numeric value.")
    return float(stripped)


def _as_bool(value: str | int | float | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "depot"}


def _grid_scale_for_vrptw(depot_xy: list[float], node_xy: list[list[float]], node_tw: list[list[float]]) -> float:
    coordinate_max = max([abs(float(item)) for item in depot_xy] + [abs(float(value)) for point in node_xy for value in point] + [1.0])
    time_max = max([abs(float(value)) for window in node_tw for value in window] + [1.0])
    return float(max(coordinate_max, time_max, 1.0))


def _normalize_json_payload(raw: dict) -> tuple[dict, str]:
    if "problem_type" in raw and "instance" in raw:
        payload = {
            "problem_type": str(raw["problem_type"]).strip().lower(),
            "instance": raw["instance"],
        }
        return payload, "json_payload"

    if "points" in raw:
        return {"problem_type": "tsp", "instance": {"points": raw["points"]}}, "json_tsp"

    if {"depot_xy", "node_xy", "node_demand", "capacity"}.issubset(raw):
        payload = {
            "problem_type": "cvrptw" if "node_tw" in raw else "cvrp",
            "instance": {
                "depot_xy": raw["depot_xy"],
                "node_xy": raw["node_xy"],
                "node_demand": raw["node_demand"],
                "capacity": raw["capacity"],
            },
        }
        if "node_tw" in raw:
            payload["instance"]["node_tw"] = raw["node_tw"]
            payload["instance"]["service_time"] = raw.get("service_time", 0)
            payload["instance"]["grid_scale"] = raw.get(
                "grid_scale",
                _grid_scale_for_vrptw(raw["depot_xy"], raw["node_xy"], raw["node_tw"]),
            )
        return payload, "json_instance"

    raise ValueError("JSON content is not a recognized solver payload or instance.")


def _parse_solomon_text(content: str) -> tuple[dict, str]:
    upper = content.upper()
    if "CUSTOMER" not in upper or "CAPACITY" not in upper:
        raise ValueError("Not a Solomon/VRPTW text instance.")

    capacity_match = re.search(r"NUMBER\s+CAPACITY\s+(\d+)\s+(\d+)", content, flags=re.IGNORECASE | re.MULTILINE)
    if not capacity_match:
        raise ValueError("Failed to parse Solomon capacity.")
    capacity = float(capacity_match.group(2))

    lines = content.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if "CUST" in line.upper() and "XCOORD" in line.upper():
            start_index = index + 1
            break
    if start_index is None:
        raise ValueError("Failed to locate Solomon customer section.")

    rows: list[list[float]] = []
    for line in lines[start_index:]:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(numbers) >= 7:
            rows.append([float(value) for value in numbers[:7]])

    if len(rows) < 2:
        raise ValueError("Solomon instance does not contain enough rows.")

    depot = rows[0]
    customers = rows[1:]
    node_xy = [[row[1], row[2]] for row in customers]
    node_demand = [row[3] for row in customers]
    node_tw = [[row[4], row[5]] for row in customers]
    service_time = [row[6] for row in customers]
    depot_xy = [depot[1], depot[2]]

    instance = {
        "depot_xy": depot_xy,
        "node_xy": node_xy,
        "node_demand": node_demand,
        "capacity": capacity,
        "node_tw": node_tw,
        "service_time": service_time,
        "grid_scale": _grid_scale_for_vrptw(depot_xy, node_xy, node_tw),
    }
    return {"problem_type": "cvrptw", "instance": instance}, "solomon_vrptw"


def _parse_tsplib_tsp(content: str) -> tuple[dict, str]:
    if "NODE_COORD_SECTION" not in content.upper():
        raise ValueError("Not a TSPLIB TSP file.")

    lines = content.splitlines()
    reading = False
    points: list[list[float]] = []
    for line in lines:
        upper = line.strip().upper()
        if upper == "NODE_COORD_SECTION":
            reading = True
            continue
        if not reading:
            continue
        if upper in {"EOF", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION"}:
            break
        parts = line.split()
        if len(parts) >= 3:
            points.append([float(parts[1]), float(parts[2])])

    if len(points) < 3:
        raise ValueError("TSPLIB TSP instance does not contain enough points.")

    return {"problem_type": "tsp", "instance": {"points": points}}, "tsplib_tsp"


def _parse_tsplib_cvrp(content: str) -> tuple[dict, str]:
    upper = content.upper()
    if "NODE_COORD_SECTION" not in upper or "DEMAND_SECTION" not in upper or "DEPOT_SECTION" not in upper:
        raise ValueError("Not a TSPLIB CVRP file.")

    capacity_match = re.search(r"CAPACITY\s*:\s*(-?\d+(?:\.\d+)?)", content, flags=re.IGNORECASE)
    if not capacity_match:
        raise ValueError("CVRP file does not contain CAPACITY.")
    capacity = float(capacity_match.group(1))

    coords: dict[int, list[float]] = {}
    demands: dict[int, float] = {}
    depot_id = None
    section = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        upper_line = line.upper()
        if not line:
            continue
        if upper_line == "NODE_COORD_SECTION":
            section = "coords"
            continue
        if upper_line == "DEMAND_SECTION":
            section = "demand"
            continue
        if upper_line == "DEPOT_SECTION":
            section = "depot"
            continue
        if upper_line == "EOF":
            break

        if section == "coords":
            parts = line.split()
            if len(parts) >= 3:
                coords[int(parts[0])] = [float(parts[1]), float(parts[2])]
        elif section == "demand":
            parts = line.split()
            if len(parts) >= 2:
                demands[int(parts[0])] = float(parts[1])
        elif section == "depot":
            if line == "-1":
                section = None
            else:
                depot_id = int(line)

    if depot_id is None or depot_id not in coords:
        raise ValueError("Failed to parse depot from CVRP file.")

    customer_ids = sorted(node_id for node_id in coords if node_id != depot_id)
    node_xy = [coords[node_id] for node_id in customer_ids]
    node_demand = [demands.get(node_id, 0.0) for node_id in customer_ids]

    return {
        "problem_type": "cvrp",
        "instance": {
            "depot_xy": coords[depot_id],
            "node_xy": node_xy,
            "node_demand": node_demand,
            "capacity": capacity,
        },
    }, "tsplib_cvrp"


def _parse_csv_table(content: str) -> tuple[dict, str]:
    sample = content[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(content.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("CSV file does not have a header row.")

    rows = list(reader)
    if not rows:
        raise ValueError("CSV file does not contain any rows.")

    normalized_headers = {field: field.strip().lower() for field in reader.fieldnames}
    reverse_headers = {value: key for key, value in normalized_headers.items()}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in reverse_headers:
                return reverse_headers[name]
        return None

    x_key = pick("x", "xcoord", "x_coord", "lon", "longitude")
    y_key = pick("y", "ycoord", "y_coord", "lat", "latitude")
    demand_key = pick("demand", "demands")
    ready_key = pick("ready", "ready_time", "tw_start", "start")
    due_key = pick("due", "due_time", "due_date", "tw_end", "end")
    service_key = pick("service", "service_time")
    capacity_key = pick("capacity", "cap")
    depot_key = pick("is_depot", "depot", "role", "type")

    if not x_key or not y_key:
        raise ValueError("CSV header must contain x/y columns.")

    has_time_window = ready_key is not None and due_key is not None
    has_demand = demand_key is not None

    if not has_demand:
        points = [[_as_number(row[x_key]), _as_number(row[y_key])] for row in rows]
        if len(points) < 3:
            raise ValueError("TSP CSV must contain at least 3 points.")
        return {"problem_type": "tsp", "instance": {"points": points}}, "csv_tsp"

    capacity = None
    if capacity_key is not None:
        for row in rows:
            value = str(row.get(capacity_key, "")).strip()
            if value:
                capacity = _as_number(value)
                break
    if capacity is None:
        raise ValueError("VRP CSV must contain a capacity column.")

    depot_index = 0
    if depot_key is not None:
        for index, row in enumerate(rows):
            if _as_bool(row.get(depot_key)):
                depot_index = index
                break

    depot_row = rows[depot_index]
    depot_xy = [_as_number(depot_row[x_key]), _as_number(depot_row[y_key])]
    customer_rows = [row for index, row in enumerate(rows) if index != depot_index]

    node_xy = [[_as_number(row[x_key]), _as_number(row[y_key])] for row in customer_rows]
    node_demand = [_as_number(row[demand_key]) for row in customer_rows]

    if has_time_window:
        node_tw = [[_as_number(row[ready_key]), _as_number(row[due_key])] for row in customer_rows]
        service_time = [_as_number(row.get(service_key, 0) or 0) for row in customer_rows]
        instance = {
            "depot_xy": depot_xy,
            "node_xy": node_xy,
            "node_demand": node_demand,
            "capacity": capacity,
            "node_tw": node_tw,
            "service_time": service_time,
            "grid_scale": _grid_scale_for_vrptw(depot_xy, node_xy, node_tw),
        }
        return {"problem_type": "cvrptw", "instance": instance}, "csv_cvrptw"

    return {
        "problem_type": "cvrp",
        "instance": {
            "depot_xy": depot_xy,
            "node_xy": node_xy,
            "node_demand": node_demand,
            "capacity": capacity,
        },
    }, "csv_cvrp"


def detect_and_parse_instance(source_path: str | Path) -> tuple[dict, str]:
    file_path = Path(source_path)
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    suffix = file_path.suffix.lower()

    parsers = []
    if suffix == ".json":
        parsers.extend([lambda: _normalize_json_payload(json.loads(content)), lambda: _parse_csv_table(content)])
    elif suffix in {".txt"}:
        parsers.extend([lambda: _parse_solomon_text(content), lambda: _parse_tsplib_tsp(content), lambda: _parse_csv_table(content)])
    elif suffix in {".vrp"}:
        parsers.extend([lambda: _parse_tsplib_cvrp(content), lambda: _parse_solomon_text(content), lambda: _parse_csv_table(content)])
    elif suffix in {".tsp"}:
        parsers.extend([lambda: _parse_tsplib_tsp(content), lambda: _parse_csv_table(content)])
    elif suffix in {".csv", ".tsv"}:
        parsers.extend([lambda: _parse_csv_table(content), lambda: _parse_solomon_text(content)])
    else:
        if content.lstrip().startswith("{"):
            parsers.append(lambda: _normalize_json_payload(json.loads(content)))
        parsers.extend(
            [
                lambda: _parse_solomon_text(content),
                lambda: _parse_tsplib_cvrp(content),
                lambda: _parse_tsplib_tsp(content),
                lambda: _parse_csv_table(content),
            ]
        )

    errors: list[str] = []
    for parser in parsers:
        try:
            return parser()
        except Exception as error:
            errors.append(str(error))
    raise ValueError(f"Unsupported instance format for '{file_path.name}'. Tried parsers: {' | '.join(errors)}")


def _write_tsp_tsplib(path: Path, payload: dict, name: str) -> None:
    points = payload["instance"]["points"]
    lines = [
        f"NAME : {name}",
        "TYPE : TSP",
        f"DIMENSION : {len(points)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for index, (x, y) in enumerate(points, start=1):
        lines.append(f"{index} {x} {y}")
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cvrp_tsplib(path: Path, payload: dict, name: str) -> None:
    instance = payload["instance"]
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    node_demand = instance["node_demand"]
    lines = [
        f"NAME : {name}",
        "TYPE : CVRP",
        f"DIMENSION : {len(node_xy) + 1}",
        f"CAPACITY : {instance['capacity']}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        f"1 {depot_xy[0]} {depot_xy[1]}",
    ]
    for index, (x, y) in enumerate(node_xy, start=2):
        lines.append(f"{index} {x} {y}")
    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for index, demand in enumerate(node_demand, start=2):
        lines.append(f"{index} {demand}")
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cvrptw_solomon(path: Path, payload: dict, name: str) -> None:
    instance = payload["instance"]
    depot_xy = instance["depot_xy"]
    node_xy = instance["node_xy"]
    node_demand = instance["node_demand"]
    node_tw = instance["node_tw"]
    service_time = instance["service_time"]
    if isinstance(service_time, (int, float)):
        service_values = [float(service_time)] * len(node_xy)
    else:
        service_values = [float(value) for value in service_time]

    lines = [
        name,
        "",
        "VEHICLE",
        "NUMBER     CAPACITY",
        f"  999         {instance['capacity']}",
        "",
        "CUSTOMER",
        "CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME",
        "",
        f"    0      {depot_xy[0]}      {depot_xy[1]}      0      0      999999      0",
    ]
    for index, (xy, demand, tw, service) in enumerate(zip(node_xy, node_demand, node_tw, service_values), start=1):
        lines.append(
            f"{index:5d}      {xy[0]}      {xy[1]}      {demand}      {tw[0]}      {tw[1]}      {service}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_canonical_file(directory: Path, payload: dict, base_name: str) -> Path:
    problem_type = payload["problem_type"]
    if problem_type == "tsp":
        target = directory / f"{base_name}.tsp"
        _write_tsp_tsplib(target, payload, base_name)
        return target
    if problem_type == "cvrp":
        target = directory / f"{base_name}.vrp"
        _write_cvrp_tsplib(target, payload, base_name)
        return target
    if problem_type == "cvrptw":
        target = directory / f"{base_name}.solomon.txt"
        _write_cvrptw_solomon(target, payload, base_name)
        return target
    raise ValueError(f"Unsupported problem_type for saving: {problem_type}")


def _build_summary(payload: dict, detected_format: str) -> dict:
    instance = payload["instance"]
    if payload["problem_type"] == "tsp":
        return {"problem_type": "tsp", "detected_format": detected_format, "node_count": len(instance["points"])}

    summary = {
        "problem_type": payload["problem_type"],
        "detected_format": detected_format,
        "node_count": len(instance["node_xy"]),
        "capacity": float(instance["capacity"]),
    }
    if payload["problem_type"] == "cvrptw":
        summary["service_time_type"] = "scalar" if isinstance(instance["service_time"], (int, float)) else "per_customer"
    return summary


def ingest_uploaded_file(source_path: str | Path, save_root: str | Path | None = None) -> dict:
    file_path = Path(source_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    payload, detected_format = detect_and_parse_instance(file_path)
    save_root_path = Path(save_root).resolve() if save_root is not None else DEFAULT_SAVE_ROOT
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{_slugify(file_path.stem)}"
    directory = save_root_path / run_name
    directory.mkdir(parents=True, exist_ok=True)

    original_copy_path = directory / file_path.name
    shutil.copy2(file_path, original_copy_path)

    payload_json_path = directory / "payload.json"
    payload_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    canonical_path = _write_canonical_file(directory, payload, _slugify(file_path.stem))
    result = IngestResult(
        payload=payload,
        detected_format=detected_format,
        source_path=str(file_path),
        saved_directory=str(directory),
        payload_json_path=str(payload_json_path),
        canonical_path=str(canonical_path),
        original_copy_path=str(original_copy_path),
        summary=_build_summary(payload, detected_format),
    )
    return result.to_dict()


def _normalize_solve_config(mode: str = "quick", config: dict | None = None) -> dict:
    raw = dict(config or {})
    normalized_mode = str(mode).strip().lower()
    quick_mode = normalized_mode in {"quick", "fast"}
    return {
        **raw,
        "mode": "fast" if quick_mode else "hybrid",
        "drl_samples": int(raw.get("drl_samples", 128)),
        "seed_trials": int(raw.get("seed_trials", 8 if quick_mode else 1)),
        "enable_lookahead": False if quick_mode else bool(raw.get("enable_lookahead", True)),
        "lookahead_depth": int(raw.get("lookahead_depth", 2)),
        "lookahead_beam_width": int(raw.get("lookahead_beam_width", 4)),
        "lookahead_k": int(raw.get("lookahead_k", 1)),
        "enable_local_search": False if quick_mode else bool(raw.get("enable_local_search", False)),
        "local_search_rounds": int(raw.get("local_search_rounds", 50)),
    }


def ingest_and_solve_uploaded_file(
    source_path: str | Path,
    mode: str = "quick",
    config: dict | None = None,
    save_root: str | Path | None = None,
) -> dict:
    ingest_result = ingest_uploaded_file(source_path, save_root=save_root)
    payload = {
        "problem_type": ingest_result["payload"]["problem_type"],
        "instance": ingest_result["payload"]["instance"],
        "config": _normalize_solve_config(mode=mode, config=config),
    }
    solve_result = solve_payload(payload)
    return {
        "payload": payload,
        "payload_source": "upload",
        "ingest_result": ingest_result,
        "result": solve_result,
    }
