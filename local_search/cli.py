from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import improve_payload


def _read_payload(input_path: str | None) -> dict:
    if input_path:
        with open(input_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return json.load(sys.stdin)


def _write_payload(payload: dict, output_path: str | None, pretty: bool) -> None:
    kwargs = {"ensure_ascii": False}
    if pretty:
        kwargs["indent"] = 2

    content = json.dumps(payload, **kwargs)
    if output_path:
        Path(output_path).write_text(content + ("\n" if pretty else ""), encoding="utf-8")
        return

    sys.stdout.write(content)
    if pretty:
        sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local search improvement on a TSP/CVRP/CVRPTW solution payload.")
    parser.add_argument("--input", help="Path to input JSON payload. Defaults to stdin.")
    parser.add_argument("--output", help="Path to write output JSON. Defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    payload = _read_payload(args.input)
    improved = improve_payload(payload)
    _write_payload(improved, args.output, args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
