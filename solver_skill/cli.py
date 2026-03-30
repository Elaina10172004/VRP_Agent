from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import solve_payload


def _read_payload(input_path: str | None) -> dict:
    if input_path:
        with open(input_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return json.load(sys.stdin)


def _write_output(payload: dict, output_path: str | None, pretty: bool) -> None:
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


def _emit_progress(step_id: str, label: str, detail: str | None = None) -> None:
    sys.stderr.write(
        "PROGRESS\t"
        + json.dumps(
            {
                "stepId": step_id,
                "label": label,
                "detail": detail,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OptiChat solve skill pipeline: DRL seed -> lookahead -> local search.")
    parser.add_argument("--input", help="Path to input JSON payload. Defaults to stdin.")
    parser.add_argument("--output", help="Path to write output JSON. Defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    payload = _read_payload(args.input)
    result = solve_payload(payload, progress=_emit_progress)
    _write_output(result, args.output, args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
