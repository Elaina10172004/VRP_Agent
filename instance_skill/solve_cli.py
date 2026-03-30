from __future__ import annotations

import argparse
import json
import sys

from .api import ingest_and_solve_uploaded_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize an uploaded routing instance, normalize it, and solve it.")
    parser.add_argument("--input-file", required=True, help="Path to the uploaded instance file.")
    parser.add_argument("--mode", choices=["quick", "thinking"], default="quick", help="Solve mode.")
    parser.add_argument("--save-root", help="Optional directory to store normalized outputs.")
    parser.add_argument("--config-json", help="Optional JSON string merged into solver config.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    config = json.loads(args.config_json) if args.config_json else None
    result = ingest_and_solve_uploaded_file(
        args.input_file,
        mode=args.mode,
        config=config,
        save_root=args.save_root,
    )

    kwargs = {"ensure_ascii": False}
    if args.pretty:
        kwargs["indent"] = 2
    sys.stdout.write(json.dumps(result, **kwargs))
    if args.pretty:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
