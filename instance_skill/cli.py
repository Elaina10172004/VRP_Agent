from __future__ import annotations

import argparse
import json
import sys

from .api import ingest_uploaded_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize an uploaded routing instance, normalize it, and save canonical files.")
    parser.add_argument("--input-file", required=True, help="Path to the uploaded instance file.")
    parser.add_argument("--save-root", help="Optional directory to store normalized outputs.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    result = ingest_uploaded_file(args.input_file, save_root=args.save_root)
    kwargs = {"ensure_ascii": False}
    if args.pretty:
        kwargs["indent"] = 2
    sys.stdout.write(json.dumps(result, **kwargs))
    if args.pretty:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
