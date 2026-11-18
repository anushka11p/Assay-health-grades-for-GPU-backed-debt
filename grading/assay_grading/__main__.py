"""Build the census and the site.

    python3 -m assay_grading --raw ../census/raw --log ../log --census ../census/census.json --site ../site

Reports should already be verified (scripts/build.sh runs `assay verify` first).
"""

import argparse
import json
import sys
from pathlib import Path

from .census import build
from .site import build_site


def main() -> int:
    ap = argparse.ArgumentParser(prog="assay_grading", description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="folder of signed reports")
    ap.add_argument("--log", type=Path, help="transparency log folder (optional)")
    ap.add_argument("--census", type=Path, required=True, help="where to write census.json")
    ap.add_argument("--site", type=Path, help="where to build the static site (optional)")
    args = ap.parse_args()

    census = build(args.raw, args.log)
    args.census.write_text(json.dumps(census, indent=1) + "\n")
    s = census["summary"]
    print(f"census: {s['gpus']} GPUs from {s['reports']} reports; grades {s['grades']}")
    for w in census["warnings"]:
        print("warning:", w, file=sys.stderr)
    for g in census["gpus"]:
        print(f"  {g['grade']['grade']}  {g['name']:<10} {g['serial'] or '':<14} {g['grade']['coverage']:<15} {g['grade']['reason']}")
    if args.site:
        pages = build_site(census, args.raw, args.log, args.site)
        print(f"site: {len(pages)} pages in {args.site}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
