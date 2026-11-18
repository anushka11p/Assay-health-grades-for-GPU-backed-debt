"""Builds the census: every distinct physical GPU seen, its latest report, grade
and history, plus summary statistics.

Input: signed reports in census/raw/ (verify them first: scripts/build.sh does).
Output: census/census.json, which the site is built from.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .grade import cohort_key, gpu_key, grade_gpu


def load_reports(raw_dir: Path) -> tuple[list[dict], list[str]]:
    """Reads every *.json report in raw_dir. Reports from the fake backend are
    skipped (with a warning): only real hardware belongs in the census."""
    reports, warnings = [], []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith(".proof.json"):
            continue
        r = json.loads(path.read_text())
        if (r.get("agent") or {}).get("backend") != "nvml":
            warnings.append(f"skipped {path.name}: not from real hardware (backend={r.get('agent', {}).get('backend')})")
            continue
        r["_file"] = path.name
        reports.append(r)
    return reports, warnings


def load_log(log_dir: Path) -> tuple[dict | None, dict[str, int]]:
    """The signed tree head and a digest -> log index map, if a log exists."""
    head_path, entries_path = log_dir / "head.json", log_dir / "entries.jsonl"
    if not head_path.exists():
        return None, {}
    head = json.loads(head_path.read_text())
    index = {}
    if entries_path.exists():
        for line in entries_path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                index[e["digest"]] = e["index"]
    return head, index


def build(raw_dir: Path, log_dir: Path | None = None) -> dict:
    reports, warnings = load_reports(raw_dir)
    head, log_index = load_log(log_dir) if log_dir else (None, {})

    # Every GPU in every report, grouped by physical GPU.
    seen: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for r in reports:
        for g in r.get("gpus") or []:
            seen[gpu_key(g)].append((r, g))

    # Grade each GPU on its latest check, against the latest check of every
    # other GPU of the same model.
    latest = {k: max(v, key=lambda rg: rg[0]["created_at"]) for k, v in seen.items()}
    cohorts: dict[str, list[dict]] = defaultdict(list)
    for r, g in latest.values():
        cohorts[cohort_key(g)].append(g)

    gpus = []
    for key, (r, g) in sorted(latest.items(), key=lambda kv: kv[1][0]["created_at"]):
        grade, perf, compared = grade_gpu(g, cohorts[cohort_key(g)])
        ident = g.get("identity", {})
        digest = (r.get("signature") or {}).get("digest")
        gpus.append({
            "key": key,
            "name": ident.get("name"),
            "architecture": ident.get("architecture"),
            "serial": ident.get("serial"),
            "uuid": ident.get("uuid"),
            "memory_total_mib": ident.get("memory_total_mib"),
            "provider": (r.get("host") or {}).get("provider", "unknown"),
            "driver_version": (r.get("host") or {}).get("driver_version"),
            "checks": len(seen[key]),
            "latest": {
                "file": r["_file"], "report_id": r.get("report_id"), "created_at": r.get("created_at"),
                "digest": digest, "key_id": (r.get("signature") or {}).get("key_id"),
                "log_index": log_index.get(digest),
            },
            "history": sorted(rr["_file"] for rr, _ in seen[key]),
            "grade": grade.to_json(),
            "peer_comparison": perf,
            "compared_with_peers": compared,
            "below_peers": any(f.area == "performance" for f in grade.findings),
            "gpu_index": g.get("index", 0),
        })

    # Worst grades first: that's what a lender scans for. Ties: most recent first.
    gpus.sort(key=lambda x: x["latest"]["created_at"], reverse=True)
    gpus.sort(key=lambda x: "FDCBA".index(x["grade"]["grade"]))
    compared = [x for x in gpus if x["compared_with_peers"]]
    summary = {
        "reports": len(reports),
        "gpus": len(gpus),
        "providers": dict(Counter(x["provider"] for x in gpus)),
        "models": dict(Counter(x["name"] for x in gpus)),
        "grades": {g: sum(1 for x in gpus if x["grade"]["grade"] == g) for g in "ABCDF"},
        "flagged": sum(1 for x in gpus if x["grade"]["grade"] in ("D", "F")),
        "compared_with_peers": len(compared),
        "below_peers": sum(1 for x in compared if x["below_peers"]),
        "below_peers_pct": round(100 * sum(1 for x in compared if x["below_peers"]) / len(compared)) if compared else None,
        "full_coverage": sum(1 for x in gpus if x["grade"]["coverage"] == "full"),
    }
    first = min((r["created_at"] for r in reports), default=None)
    last = max((r["created_at"] for r in reports), default=None)
    return {"schema_version": 1, "period": {"first": first, "last": last}, "summary": summary,
            "log": head, "gpus": gpus, "warnings": warnings}
