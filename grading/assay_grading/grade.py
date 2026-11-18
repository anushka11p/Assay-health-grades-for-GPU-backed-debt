"""Turns one GPU's signed health report into findings and an A–F grade.

A grade is the worst finding's severity, so every grade comes with the exact
reason for it. Nothing is invented: a value the GPU couldn't report produces no
finding (it's listed under coverage instead), and peers are only compared when
there are enough of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median

# Severity -> grade. A GPU with no findings is an A.
CRITICAL, MAJOR, MINOR, NOTE = "critical", "major", "minor", "note"
GRADE_FOR = {CRITICAL: "F", MAJOR: "D", MINOR: "C", NOTE: "B"}
ORDER = [CRITICAL, MAJOR, MINOR, NOTE]

# XID codes by what they say about the hardware (NVIDIA "XID Errors" docs).
# Others (13, 31, 43, 45...) are often caused by the application, not the GPU.
XID_CRITICAL = {48: "double-bit ECC error", 64: "ECC page retirement failure",
                79: "GPU fell off the bus", 95: "uncontained ECC error"}
XID_MAJOR = {63: "ECC page retirement / row remap event", 74: "NVLink error", 92: "high single-bit ECC rate",
             94: "contained ECC error", 119: "GSP RPC timeout", 120: "GSP error"}

# Peer comparison needs at least this many GPUs of the same model.
MIN_PEERS = 3
# (probe, result key, label, unit). Higher is better for all of them.
PEER_METRICS = [
    ("gemm", "fp16_gflops", "FP16 matrix-multiply", "GFLOPS"),
    ("gemm", "fp32_gflops", "FP32 matrix-multiply", "GFLOPS"),
    ("membw", "triad_mb_per_s", "memory bandwidth", "MB/s"),
    ("soak", "last_gflops", "sustained (end of soak) throughput", "GFLOPS"),
]
LOADED_UTIL_PCT = 50  # readings at or above this count as "under load"


def n_of(n: int, word: str, plural: str | None = None) -> str:
    """'1 error' / '3 errors'."""
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


@dataclass
class Finding:
    severity: str
    area: str  # memory, errors, pcie, thermal, probes, performance
    text: str


@dataclass
class Grade:
    grade: str
    reason: str
    coverage: str  # "full", "partial" or "telemetry-only"
    findings: list[Finding] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def gpu_key(gpu: dict) -> str:
    """Identifies a physical GPU across reports: board serial + UUID."""
    ident = gpu.get("identity", {})
    return f"{ident.get('serial') or 'noserial'}-{ident.get('uuid', 'nouuid')}"


def cohort_key(gpu: dict) -> str:
    """GPUs compared with each other: same model name."""
    return gpu.get("identity", {}).get("name") or "unknown"


def probe(gpu: dict, name: str) -> dict | None:
    for p in gpu.get("probes") or []:
        if p.get("name") == name:
            return p
    return None


def metric(gpu: dict, probe_name: str, key: str) -> int | None:
    p = probe(gpu, probe_name)
    if p and p.get("status") == "ok":
        v = (p.get("result") or {}).get(key)
        if isinstance(v, int):
            return v
    return None


def loaded_link(gpu: dict) -> tuple[int, int] | None:
    """Best PCIe (gen, width) seen while the GPU was busy; None if never measured under load."""
    best = None
    for s in gpu.get("samples") or []:
        if (s.get("util_gpu_pct") or 0) < LOADED_UTIL_PCT:
            continue
        gen, width = s.get("pcie_gen"), s.get("pcie_width")
        if gen is None or width is None:
            continue
        if best is None or (gen, width) > best:
            best = (gen, width)
    return best


def findings_for(gpu: dict) -> tuple[list[Finding], list[str]]:
    """Health findings from one GPU's own report, plus what couldn't be checked."""
    out: list[Finding] = []
    not_checked: list[str] = []
    mem = gpu.get("memory") or {}
    agg = mem.get("aggregate") or {}

    # --- memory health ---
    if mem.get("ecc_enabled") is False:
        out.append(Finding(NOTE, "memory", "ECC is turned off, so memory errors can't be counted"))
    unc = agg.get("uncorrected")
    if unc is None:
        not_checked.append("lifetime ECC error counters")
    elif unc > 0:
        out.append(Finding(CRITICAL, "memory", f"{n_of(unc, 'uncorrectable (double-bit) ECC error')} in its lifetime"))
    corr = agg.get("corrected")
    if corr is not None and corr >= 1000:
        out.append(Finding(MINOR, "memory", f"{corr:,} corrected ECC errors in its lifetime (memory wearing)"))

    rp = mem.get("retired_pages")
    rr = mem.get("row_remap")
    if rp is None and rr is None:
        not_checked.append("retired pages / row remapping")
    if rp:
        if rp.get("double_bit", 0) > 0:
            out.append(Finding(CRITICAL, "memory", f"{n_of(rp['double_bit'], 'memory page')} retired after double-bit errors"))
        if rp.get("pending_retirement"):
            out.append(Finding(MAJOR, "memory", "a memory page retirement is pending (needs a driver reload)"))
        if rp.get("single_bit", 0) > 0:
            out.append(Finding(MINOR, "memory", f"{n_of(rp['single_bit'], 'memory page')} retired after repeated single-bit errors"))
    if rr:
        if rr.get("failed"):
            out.append(Finding(CRITICAL, "memory", "row remapping failed: no spare memory rows left"))
        elif rr.get("uncorrectable", 0) > 0:
            out.append(Finding(MAJOR, "memory", f"{n_of(rr['uncorrectable'], 'memory row')} remapped after uncorrectable errors"))
        if rr.get("pending"):
            out.append(Finding(MAJOR, "memory", "a row remap is pending (needs a GPU reset)"))

    # --- driver error events ---
    events = gpu.get("events")
    if events is None:
        not_checked.append("XID error events")
    for xid in sorted({e.get("xid") for e in events or []}):
        n = sum(1 for e in events if e.get("xid") == xid)
        times = f"{n}x" if n > 1 else "once"
        if xid in XID_CRITICAL:
            out.append(Finding(CRITICAL, "errors", f"XID {xid} ({XID_CRITICAL[xid]}) reported {times}"))
        elif xid in XID_MAJOR:
            out.append(Finding(MAJOR, "errors", f"XID {xid} ({XID_MAJOR[xid]}) reported {times}"))
        else:
            out.append(Finding(MINOR, "errors", f"XID {xid} reported {times} (often application-caused)"))

    # --- PCIe: only judged under load; idle GPUs slow their link on purpose ---
    ic = gpu.get("interconnect") or {}
    link = loaded_link(gpu)
    gen_max, width_max = ic.get("pcie_gen_max"), ic.get("pcie_width_max")
    if link is None:
        not_checked.append("PCIe link speed under load")
    elif gen_max and width_max and (link[0] < gen_max or link[1] < width_max):
        out.append(Finding(MAJOR, "pcie", f"PCIe link under load is Gen{link[0]} x{link[1]}, below its max Gen{gen_max} x{width_max}"))
    replays = ic.get("pcie_replay_counter")
    if replays is not None and replays >= 100:
        out.append(Finding(MINOR, "pcie", f"{replays:,} PCIe replays (link-level retransmits)"))

    # --- thermals ---
    reasons = [r for s in gpu.get("samples") or [] for r in s.get("throttle_reasons") or []]
    hw = sum(1 for r in reasons if r in ("hw_thermal_slowdown", "hw_slowdown"))
    sw = sum(1 for r in reasons if r == "sw_thermal_slowdown")
    if hw:
        out.append(Finding(MAJOR, "thermal", f"hardware slowdown (overheating protection) in {n_of(hw, 'reading')}"))
    elif sw:
        out.append(Finding(MINOR, "thermal", f"slowed itself down for heat in {n_of(sw, 'reading')}"))

    # --- stress tests ---
    probes = gpu.get("probes") or []
    for p in probes:
        if p.get("status") in ("failed", "timeout") and p.get("name") != "allreduce":
            out.append(Finding(MAJOR, "probes", f"{p['name']} stress test {p['status']}: {p.get('reason', '')}".strip()))
    bad = metric(gpu, "memtest", "mismatches")
    if bad:
        out.append(Finding(CRITICAL, "probes", f"memory test found {n_of(bad, 'word')} that read back wrong"))
    drop = metric(gpu, "soak", "drop_pct")
    if drop is not None and drop >= 25:
        out.append(Finding(MAJOR, "probes", f"lost {drop}% of its speed during the sustained-load test"))
    elif drop is not None and drop >= 10:
        out.append(Finding(MINOR, "probes", f"lost {drop}% of its speed during the sustained-load test"))
    if not any(p.get("status") == "ok" for p in probes):
        not_checked.append("stress tests (telemetry-only check)")
    return out, not_checked


def peer_findings(gpu: dict, cohort: list[dict]) -> tuple[list[Finding], dict, bool]:
    """Compares a GPU with others of its model. Returns findings, per-metric
    detail, and whether any comparison was possible."""
    out: list[Finding] = []
    detail: dict = {}
    compared = False
    for probe_name, key, label, unit in PEER_METRICS:
        values = [v for g in cohort if (v := metric(g, probe_name, key)) is not None]
        mine = metric(gpu, probe_name, key)
        if mine is None:
            continue
        if len(values) < MIN_PEERS:
            detail[key] = {"value": mine, "peers": len(values), "median": None}
            continue
        compared = True
        med = median(values)
        below_pct = round((med - mine) * 100 / med) if med else 0
        detail[key] = {"value": mine, "peers": len(values), "median": med, "below_median_pct": below_pct}
        if below_pct >= 30:
            out.append(Finding(MAJOR, "performance", f"{label} {below_pct}% below the {cohort_key(gpu)} median ({mine:,} vs {med:,.0f} {unit})"))
        elif below_pct >= 15:
            out.append(Finding(MINOR, "performance", f"{label} {below_pct}% below the {cohort_key(gpu)} median ({mine:,} vs {med:,.0f} {unit})"))
    return out, detail, compared


def grade_gpu(gpu: dict, cohort: list[dict]) -> tuple[Grade, dict, bool]:
    """Grades one GPU against its cohort (which includes itself)."""
    found, not_checked = findings_for(gpu)
    perf, detail, compared = peer_findings(gpu, cohort)
    found += perf
    if not compared:
        not_checked.append(f"comparison with peers (needs {MIN_PEERS}+ {cohort_key(gpu)} GPUs with stress-test results)")
    found.sort(key=lambda f: ORDER.index(f.severity))

    probes = [p for p in gpu.get("probes") or [] if p.get("name") != "allreduce"]
    ok = [p for p in probes if p.get("status") == "ok"]
    coverage = "telemetry-only" if not ok else ("full" if len(ok) == len(probes) else "partial")
    if found:
        g = Grade(GRADE_FOR[found[0].severity], found[0].text, coverage, found, not_checked)
    else:
        g = Grade("A", "no problems found in any check that ran", coverage, found, not_checked)
    return g, detail, compared
