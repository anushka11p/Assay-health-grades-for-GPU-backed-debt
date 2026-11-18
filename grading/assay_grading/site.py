"""Builds the static census website from census.json and the raw reports.

Plain HTML + one CSS file + one small script, so it can be served from GitHub
Pages with no build tools. Every number on the site links back to a signed
report anyone can download and verify with `assay verify`.
"""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

REPO = "https://github.com/anushka11p/anushka11p-Assay-health-grades-for-GPU-backed-debt"
ASSETS = Path(__file__).parent / "assets"

# Grades use the fixed status palette and always appear with their letter and a word.
GRADE_STATUS = {"A": ("good", "Healthy"), "B": ("good", "Healthy, with a note"), "C": ("warning", "Minor issue"),
                "D": ("serious", "Major issue"), "F": ("critical", "Failing")}
SEVERITY_LABEL = {"critical": "Critical", "major": "Major", "minor": "Minor", "note": "Note"}
SEVERITY_STATUS = {"critical": "critical", "major": "serious", "minor": "warning", "note": "good"}
HEADLINE = {"gemm": ["fp16_gflops", "fp32_gflops"], "membw": ["copy_mb_per_s", "triad_mb_per_s"],
            "memtest": ["mismatches", "mib_tested"], "soak": ["first_gflops", "last_gflops", "drop_pct"],
            "allreduce": ["busbw_mb_per_s", "gpus"]}


def e(x) -> str:
    return escape("" if x is None else str(x))


def num(x) -> str:
    return f"{x:,}" if isinstance(x, int) else e(x if x is not None else "n/a")


def page(title: str, body: str, depth: int = 0) -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} · Assay</title>
<link rel="stylesheet" href="{up}assets/style.css">
</head><body>
<header class="top"><a class="brand" href="{up}index.html">Assay</a>
<nav><a href="{up}index.html">Census</a><a href="{up}methodology.html">Methodology</a><a href="{REPO}">Source</a></nav></header>
<main>{body}</main>
<footer>Every number here comes from a report signed on the GPU's own machine. Download any report and check it with <code>assay verify</code>.</footer>
<script src="{up}assets/chart.js"></script>
</body></html>
"""


def badge(grade: str, big: bool = False) -> str:
    status, word = GRADE_STATUS[grade]
    cls = "badge big" if big else "badge"
    return f'<span class="{cls} s-{status}" title="{e(word)}"><b>{grade}</b><span>{e(word)}</span></span>'


def tile(value, label: str, note: str = "") -> str:
    n = f'<div class="tile-note">{e(note)}</div>' if note else ""
    return f'<div class="tile"><div class="tile-value">{e(value)}</div><div class="tile-label">{e(label)}</div>{n}</div>'


# ---------------------------------------------------------------- charts

def nice_step(span: float, target: int) -> float:
    """A round tick step (1, 2, 2.5 or 5 x 10^n) giving about `target` ticks."""
    import math
    raw = max(span, 1e-9) / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def grade_bars(grades: dict) -> str:
    """Horizontal bars, one per grade: a magnitude per category."""
    total = max(sum(grades.values()), 1)
    peak = max(max(grades.values()), 1)
    rows = []
    for g in "ABCDF":
        n = grades.get(g, 0)
        status, word = GRADE_STATUS[g]
        width = 0 if n == 0 else max(2, round(100 * n / peak))
        rows.append(f"""<div class="gbar" tabindex="0" title="Grade {g} ({word}): {n} GPU{'s' if n != 1 else ''}, {round(100*n/total)}%">
  <span class="gbar-label">{g} <small>{e(word)}</small></span>
  <span class="gbar-track"><span class="gbar-fill s-{status}" style="width:{width}%"></span></span>
  <span class="gbar-value">{n}</span></div>""")
    return '<div class="gbars" role="table" aria-label="GPUs per grade">' + "".join(rows) + "</div>"


def line_chart(samples: list, key: str, title: str, unit: str, probes: list) -> str:
    """One measure over time (single series, so no legend). Probe windows are
    shaded so readings can be matched to what the GPU was doing."""
    pts = [(s["t_ms"], s[key]) for s in samples if s.get(key) is not None]
    if len(pts) < 2:
        return f'<div class="chart empty"><h4>{e(title)}</h4><p>Not enough readings to chart.</p></div>'
    step = max(1, len(pts) // 400)  # keep the SVG light
    pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
    W, H, L, R, T, B = 640, 180, 48, 10, 10, 26
    t0, t1 = pts[0][0], pts[-1][0]
    # Temperatures and clocks are magnitudes: the axis starts at zero, so flat
    # data looks flat and small changes aren't exaggerated.
    step_v = nice_step(max(v for _, v in pts) * 1.1 or 1, 4)
    lo, hi = 0, step_v * max(1, -(-max(v for _, v in pts) * 1.05 // step_v))
    x = lambda t: L + (t - t0) * (W - L - R) / max(t1 - t0, 1)
    y = lambda v: T + (hi - v) * (H - T - B) / max(hi - lo, 1)

    bands = []
    for p in probes:
        if p.get("status") != "ok" or p.get("end_ms") is None:
            continue
        a, b = max(p["start_ms"], t0), min(p["end_ms"], t1)
        if b <= a:
            continue
        # 2px gap between neighbouring windows so each test reads as its own block.
        bands.append(f'<rect class="band" x="{x(a)+1:.1f}" y="{T}" width="{max(x(b)-x(a)-2, 1):.1f}" height="{H-T-B}"/>'
                     f'<text class="band-label" x="{x(a)+3:.1f}" y="{T+11}">{e(p["name"])}</text>')
    grid = []
    v = lo
    while v <= hi + 1e-9:
        grid.append(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                    f'<text class="tick" x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end">{round(v):,}</text>')
        v += step_v
    secs = (t1 - t0) / 1000
    step_s = nice_step(secs, 5)
    s_ = 0.0
    while s_ <= secs + 1e-9:
        grid.append(f'<text class="tick" x="{x(t0 + s_*1000):.1f}" y="{H-8}" text-anchor="middle">{s_:g}s</text>')
        s_ += step_s
    path = "M" + " L".join(f"{x(t):.1f},{y(v):.1f}" for t, v in pts)
    data = json.dumps({"x": [round(x(t), 1) for t, _ in pts], "t": [t for t, _ in pts], "v": [v for _, v in pts],
                       "unit": unit, "top": T, "bottom": H - B})
    return f"""<figure class="chart" data-series='{e(data)}'>
<figcaption><h4>{e(title)} ({e(unit)})</h4><span>{len(pts)} readings over {secs:.0f}s{' · shaded = stress test running' if bands else ''}</span></figcaption>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(title)} over time">
{''.join(bands)}{''.join(grid)}
<path class="line" d="{path}"/>
<line class="cross" x1="0" x2="0" y1="{T}" y2="{H-B}" visibility="hidden"/>
</svg><div class="tip" hidden></div></figure>"""


# ---------------------------------------------------------------- pages

def index_page(census: dict) -> str:
    s = census["summary"]
    log = census.get("log")
    below = f"{s['below_peers_pct']}%" if s["below_peers_pct"] is not None else "n/a"
    below_note = (f"of {s['compared_with_peers']} GPUs with enough peers" if s["compared_with_peers"]
                  else "needs 3+ GPUs of one model with stress tests")
    period = census["period"]
    rows = []
    for g in census["gpus"]:
        gr = g["grade"]
        rows.append(f"""<tr><td>{badge(gr['grade'])}</td>
<td><a href="gpu/{e(g['key'])}.html">{e(g['name'])}</a><div class="sub mono">serial {e(g['serial'] or 'n/a')}</div></td>
<td>{e(g['provider'])}</td><td class="num">{g['checks']}</td><td>{e(gr['coverage'])}</td><td>{e(gr['reason'])}</td></tr>""")
    log_html = ""
    if log:
        log_html = f"""<section class="panel"><h2>Transparency log</h2>
<p>Every report is logged in an append-only Merkle tree. The log operator signs its root; anyone can check a report is in it, and that nothing logged was later removed or edited.</p>
<dl class="kv"><dt>Entries</dt><dd>{log['tree_size']}</dd><dt>Root hash</dt><dd class="mono">{e(log['root_hash'])}</dd>
<dt>Signed</dt><dd>{e(log['timestamp'])} by log key <span class="mono">{e(log['log_key_id'])}</span></dd></dl>
<p><a href="data/log/head.json">head.json</a> · <a href="data/log/entries.jsonl">entries.jsonl</a> · <a href="data/log/log.pub">log.pub</a></p></section>"""
    else:
        log_html = '<section class="panel"><h2>Transparency log</h2><p>No reports have been logged yet.</p></section>'
    body = f"""<section class="hero"><h1>Rented GPU Census</h1>
<p class="lede">Health records for free cloud GPUs (Google Colab and Kaggle), measured by the Assay agent: memory errors, crash codes, PCIe link, thermals and CUDA stress tests, each report signed where it was taken and graded A–F against its peers.</p>
<p class="sub">Reports from {e((period['first'] or '')[:10])} to {e((period['last'] or '')[:10])}.</p></section>
<section class="tiles">{tile(s['gpus'], 'distinct physical GPUs', 'by board serial + UUID')}{tile(s['reports'], 'signed reports')}
{tile(s['flagged'], 'graded D or F')}{tile(below, 'below their peers', below_note)}</section>
<section class="panel"><h2>Grades</h2>{grade_bars(s['grades'])}
<p class="sub">A grade is the most serious problem found. See the <a href="methodology.html">methodology</a> for every rule.</p></section>
<section class="panel"><h2>Every GPU</h2><div class="table-wrap"><table>
<thead><tr><th>Grade</th><th>GPU</th><th>Provider</th><th class="num">Checks</th><th>Coverage</th><th>Why</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="6">No reports yet.</td></tr>'}</tbody></table></div></section>
{log_html}"""
    return page("Rented GPU Census", body)


def kv(pairs) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{e(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def gpu_page(g: dict, report: dict, gpu: dict, census: dict) -> str:
    gr = g["grade"]
    status, _ = GRADE_STATUS[gr["grade"]]
    findings = "".join(
        f'<li><span class="sev s-{SEVERITY_STATUS[f["severity"]]}">{SEVERITY_LABEL[f["severity"]]}</span> {e(f["text"])}</li>'
        for f in gr["findings"]) or '<li><span class="sev s-good">OK</span> No problems found in any check that ran.</li>'
    not_checked = "".join(f"<li>{e(n)}</li>" for n in gr["not_checked"])

    ident = gpu.get("identity", {})
    mem = gpu.get("memory") or {}
    agg, rp, rr = mem.get("aggregate") or {}, mem.get("retired_pages"), mem.get("row_remap")
    ic = gpu.get("interconnect") or {}
    lim = gpu.get("limits") or {}
    loaded = [s for s in gpu.get("samples") or [] if (s.get("util_gpu_pct") or 0) >= 50 and s.get("pcie_gen")]
    under_load = (f"Gen{max(s['pcie_gen'] for s in loaded)} x{max(s['pcie_width'] for s in loaded)}"
                  if loaded else "not measured (no readings under load)")

    probe_rows = []
    for p in gpu.get("probes") or []:
        res = p.get("result") or {}
        nums = ", ".join(f"{k} = {num(res[k])}" for k in HEADLINE.get(p["name"], []) if k in res)
        d = p.get("during") or {}
        during = (f"max {num(d.get('max_temp_c'))}°C, min clock {num(d.get('min_sm_clock_mhz'))} MHz, "
                  f"{d.get('slowdown_readings', 0)}/{d.get('readings', 0)} readings slowed down") if d else ""
        probe_rows.append(f"<tr><td>{e(p['name'])}</td><td>{e(p['status'])}</td><td>{e(nums or p.get('reason', ''))}</td><td>{e(during)}</td></tr>")
    probes_html = (f"""<div class="table-wrap"><table><thead><tr><th>Test</th><th>Status</th><th>Result</th><th>GPU during the test</th></tr></thead>
<tbody>{''.join(probe_rows)}</tbody></table></div>""" if probe_rows else
                   "<p>No stress tests in this report (telemetry-only check, before stress tests existed or with probes turned off).</p>")

    peers = g.get("peer_comparison") or {}
    peer_rows = "".join(
        f"<tr><td>{e(k)}</td><td class='num'>{num(v['value'])}</td><td class='num'>{num(round(v['median'])) if v.get('median') else 'n/a'}</td>"
        f"<td class='num'>{e(str(v['below_median_pct']) + '%') if v.get('median') else 'n/a'}</td><td class='num'>{v['peers']}</td></tr>"
        for k, v in peers.items())
    peers_html = (f"""<div class="table-wrap"><table><thead><tr><th>Metric</th><th class="num">This GPU</th><th class="num">Model median</th>
<th class="num">Below median</th><th class="num">GPUs compared</th></tr></thead><tbody>{peer_rows}</tbody></table></div>"""
                  if peer_rows else "<p>No stress-test results to compare yet.</p>")

    latest = g["latest"]
    log_line = (f"entry #{latest['log_index']} in the transparency log" if latest.get("log_index") is not None
                else "not in the transparency log yet")
    samples = gpu.get("samples") or []
    probes = gpu.get("probes") or []
    history = "".join(f'<li><a href="../data/raw/{e(f)}">{e(f)}</a></li>' for f in g["history"])
    body = f"""<p class="crumb"><a href="../index.html">Census</a> / {e(g['name'])}</p>
<section class="statement s-edge-{status}">
<div class="statement-head">{badge(gr['grade'], big=True)}
<div><h1>{e(g['name'])}</h1><p class="mono sub">serial {e(g['serial'] or 'n/a')} · {e(g['uuid'])}</p>
<p class="reason">{e(gr['reason'])}</p>
<p class="sub">Coverage: {e(gr['coverage'])} · {g['checks']} check{'s' if g['checks'] != 1 else ''} · latest {e(latest['created_at'])} on {e(g['provider'])}</p></div></div>
<h3>Findings</h3><ul class="findings">{findings}</ul>
{f'<h3>Not checked</h3><ul class="plain">{not_checked}</ul>' if not_checked else ''}
</section>
<section class="grid2">
<div class="panel"><h2>Identity</h2>{kv([("Model", e(ident.get('name'))), ("Architecture", e(ident.get('architecture'))),
    ("Memory", f"{num(ident.get('memory_total_mib'))} MiB"), ("Board part", e(ident.get('board_part_number') or 'n/a')),
    ("VBIOS", e(ident.get('vbios_version'))), ("Driver", e((report.get('host') or {}).get('driver_version')))])}</div>
<div class="panel"><h2>Memory health</h2>{kv([("ECC", "on" if mem.get('ecc_enabled') else ("off" if mem.get('ecc_enabled') is False else "unknown")),
    ("Lifetime ECC errors", f"{num(agg.get('corrected'))} corrected, {num(agg.get('uncorrected'))} uncorrected"),
    ("Retired pages", f"{rp['single_bit']} single-bit, {rp['double_bit']} double-bit" + (", pending" if rp.get('pending_retirement') else "") if rp else "n/a"),
    ("Row remapping", f"{rr['correctable']} correctable, {rr['uncorrectable']} uncorrectable" if rr else "not on this GPU")])}</div>
<div class="panel"><h2>PCIe link</h2>{kv([("Under load", e(under_load)), ("Maximum", f"Gen{num(ic.get('pcie_gen_max'))} x{num(ic.get('pcie_width_max'))}"),
    ("At start (idle)", f"Gen{num(ic.get('pcie_gen_current'))} x{num(ic.get('pcie_width_current'))}"), ("Replays", num(ic.get('pcie_replay_counter')))])}
<p class="sub">Idle GPUs slow their link to save power, so only readings under load count.</p></div>
<div class="panel"><h2>Limits</h2>{kv([("Power limit", f"{num((lim.get('power_limit_mw') or 0)//1000) if lim.get('power_limit_mw') else 'n/a'} W"),
    ("Max SM clock", f"{num(lim.get('max_sm_clock_mhz'))} MHz"), ("Slowdown at", f"{num(lim.get('slowdown_temp_c'))}°C"),
    ("Shutdown at", f"{num(lim.get('shutdown_temp_c'))}°C")])}</div>
</section>
<section class="panel"><h2>Stress tests</h2>{probes_html}</section>
<section class="panel"><h2>Over time</h2>
{line_chart(samples, 'temp_c', 'Temperature', '°C', probes)}
{line_chart(samples, 'sm_clock_mhz', 'SM clock', 'MHz', probes)}</section>
<section class="panel"><h2>Compared with other {e(g['name'])} GPUs</h2>{peers_html}</section>
<section class="panel verify"><h2>Verify this report yourself</h2>
{kv([("Report", f'<a href="../data/raw/{e(latest["file"])}">{e(latest["file"])}</a>'), ("Digest", f'<span class="mono">{e(latest["digest"])}</span>'),
     ("Signing key", f'<span class="mono">{e(latest["key_id"])}</span>'), ("Log", e(log_line))])}
<pre><code>assay verify {e(latest['file'])}
assay log prove --dir log {e(latest['file'])}
assay verify --proof {e(latest['file'][:-5])}.proof.json --log-pubkey log/log.pub {e(latest['file'])}</code></pre>
<p class="sub">The signature covers every byte of the report: change any number and verification fails.</p>
<h3>All checks of this GPU</h3><ul class="plain">{history}</ul></section>"""
    return page(f"{g['name']} {g['serial']}", body, depth=1)


def methodology_page() -> str:
    body = f"""<section class="hero"><h1>Methodology</h1>
<p class="lede">How Assay measures a GPU, how a grade is decided, and exactly what a signature does and doesn't prove.</p></section>
<section class="panel"><h2>What a check measures</h2>
<ol><li><b>Telemetry</b> through NVIDIA's NVML library, read-only: identity (serial, UUID), ECC error counters, retired pages / row remapping, XID driver errors, PCIe link, temperature, power, clocks and throttle reasons. Sampled every 250&nbsp;ms throughout.</li>
<li><b>Stress tests</b> (CUDA programs), run only if the GPU is idle: matrix multiply (cuBLAS FP16/FP32), memory bandwidth (copy/triad), memory pattern test (6 patterns over 75% of free memory), a 3-minute sustained-load soak, and NCCL all-reduce on multi-GPU machines.</li>
<li><b>Signature</b>: the agent signs the whole report (Ed25519 over canonical JSON) before it leaves the machine.</li>
<li><b>Log</b>: reports are appended to an RFC 6962 Merkle transparency log whose root is signed and published.</li></ol></section>
<section class="panel"><h2>How a grade is decided</h2>
<p>Each problem becomes a finding with a severity. The grade is the most serious finding, and the first line of every GPU's page says which one.</p>
<div class="table-wrap"><table><thead><tr><th>Grade</th><th>Severity</th><th>Examples</th></tr></thead><tbody>
<tr><td>{badge('F')}</td><td>Critical</td><td>uncorrectable ECC errors; pages retired after double-bit errors; row remapping failed; memtest mismatches; XID 48, 64, 79, 95</td></tr>
<tr><td>{badge('D')}</td><td>Major</td><td>PCIe link below max <i>under load</i>; hardware thermal slowdown; soak loses ≥25% speed; a stress test failed or timed out; pending page retirement; XID 63, 74, 92, 94, 119, 120; ≥30% below the model median</td></tr>
<tr><td>{badge('C')}</td><td>Minor</td><td>software thermal slowdown; soak loses 10–25%; ≥1,000 corrected ECC errors; single-bit retired pages; ≥100 PCIe replays; other XIDs; 15–30% below the model median</td></tr>
<tr><td>{badge('B')}</td><td>Note</td><td>ECC turned off (errors can't be counted)</td></tr>
<tr><td>{badge('A')}</td><td>None</td><td>no problems found in any check that ran</td></tr></tbody></table></div>
<h3>Deliberately not counted</h3><ul class="plain">
<li><b>Idle PCIe link speed.</b> Idle GPUs drop to Gen1 to save power; the first real Colab T4 did exactly this. Only readings at ≥50% utilisation count.</li>
<li><b>Running at the power cap.</b> A 70&nbsp;W T4 hits its power limit on every heavy workload; that's normal, not a fault.</li>
<li><b>Values the GPU can't report.</b> They're listed under "Not checked", never treated as zero.</li></ul>
<h3>Peers</h3><p>Performance is compared with the median of the same model, only when at least 3 GPUs of that model have stress-test results. Each GPU is graded on its most recent check.</p>
<h3>Coverage</h3><p><b>full</b>: every stress test passed or ran; <b>partial</b>: some failed; <b>telemetry-only</b>: no stress tests ran (early reports, or a busy GPU).</p></section>
<section class="panel"><h2>What a signature proves (threat model)</h2>
<ul class="plain"><li><b>Proves:</b> the report is byte-for-byte what the agent wrote, signed by the key it names. Pin a known key (<code>--pubkey</code>) to check <i>which</i> install signed it.</li>
<li><b>Proves (log):</b> the report was logged, and the log only grew: no logged report was later removed or edited (consistency proofs).</li>
<li><b>Does not prove:</b> which physical GPU was measured. On T4/P100 a modified agent on a dishonest host could sign false readings. Binding readings to the silicon needs hardware attestation (NVIDIA nvtrust on H100/Blackwell in confidential-computing mode), the planned next step.</li>
<li><b>Free notebook keys:</b> each Colab/Kaggle session makes a fresh key, so reports prove integrity, not a long-lived identity.</li></ul></section>
<section class="panel"><h2>Limits of this census</h2><p>Free tiers hand out whatever GPUs they have, so the sample is small and not random. Use it as evidence that condition varies and can be measured, not as a fleet-wide failure rate.</p></section>"""
    return page("Methodology", body)


def build_site(census: dict, raw_dir: Path, log_dir: Path | None, out: Path) -> list[str]:
    """Writes the whole site into out/ and returns the pages written."""
    if out.exists():
        shutil.rmtree(out)
    (out / "gpu").mkdir(parents=True)
    shutil.copytree(ASSETS, out / "assets")
    (out / "data" / "raw").mkdir(parents=True)
    for f in raw_dir.glob("*.json"):
        shutil.copy(f, out / "data" / "raw" / f.name)
    for f in raw_dir.glob("*.pub"):
        shutil.copy(f, out / "data" / "raw" / f.name)
    if log_dir and log_dir.exists():
        shutil.copytree(log_dir, out / "data" / "log")
    (out / "data" / "census.json").write_text(json.dumps(census, indent=1) + "\n")

    written = ["index.html", "methodology.html"]
    (out / "index.html").write_text(index_page(census))
    (out / "methodology.html").write_text(methodology_page())
    for g in census["gpus"]:
        report = json.loads((raw_dir / g["latest"]["file"]).read_text())
        gpu = next(x for x in report["gpus"] if x.get("index", 0) == g["gpu_index"])
        name = f"gpu/{g['key']}.html"
        (out / name).write_text(gpu_page(g, report, gpu, census))
        written.append(name)
    (out / ".nojekyll").write_text("")  # serve files as-is on GitHub Pages
    return written
