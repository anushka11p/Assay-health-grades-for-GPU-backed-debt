"""Tests for grading. Run from grading/:  python3 -m unittest -v"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from assay_grading.census import build
from assay_grading.grade import grade_gpu

REAL_RAW = Path(__file__).resolve().parents[2] / "census" / "raw"


def sample(t, util=100, gen=3, width=16, reasons=None, temp=65):
    return {"t_ms": t, "temp_c": temp, "util_gpu_pct": util, "pcie_gen": gen, "pcie_width": width,
            "throttle_reasons": reasons if reasons is not None else ["sw_power_cap"]}


def gpu(serial="S1", fp16=38000, triad=246000, last=38000, drop=1, mismatches=0, **over):
    """A healthy T4 with every stress test passed. Override any part via keyword args."""
    g = {
        "index": 0,
        "identity": {"uuid": f"GPU-{serial}", "serial": serial, "name": "Tesla T4"},
        "memory": {"ecc_enabled": True, "volatile": {"corrected": 0, "uncorrected": 0},
                   "aggregate": {"corrected": 3, "uncorrected": 0},
                   "retired_pages": {"single_bit": 0, "double_bit": 0, "pending_retirement": False}, "row_remap": None},
        "interconnect": {"pcie_gen_current": 1, "pcie_gen_max": 3, "pcie_width_current": 16, "pcie_width_max": 16,
                         "pcie_replay_counter": 0},
        "samples": [sample(0, util=0, gen=1, reasons=["gpu_idle"]), sample(1000), sample(2000)],
        "events": [],
        "probes": [
            {"name": "gemm", "status": "ok", "result": {"fp16_gflops": fp16, "fp32_gflops": 6900}},
            {"name": "membw", "status": "ok", "result": {"triad_mb_per_s": triad}},
            {"name": "memtest", "status": "ok", "result": {"mismatches": mismatches}},
            {"name": "soak", "status": "ok", "result": {"last_gflops": last, "drop_pct": drop}},
        ],
    }
    for k, v in over.items():
        g[k] = v
    return g


def grade(g, cohort=None):
    return grade_gpu(g, cohort if cohort is not None else [g])[0]


class Health(unittest.TestCase):
    def test_healthy_gpu_is_A_with_full_coverage(self):
        g = grade(gpu())
        self.assertEqual((g.grade, g.coverage), ("A", "full"))
        self.assertEqual(g.findings, [])

    def test_uncorrectable_ecc_is_F(self):
        x = gpu()
        x["memory"]["aggregate"]["uncorrected"] = 2
        g = grade(x)
        self.assertEqual(g.grade, "F")
        self.assertIn("2 uncorrectable", g.reason)

    def test_memtest_mismatch_is_F(self):
        self.assertEqual(grade(gpu(mismatches=3)).grade, "F")

    def test_idle_slow_link_is_fine_but_loaded_slow_link_is_D(self):
        # The lesson from the first real Colab report: idle T4s drop to Gen1.
        self.assertEqual(grade(gpu()).grade, "A")
        x = gpu(samples=[sample(0, util=0, gen=1), sample(1000, gen=1, width=8)])
        g = grade(x)
        self.assertEqual(g.grade, "D")
        self.assertIn("Gen1 x8", g.reason)

    def test_soak_drop_thresholds(self):
        self.assertEqual(grade(gpu(drop=60)).grade, "D")
        self.assertEqual(grade(gpu(drop=12)).grade, "C")
        self.assertEqual(grade(gpu(drop=5)).grade, "A")

    def test_xids_by_meaning(self):
        self.assertEqual(grade(gpu(events=[{"xid": 79}])).grade, "F")
        self.assertEqual(grade(gpu(events=[{"xid": 13}, {"xid": 13}])).grade, "C")

    def test_hardware_thermal_slowdown_is_D(self):
        x = gpu(samples=[sample(0), sample(1000, reasons=["sw_thermal_slowdown", "hw_thermal_slowdown"])])
        self.assertEqual(grade(x).grade, "D")

    def test_power_cap_is_not_a_problem(self):
        self.assertEqual(grade(gpu()).grade, "A")  # every loaded sample has sw_power_cap

    def test_worst_finding_decides_and_all_are_listed(self):
        x = gpu(drop=12, events=[{"xid": 48}])
        g = grade(x)
        self.assertEqual(g.grade, "F")
        self.assertEqual([f.severity for f in g.findings], ["critical", "minor"])

    def test_failed_probe_is_D_and_coverage_partial(self):
        x = gpu()
        x["probes"][1] = {"name": "membw", "status": "timeout", "reason": "did not finish"}
        g = grade(x)
        self.assertEqual((g.grade, g.coverage), ("D", "partial"))

    def test_unknown_is_not_a_finding(self):
        x = gpu(events=None)
        x["memory"]["aggregate"]["uncorrected"] = None
        g = grade(x)
        self.assertEqual(g.grade, "A")
        self.assertTrue(any("XID" in n for n in g.not_checked))
        self.assertTrue(any("ECC" in n for n in g.not_checked))


class Peers(unittest.TestCase):
    def test_too_few_peers_means_no_comparison(self):
        a, b = gpu("A"), gpu("B", fp16=10000)
        g, detail, compared = grade_gpu(b, [a, b])
        self.assertFalse(compared)
        self.assertEqual(g.grade, "A")
        self.assertTrue(any("peers" in n for n in g.not_checked))

    def test_far_below_median_is_D_somewhat_below_is_C(self):
        cohort = [gpu("A"), gpu("B"), gpu("C"), gpu("D", fp16=24000), gpu("E", fp16=31000)]
        self.assertEqual(grade(cohort[3], cohort).grade, "D")  # 37% below 38000
        self.assertEqual(grade(cohort[4], cohort).grade, "C")  # 18% below
        self.assertEqual(grade(cohort[0], cohort).grade, "A")


class Census(unittest.TestCase):
    def write(self, d, name, report):
        (d / name).write_text(json.dumps(report))

    def report(self, created, *gpus, backend="nvml"):
        return {"created_at": created, "report_id": created, "agent": {"backend": backend},
                "host": {"provider": "colab"}, "gpus": list(gpus), "signature": {"digest": "sha256:" + created}}

    def test_dedupes_by_physical_gpu_and_skips_fake(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old, new = gpu("S1"), gpu("S1", drop=60)
            self.write(d, "a.json", self.report("2026-11-10T00:00:00Z", old))
            self.write(d, "b.json", self.report("2026-11-12T00:00:00Z", new))
            self.write(d, "c.json", self.report("2026-11-11T00:00:00Z", gpu("S2")))
            self.write(d, "fake.json", self.report("2026-11-13T00:00:00Z", gpu("S3"), backend="fake"))
            c = build(d)
        s = c["summary"]
        self.assertEqual((s["reports"], s["gpus"]), (3, 2))
        s1 = next(x for x in c["gpus"] if x["serial"] == "S1")
        self.assertEqual(s1["checks"], 2)
        self.assertEqual(s1["grade"]["grade"], "D")  # graded on its latest check
        self.assertTrue(any("fake" in w for w in c["warnings"]))

    @unittest.skipUnless(any(REAL_RAW.glob("*.json")), "no real reports collected yet")
    def test_real_reports(self):
        c = build(REAL_RAW)
        self.assertGreaterEqual(c["summary"]["gpus"], 1)
        for x in c["gpus"]:
            self.assertIn(x["grade"]["grade"], "ABCDF")
            self.assertTrue(x["grade"]["reason"])


if __name__ == "__main__":
    unittest.main()
