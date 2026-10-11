"""CPU-only tests for evidence validation and comparisons; fixtures are not GPU measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import public_benchmark as public
import test_kit as kit
workspace_temp = kit.workspace_temp


def log_fixture(gpu="NVIDIA B300 SXM6 AC", ratio=1.0):
    # Test-only synthetic protocol fixture, deliberately never shipped as public data.
    lines = ["# nccl-tests version 2.17.6 nccl-headers=22809 nccl-library=22809",
             "# Collective test starting: all_reduce_perf",
             "# nThread 1 nGpus 8 minBytes 524288 maxBytes 8589934592 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0"]
    lines += [f"#  Rank {r} Group 0 Pid 400 on test-node device {r} [0000:01:00] {gpu}" for r in range(8)]
    size = 524288
    while size <= 8589934592:
        lines.append(f"{size} {size//4} float sum -1 {18006/ratio:.4f} {477.05*ratio:.4f} {834.84*ratio:.4f} 0 "
                     f"{17990/ratio:.4f} {477.49*ratio:.4f} {835.61*ratio:.4f} 0")
        size *= 2
    lines += ["# Out of bounds values : 0 OK", "# Avg bus bandwidth : 440.608", "# Collective test concluded: all_reduce_perf"]
    return "\n".join(lines) + "\n"


def save_evidence(out, log=None):
    path = out / "nccl-public.log"
    path.write_text(log or log_fixture(), encoding="utf-8")
    return public.capture(out, path, dict(cuda="13.1"))


class PublicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_actual_protocol_row_columns_and_decimal_units(self):
        result = public.parse_nccl(log_fixture())
        self.assertEqual(len(result["measurements"]), 15)
        last = result["measurements"][-1]
        self.assertEqual(last["size_bytes"], 8 * 2**30)
        self.assertEqual(last["out_of_place"]["bus_gbps"], 834.84)
        self.assertEqual(last["in_place"]["bus_gbps"], 835.61)
        self.assertEqual(result["runtime"]["nccl"], "2.28.9")

    def test_no_derived_ratio_without_public_run(self):
        data = public.compare(self.out)
        self.assertEqual(data["comparisons"][0]["status"], "NOT_RUN")
        self.assertTrue(all(m["relative_percent"] is None for r in data["comparisons"] for m in r["metrics"]))

    def test_ratio_and_difference_are_system_values_not_times_eight(self):
        save_evidence(self.out, log_fixture(ratio=0.8))
        data = public.compare(self.out)
        ref = data["comparisons"][0]
        self.assertEqual(ref["status"], "WORKLOAD_MATCHED")
        self.assertAlmostEqual(ref["metrics"][2]["relative_percent"], 80)
        self.assertAlmostEqual(ref["metrics"][2]["delta_percent"], -20)
        self.assertAlmostEqual(ref["metrics"][0]["relative_percent"], 125)
        self.assertTrue(any("nccl_tests" in x for x in ref["differences"]))

    def test_all_workload_mismatches_suppress_ratio(self):
        replacements = [("iters: 20", "iters: 50"), ("warmup iters: 1", "warmup iters: 5"),
                        ("graph: 0", "graph: 1"), ("agg iters: 1", "agg iters: 2"),
                        ("validation: 1", "validation: 2")]
        for old, new in replacements:
            with self.subTest(new=new):
                save_evidence(self.out, log_fixture().replace(old, new))
                ref = public.compare(self.out)["comparisons"][0]
                self.assertEqual(ref["status"], "CONFIG_MISMATCH")
                self.assertTrue(all(m["measured"] is None for m in ref["metrics"]))

    def test_corruption_missing_validation_and_truncation_fail_closed(self):
        log = log_fixture()
        invalid = [log.replace("0 OK", "1 FAILED"), log.replace("validation: 1", "validation: 0"),
                   log.replace("Collective test concluded", "NOT_CONCLUDED"),
                   log.replace("float sum", "float max"), log.replace("float sum", "half sum"),
                   log.replace("834.8400 0", "834.8400 1"), log.replace("835.6100 0", "835.6100 N/A"),
                   log.replace("477.0500", "nan"), log.replace("477.0500", "inf"),
                   log.replace("477.0500", "-3"), log.replace("835.6100", "999.9"),
                   log.replace("nccl-library=22809", "no-library-version")]
        for text in invalid:
            with self.subTest(text=text[-200:]), self.assertRaises(ValueError):
                public.parse_nccl(text)

    def test_missing_duplicate_and_malformed_rows_fail(self):
        lines = log_fixture().splitlines()
        row = next(i for i, s in enumerate(lines) if s.startswith("8589934592 "))
        for changed in [lines[:row]+lines[row+1:], lines[:row]+[lines[row]]+lines[row:],
                        lines[:row]+[lines[row]+" extra"]+lines[row+1:]]:
            with self.assertRaises(ValueError):
                public.parse_nccl("\n".join(changed))

    def test_wrong_gpu_count_node_group_and_gb300_rejected(self):
        invalid = [log_fixture("NVIDIA GB300"), log_fixture("NVIDIA B200"),
                   log_fixture().replace("Rank 7", "Rank 6"), log_fixture().replace("nGpus 8", "nGpus 4"),
                   log_fixture().replace("Rank 7 Group 0", "Rank 7 Group 1"),
                   log_fixture().replace("on test-node device 7", "on other-node device 7")]
        for text in invalid:
            with self.assertRaises(ValueError):
                public.parse_nccl(text)

    def test_context_model_scores_cannot_be_compared_to_qwen(self):
        save_evidence(self.out)
        (self.out / "llm-Qwen3-8B-gpu0.json").write_text(json.dumps(dict(status="PASS", aggregate_tokens_per_second=165432)), encoding="utf-8")
        refs = public.compare(self.out)["comparisons"][1:]
        self.assertEqual(len(refs), 3)
        self.assertTrue(all(r["status"] == "REFERENCE_ONLY" for r in refs))
        self.assertTrue(all(m["measured"] is None for r in refs for m in r["metrics"]))

    def test_software_differences_remain_visible(self):
        save_evidence(self.out, log_fixture().replace("22809", "22703"))
        ref = public.compare(self.out)["comparisons"][0]
        self.assertEqual(ref["status"], "WORKLOAD_MATCHED")
        self.assertTrue(any("2.27.3" in x for x in ref["differences"]))

    def test_hash_tampering_and_failed_process_cannot_compare(self):
        result = save_evidence(self.out)
        with (self.out / "nccl-public.log").open("a") as stream:
            stream.write("changed\n")
        self.assertEqual(public.compare(self.out)["comparisons"][0]["status"], "INVALID")
        result = save_evidence(self.out)
        result.update(status="FAIL", exit_code=1)
        public.write_json(self.out / "nccl-public.json", result)
        self.assertEqual(public.compare(self.out)["comparisons"][0]["status"], "INVALID")

    def test_cached_metrics_and_library_versions_are_not_trusted(self):
        result = save_evidence(self.out)
        result["measurements"][-1]["in_place"]["bus_gbps"] = 12345
        result["runtime"]["nccl"] = "99.0.0"
        public.write_json(self.out / "nccl-public.json", result)
        data = public.compare(self.out)
        self.assertEqual(data["comparisons"][0]["metrics"][5]["measured"], 835.61)
        self.assertEqual(data["measured_runtime"]["nccl"], "2.28.9")

    def test_baseline_snapshot_overrides_current_file(self):
        baseline = public.read_json(public.BASELINES)
        baseline["references"][0]["metrics"][2]["value"] *= 2
        public.write_json(self.out / "public-baselines.json", baseline)
        save_evidence(self.out)
        self.assertAlmostEqual(public.compare(self.out)["comparisons"][0]["metrics"][2]["relative_percent"], 50)

    def test_invalid_baseline_metric_or_gpu_is_rejected(self):
        for field, value in [("value", 0), ("value", float("nan")), ("unit", "Mb/s"), ("direction", "lower")]:
            data = public.read_json(public.BASELINES)
            data["references"][0]["metrics"][2][field] = value
            with self.assertRaises(ValueError):
                public.validate_baselines(data)
        data = public.read_json(public.BASELINES)
        data["references"][0]["system"]["gpu"] = "NVIDIA GB300"
        with self.assertRaises(ValueError):
            public.validate_baselines(data)

    def test_html_escapes_public_metadata(self):
        data = public.compare(self.out)
        data["comparisons"][0]["title"] = '<script>bad()</script>'
        page = public.render(data)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_run_binary_failure_retains_evidence_and_fail_status(self):
        def fail(command, **kwargs):
            kwargs["stdout"].write(log_fixture())
            return subprocess.CompletedProcess(command, 3)
        with patch.object(sys, "argv", ["public_benchmark.py", "run", "--out", str(self.out)]), \
             patch.object(public.subprocess, "run", side_effect=fail), patch("builtins.print"):
            self.assertEqual(public.main(), 1)
        result = public.read_json(self.out / "nccl-public.json")
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["status"], "FAIL")

    def test_vendored_source_matches_pinned_checksum(self):
        import build_nccl_tests
        source = public.ROOT / "vendor/nccl-tests-v2.17.6.tar.gz"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), build_nccl_tests.SOURCE_SHA256)


class PublicReportTests(unittest.TestCase):
    setUp = kit.ReportTests.setUp
    tearDown = kit.ReportTests.tearDown
    save = kit.ReportTests.save
    snapshot = kit.ReportTests.snapshot

    def require_public(self):
        self.plan["required"].append("nccl-public")
        self.save("plan.json", self.plan)
        with (self.out / "stages.tsv").open("a") as stream:
            stream.write("nccl-public\tPASS\t0\n")

    def test_missing_public_evidence_cannot_pass(self):
        import report
        self.require_public()
        self.assertEqual(report.generate(self.out)["overall"], "INCOMPLETE")

    def test_low_performance_does_not_invent_hardware_failure(self):
        import report
        self.require_public()
        save_evidence(self.out, log_fixture(ratio=0.5))
        self.assertEqual(report.generate(self.out)["overall"], "PASS")
        self.assertTrue((self.out / "public-comparison.json").exists())
        page = (self.out / "report.html").read_text(encoding="utf-8")
        self.assertIn("50.0%", page)
        self.assertIn("834.84", page)

    def test_corrupt_public_log_fails_report(self):
        import report
        self.require_public()
        save_evidence(self.out)
        (self.out / "nccl-public.log").write_text("truncated", encoding="utf-8")
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")


if __name__ == "__main__":
    unittest.main()
