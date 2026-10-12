"""Synthetic protocol fixtures; these tests do not contain GPU measurements."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import URLError
import subprocess
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepseek_mlperf as ds
import deepseek_assets
import public_benchmark as public
import test_kit as kit

PERFORMANCE = """Scenario : Offline
Mode : PerformanceOnly
Samples per second : 18.7489
Tokens per second : 70325.7
Result is : VALID
Min duration satisfied : Yes
Min queries satisfied : Yes
min_duration (ms) : 600000
samples_per_query : 105312
performance_sample_count : 4388
accuracy_sample_count : 4388
No errors encountered during test.
"""
ACCURACY = "Results\n{'exact_match': 81.10756608933455, 'tokens_per_sample': 3500.0, 'num-samples': 4388}\n"


def save_evidence(out):
    perf = out / "deepseek-performance/system/deepseek-r1/Offline"
    acc = out / "deepseek-accuracy/system/deepseek-r1/Offline"
    perf.mkdir(parents=True, exist_ok=True)
    acc.mkdir(parents=True, exist_ok=True)
    (perf / "mlperf_log_summary.txt").write_text(PERFORMANCE, encoding="utf-8")
    (perf / "mlperf_log_detail.txt").write_text(':::MLLOG {"key": "loadgen_version", "value": "6.0.9 @ synthetic fixture", "metadata": {"is_error": false}}\n', encoding="utf-8")
    (acc / "accuracy.txt").write_text(ACCURACY, encoding="utf-8")
    (acc / "mlperf_log_summary.txt").write_text("Scenario : Offline\nMode : AccuracyOnly\n", encoding="utf-8")
    (acc / "mlperf_log_detail.txt").write_text((perf / "mlperf_log_detail.txt").read_text(encoding="utf-8"), encoding="utf-8")
    raw = acc / "mlperf_log_accuracy.json"
    # Full-length synthetic int32 outputs exercise summary/response consistency.
    with raw.open("w", encoding="utf-8") as stream:
        stream.write('[\n')
        for i in range(4388):
            stream.write(json.dumps({"qsl_idx": i, "data": "01000000" * 3500}) + (',\n' if i < 4387 else '\n]'))
    context = dict(suite="MLPerf Inference v6.0", model="deepseek-r1", scenario="Offline", precision="fp4",
                   gpus=8, nodes=1, tensor_parallelism=8, expert_parallelism=8, speculative_decoding=False,
                   source_revision=ds.SOURCE_REVISION, mlcommons_revision=ds.MLCOMMONS_REVISION)
    result = dict(status="PASS", context=context,
                  evidence_sha256={p.relative_to(out).as_posix(): ds.sha(p) for p in list(perf.iterdir()) + list(acc.iterdir())})
    ds.write_json(out / "deepseek-mlperf.json", result)
    return result, perf, acc


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.tmp = kit.workspace_temp()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_protocol_parses(self):
        self.assertEqual(ds.parse_performance(PERFORMANCE)["tokens_per_second"], 70325.7)
        self.assertEqual(ds.parse_accuracy(ACCURACY)["samples"], 4388)

    def test_invalid_or_wrong_scenario_rejected(self):
        for changed in [PERFORMANCE.replace("Offline", "Server"), PERFORMANCE.replace("PerformanceOnly", "AccuracyOnly"),
                        PERFORMANCE.replace("VALID", "INVALID"), PERFORMANCE.replace("satisfied : Yes", "satisfied : No"),
                        PERFORMANCE.replace("No errors encountered during test.", "Errors occurred")]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                ds.parse_performance(changed)

    def test_short_or_subset_run_rejected(self):
        for changed in [PERFORMANCE.replace("600000", "60000"), PERFORMANCE.replace("105312", "1024"),
                        PERFORMANCE.replace("count : 4388", "count : 100")]:
            with self.assertRaises(ValueError):
                ds.parse_performance(changed)

    def test_nan_zero_duplicate_throughput_rejected(self):
        for changed in [PERFORMANCE.replace("70325.7", x) for x in ["nan", "inf", "0", "-1"]] + [PERFORMANCE + "Tokens per second : 1\n"]:
            with self.assertRaises(ValueError):
                ds.parse_performance(changed)

    def test_accuracy_below_threshold_or_incomplete_rejected(self):
        for changed in [ACCURACY.replace("81.10756608933455", "70.0"), ACCURACY.replace("4388", "100"),
                        ACCURACY.replace("3500.0", "0"), ACCURACY.replace("3500.0", "5000"), ACCURACY + ACCURACY]:
            with self.assertRaises(ValueError):
                ds.parse_accuracy(changed)

    def test_complete_evidence_is_comparable(self):
        save_evidence(self.out)
        ref = public.compare(self.out)["comparisons"][1]
        self.assertEqual(ref["status"], "WORKLOAD_MATCHED")
        self.assertEqual(ref["metrics"][0]["relative_percent"], 100)
        self.assertIn("本次准确率", public.render(public.compare(self.out)))

    def test_cached_throughput_is_ignored(self):
        result, _, _ = save_evidence(self.out)
        result["performance"] = {"tokens_per_second": 1e9}
        ds.write_json(self.out / "deepseek-mlperf.json", result)
        self.assertEqual(public.compare(self.out)["comparisons"][1]["metrics"][0]["measured"], 70325.7)

    def test_tampered_missing_or_failed_evidence_never_compares(self):
        result, perf, acc = save_evidence(self.out)
        (perf / "mlperf_log_summary.txt").write_text("truncated", encoding="utf-8")
        self.assertEqual(public.compare(self.out)["comparisons"][1]["status"], "INVALID")
        result, perf, acc = save_evidence(self.out)
        (acc / "accuracy.txt").unlink()
        self.assertEqual(public.compare(self.out)["comparisons"][1]["status"], "INVALID")
        result, _, _ = save_evidence(self.out)
        result["status"] = "FAIL"
        ds.write_json(self.out / "deepseek-mlperf.json", result)
        self.assertIsNotNone(public.compare(self.out)["measurement_error"])

    def test_distilled_model_wrong_precision_topology_and_specdecode_rejected(self):
        for key, value in [("model", "deepseek-r1-distill"), ("precision", "fp8"), ("gpus", 4),
                           ("nodes", 2), ("tensor_parallelism", 4), ("speculative_decoding", True), ("source_revision", "0" * 40)]:
            result, _, _ = save_evidence(self.out)
            result["context"][key] = value
            ds.write_json(self.out / "deepseek-mlperf.json", result)
            self.assertEqual(public.compare(self.out)["comparisons"][1]["status"], "INVALID")

    def test_empty_duplicate_or_subset_accuracy_outputs_rejected(self):
        for responses in [[], [{"qsl_idx": 0, "data": "01"}] * 4388,
                          [{"qsl_idx": i, "data": ""} for i in range(4388)]]:
            result, _, acc = save_evidence(self.out)
            path = acc / "mlperf_log_accuracy.json"
            ds.write_json(path, responses)
            result["evidence_sha256"][path.relative_to(self.out).as_posix()] = ds.sha(path)
            ds.write_json(self.out / "deepseek-mlperf.json", result)
            self.assertEqual(public.compare(self.out)["comparisons"][1]["status"], "INVALID")

    def test_baseline_workload_mismatch_does_not_compute_percentage(self):
        save_evidence(self.out)
        baseline = public.read_json(public.BASELINES)
        baseline["references"][1]["workload"]["scenario"] = "Server"
        ds.write_json(self.out / "public-baselines.json", baseline)
        ref = public.compare(self.out)["comparisons"][1]
        self.assertEqual(ref["status"], "CONFIG_MISMATCH")
        self.assertIsNone(ref["metrics"][0]["relative_percent"])

    def test_missing_assets_fail_before_gpu_import(self):
        with self.assertRaisesRegex(RuntimeError, "assets missing"), patch.object(ds.importlib, "import_module") as imports:
            ds.check(self.out)
        imports.assert_not_called()

    def test_missing_assets_write_failed_run(self):
        with patch.object(sys, "argv", ["deepseek_mlperf.py", "run", "--workspace", str(self.out), "--out", str(self.out)]), patch("builtins.print"):
            self.assertEqual(ds.main(), 1)
        self.assertEqual(ds.read_json(self.out / "deepseek-mlperf.json")["status"], "FAIL")

    def test_phase_failure_never_becomes_pass(self):
        with patch.object(sys, "argv", ["deepseek_mlperf.py", "run", "--workspace", str(self.out), "--out", str(self.out)]), \
             patch.object(ds, "check", return_value={}), patch.object(ds, "run_phase", side_effect=RuntimeError("server failed")), patch("builtins.print"):
            self.assertEqual(ds.main(), 1)
        self.assertEqual(ds.read_json(self.out / "deepseek-mlperf.json")["error"], "server failed")

    def test_fresh_server_per_phase_and_cleanup_after_harness_failure(self):
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        proc = unittest.mock.Mock(pid=123)
        proc.poll.return_value = None
        with patch.object(ds.urllib.request, "urlopen", side_effect=[URLError("not started"), Response()] * 2), \
             patch.object(ds.subprocess, "Popen", return_value=proc) as servers, \
             patch.object(ds, "run_harness_process", side_effect=[None, subprocess.CalledProcessError(1, "harness")]), \
             patch.object(ds, "stop") as stopped:
            ds.run_phase(self.out, self.out, "PerformanceOnly", 10, 10)
            with self.assertRaises(subprocess.CalledProcessError):
                ds.run_phase(self.out, self.out, "AccuracyOnly", 10, 10)
        self.assertEqual(servers.call_count, 2)
        self.assertEqual(stopped.call_count, 2)
        self.assertIn("--server_in_foreground=True", servers.call_args_list[0].args[0][-1])

    def test_wrong_reference_units_rejected(self):
        baseline = public.read_json(public.BASELINES)
        baseline["references"][1]["metrics"][0]["unit"] = "Samples/s"
        with self.assertRaises(ValueError):
            public.validate_baselines(baseline)

    def test_wrong_loadgen_version_and_detail_errors_rejected(self):
        for text in [':::MLLOG {"key":"loadgen_version","value":"5.1.0"}\n',
                     ':::MLLOG {"key":"loadgen_version","value":"6.0.9","metadata":{"is_error":true}}\n']:
            result, perf, _ = save_evidence(self.out)
            path = perf / "mlperf_log_detail.txt"
            path.write_text(text, encoding="utf-8")
            result["evidence_sha256"][path.relative_to(self.out).as_posix()] = ds.sha(path)
            ds.write_json(self.out / "deepseek-mlperf.json", result)
            self.assertEqual(public.compare(self.out)["comparisons"][1]["status"], "INVALID")

    def test_optional_model_catalog_is_pinned_and_complete(self):
        root, models = deepseek_assets.catalog(self.out)
        self.assertEqual(root, self.out / "build")
        self.assertEqual(len([f for f in models[0]["files"] if f["path"].endswith(".safetensors")]), 80)
        self.assertEqual(models[0]["revision"], "4bedb8a695a119b1a38d16a675c4665e58708aea")
        for m in models:
            for f in m["files"]:
                self.assertIn(f["hash_type"], ["git-sha1", "sha256"])
                self.assertGreater(f["size"], 0)


class DeepSeekReportTests(unittest.TestCase):
    setUp = kit.ReportTests.setUp
    tearDown = kit.ReportTests.tearDown
    save = kit.ReportTests.save
    snapshot = kit.ReportTests.snapshot
    telemetry = kit.ReportTests.telemetry

    def require_deepseek(self):
        self.plan.update(profile="deepseek", required=["deepseek-r1-mlperf"])
        self.save("plan.json", self.plan)
        (self.out / "stages.tsv").write_text("deepseek-r1-mlperf\tPASS\t0\n", encoding="utf-8")

    def test_deepseek_has_whole_system_evidence_not_replica_files(self):
        import report
        self.require_deepseek()
        save_evidence(self.out)
        self.assertEqual(report.generate(self.out)["overall"], "PASS")

    def test_performance_without_accuracy_is_incomplete_or_fail(self):
        import report
        self.require_deepseek()
        self.assertEqual(report.generate(self.out)["overall"], "INCOMPLETE")
        _, _, acc = save_evidence(self.out)
        (acc / "accuracy.txt").unlink()
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")

    def test_low_throughput_does_not_invent_hardware_failure(self):
        import report
        self.require_deepseek()
        result, perf, _ = save_evidence(self.out)
        path = perf / "mlperf_log_summary.txt"
        path.write_text(PERFORMANCE.replace("70325.7", "35162.85"), encoding="utf-8")
        result["evidence_sha256"][path.relative_to(self.out).as_posix()] = ds.sha(path)
        ds.write_json(self.out / "deepseek-mlperf.json", result)
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "PASS")
        self.assertEqual(summary["public_comparison"]["comparisons"][1]["metrics"][0]["relative_percent"], 50)


if __name__ == "__main__":
    unittest.main()
