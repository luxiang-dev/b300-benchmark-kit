"""CPU fault-injection for authenticity workflow/report, using synthetic hardware.

These fixtures never execute CUDA, Docker or DCGM, and never represent hardware
measurements. They check report truthfulness and launcher error propagation.
"""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import authenticity
import memory_probe
import report
import run_authenticity as launcher
import test_authenticity as identity_fixture


def arguments(out, profile="authenticity", gpus=2):
    return SimpleNamespace(out=out, profile=profile, gpus=gpus, seconds=1,
                           image="synthetic-image:never-pull", memory_fraction=0.85, max_temp=85)


class WorkflowReportTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / ".test-tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.out = Path(self.temp.name) / "results with spaces"
        self.out.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        authenticity.write_json(self.out / name, value)

    def identities(self, count=2, **xml_args):
        with patch.object(authenticity.platform, "system", return_value="Linux"), \
             patch.object(authenticity.platform, "release", return_value="synthetic-kernel"), \
             patch.object(authenticity.platform, "machine", return_value="x86_64"), \
             patch.object(authenticity.platform, "node", return_value="synthetic-host"):
            return identity_fixture.IdentityTests.collect_fixture(self, count=count, **xml_args)

    def hardware(self, count=2):
        xml = identity_fixture.fixture_xml(count).decode().replace("</gpu>",
              "<ecc_errors><volatile><dram_uncorrectable>0</dram_uncorrectable></volatile></ecc_errors></gpu>")
        for name in ("before.xml", "after.xml"):
            (self.out / name).write_text(xml, encoding="utf-8")
        (self.out / "kernel-during.log").write_text("kernel: explicitly synthetic test-window event\n", encoding="utf-8")
        rows = [f"2026/10/12 10:00:0{second}.000,{rank},GPU-00000000-0000-0000-0000-{rank + 1:012x},50,500,1000,98\n"
                for second in (0, 2) for rank in range(count)]
        (self.out / "telemetry.csv").write_text("timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n" + "".join(rows), encoding="utf-8")
        start = datetime(2026, 10, 12, 10, 0, 0).timestamp()
        self.save("telemetry-window.json", dict(start_epoch=start, end_epoch=start+3))
        shutil.copyfile(launcher.ROOT / "public_baselines.json", self.out / "public-baselines.json")

    def launch_host(self, host):
        # Reproduce collection during this run, rather than bypassing the new
        # production guard against reusing stale result directories.
        paths = list(self.out.rglob('*'))
        files = {p.relative_to(self.out): p.read_bytes() for p in paths if p.is_file()}
        for p in sorted(paths, key=lambda p: len(p.parts), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        def restore(_out):
            for name, raw in files.items():
                target = self.out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            return host
        return restore

    def report_fixture(self, required=None, count=2):
        self.identities(count)
        self.hardware(count)
        plan = dict(profile="authenticity", gpus=count, required=required or ["identity"], max_gpu_temp=85)
        self.save("plan.json", plan)
        (self.out / "stages.tsv").write_text("identity\tPASS\t0\n", encoding="utf-8")
        return plan

    def stage(self, name, status="PASS", code=0):
        with (self.out / "stages.tsv").open("a", encoding="utf-8") as stream:
            stream.write(f"{name}\t{status}\t{code}\n")

    def memory_result(self, rank):
        allocated = memory_probe.allocation_budget(270 * 2**30, 269 * 2**30, 0.85)
        return dict(test="memory", rank=rank, status="PASS", model="", synthetic_only=True,
                    dtype="int32", patterns=list(memory_probe.PATTERNS), total_memory_bytes=270 * 2**30,
                    free_memory_before_bytes=269 * 2**30, requested_memory_fraction=0.85,
                    reserve_bytes=memory_probe.RESERVE_BYTES, scratch_bytes=memory_probe.CHUNK_BYTES,
                    chunk_bytes=memory_probe.CHUNK_BYTES, target_allocation_bytes=allocated,
                    allocated_bytes=allocated, verified_bytes_per_pass=allocated,
                    total_verified_bytes=allocated * 3, coverage_percent=allocated / (270 * 2**30) * 100,
                    passes=[dict(pattern=pattern, status="PASS", written_bytes=allocated, verified_bytes=allocated,
                                 write_seconds=1, verify_seconds=1) for pattern in memory_probe.PATTERNS],
                    duration_s=6, memory_tested_gib=allocated / 2**30, peak_temperature_c=50, free_memory_limited=False)

    def test_report_contains_revalidated_identity_and_actual_capacity(self):
        self.report_fixture(["identity", "memory"])
        self.stage("memory")
        for rank in range(2):
            self.save(f"memory-gpu{rank}.json", self.memory_result(rank))
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "PASS", summary)
        self.assertEqual(summary["identity_verification"]["status"], "CONSISTENT")
        self.assertEqual(summary["identity_verification"]["vendor_confirmation"], "NOT_PERFORMED")
        memory = [item for item in summary["results"] if item["test"] == "memory"]
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory[0]["total_verified_bytes"], memory[0]["allocated_bytes"] * 3)
        page = (self.out / "report.html").read_text(encoding="utf-8")
        self.assertIn("硬件身份交叉核验", page)
        self.assertIn("inverted-address", page)
        self.assertIn("不能替代", page)
        self.assertIn("<title>B300 2卡测试报告</title>", page)
        self.assertIn("<h1>NVIDIA B300 · 2卡测试报告</h1>", page)
        self.assertNotIn("<h1>NVIDIA B300 · 8卡", page)

    def test_incomplete_or_inflated_memory_claims_cannot_pass(self):
        self.report_fixture(["identity", "memory"])
        self.stage("memory")
        self.save("memory-gpu1.json", self.memory_result(1))
        for case in ("bare-pass", "capacity-is-cumulative", "wrong-byte-total", "missing-pattern",
                     "partial-read", "repeated-pattern", "fake-full-coverage", "bool-capacity"):
            record = self.memory_result(0)
            if case == "bare-pass":
                record = dict(test="memory", rank=0, status="PASS", model="")
            elif case == "capacity-is-cumulative":
                record["allocated_bytes"] = record["total_verified_bytes"]
            elif case == "wrong-byte-total":
                record["total_verified_bytes"] -= 4
            elif case == "missing-pattern":
                record["passes"].pop()
            elif case == "partial-read":
                record["passes"][1]["verified_bytes"] -= 4
            elif case == "repeated-pattern":
                record["passes"][1]["pattern"] = record["passes"][0]["pattern"]
            elif case == "fake-full-coverage":
                record["coverage_percent"] = 100
            elif case == "bool-capacity":
                record["allocated_bytes"] = True
            with self.subTest(case=case):
                self.save("memory-gpu0.json", record)
                summary = report.generate(self.out)
                self.assertEqual(summary["overall"], "FAIL", summary)

    def test_bool_or_out_of_range_rank_is_invalid_evidence(self):
        self.report_fixture(["identity", "environment"])
        self.stage("environment")
        self.save("environment-gpu0.json", dict(test="environment", rank=0, status="PASS", model=""))
        for rank in (True, -1, 2):
            with self.subTest(rank=rank):
                self.save("environment-gpu1.json", dict(test="environment", rank=rank, status="PASS", model=""))
                self.assertEqual(report.generate(self.out)["overall"], "FAIL")

    def test_missing_cuda_cannot_use_cached_consistent_verdict(self):
        self.report_fixture()
        authenticity.evaluate_identity(self.out, 2)
        (self.out / "identity-gpu1.json").unlink()
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "REVIEW", summary)
        self.assertEqual(summary["identity_verification"]["status"], "REVIEW")
        self.assertIn("cuda_evidence_missing", {f["code"] for f in summary["identity_verification"]["findings"]})

    def test_identity_cuda_contradiction_overrides_successful_stage(self):
        self.report_fixture()
        path = self.out / "identity-gpu0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["capability"] = [10, 0]
        authenticity.write_json(path, raw)
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "FAIL")
        self.assertEqual(summary["identity_verification"]["status"], "FAIL")

    def test_memory_success_must_cover_every_rank(self):
        self.report_fixture(["identity", "memory"])
        self.stage("memory")
        self.save("memory-gpu0.json", self.memory_result(0))
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "INCOMPLETE")
        self.assertTrue(any("memory" in item for item in summary["incomplete"]))

    def test_boolean_identity_rank_cannot_impersonate_worker_one(self):
        self.identities(2)
        path = self.out / 'identity-gpu1.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['rank'] = True
        self.save(path.name, value)
        self.assertEqual(authenticity.evaluate_identity(self.out, 2)['status'], 'FAIL')

    def test_public_comparison_and_dcgm_skip_remain_incomplete(self):
        self.report_fixture(["identity", "nccl-public", "dcgm"])
        self.stage("nccl-public", "SKIP")
        self.stage("dcgm", "SKIP")
        summary = report.generate(self.out)
        self.assertEqual(summary["overall"], "INCOMPLETE")
        self.assertTrue(any("nccl-public" in item for item in summary["incomplete"]))
        self.assertTrue(any("dcgm" in item for item in summary["incomplete"]))

    def test_nonzero_process_result_records_failure_and_raises(self):
        workflow = launcher.Workflow(arguments(self.out))
        process = MagicMock()
        process.poll.return_value = 42
        process.returncode = 42
        with patch.object(launcher.subprocess, "Popen", return_value=process), patch.object(workflow, "stop_container") as stopped:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                workflow.execute("identity-cuda", ["synthetic-command"])
        self.assertEqual(caught.exception.returncode, 42)
        self.assertIn("identity-cuda\tFAIL\t42", (self.out / "stages.tsv").read_text(encoding="utf-8"))
        self.assertIsNone(workflow.current)
        stopped.assert_called_once()

    def test_container_uses_local_image_pinned_id_no_network(self):
        workflow = launcher.Workflow(arguments(self.out))
        workflow.image_id = "sha256:" + "0" * 64
        command = workflow.torchrun("authenticity.py", "cuda")
        self.assertEqual(command[command.index("--pull") + 1], "never")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn(workflow.image_id, command)
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("/kit/authenticity.py", command)
        self.assertEqual(command[-4:], ["--out", "/out", "--gpus", "2"])

    def test_telemetry_missing_gpu_cancels_before_any_positive_result(self):
        self.hardware(2)
        workflow = launcher.Workflow(arguments(self.out))
        sample = "2026/10/12 10:00:00.000,0,GPU-SYNTHETIC,50,500,1000,98\n"
        completed = subprocess.CompletedProcess(["nvidia-smi"], 0, sample, "")
        with patch.object(launcher.subprocess, "run", return_value=completed), patch.object(workflow, "stop_current") as stopped:
            workflow.telemetry()
        self.assertTrue(workflow.cancelled.is_set())
        self.assertIn("expected GPU count", workflow.watchdog_error)
        stopped.assert_called_once()
        self.assertTrue((self.out / "telemetry.stderr").exists())

    def test_watchdog_rejects_duplicate_uuid_and_invalid_live_values(self):
        self.hardware(2)
        base_rows = [["2026/10/12 10:00:00.000", str(rank), f"GPU-00000000-0000-0000-0000-{rank + 1:012x}", "50", "500", "1000", "98"] for rank in range(2)]
        mutations = (("duplicate-uuid", 1, 2, base_rows[0][2]), ("unknown-uuid", 1, 2, "GPU-UNKNOWN"),
                     ("duplicate-index", 1, 1, "0"), ("negative-temperature", 1, 3, "-1"),
                     ("invalid-temperature", 1, 3, "120"), ("nonfinite-power", 1, 4, "nan"),
                     ("negative-power", 1, 4, "-1"), ("negative-memory", 1, 5, "-1"),
                     ("invalid-utilization", 1, 6, "101"))
        for name, row, column, value in mutations:
            with self.subTest(case=name):
                rows = [list(item) for item in base_rows]
                rows[row][column] = value
                sample = "".join(",".join(item) + "\n" for item in rows)
                completed = subprocess.CompletedProcess(["nvidia-smi"], 0, sample, "")
                workflow = launcher.Workflow(arguments(self.out))
                workflow.expected_uuids = [item[2] for item in base_rows]
                with patch.object(launcher.subprocess, "run", return_value=completed), \
                     patch.object(workflow.monitor_done, "wait", side_effect=lambda _seconds: workflow.monitor_done.set()), \
                     patch.object(workflow, "stop_current") as stopped:
                    workflow.telemetry()
                self.assertTrue(workflow.cancelled.is_set(), name)
                self.assertTrue(workflow.watchdog_error, name)
                stopped.assert_called()

    def test_host_only_identity_fallback_is_review_and_keeps_evidence(self):
        workflow = launcher.Workflow(arguments(self.out, profile="identity"))
        host = self.identities(2)
        for path in self.out.glob("identity-gpu*.json"):
            path.unlink()
        def capture(argv, name, timeout=20):
            self.hardware(2)
            return subprocess.CompletedProcess(argv, 0, "synthetic host capture", "")
        with patch.object(launcher.authenticity, "collect_host", side_effect=self.launch_host(host)), \
             patch.object(workflow, "preflight", side_effect=RuntimeError("synthetic local Docker image missing")), \
             patch.object(workflow, "capture", side_effect=capture), \
             patch.object(workflow, "stop_current"), patch.object(workflow, "stop_container"), \
             patch.object(launcher.signal, "signal"), patch.object(workflow, "execute") as execute:
            status = workflow.run()
        self.assertEqual(status, 2)
        execute.assert_not_called()
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["overall"], "REVIEW")
        self.assertEqual(summary["identity_verification"]["status"], "REVIEW")
        self.assertTrue((self.out / "identity-cuda-unavailable.txt").is_file())
        self.assertTrue((self.out / "evidence.zip").is_file())

    def test_host_only_fallback_hardware_failure_must_return_failure(self):
        workflow = launcher.Workflow(arguments(self.out, profile="identity"))
        host = self.identities(2)
        for path in self.out.glob("identity-gpu*.json"):
            path.unlink()
        def capture(argv, name, timeout=20):
            self.hardware(2)
            (self.out / "kernel-during.log").write_text("NVRM: Xid (PCI:0000:0a:00): 79, synthetic fault", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "synthetic capture", "")
        with patch.object(launcher.authenticity, "collect_host", side_effect=self.launch_host(host)), \
             patch.object(workflow, "preflight", side_effect=RuntimeError("synthetic image missing")), \
             patch.object(workflow, "capture", side_effect=capture), \
             patch.object(workflow, "stop_current"), patch.object(workflow, "stop_container"), \
             patch.object(launcher.signal, "signal"):
            status = workflow.run()
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8"))["overall"], "FAIL")
        self.assertEqual(status, 1)

    def test_host_only_wrong_model_is_failed_identity(self):
        workflow = launcher.Workflow(arguments(self.out, profile="identity"))
        host = self.identities(2, capability_name="NVIDIA H100")
        for path in self.out.glob("identity-gpu*.json"):
            path.unlink()
        def capture(argv, name, timeout=20):
            self.hardware(2)
            return subprocess.CompletedProcess(argv, 0, "synthetic capture", "")
        with patch.object(launcher.authenticity, "collect_host", side_effect=self.launch_host(host)), \
             patch.object(workflow, "preflight", side_effect=RuntimeError("synthetic model mismatch")), \
             patch.object(workflow, "capture", side_effect=capture), \
             patch.object(workflow, "stop_current"), patch.object(workflow, "stop_container"), \
             patch.object(launcher.signal, "signal"), patch.object(workflow, "execute") as execute:
            status = workflow.run()
        execute.assert_not_called()
        self.assertEqual(status, 1)
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["overall"], "FAIL")
        self.assertEqual(summary["identity_verification"]["status"], "FAIL")

    def test_host_only_report_error_upgrades_incomplete_exit_to_failure(self):
        workflow = launcher.Workflow(arguments(self.out, profile="identity"))
        host = self.identities(2)
        for path in self.out.glob("identity-gpu*.json"):
            path.unlink()
        def capture(argv, name, timeout=20):
            self.hardware(2)
            return subprocess.CompletedProcess(argv, 0, "synthetic capture", "")
        with patch.object(launcher.authenticity, "collect_host", side_effect=self.launch_host(host)), \
             patch.object(workflow, "preflight", side_effect=RuntimeError("synthetic image missing")), \
             patch.object(workflow, "capture", side_effect=capture), \
             patch.object(workflow, "stop_current"), patch.object(workflow, "stop_container"), \
             patch.object(launcher.signal, "signal"), \
             patch.object(launcher.report, "generate", side_effect=ValueError("synthetic malformed report evidence")):
            status = workflow.run()
        self.assertEqual(status, 1)
        self.assertIn("synthetic malformed report evidence", (self.out / "report-error.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.out / "evidence.zip").is_file())

    def test_preflight_failure_still_writes_report_and_export(self):
        workflow = launcher.Workflow(arguments(self.out))
        host = self.identities(2)
        def capture(_argv, name, timeout=20):
            self.hardware(2)
            return subprocess.CompletedProcess(_argv, 0, "synthetic capture", "")
        with patch.object(launcher.authenticity, "collect_host", side_effect=self.launch_host(host)), \
             patch.object(workflow, "preflight", side_effect=RuntimeError("synthetic Docker unavailable")), \
             patch.object(workflow, "capture", side_effect=capture), \
             patch.object(workflow, "stop_current"), patch.object(workflow, "stop_container"), \
             patch.object(launcher.signal, "signal"), patch.object(workflow, "execute") as execute:
            status = workflow.run()
        self.assertEqual(status, 1)
        execute.assert_not_called()
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8"))["overall"], "FAIL")
        self.assertTrue((self.out / "evidence.zip").is_file())
        self.assertIn("synthetic Docker unavailable", (self.out / "workflow-error.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
