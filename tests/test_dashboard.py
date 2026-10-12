"""Local HTTP and process regressions; all hardware fixtures are synthetic.

No browser preview, real GPU work or benchmark measurements are created here.
"""
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard


def temporary_workspace():
    base = Path(__file__).resolve().parent / ".test-tmp"
    base.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


def readiness_fixture(count=0):
    """Private schema fixture, never stored as hardware observations."""
    return dict(synthetic_only=True,
                gpus=[dict(index=i, name="SYNTHETIC NVIDIA B300", uuid=f"synthetic-{i}",
                           mig="Disabled", ecc="Enabled") for i in range(count)],
                tools=dict(python3=True, docker=True, nvidia_smi=True, dcgmi=True),
                image=dict(available=True, id="synthetic-only", architecture="amd64", tag="synthetic"),
                bundle=dict(sealed=True, hardware_sealed=True, archive_available=True),
                deepseek=dict(configured=True, fields={}), busy_pids=[],
                busy_check_available=True, warnings=["CPU fixture only; no observed hardware"])


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_workspace()
        self.root = Path(self.temporary.name).resolve()
        self.manager = dashboard.JobManager(self.root / "runs", preview=True)
        self.ready_patch = patch.object(self.manager, "ready", return_value=readiness_fixture())
        self.ready_patch.start()
        self.server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(self.manager))
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.manager.active = None  # Only private metadata fixtures; no process was started.
        self.manager.close()
        self.ready_patch.stop()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None, authenticated=False):
        supplied = dict(headers or {})
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        if body is not None:
            supplied.setdefault("Content-Type", "application/json")
        if authenticated:
            supplied.setdefault("X-CSRF-Token", self.manager.token)
            supplied.setdefault("Origin", f"http://127.0.0.1:{self.port}")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        try:
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def add_job(self):
        key = "20261012-123456-0123abcd"
        out = self.manager.root / key
        out.mkdir()
        job = dict(id=key, profile="identity", status="FAIL", started_at="2026-10-12T00:00:00Z",
                   ended_at="2026-10-12T00:00:01Z", exit_code=1, cancelled=False,
                   thermal_stop=False, seconds=600, synthetic_only=True)
        self.manager.jobs[key] = job
        dashboard.write_json(out / "job.json", job)
        return key, out

    def test_host_is_checked_on_read_and_mutation_routes(self):
        for method, path, body in (("GET", "/api/session", None), ("GET", "/api/state", None),
                                   ("POST", "/api/jobs", dict(profile="identity"))):
            with self.subTest(method=method, path=path):
                status, _, _ = self.request(method, path, body, headers={"Host": "external.example"}, authenticated=True)
                self.assertEqual(status, 403)
        status, _, raw = self.request("GET", "/api/session", headers={"Host": f"localhost:{self.port}"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["csrf_token"], self.manager.token)

    def test_origin_and_csrf_are_both_required_for_browser_mutations(self):
        for headers in ({}, {"X-CSRF-Token": "wrong"},
                        {"X-CSRF-Token": self.manager.token, "Origin": "https://external.example"},
                        {"X-CSRF-Token": self.manager.token, "Origin": "null"}):
            with self.subTest(headers=headers):
                status, _, _ = self.request("POST", "/api/jobs", dict(profile="identity"), headers=headers)
                self.assertEqual(status, 403)
        self.assertFalse(self.manager.jobs)

    def test_json_type_size_and_invalid_body_are_rejected(self):
        status, _, _ = self.request("POST", "/api/jobs", b"{}", headers={"Content-Type": "text/plain"}, authenticated=True)
        self.assertEqual(status, 415)
        for body in (b"x" * 4097, b""):
            status, _, _ = self.request("POST", "/api/jobs", body, authenticated=True)
            self.assertEqual(status, 413)
        for body in (b"{incomplete", [], dict(profile="does-not-exist"), dict(profile=["identity"]),
                     dict(profile="identity", seconds=True), dict(profile="identity", seconds=86401)):
            with self.subTest(body=body):
                status, _, _ = self.request("POST", "/api/jobs", body, authenticated=True)
                self.assertEqual(status, 400)
        self.assertFalse(self.manager.jobs)

    def test_preview_is_read_only_even_when_fixture_has_eight_gpus(self):
        with patch.object(self.manager, "ready", return_value=readiness_fixture(8)), patch.object(dashboard.sys, "platform", "linux"):
            status, _, raw = self.request("POST", "/api/jobs", dict(profile="identity"), authenticated=True)
        self.assertEqual(status, 400)
        self.assertIn("只读", json.loads(raw)["error"])
        self.assertFalse(self.manager.jobs)
        self.assertEqual(list(self.manager.root.iterdir()), [])

    def test_get_cannot_start_or_cancel_a_job(self):
        for path in ("/api/jobs", "/api/jobs/anything/stop"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)
        self.assertFalse(self.manager.jobs)

    def test_controller_refuses_second_active_job_with_409(self):
        key, _ = self.add_job()
        self.manager.active = key
        status, _, raw = self.request("POST", "/api/jobs", dict(profile="quick"), authenticated=True)
        self.assertEqual(status, 409)
        self.assertIn("active", json.loads(raw)["error"])
        self.assertEqual(len(self.manager.jobs), 1)

    def test_unknown_job_is_404_and_inactive_stop_is_400(self):
        key, _ = self.add_job()
        status, _, _ = self.request("GET", "/api/jobs/missing/log")
        self.assertEqual(status, 404)
        status, _, _ = self.request("POST", f"/api/jobs/{key}/stop", {}, authenticated=True)
        self.assertEqual(status, 400)

    def test_exporting_job_cannot_be_cancelled_or_mutated(self):
        key, _ = self.add_job()
        self.manager.jobs[key]["status"] = "EXPORTING"
        self.manager.active = key
        status, _, _ = self.request("POST", f"/api/jobs/{key}/stop", {}, authenticated=True)
        self.assertEqual(status, 400)
        self.assertEqual(self.manager.jobs[key]["status"], "EXPORTING")
        self.assertFalse(self.manager.jobs[key]["cancelled"])

    def test_logs_remain_literal_data_and_report_html_is_sandboxed(self):
        key, out = self.add_job()
        content = "Synthetic log only\n<script>alert('must be literal')</script>"
        (out / "launcher.log").write_bytes(content.encode("utf-8"))
        (out / "report.html").write_text("<html>CPU fixture report only</html>", encoding="utf-8")
        status, headers, raw = self.request("GET", f"/api/jobs/{key}/log")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(json.loads(raw)["log"], content)
        status, headers, raw = self.request("GET", f"/runs/{key}/launcher.log")
        self.assertEqual((status, raw.decode()), (200, content))
        self.assertNotIn("text/html", headers["Content-Type"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        status, headers, _ = self.request("GET", f"/runs/{key}/report.html")
        self.assertEqual(status, 200)
        self.assertIn("sandbox", headers["Content-Security-Policy"])

    def test_partial_result_json_and_stage_rows_do_not_crash_state(self):
        key, out = self.add_job()
        (out / "summary.json").write_text('{"overall":', encoding="utf-8")
        (out / "identity-summary.json").write_text('{"identity_status":', encoding="utf-8")
        (out / "stages.tsv").write_text("environment\tRUNNING\t0\npartial\trow\nstress\tFAIL\tnot-an-int\n", encoding="utf-8")
        (out / "environment.log").write_text("Synthetic stage log", encoding="utf-8")
        status, _, raw = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        job = next(job for job in json.loads(raw)["jobs"] if job["id"] == key)
        self.assertIsNone(job["summary"])
        self.assertIsNone(job["identity_summary"])
        self.assertEqual(len(job["stages"]), 1)
        self.assertIn("Synthetic stage log", job["log_tail"])

    def test_encoded_traversal_and_null_paths_do_not_expose_files(self):
        key, _ = self.add_job()
        (self.root / "outside-secret.txt").write_text("private-outside-fixture", encoding="utf-8")
        for suffix in ("%2e%2e/outside-secret.txt", "sub/%2e%2e/%2e%2e/outside-secret.txt", "%00"):
            with self.subTest(suffix=suffix):
                status, _, raw = self.request("GET", f"/runs/{key}/{suffix}")
                self.assertIn(status, (400, 403, 404))
                self.assertNotIn(b"private-outside-fixture", raw)

    def test_external_file_and_directory_symlinks_are_blocked(self):
        key, out = self.add_job()
        external = self.root / "external"
        external.mkdir()
        secret = external / "private.txt"
        secret.write_text("private-outside-fixture", encoding="utf-8")
        try:
            (out / "linked-file.txt").symlink_to(secret)
            (out / "linked-directory").symlink_to(external, target_is_directory=True)
            (out / "launcher.log").symlink_to(secret)
            (out / "summary.json").symlink_to(secret)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"OS does not permit symlink fixture: {error}")
        for suffix in ("linked-file.txt", "linked-directory/private.txt"):
            status, _, raw = self.request("GET", f"/runs/{key}/{suffix}")
            self.assertEqual(status, 403)
            self.assertNotIn(b"private-outside-fixture", raw)
        for path in (f"/api/jobs/{key}/log", "/api/state"):
            status, _, raw = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertNotIn(b"private-outside-fixture", raw)

    def test_links_in_stage_log_ancestors_are_not_followed(self):
        external = self.root / "outside-run"
        external.mkdir()
        (external / "launcher.log").write_text("private-outside-fixture", encoding="utf-8")
        try:
            linked = self.manager.root / "linked-run"
            linked.symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"OS does not permit symlink fixture: {error}")
        self.assertEqual(dashboard.tail(linked / "launcher.log"), "")
        (external / "summary.json").write_text('{"private":"outside-fixture"}', encoding="utf-8")
        self.assertIsNone(dashboard.optional_json(linked / "summary.json"))


class DashboardEligibilityTests(unittest.TestCase):
    def test_eight_idle_b300_fixture_enables_all_prepared_profiles(self):
        with patch.object(dashboard.sys, "platform", "linux"):
            for profile in dashboard.PROFILES:
                with self.subTest(profile=profile["id"]):
                    self.assertTrue(dashboard.profile_eligibility(profile, readiness_fixture(8))["available"])

    def test_gpu_profiles_fail_closed_on_count_model_mig_ecc_or_busy(self):
        cases = [readiness_fixture(7), readiness_fixture(9)]
        for key, value in (("name", "NVIDIA GB300"), ("mig", "Enabled"), ("ecc", "N/A")):
            ready = readiness_fixture(8)
            ready["gpus"][0][key] = value
            cases.append(ready)
        ready = readiness_fixture(8)
        ready["busy_pids"] = ["synthetic-pid"]
        cases.append(ready)
        ready = readiness_fixture(8)
        ready["busy_check_available"] = False
        cases.append(ready)
        with patch.object(dashboard.sys, "platform", "linux"):
            for ready in cases:
                with self.subTest(ready=ready):
                    result = dashboard.profile_eligibility(dashboard.PROFILE_MAP["authenticity"], ready)
                    self.assertFalse(result["available"])
                    self.assertTrue(result["unavailable_reason"])

    def test_missing_bundle_and_deepseek_dependencies_affect_their_profiles(self):
        ready = readiness_fixture(8)
        ready["bundle"]["sealed"] = False
        ready["deepseek"]["configured"] = False
        with patch.object(dashboard.sys, "platform", "linux"):
            self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP["identity"], ready)["available"])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP["quick"], ready)["available"])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP["deepseek"], ready)["available"])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP["identity"], ready, preview=True)["available"])

    def test_host_identity_collection_does_not_require_prevalidated_gpu(self):
        ready = readiness_fixture()
        ready["tools"]["docker"] = False
        ready["image"]["available"] = False
        ready["busy_check_available"] = False
        with patch.object(dashboard.sys, "platform", "linux"):
            self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP["identity"], ready)["available"])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP["authenticity"], ready)["available"])


@unittest.skipUnless(sys.platform == "linux", "Requires real Linux process groups and flock")
class DashboardProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = temporary_workspace()
        self.root = Path(self.temporary.name).resolve()
        self.manager = dashboard.JobManager(self.root / "runs")
        self.patches = [patch.object(self.manager, "ready", return_value=readiness_fixture(8)),
                        patch.object(self.manager, "_watch", side_effect=lambda _job, done: done.wait()),
                        patch.object(self.manager, "_cleanup_containers", return_value=None)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.manager.close()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def wait_for(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("Timed out waiting for real CPU child process")
            time.sleep(.02)

    def test_cancel_exits_process_group_and_exports_final_record(self):
        script = """import pathlib,signal,subprocess,sys,time
out=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
def stop(*args):
    child.terminate()
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait()
    raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
(out/'child-pid.txt').write_text(str(child.pid))
print('Synthetic CPU process cancellation fixture only',flush=True)
time.sleep(120)
"""
        with patch.object(self.manager, "command", side_effect=lambda _profile, out, _seconds: [sys.executable, "-u", "-c", script, str(out)]):
            key = self.manager.start(dict(profile="identity"))
            out = self.manager.root / key
            self.wait_for(lambda: (out / "child-pid.txt").is_file())
            process = self.manager.process
            self.assertIsNotNone(process)
            pgid = process.pid
            other = dashboard.JobManager(self.root / "second-controller")
            with patch.object(other, "ready", return_value=readiness_fixture(8)):
                with self.assertRaisesRegex(RuntimeError, "另一个控制器"):
                    other.start(dict(profile="authenticity"))
            other.close()
            self.manager.stop(key)
            self.manager.worker.join(timeout=15)
        self.assertFalse(self.manager.worker.is_alive())
        self.assertIsNotNone(process.poll())
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)
        self.assertEqual(self.manager.jobs[key]["status"], "CANCELLED")
        self.assertIsNone(self.manager.active)
        final_bytes = (out / "run-result.json").read_bytes()
        self.assertEqual(json.loads(final_bytes)["status"], "CANCELLED")
        self.assertEqual(json.loads(final_bytes), json.loads((out / "job.json").read_bytes()))
        manifest = json.loads((out / "evidence-manifest.json").read_text())
        self.assertNotIn("job.json", manifest["files"])
        self.assertEqual(manifest["files"]["run-result.json"]["sha256"], hashlib.sha256(final_bytes).hexdigest())
        with zipfile.ZipFile(out / "evidence.zip") as archive:
            self.assertEqual(archive.read("run-result.json"), final_bytes)
        with self.assertRaisesRegex(ValueError, "not active"):
            self.manager.stop(key)

    def test_nonzero_exit_cannot_be_overridden_by_pass_summary(self):
        script = "import json,pathlib,sys; pathlib.Path(sys.argv[1],'summary.json').write_text(json.dumps({'overall':'PASS','synthetic_only':True})); sys.exit(7)"
        with patch.object(self.manager, "command", side_effect=lambda _profile, out, _seconds: [sys.executable, "-c", script, str(out)]):
            key = self.manager.start(dict(profile="identity"))
            self.manager.worker.join(timeout=15)
        self.assertFalse(self.manager.worker.is_alive())
        self.assertEqual(self.manager.jobs[key]["exit_code"], 7)
        self.assertEqual(self.manager.jobs[key]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
