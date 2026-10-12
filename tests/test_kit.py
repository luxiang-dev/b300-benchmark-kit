"""CPU-only fault-injection tests; does not simulate GPU benchmark results."""
import hashlib
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_models as dl
import report
import bundle
import image_constraints


def workspace_temp():
    base = Path(__file__).resolve().parent / ".test-tmp"
    base.mkdir(exist_ok=True)
    temp = tempfile.TemporaryDirectory(dir=base)
    assert Path(temp.name).resolve().is_relative_to(base.resolve())
    return temp


class Response(io.BytesIO):
    def __init__(self, data, status, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.root = Path(self.tmp.name)
        self.model = {"name": "sample", "repo": "owner/sample", "revision": "a" * 40}
        self.data = b"safetensors-test-data"
        self.item = {"path": "weights/test.safetensors", "size": len(self.data),
                     "hash_type": "sha256", "hash": hashlib.sha256(self.data).hexdigest()}
        self.path = self.root / "models/sample/weights/test.safetensors"
        self.path.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_with_range(self):
        self.path.with_name(self.path.name + ".part").write_bytes(self.data[:5])
        def request(req, **kw):
            self.assertEqual(req.get_header("Range"), "bytes=5-")
            self.assertIn(self.model["revision"], req.full_url)
            return Response(self.data[5:], 206, {"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})
        with patch.object(dl.urllib.request, "urlopen", side_effect=request):
            dl.download_one(self.root, self.model, self.item)
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_server_ignores_range_restarts(self):
        self.path.with_name(self.path.name + ".part").write_bytes(self.data[:5])
        with patch.object(dl.urllib.request, "urlopen", return_value=Response(self.data, 200)):
            dl.download_one(self.root, self.model, self.item)
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_bad_hash_is_never_promoted(self):
        with patch.object(dl.urllib.request, "urlopen", side_effect=lambda *a, **k: Response(b"x" * len(self.data), 200)), patch.object(dl.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "hash check"):
                dl.download_one(self.root, self.model, self.item)
        self.assertFalse(self.path.exists())

    def test_missing_and_corrupted_weights(self):
        model = {**self.model, "files": [self.item]}
        self.assertTrue(dl.verify(self.root, [model]))
        self.path.write_bytes(self.data)
        self.assertFalse(dl.verify(self.root, [model]))
        self.path.write_bytes(b"x" * len(self.data))
        self.assertTrue(dl.verify(self.root, [model]))

    def test_git_blob_hash(self):
        self.path.write_bytes(self.data)
        expected = hashlib.sha1(f"blob {len(self.data)}\0".encode() + self.data).hexdigest()
        self.assertEqual(dl.file_hash(self.path, git=True), expected)

    def test_path_traversal_blocked(self):
        for name in ("../escape", "/absolute", "dir/../../escape", "C:/escape", "dir\\escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dl.safe_path(self.root, name)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = workspace_temp()
        self.out = Path(self.tmp.name)
        self.plan = {"profile": "quick", "gpus": 8, "required": ["environment"]}
        self.save("plan.json", self.plan)
        (self.out / "stages.tsv").write_text("environment\tPASS\t0\n", encoding="utf-8")
        for rank in range(8):
            self.save(f"environment-gpu{rank}.json", {"test": "environment", "rank": rank, "model": "", "status": "PASS", "note": "<script>bad()</script>"})
        self.snapshot("before.xml", 0)
        self.snapshot("after.xml", 0)
        (self.out / "kernel-during.log").write_text("kernel: normal test-window event\n", encoding="utf-8")
        self.telemetry()

    def telemetry(self):
        rows = [f"2026/10/12 10:00:0{second}.000,{rank},gpu{rank},50,500,1000,98\n"
                for second in (0, 2) for rank in range(8)]
        (self.out / "telemetry.csv").write_text("timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n" + "".join(rows), encoding="utf-8")
        start = datetime(2026, 10, 12, 10, 0, 0).timestamp()
        self.save("telemetry-window.json", dict(start_epoch=start, end_epoch=start+3))

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, data):
        (self.out / name).write_text(json.dumps(data), encoding="utf-8")

    def snapshot(self, name, first):
        xml = "<nvidia_smi_log>" + "".join(f"<gpu><uuid>gpu{rank}</uuid><ecc_errors><volatile><dram_uncorrectable>{first if rank==0 else 0}</dram_uncorrectable></volatile></ecc_errors></gpu>" for rank in range(8)) + "</nvidia_smi_log>"
        (self.out / name).write_text(xml, encoding="utf-8")

    def test_complete_results_and_html_escaping(self):
        self.assertEqual(report.generate(self.out)["overall"], "PASS")
        page = (self.out / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_missing_rank_cannot_pass(self):
        (self.out / "environment-gpu7.json").unlink()
        self.assertEqual(report.generate(self.out)["overall"], "INCOMPLETE")

    def test_ecc_delta_overrides_successful_exit(self):
        self.snapshot("after.xml", 1)
        result = report.generate(self.out)
        self.assertEqual(result["overall"], "FAIL")
        self.assertTrue(any("增加 1" in s for s in result["failures"]))

    def test_xid_overrides_test_success(self):
        (self.out / "kernel-during.log").write_text("NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus", encoding="utf-8")
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")

    def test_dcgm_skip_not_silent_pass(self):
        self.plan["required"].append("dcgm")
        self.save("plan.json", self.plan)
        with (self.out / "stages.tsv").open("a") as f:
            f.write("dcgm\tPASS\t0\n")
        self.save("dcgm.json", {"results": [{"status": "Pass"}, {"status": "Skip"}]})
        self.assertEqual(report.generate(self.out)["overall"], "REVIEW")
        self.save("dcgm.json", {"results": [{"status": "Fail"}]})
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")

    def test_invalid_result_fails(self):
        self.save("environment-gpu0.json", {"unexpected": 1})
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")

    def test_no_kernel_evidence_requires_review(self):
        (self.out / "kernel-during.log").write_text("-- No entries --", encoding="utf-8")
        self.assertEqual(report.generate(self.out)["overall"], "REVIEW")

    def test_history_ecc_requires_review(self):
        self.snapshot("before.xml", 1)
        self.snapshot("after.xml", 1)
        self.assertEqual(report.generate(self.out)["overall"], "REVIEW")

    def test_missing_partial_nonfinite_or_interrupted_telemetry_requires_review(self):
        path = self.out / "telemetry.csv"
        valid = path.read_text(encoding="utf-8")
        for raw in ("", "\n".join(line for line in valid.splitlines() if 'gpu7' not in line),
                    valid.replace(',50,500,', ',nan,500,'), valid.replace('10:00:02', '10:01:02')):
            with self.subTest(raw=raw[:50]):
                path.write_text(raw, encoding="utf-8")
                self.assertEqual(report.generate(self.out)["overall"], "REVIEW")
        path.unlink()
        self.assertEqual(report.generate(self.out)["overall"], "REVIEW")

    def test_telemetry_overheat_fails_otherwise_successful_run(self):
        path = self.out / "telemetry.csv"
        path.write_text(path.read_text(encoding="utf-8").replace(',50,500,', ',90,500,'), encoding="utf-8")
        self.assertEqual(report.generate(self.out)["overall"], "FAIL")


class BundleTests(unittest.TestCase):
    def test_image_app_upgrade_preserves_gpu_constraints(self):
        constraints = 'torch==2.9.0a0\nnvidia-cublas-cu13==13.0\ntransformers==4.55.0\n# comment\n'
        filtered = image_constraints.filtered_constraints(constraints, 'transformers==4.57.1\n')
        self.assertIn('torch==2.9.0a0', filtered)
        self.assertIn('nvidia-cublas-cu13==13.0', filtered)
        self.assertNotIn('transformers==4.55.0', filtered)
        with self.assertRaises(ValueError):
            image_constraints.filtered_constraints(constraints, 'torch==2.8.0')

    def test_incomplete_package_cannot_be_sealed(self):
        with workspace_temp() as tmp, patch.object(sys, "argv", ["bundle.py", "seal", "--root", tmp]):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                bundle.main()


if __name__ == "__main__":
    unittest.main()
