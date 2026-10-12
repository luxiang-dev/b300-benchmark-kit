"""Evidence integrity regression fixtures, not hardware benchmark observations."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence


class EvidenceExportTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / ".test-tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temporary.name).resolve()
        self.out = self.root / "run"
        self.out.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def assert_archive_matches_manifest(self, manifest):
        with zipfile.ZipFile(self.out / "evidence.zip") as archive:
            self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"evidence-manifest.json"})
            self.assertEqual(json.loads(archive.read("evidence-manifest.json")), manifest)
            for name, metadata in manifest["files"].items():
                content = archive.read(name)
                self.assertEqual(len(content), metadata["size"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])
                self.assertEqual(content, (self.out / name).read_bytes())

    def test_final_job_and_nested_files_have_consistent_zip_hashes(self):
        (self.out / "run-result.json").write_text(json.dumps(dict(status="CANCELLED", synthetic_only=True)), encoding="utf-8")
        (self.out / "job.json").write_text(json.dumps(dict(status="EXPORTING", synthetic_only=True)), encoding="utf-8")
        (self.out / "raw").mkdir()
        (self.out / "raw/log.txt").write_text("Synthetic evidence fixture only\n", encoding="utf-8")
        manifest = evidence.export_evidence(self.out)
        self.assertEqual(set(manifest["files"]), {"run-result.json", "raw/log.txt"})
        self.assert_archive_matches_manifest(manifest)

    def test_repeated_exports_do_not_include_their_previous_archives(self):
        (self.out / "run-result.json").write_text('{"status":"FAIL","synthetic_only":true}', encoding="utf-8")
        (self.out / "job.json").write_text('{"status":"EXPORTING","synthetic_only":true}', encoding="utf-8")
        first = evidence.export_evidence(self.out)
        (self.out / "evidence.zip.tmp").write_bytes(b"excluded interrupted archive fixture")
        (self.out / "job.json").write_text('{"status":"FAIL","synthetic_only":true}', encoding="utf-8")
        (self.out / "job.json.tmp").write_text("mutable excluded fixture", encoding="utf-8")
        second = evidence.export_evidence(self.out)
        self.assertEqual(first, second)
        self.assertEqual(set(second["files"]), {"run-result.json"})
        self.assert_archive_matches_manifest(second)

    def test_changed_final_record_requires_and_receives_new_hash(self):
        job = self.out / "run-result.json"
        job.write_text('{"status":"RUNNING","synthetic_only":true}', encoding="utf-8")
        before = evidence.export_evidence(self.out)
        job.write_text('{"status":"CANCELLED","synthetic_only":true}', encoding="utf-8")
        after = evidence.export_evidence(self.out)
        self.assertNotEqual(before["files"]["run-result.json"]["sha256"], after["files"]["run-result.json"]["sha256"])
        self.assert_archive_matches_manifest(after)

    def test_external_symlinks_are_not_copied_or_hashed(self):
        external = self.root / "outside"
        external.mkdir()
        secret = external / "private.txt"
        secret.write_text("outside-private-fixture", encoding="utf-8")
        (self.out / "log.txt").write_text("Synthetic local evidence fixture", encoding="utf-8")
        try:
            (self.out / "linked-file").symlink_to(secret)
            (self.out / "linked-directory").symlink_to(external, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"OS does not permit symlink fixture: {error}")
        manifest = evidence.export_evidence(self.out)
        self.assertEqual(set(manifest["files"]), {"log.txt"})
        self.assert_archive_matches_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
