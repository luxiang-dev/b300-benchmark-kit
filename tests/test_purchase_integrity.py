"""Procurement fault injection only; no server or GPU measurements."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard
import evidence
import run_authenticity as launcher
import test_kit


class PurchaseEvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = test_kit.workspace_temp()
        self.out = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_concurrent_producer_cannot_publish_inconsistent_zip(self):
        log = self.out / 'telemetry.csv'
        log.write_bytes(b'synthetic first sample\n')
        original_digest = evidence.digest
        changed = False

        def change_after_first_hash(path):
            nonlocal changed
            value = original_digest(path)
            if path == log and not changed:
                changed = True
                log.write_bytes(b'synthetic second sample, longer\n')
            return value

        # Deterministic representation of an independent writer changing the
        # log between the manifest read and archive read. Either refuse the
        # export, or publish a manifest matching the exact archived bytes.
        with patch.object(evidence, 'digest', side_effect=change_after_first_hash):
            try:
                manifest = evidence.export_evidence(self.out)
            except (RuntimeError, ValueError):
                self.assertFalse((self.out / 'evidence.zip').exists())
                return
        with zipfile.ZipFile(self.out / 'evidence.zip') as archive:
            self.assertEqual(json.loads(archive.read('evidence-manifest.json')), manifest)
            for name, expected in manifest['files'].items():
                content = archive.read(name)
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected['sha256'])
                self.assertEqual(len(content), expected['size'])

    def test_existing_stage_without_plan_cannot_be_reused(self):
        (self.out / 'stages.tsv').write_text('memory\tPASS\t0\n', encoding='utf-8')
        args = SimpleNamespace(out=self.out, profile='identity', gpus=8, seconds=1,
                               image='synthetic-image', memory_fraction=.85, max_temp=85)
        workflow = launcher.Workflow(args)
        with patch.object(launcher.authenticity, 'collect_host', side_effect=RuntimeError('stop synthetic fixture')) as collector, \
             patch.object(launcher.authenticity, 'evaluate_identity', return_value={'status': 'REVIEW'}), \
             patch.object(launcher.report, 'generate', return_value={'overall': 'REVIEW'}), \
             patch.object(launcher, 'export_evidence'), patch.object(launcher.signal, 'signal'), \
             patch.object(workflow, 'capture'), patch.object(workflow, 'stop_current'), patch.object(workflow, 'stop_container'):
            with self.assertRaisesRegex(RuntimeError, '(?i)fresh|new|empty|result'):
                workflow.run()
            collector.assert_not_called()
        self.assertFalse((self.out / 'plan.json').exists())

    def test_failed_archive_publication_retains_previous_consistent_evidence(self):
        log = self.out / 'launcher.log'
        log.write_bytes(b'synthetic old evidence\n')
        evidence.export_evidence(self.out)
        original_zip = (self.out / 'evidence.zip').read_bytes()
        original_manifest = (self.out / 'evidence-manifest.json').read_bytes()
        log.write_bytes(b'synthetic changed evidence\n')
        original_replace = Path.replace

        def fail_zip_replace(path, target):
            if path.name == 'evidence.zip.tmp':
                raise OSError('synthetic archive locked')
            return original_replace(path, target)

        with patch.object(Path, 'replace', new=fail_zip_replace):
            with self.assertRaisesRegex(OSError, 'archive locked'):
                evidence.export_evidence(self.out)
        self.assertEqual((self.out / 'evidence.zip').read_bytes(), original_zip)
        self.assertEqual((self.out / 'evidence-manifest.json').read_bytes(), original_manifest)

    def test_new_file_during_final_check_prevents_publication(self):
        log = self.out / 'telemetry.csv'
        log.write_bytes(b'synthetic sample\n')
        original_digest = evidence.digest

        def additional_producer_file(path):
            value = original_digest(path)
            (self.out / 'late-result.json').write_text('{"synthetic_only":true}', encoding='utf-8')
            return value

        with patch.object(evidence, 'digest', side_effect=additional_producer_file):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                evidence.export_evidence(self.out)
        self.assertFalse((self.out / 'evidence.zip').exists())


class PurchaseControllerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = test_kit.workspace_temp()
        self.root = Path(self.temporary.name).resolve()
        self.manager = dashboard.JobManager(self.root / 'runs', preview=True)
        self.key = '20261012-120000-00000001'
        self.out = self.manager.root / self.key
        self.out.mkdir()
        self.manager.jobs[self.key] = dict(id=self.key, profile='identity', status='RUNNING',
                                          seconds=600, cancelled=False, thermal_stop=False,
                                          started_at=dashboard.utc_now(), exit_code=None, ended_at=None)
        self.manager.active = self.key
        self.manager.gpu_lock = os.open(self.root / 'synthetic-lock', os.O_CREAT | os.O_RDWR, 0o600)

    def tearDown(self):
        if self.manager.gpu_lock is not None:
            os.close(self.manager.gpu_lock)
            self.manager.gpu_lock = None
        self.manager.active = None
        self.manager.process = None
        self.temporary.cleanup()

    def test_disk_full_on_final_metadata_cannot_leak_gpu_lock_or_active_job(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(self.manager, 'command', return_value=['synthetic-cpu-fixture']), \
             patch.object(dashboard.subprocess, 'Popen', return_value=process), \
             patch.object(self.manager, '_cleanup_containers'), \
             patch.object(dashboard, 'write_json', side_effect=OSError('synthetic ENOSPC final metadata')):
            try:
                self.manager._run(self.key, self.out)
            except OSError:
                pass
        self.assertIsNone(self.manager.active, 'Disk full must not leave the controller permanently busy')
        self.assertIsNone(self.manager.process)
        self.assertIsNone(self.manager.gpu_lock, 'The process lock must be released even when writing metadata fails')
        self.assertEqual(self.manager.jobs[self.key]['status'], 'FAIL')

    def test_disk_full_cannot_prevent_thermal_stop_signal(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        self.manager.process = process
        def persistence_failure(*_):
            self.assertTrue(terminate.called, 'Stop signal must precede potentially hung result-disk I/O')
            raise OSError('synthetic ENOSPC')
        with patch.object(dashboard, 'write_json', side_effect=persistence_failure), \
             patch.object(dashboard.os, 'killpg', create=True) as terminate, \
             patch.object(dashboard.threading, 'Thread') as cleanup_thread:
            self.manager.stop(self.key, 'Synthetic temperature threshold')
        terminate.assert_called_once_with(12345, signal.SIGTERM)
        cleanup_thread.return_value.start.assert_called_once()
        self.assertTrue(self.manager.jobs[self.key]['thermal_stop'])
        self.assertIn('ENOSPC', self.manager.jobs[self.key]['error'])

    def test_unconfirmed_container_cleanup_cannot_finish_as_pass(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        (self.out / 'summary.json').write_text('{"overall":"PASS","synthetic_only":true}', encoding='utf-8')
        with patch.object(self.manager, 'command', return_value=['synthetic-cpu-fixture']), \
             patch.object(dashboard.subprocess, 'Popen', return_value=process), \
             patch.object(self.manager, '_cleanup_containers', side_effect=RuntimeError('synthetic owned container still alive')):
            self.manager._run(self.key, self.out)
        self.assertEqual(self.manager.jobs[self.key]['status'], 'FAIL')
        self.assertIn('still alive', self.manager.jobs[self.key]['error'])
        self.assertIsNone(self.manager.active)
        self.assertIsNone(self.manager.gpu_lock)
        self.assertFalse((self.out / 'evidence.zip').exists())

    def test_cancellation_at_finalization_cannot_be_overwritten_by_stale_pass(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        (self.out / 'summary.json').write_text('{"overall":"PASS","synthetic_only":true}', encoding='utf-8')
        original_now = dashboard.utc_now
        def cancellation_boundary():
            self.manager.stop(self.key)
            return original_now()
        with patch.object(self.manager, 'command', return_value=['synthetic-cpu-fixture']), \
             patch.object(dashboard.subprocess, 'Popen', return_value=process), \
             patch.object(self.manager, '_cleanup_containers'), \
             patch.object(dashboard, 'utc_now', side_effect=cancellation_boundary):
            self.manager._run(self.key, self.out)
        self.assertEqual(self.manager.jobs[self.key]['status'], 'CANCELLED')
        final = json.loads((self.out / 'run-result.json').read_text())
        self.assertEqual(final['status'], 'CANCELLED')
        with zipfile.ZipFile(self.out / 'evidence.zip') as archive:
            self.assertEqual(json.loads(archive.read('run-result.json'))['status'], 'CANCELLED')

    def test_cleanup_failure_does_not_prevent_wait_and_second_cleanup(self):
        process = Mock(pid=12345)
        process.wait.side_effect = subprocess.TimeoutExpired('synthetic-launcher', 45)
        process.poll.return_value = None
        with patch.object(self.manager, '_cleanup_containers', side_effect=RuntimeError('synthetic Docker unavailable')) as cleanup, \
             patch.object(dashboard.signal, 'SIGKILL', 9, create=True), \
             patch.object(dashboard.os, 'killpg', create=True, side_effect=PermissionError('synthetic denied signal')):
            self.manager._finish_stop(self.key, process)
        process.wait.assert_called_once_with(timeout=45)
        self.assertEqual(cleanup.call_count, 2)
        self.assertIn('Docker unavailable', self.manager.jobs[self.key]['cleanup_error'])


if __name__ == '__main__':
    unittest.main()
