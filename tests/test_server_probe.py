"""Tiny real spot checks plus synthetic inventory; no full server/GPU claims."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import server_probe as probe
from evidence import digest
import test_kit
import report


class ServerProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_kit.workspace_temp()
        self.out = Path(self.temp.name)
        (self.out / 'server-evidence').mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def capture_fixture(self, out, label, argv):
        files = {}
        for kind, content in [('stdout', b'Synthetic inventory only\n'), ('stderr', b'')]:
            path = out / 'server-evidence' / f'{label}.{kind}.txt'
            path.write_bytes(content)
            files[kind] = dict(path=path.relative_to(out).as_posix(), sha256=digest(path))
        return dict(argv=argv, returncode=0, error=None, status='COLLECTED', files=files)

    def collect_fixture(self, full=True):
        with patch.object(probe, 'capture', side_effect=self.capture_fixture), patch.object(probe, 'available_ram', return_value=2**30):
            return probe.collect(self.out, full, ram_bytes=probe.MIB, disk_bytes=probe.MIB)

    def test_real_tiny_cpu_ram_and_result_file_checks(self):
        result = self.collect_fixture()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['checks']['ram']['allocated_bytes'], probe.MIB)
        self.assertEqual(result['checks']['ram']['total_verified_bytes'], 2*probe.MIB)
        self.assertEqual(result['checks']['result-disk']['read_bytes'], probe.MIB)
        self.assertFalse(list(self.out.glob('.b300-disk-probe-*')))
        self.assertIn('not', result['limitation'])

    def test_insufficient_available_ram_skips_without_allocating(self):
        with patch.object(probe, 'available_ram', return_value=probe.MIB), patch.object(probe.mmap, 'mmap') as allocation:
            self.assertEqual(probe.ram_check(probe.MIB)['status'], 'SKIP')
            allocation.assert_not_called()

    def test_inventory_only_is_review_without_functional_claims(self):
        result = self.collect_fixture(full=False)
        self.assertEqual(result['status'], 'REVIEW')
        self.assertEqual(result['checks'], {})

    def test_unavailable_inventory_keeps_error_and_is_review(self):
        with patch.object(probe.subprocess, 'run', side_effect=FileNotFoundError('synthetic missing tool')):
            record = probe.capture(self.out, 'absent-tool', ['absent-tool'])
        self.assertEqual(record['status'], 'UNAVAILABLE')
        self.assertIsNone(record['returncode'])
        self.assertIn('synthetic missing', record['error'])
        result = self.collect_fixture()
        path = self.out / 'server-summary.json'
        result['commands']['bmc-fru'].update(returncode=1, error='synthetic permission denied', status='UNAVAILABLE')
        probe.save(path, result)
        self.assertEqual(probe.read_result(self.out)['status'], 'REVIEW')

    def test_raw_tampering_and_escape_are_rejected(self):
        self.collect_fixture()
        raw = self.out / 'server-evidence/cpu.stdout.txt'
        raw.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            probe.read_result(self.out)
        result = self.collect_fixture()
        result['commands']['cpu']['files']['stdout']['path'] = '../outside.txt'
        probe.save(self.out / 'server-summary.json', result)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            probe.read_result(self.out)

    def test_short_or_inflated_ram_claim_rejected_even_with_matching_hash(self):
        result = self.collect_fixture()
        result['checks']['ram']['allocated_bytes'] *= 2
        raw = self.out / 'server-evidence/check-ram.json'
        probe.save(raw, result['checks']['ram'])
        result['check_files']['ram']['sha256'] = digest(raw)
        probe.save(self.out / 'server-summary.json', result)
        with self.assertRaisesRegex(ValueError, 'RAM coverage'):
            probe.read_result(self.out)

    def test_disk_failure_removes_only_its_temporary_file(self):
        retained = self.out / 'user-file.txt'
        retained.write_text('retain', encoding='utf-8')
        with patch('os.fsync', side_effect=OSError('synthetic disk error')):
            with self.assertRaises(OSError):
                probe.disk_check(self.out, probe.MIB)
        self.assertEqual(retained.read_text(), 'retain')
        self.assertFalse(list(self.out.glob('.b300-disk-probe-*')))

    def test_failed_check_overrides_inaccurate_summary_status(self):
        result = self.collect_fixture()
        result['status'] = 'PASS'
        result['checks']['cpu'] = dict(status='FAIL', error='synthetic known-answer error')
        raw = self.out / 'server-evidence/check-cpu.json'
        probe.save(raw, result['checks']['cpu'])
        result['check_files']['cpu']['sha256'] = digest(raw)
        probe.save(self.out / 'server-summary.json', result)
        self.assertEqual(probe.read_result(self.out)['status'], 'FAIL')

    def test_checks_are_bounded_and_do_not_write_raw_devices(self):
        for size in (0, True, probe.MIB//2, 257*probe.MIB):
            with self.subTest(size=size), self.assertRaises(ValueError):
                probe.ram_check(size)
        for argv in probe.COMMANDS.values():
            self.assertNotIn('sudo', argv)
            self.assertNotIn('dd', argv)


class ServerReportTests(unittest.TestCase):
    setUp = test_kit.ReportTests.setUp
    tearDown = test_kit.ReportTests.tearDown
    save = test_kit.ReportTests.save
    snapshot = test_kit.ReportTests.snapshot
    telemetry = test_kit.ReportTests.telemetry

    def test_missing_server_stage_evidence_cannot_pass(self):
        self.plan['required'] = ['server']
        self.save('plan.json', self.plan)
        (self.out / 'stages.tsv').write_text('server\tPASS\t0\n', encoding='utf-8')
        self.assertEqual(report.generate(self.out)['overall'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
