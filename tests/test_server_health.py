"""Synthetic SMART/BMC parser and raw-integrity fixtures; no hardware claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server_health as health
import test_kit


def disk(path='/dev/nvme0n1', serial='SYNTHETIC-NVME'):
    return dict(name=path.split('/')[-1], path=path, type='disk', size=2**40,
                model='SYNTHETIC-MODEL', serial=serial)


def smart(path='/dev/nvme0n1', serial='SYNTHETIC-NVME'):
    data = dict(smartctl=dict(exit_status=0), smart_status=dict(passed=True),
                serial_number=serial, model_name='SYNTHETIC-MODEL',
                device=dict(name=path, protocol='NVMe' if 'nvme' in path else 'ATA'))
    if 'nvme' in path:
        data['nvme_smart_health_information_log'] = dict(critical_warning=0, media_errors=0,
            percentage_used=0, power_on_hours=0, unsafe_shutdowns=0, num_err_log_entries=0,
            available_spare=100, available_spare_threshold=10)
    else:
        data['ata_smart_attributes'] = dict(table=[dict(id=identifier, name='Synthetic attribute',
            raw=dict(value=0), when_failed='') for identifier in (5, 9, 187, 197, 198, 199)])
    return data


class ServerHealthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = test_kit.workspace_temp()
        self.out = Path(self.temporary.name).resolve()
        self.disks = [disk()]
        self.smart = {self.disks[0]['path']: smart()}
        self.sdr = 'Synthetic Ambient | 01h | ok | 7.1 | 30 degrees C\n'
        self.sel = 'SEL has no entries\n'
        self.sensors = {}
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, argv, **kwargs):
        self.assertTrue(kwargs['capture_output'])
        self.assertEqual(kwargs['timeout'], 8)
        self.assertFalse(kwargs['check'])
        self.calls.append(argv)
        rc = 0
        if argv == health.COMMANDS['lsblk']:
            stdout = json.dumps(dict(blockdevices=self.disks)).encode()
        elif argv == health.COMMANDS['bmc-sensors']:
            stdout = self.sdr.encode()
        elif argv == health.COMMANDS['bmc-events']:
            stdout = self.sel.encode()
        elif argv == health.COMMANDS['sensors']:
            stdout = json.dumps(self.sensors).encode()
        else:
            self.assertEqual(argv[:3], ['smartctl', '-a', '-j'])
            self.assertEqual(len(argv), 4)
            self.assertRegex(argv[3], health.DEVICE)
            data = self.smart[argv[3]]
            stdout = json.dumps(data).encode()
            rc = data.get('smartctl', {}).get('exit_status', 0)
        return subprocess.CompletedProcess(argv, rc, stdout, b'')

    def collect(self, phase='before'):
        with patch.object(health.subprocess, 'run', side_effect=self.run_fixture):
            return health.collect(self.out, phase)

    def test_two_complete_healthy_snapshots_are_valid_health_evidence_only(self):
        first = self.collect()
        self.assertEqual(first['status'], 'REVIEW')
        self.assertIn('Missing server health after snapshot', first['review'])
        result = self.collect('after')
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['inventory']['after']['disk_health'][0]['counters']['media_errors'], 0)
        self.assertIn('not provenance', result['limitation'])
        self.assertEqual(health.evaluate(self.out), result)

    def test_nvme_critical_warning_and_media_errors_fail(self):
        for key in ('critical_warning', 'media_errors'):
            with self.subTest(key=key):
                record = smart()
                record['nvme_smart_health_information_log'][key] = 1
                result = health.smart_health(json.dumps(record), disk(), 0)
                self.assertTrue(any(key in item for item in result['failures']))

    def test_nvme_lifetime_wear_and_errlog_are_review_not_counterfeit(self):
        data = smart()
        data['nvme_smart_health_information_log'].update(percentage_used=4, power_on_hours=7000, num_err_log_entries=10)
        parsed = health.smart_health(json.dumps(data), disk(), 0)
        self.assertEqual(parsed['failures'], [])
        self.assertEqual(len(parsed['review']), 3)
        self.assertTrue(all('OEM review' in item for item in parsed['review']))

    def test_nvme_missing_or_nonfinite_health_values_cannot_pass(self):
        for value in (None, True, '0', -1, float('nan')):
            with self.subTest(value=value):
                data = smart()
                data['nvme_smart_health_information_log']['critical_warning'] = value
                self.assertTrue(health.smart_health(json.dumps(data), disk(), 0)['review'])

    def test_missing_overall_smart_health_cannot_pass(self):
        data = smart()
        data.pop('smart_status')
        self.assertTrue(health.smart_health(json.dumps(data), disk(), 0)['review'])

    def test_ata_failed_health_and_pending_media_fail(self):
        node = disk('/dev/sda', 'SYNTHETIC-ATA')
        data = smart('/dev/sda', 'SYNTHETIC-ATA')
        data['smart_status']['passed'] = False
        data['ata_smart_attributes']['table'][3]['raw']['value'] = 1
        parsed = health.smart_health(json.dumps(data), node, 8)
        self.assertTrue(any('overall' in item for item in parsed['failures']))
        self.assertTrue(any('pending_sectors' in item for item in parsed['failures']))

    def test_smart_exit_bits_distinguish_unavailable_current_and_historical(self):
        node = disk('/dev/sda', 'SYNTHETIC-ATA')
        for rc, failed in [(2, False), (4, False), (8, True), (16, True), (32, False), (64, False), (128, False)]:
            with self.subTest(rc=rc):
                data = smart('/dev/sda', 'SYNTHETIC-ATA')
                data['smartctl']['exit_status'] = rc
                result = health.smart_health(json.dumps(data), node, rc)
                self.assertEqual(bool(result['failures']), failed)
                if not failed:
                    self.assertTrue(result['review'])

    def test_raw_health_tampering_is_failure_despite_cached_pass(self):
        self.collect()
        self.collect('after')
        path = self.out / 'server-health-evidence/after-smart-000.stdout.txt'
        data = json.loads(path.read_text())
        data['nvme_smart_health_information_log']['critical_warning'] = 1
        path.write_text(json.dumps(data), encoding='utf-8')
        result = health.evaluate(self.out)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('hash/size changed' in item for item in result['failures']))

    def test_cached_health_status_is_ignored(self):
        self.collect()
        self.smart['/dev/nvme0n1']['nvme_smart_health_information_log']['media_errors'] = 1
        self.collect('after')
        path = self.out / 'server-health-after.json'
        data = json.loads(path.read_text())
        data.update(status='PASS', failures=[], review=[])
        health.save(path, data)
        result = health.evaluate(self.out)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('media_errors' in item for item in result['failures']))

    def test_raw_path_escape_or_wrong_command_is_rejected(self):
        self.collect()
        path = self.out / 'server-health-before.json'
        original = json.loads(path.read_text())
        for mutation in ('path', 'command'):
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(original)
                if mutation == 'path':
                    data['commands']['lsblk']['files']['stdout']['path'] = '../outside.txt'
                else:
                    data['commands']['smart-000']['argv'] = ['smartctl', '-t', 'long', '/dev/nvme0n1']
                health.save(path, data)
                self.assertEqual(health.evaluate(self.out)['status'], 'FAIL')

    def test_unsafe_device_argument_is_never_executed(self):
        self.disks = [disk('/dev/sda;touch /tmp/unsafe', 'SYNTHETIC-INVALID')]
        result = self.collect()
        self.assertTrue(result['review'])
        self.assertFalse(any(argv[0] == 'smartctl' for argv in self.calls))

    def test_only_type_disk_is_queried_and_collection_is_bounded(self):
        self.disks = [disk('/dev/sd' + 'a'*(i//26+1) + chr(97+i%26), 'SYNTHETIC-' + str(i)) for i in range(34)]
        self.disks[0]['children'] = [dict(name='synthetic-part', path='/dev/sda1', type='part', size=2**20)]
        self.smart = {node['path']: smart(node['path'], node['serial']) for node in self.disks}
        result = self.collect()
        smart_calls = [argv for argv in self.calls if argv[0] == 'smartctl']
        self.assertEqual(len(smart_calls), 32)
        self.assertTrue(any('first 32' in item for item in result['review']))
        self.assertNotIn('/dev/sda1', [argv[-1] for argv in smart_calls])
        for argv in self.calls:
            self.assertNotIn('sudo', argv)
            self.assertNotIn('clear', argv)
            self.assertNotIn('-t', argv)

    def test_missing_tools_and_permissions_remain_review(self):
        with patch.object(health.subprocess, 'run', side_effect=PermissionError('synthetic denied read')):
            result = health.collect(self.out, 'before')
        self.assertEqual(result['status'], 'REVIEW')
        self.assertEqual(result['failures'], [])
        self.assertTrue(any('unavailable' in item for item in result['review']))

    def test_timeout_keeps_partial_raw_output_and_cannot_pass(self):
        (self.out / 'server-health-evidence').mkdir()
        with patch.object(health.subprocess, 'run', side_effect=subprocess.TimeoutExpired('synthetic', 8, output=b'partial synthetic', stderr=b'synthetic timeout')):
            record = health.capture(self.out, 'before', 'lsblk', health.COMMANDS['lsblk'])
        self.assertIsNone(record['returncode'])
        self.assertIn('8 second', record['error'])
        self.assertEqual((self.out / record['files']['stdout']['path']).read_bytes(), b'partial synthetic')

    def test_current_bmc_threshold_statuses_fail_unknown_is_review(self):
        for state in ('cr', 'nr', 'nc', 'lnr', 'unr', 'lcr', 'ucr', 'lnc', 'unc'):
            with self.subTest(state=state):
                _, failures, _ = health.bmc_sensors(f'Synthetic temp | 01h | {state} | 7.1 | 90 degrees C\n')
                self.assertTrue(failures)
        _, failures, review = health.bmc_sensors('Synthetic unused bay | 01h | ns | 7.1 | No reading\n')
        self.assertFalse(failures)
        self.assertTrue(review)

    def test_current_discrete_bmc_fault_is_not_masked_by_ok_state(self):
        _, failures, _ = health.bmc_sensors('Synthetic PSU | 01h | ok | 10.1 | Failure detected\n')
        self.assertTrue(failures)
        _, failures, _ = health.bmc_sensors('Synthetic failed-label only | 01h | ok | 10.1 | Presence detected\n')
        self.assertFalse(failures)

    def test_unknown_bmc_reading_cannot_be_assumed_healthy(self):
        for reading in ('No reading', 'Synthetic OEM state 44', '0x01', ''):
            with self.subTest(reading=reading):
                _, failures, review = health.bmc_sensors(f'Synthetic sensor | 01h | ok | 10.1 | {reading}\n')
                self.assertFalse(failures)
                self.assertTrue(review)

    def test_sel_history_is_review_and_is_never_cleared(self):
        self.sel = '1 | 01/01/2020 | 00:00:00 | Synthetic PSU | Failure detected | Asserted\n'
        self.collect()
        result = self.collect('after')
        self.assertEqual(result['status'], 'REVIEW')
        self.assertFalse(result['failures'])
        self.assertTrue(any('historical' in item for item in result['review']))
        self.assertFalse(any('clear' in argv for argv in self.calls))

    def test_only_known_alarm_fields_are_treated_as_board_alarm(self):
        failures, review = health.sensor_alarms(json.dumps({'synthetic-chip': {'temp1': {'temp1_crit_alarm': 1}}}))
        self.assertTrue(failures)
        failures, review = health.sensor_alarms(json.dumps({'synthetic-failed-label': {'temp1': {'temp1_input': 40}}}))
        self.assertFalse(failures)

    def test_disk_inventory_or_serial_changes_fail(self):
        self.collect()
        self.disks[0]['serial'] = 'SYNTHETIC-CHANGED'
        self.smart['/dev/nvme0n1']['serial_number'] = 'SYNTHETIC-CHANGED'
        result = self.collect('after')
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('serial changed' in item for item in result['failures']))

    def test_media_counter_growth_is_recorded_and_fails(self):
        self.disks = [disk('/dev/sda', 'SYNTHETIC-ATA')]
        self.smart = {'/dev/sda': smart('/dev/sda', 'SYNTHETIC-ATA')}
        self.collect()
        self.smart['/dev/sda']['ata_smart_attributes']['table'][0]['raw']['value'] = 1
        result = self.collect('after')
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['counter_changes'][0]['delta'], 1)
        self.assertTrue(any('counter increased' in item for item in result['failures']))

    def test_existing_phase_is_preserved_and_not_overwritten(self):
        self.collect()
        path = self.out / 'server-health-before.json'
        original = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            self.collect()
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
