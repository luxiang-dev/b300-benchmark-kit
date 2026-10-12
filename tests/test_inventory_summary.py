"""Synthetic evidence parsing only; no claim about any purchased server."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import inventory_summary as inventory
import server_probe
from evidence import digest
import test_authenticity
import test_kit

CPU = json.dumps({'lscpu': [{'field': 'Architecture:', 'data': 'x86_64'},
                           {'field': 'Model name:', 'data': 'SYNTHETIC CPU'},
                           {'field': 'Socket(s):', 'data': '2'}, {'field': 'CPU(s):', 'data': '128'}]})
FREE = '              total        used        free      shared  buff/cache   available\nMem:    68719476736 12345678 61234567890 0 12345678 60000000000\nSwap: 0 0 0\n'
SMBIOS = '''# Synthetic only
Handle 0x0001, DMI type 1, 27 bytes
System Information
    Manufacturer: SYNTHETIC OEM
    Product Name: SYNTHETIC SERVER
    Version: TEST ONLY
    Serial Number: SYNTHETIC-SERVER-SERIAL

Handle 0x0011, DMI type 17, 92 bytes
Memory Device
    Size: 128 GB
    Locator: DIMM_A1
    Bank Locator: P0
    Manufacturer: SYNTHETIC MEMORY
    Serial Number: SYNTHETIC-DIMM-SERIAL
    Part Number: SYNTHETIC-DIMM-PART
    Configured Memory Speed: 5600 MT/s

Handle 0x0012, DMI type 17, 92 bytes
Memory Device
    Size: No Module Installed
    Locator: DIMM_A2
'''
STORAGE = json.dumps({'blockdevices': [{'name': 'nvme0n1', 'path': '/dev/nvme0n1', 'type': 'disk', 'model': 'SYNTHETIC SSD',
                                      'serial': 'SYNTHETIC-SSD-SERIAL', 'size': 1000000000000, 'tran': 'nvme', 'rota': False,
                                      'children': [{'name': 'nvme0n1p1', 'type': 'part', 'size': 999000000000}]},
                                     {'name': 'loop0', 'type': 'loop', 'size': 10000}]})
LINKS = json.dumps([{'ifindex': 1, 'ifname': 'lo', 'link_type': 'loopback', 'mtu': 65536, 'operstate': 'UNKNOWN'},
                   {'ifindex': 2, 'ifname': 'eth0', 'link_type': 'ether', 'address': '00:11:22:33:44:55', 'mtu': 1500, 'operstate': 'UP'},
                   {'ifindex': 3, 'ifname': 'test-veth', 'link_type': 'ether', 'linkinfo': {'info_kind': 'veth'}, 'mtu': 1500}])


class InventorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_kit.workspace_temp()
        self.out = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def server_fixture(self, mode='inventory'):
        (self.out / 'server-evidence').mkdir(exist_ok=True)
        commands = {}
        samples = {'cpu': CPU, 'ram': FREE, 'storage': STORAGE, 'nic-link': LINKS, 'smbios': SMBIOS}
        for label, argv in server_probe.COMMANDS.items():
            files = {}
            for kind, content in [('stdout', samples.get(label, 'Synthetic unrelated inventory\n')), ('stderr', '')]:
                path = self.out / 'server-evidence' / f'{label}.{kind}.txt'
                path.write_text(content, encoding='utf-8')
                files[kind] = {'path': path.relative_to(self.out).as_posix(), 'sha256': digest(path)}
            commands[label] = {'argv': argv, 'returncode': 0, 'error': None, 'status': 'COLLECTED', 'files': files}
        result = {'schema': 1, 'test': 'server', 'mode': mode, 'commands': commands, 'checks': {}, 'check_files': {}}
        server_probe.save(self.out / 'server-summary.json', result)
        return result

    def change_stream(self, label, text, kind='stdout', **command_changes):
        result = json.loads((self.out / 'server-summary.json').read_text(encoding='utf-8'))
        command = result['commands'][label]
        path = self.out / command['files'][kind]['path']
        path.write_text(text, encoding='utf-8')
        command['files'][kind]['sha256'] = digest(path)
        command.update(command_changes)
        server_probe.save(self.out / 'server-summary.json', result)

    def identity_fixture(self, **args):
        with patch.multiple(test_authenticity.identity.platform, system=lambda: 'Linux', release=lambda: 'SYNTHETIC',
                            machine=lambda: 'x86_64', node=lambda: 'SYNTHETIC-TEST-HOST'):
            test_authenticity.IdentityTests.collect_fixture(self, **args)
        server_probe.save(self.out / 'plan.json', {'gpus': args.get('count', 1)})

    def test_complete_observations_never_become_purchase_bom(self):
        self.server_fixture()
        self.identity_fixture()
        result = inventory.derive(self.out)
        self.assertEqual(result['purpose'], 'OBSERVED_NOT_PURCHASE_BOM')
        self.assertEqual(result['comparison_status'], 'NOT_COMPARED')
        self.assertEqual(result['procurement_checks_cleared'], [])
        self.assertEqual(result['status'], 'OBSERVED', result['missing_evidence'])
        self.assertEqual(result['cpu']['sockets'], 2)
        self.assertEqual(result['cpu']['logical_cpus'], 128)
        self.assertEqual(result['system']['records'][0]['serial_number'], 'SYNTHETIC-SERVER-SERIAL')
        self.assertEqual(result['dimms']['devices'][0]['size_raw'], '128 GB')
        self.assertEqual(result['dimms']['devices'][0]['configured_speed'], {'value': 5600, 'unit': 'MT/s'})
        self.assertIs(result['dimms']['devices'][1]['populated'], False)
        self.assertEqual(result['gpus']['devices'][0]['serial_number'], 'SYNTHETIC-SERIAL-0')
        self.assertEqual(json.loads((self.out / 'observed-inventory.json').read_text(encoding='utf-8')), result)

    def test_os_ram_is_not_installed_or_fully_tested_ram(self):
        parsed = inventory.parse_free(FREE)
        self.assertEqual(parsed['os_reported_total_bytes'], 64 * 2**30)
        self.assertIsNone(parsed['installed_dimm_bytes'])
        self.assertIn('not installed', parsed['scope'])

    def test_disk_list_excludes_partitions_and_does_not_claim_physical_ssd_count(self):
        parsed = inventory.parse_lsblk(STORAGE)
        self.assertEqual(parsed['observed_disk_entries'], 1)
        self.assertEqual(parsed['disks'][0]['size_bytes'], 1000000000000)
        self.assertIsNone(parsed['physical_ssd_count'])
        self.assertIn('RAID/virtual', parsed['disks'][0]['scope'])

    def test_interface_list_does_not_count_nic_cards(self):
        parsed = inventory.parse_interfaces(LINKS)
        self.assertEqual(parsed['observed_interface_entries'], 3)
        self.assertIsNone(parsed['physical_nic_card_count'])
        self.assertIs(parsed['interfaces'][0]['is_known_virtual'], True)
        self.assertIsNone(parsed['interfaces'][1]['is_known_virtual'], 'Ethernet alone does not prove a physical NIC')
        self.assertIs(parsed['interfaces'][2]['is_known_virtual'], True)

    def test_missing_raw_inventory_stays_unknown(self):
        result = inventory.derive(self.out)
        for component in ('system', 'cpu', 'ram', 'dimms', 'storage', 'interfaces', 'gpus'):
            self.assertEqual(result[component]['status'], 'UNKNOWN')
        self.assertTrue(result['missing_evidence'])
        self.assertEqual(result['gpus']['devices'], [])
        self.assertNotIn('B300', json.dumps(result['gpus']))

    def test_unrecognized_formats_are_not_guessed(self):
        self.server_fixture()
        for label in ('cpu', 'ram', 'storage', 'nic-link', 'smbios'):
            self.change_stream(label, 'Unrecognized synthetic output')
        result = inventory.derive(self.out)
        self.assertEqual(result['cpu']['status'], 'UNKNOWN')
        self.assertIsNone(result['ram']['os_reported_total_bytes'])
        self.assertEqual(result['storage']['disks'], [])
        self.assertEqual(result['interfaces']['interfaces'], [])
        self.assertEqual(result['dimms']['devices'], [])

    def test_unknown_localized_cpu_fields_and_empty_interfaces_stay_unknown(self):
        self.server_fixture()
        self.change_stream('cpu', json.dumps({'lscpu': [{'field': 'unrecognized localized field:', 'data': '128'}]}))
        self.change_stream('nic-link', '[]')
        result = inventory.derive(self.out)
        self.assertEqual(result['cpu']['status'], 'UNKNOWN')
        self.assertIsNone(result['cpu']['logical_cpus'])
        self.assertEqual(result['interfaces']['status'], 'UNKNOWN')
        self.assertIsNone(result['interfaces']['physical_nic_card_count'])

    def test_escape_in_server_raw_record_leaves_observations_unknown(self):
        summary = self.server_fixture()
        summary['commands']['cpu']['files']['stdout']['path'] = '../outside-cpu.txt'
        server_probe.save(self.out / 'server-summary.json', summary)
        result = inventory.derive(self.out)
        self.assertEqual(result['cpu']['status'], 'UNKNOWN')
        self.assertEqual(result['system']['records'], [])
        self.assertEqual(result['source_evidence_sha256'], {})

    def test_permission_denied_stdout_cannot_supply_system_claims(self):
        self.server_fixture()
        self.change_stream('smbios', SMBIOS, returncode=1, status='UNAVAILABLE', error='synthetic permission denied')
        result = inventory.derive(self.out)
        self.assertEqual(result['system']['records'], [])
        self.assertEqual(result['dimms']['devices'], [])
        self.assertIn('permission denied', json.dumps(result['missing_evidence']))

    def test_stderr_warning_keeps_component_unknown(self):
        self.server_fixture()
        self.change_stream('cpu', 'synthetic tool warning', kind='stderr')
        self.assertEqual(inventory.derive(self.out)['cpu']['status'], 'UNKNOWN')

    def test_tampered_raw_invalidates_server_cached_claims(self):
        self.server_fixture()
        (self.out / 'server-evidence/cpu.stdout.txt').write_text('modified', encoding='utf-8')
        result = inventory.derive(self.out)
        self.assertEqual(result['cpu']['status'], 'UNKNOWN')
        self.assertEqual(result['storage']['disks'], [])
        self.assertIn('hash-validated', json.dumps(result['missing_evidence']))

    def test_cached_identity_summary_alone_is_not_trusted(self):
        server_probe.save(self.out / 'identity-summary.json', {'status': 'CONSISTENT', 'per_gpu': [{'name': 'NVIDIA B300', 'serial': 'invented'}]})
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['devices'], [])
        self.assertIsNone(result['gpus']['observed_count'])

    def test_gpu_name_comes_from_validated_raw_even_when_contradictory(self):
        self.identity_fixture(capability_name='SYNTHETIC B200 DEVICE')
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['devices'][0]['model_name'], 'SYNTHETIC B200 DEVICE')
        self.assertEqual(result['gpus']['identity_verdict'], 'FAIL')
        self.assertEqual(result['gpus']['status'], 'PARTIAL')
        self.assertEqual(result['procurement_checks_cleared'], [])

    def test_cached_identity_metadata_cannot_rewrite_observed_raw_name(self):
        self.identity_fixture()
        path = self.out / 'identity-host.json'
        host = json.loads(path.read_text(encoding='utf-8'))
        host['observations']['gpus'][0]['name'] = 'SYNTHETIC CACHE TAMPER'
        server_probe.save(path, host)
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['devices'][0]['model_name'], 'NVIDIA B300 SXM6 AC')
        self.assertEqual(result['gpus']['identity_verdict'], 'FAIL')

    def test_gpu_raw_hash_tampering_stays_unknown(self):
        self.identity_fixture()
        (self.out / 'identity-evidence/nvidia-smi-xml.stdout.txt').write_text('<modified/>', encoding='utf-8')
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['devices'], [])
        self.assertEqual(result['gpus']['status'], 'UNKNOWN')

    def test_hashed_malformed_gpu_xml_is_unknown_and_does_not_crash_report(self):
        self.identity_fixture()
        host_path = self.out / 'identity-host.json'
        host = json.loads(host_path.read_text(encoding='utf-8'))
        record = host['commands']['nvidia-smi-xml']['files']['stdout']
        raw_path = self.out / record['path']
        raw_path.write_text('<malformed', encoding='utf-8')
        record['sha256'] = digest(raw_path)
        server_probe.save(host_path, host)
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['status'], 'UNKNOWN')
        self.assertEqual(result['gpus']['devices'], [])

    def test_unavailable_fields_and_placeholder_dimm_values_stay_unknown(self):
        parsed = inventory.parse_smbios(SMBIOS.replace('Size: 128 GB', 'Size: Unknown').replace('SYNTHETIC OEM', 'Unknown'))
        self.assertIsNone(parsed['systems'][0]['manufacturer'])
        self.assertIsNone(parsed['dimms'][0]['populated'])
        self.assertIsNone(parsed['dimms'][0]['size_value'])
        self.assertEqual(parsed['dimms'][0]['size_raw'], 'Unknown')

    def test_malformed_numeric_fields_do_not_supply_capacity(self):
        self.assertIsNone(inventory.positive_integer(True))
        self.assertIsNone(inventory.positive_integer('64 GiB'))
        with self.assertRaises(ValueError):
            inventory.parse_free(FREE.replace('68719476736', '-1'))
        parsed = inventory.parse_lsblk(STORAGE.replace('1000000000000', 'true'))
        self.assertIsNone(parsed['disks'][0]['size_bytes'])

    def test_duplicate_disks_and_conflicting_cpu_counts_are_rejected(self):
        disk = json.loads(STORAGE)['blockdevices'][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate disk'):
            inventory.parse_lsblk(json.dumps({'blockdevices': [disk, disk]}))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            inventory.parse_lscpu(json.dumps({'lscpu': [{'field': 'CPU(s):', 'data': '8'}, {'field': 'CPU(s):', 'data': '16'}]}))

    def test_inventory_mode_is_accepted_without_fabricated_functional_checks(self):
        self.server_fixture(mode='inventory')
        result = inventory.derive(self.out)
        self.assertEqual(result['cpu']['model_name'], 'SYNTHETIC CPU')
        self.assertEqual(result['comparison_status'], 'NOT_COMPARED')
        self.assertNotIn('PASS', json.dumps(result['cpu']))

    def test_gpu_expected_count_is_from_plan_not_fake_purchase_bom(self):
        self.identity_fixture(count=2)
        result = inventory.derive(self.out)
        self.assertEqual(result['gpus']['expected_gpus_from_plan'], 2)
        self.assertEqual(result['gpus']['observed_count'], 2)
        self.assertEqual(result['purpose'], 'OBSERVED_NOT_PURCHASE_BOM')
        (self.out / 'plan.json').unlink()
        result = inventory.derive(self.out)
        self.assertIsNone(result['gpus']['expected_gpus_from_plan'])
        self.assertTrue(any('Expected GPU count' in item['detail'] for item in result['missing_evidence']))


if __name__ == '__main__':
    unittest.main()
