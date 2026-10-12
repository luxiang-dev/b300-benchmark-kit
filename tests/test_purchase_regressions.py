"""Private CPU fixtures for purchase-review failures, never hardware measurements."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET

import report
import gpu_evidence
import memory_probe
import dcgm_evidence
import hardware_health


class PurchaseHardwareRegressions(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1] / 'b300-results' / 'purchase-regressions'
        work.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.out = Path(self.tmp.name)
        self.before = self.snapshot()
        self.after = self.snapshot()
        (self.out / 'kernel-during.log').write_text('kernel: private fixture\n', encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        tree = ET.Element('nvidia_smi_log')
        for rank in range(8):
            gpu = ET.SubElement(tree, 'gpu')
            ET.SubElement(gpu, 'uuid').text = f'GPU-00000000-0000-0000-0000-{rank:012d}'
            ecc = ET.SubElement(gpu, 'ecc_errors')
            for scope in ('volatile', 'aggregate'):
                values = ET.SubElement(ecc, scope)
                for field in ('dram_correctable', 'dram_uncorrectable', 'sram_correctable', 'sram_uncorrectable_parity', 'sram_uncorrectable_sec_ded'):
                    ET.SubElement(values, field).text = '0'
            remap = ET.SubElement(gpu, 'remapped_rows')
            for field, value in [('correctable_error', '0'), ('uncorrectable_error', '0'), ('pending', 'No'), ('remapping_failure_occurred', 'No')]:
                ET.SubElement(remap, field).text = value
            retirement = ET.SubElement(gpu, 'retired_pages')
            ET.SubElement(retirement, 'pending_retirement').text = 'No'
        return tree

    def change(self, tree, path, value):
        tree.find('gpu/' + path).text = str(value)

    def issues(self, strict=False):
        for name, tree in [('before.xml', self.before), ('after.xml', self.after)]:
            ET.ElementTree(tree).write(self.out / name, encoding='utf-8')
        failures, review = report.hardware_issues(self.out, 8, strict=strict)
        return failures + review

    def assert_issue(self, *needles, strict=False):
        issues = self.issues(strict)
        self.assertTrue(any(any(needle.lower() in issue.lower() for needle in needles) for issue in issues), issues)

    def test_lifetime_uncorrectable_ecc_cannot_disappear_behind_zero_volatile(self):
        for tree in (self.before, self.after):
            self.change(tree, 'ecc_errors/aggregate/dram_uncorrectable', 4)
        self.assert_issue('aggregate', '寿命', '历史')

    def test_lifetime_ecc_increment_is_detected_if_driver_reset_volatile(self):
        self.change(self.after, 'ecc_errors/aggregate/dram_uncorrectable', 1)
        self.assert_issue('aggregate', '寿命')

    def test_correctable_ecc_growth_requires_purchase_review(self):
        self.change(self.after, 'ecc_errors/volatile/dram_correctable', 7)
        self.change(self.after, 'ecc_errors/aggregate/dram_correctable', 7)
        self.assert_issue('correctable', '可纠正')

    def test_pending_row_remapping_blocks_clear_health(self):
        self.change(self.after, 'remapped_rows/pending', 'Yes')
        self.assert_issue('重映射', 'remap')

    def test_existing_row_remap_requires_disclosure(self):
        for tree in (self.before, self.after):
            self.change(tree, 'remapped_rows/uncorrectable_error', 3)
        self.assert_issue('重映射', 'remap')

    def test_new_row_remap_cannot_be_silently_ignored(self):
        self.change(self.after, 'remapped_rows/correctable_error', 1)
        self.assert_issue('重映射', 'remap')

    def test_nvlink5_effective_errors_are_reported(self):
        prefix = 'GPU 0: NVIDIA B300 (UUID: GPU-00000000-0000-0000-0000-000000000000)\n    Link 0:\n'
        for name, count in [('before', 0), ('after', 3)]:
            (self.out / f'nvlink-errors-{name}.txt').write_text(prefix + f'        Effective Errors: {count}\n', encoding='utf-8')
        self.assert_issue('NVLink', 'nvlink')

    def test_nvlink_counter_reset_requires_review(self):
        prefix = 'GPU 0: NVIDIA B300 (UUID: GPU-00000000-0000-0000-0000-000000000000)\n    Link 0:\n'
        for name, count in [('before', 3), ('after', 0)]:
            (self.out / f'nvlink-errors-{name}.txt').write_text(prefix + f'        Effective Errors: {count}\n', encoding='utf-8')
        self.assert_issue('NVLink', 'nvlink')

    def fabric(self, tree):
        fabric = ET.SubElement(tree.find('gpu'), 'fabric')
        ET.SubElement(fabric, 'state').text = 'Completed'
        ET.SubElement(fabric, 'status').text = 'Success'
        health = ET.SubElement(fabric, 'health')
        for field, value in [('bandwidth', 'Full'), ('route_recovery_in_progress', 'False'),
                             ('route_unhealthy', 'False'), ('access_timeout_recovery', 'False')]:
            ET.SubElement(health, field).text = value
        return fabric

    def test_completed_fabric_probe_does_not_hide_degraded_bandwidth(self):
        self.fabric(self.before)
        fabric = self.fabric(self.after)
        fabric.find('health/bandwidth').text = 'Degraded'
        self.assert_issue('Degraded', '降级', 'bandwidth')

    def test_completed_fabric_probe_does_not_hide_unhealthy_or_recovering_routes(self):
        self.fabric(self.before)
        fabric = self.fabric(self.after)
        for field in ('route_unhealthy', 'route_recovery_in_progress', 'access_timeout_recovery'):
            with self.subTest(field=field):
                fabric.find('health/' + field).text = 'True'
                self.assert_issue(field)
                fabric.find('health/' + field).text = 'False'

    def test_gpu_recovery_action_prevents_clear_health(self):
        ET.SubElement(self.before.find('gpu'), 'gpu_recovery_action').text = 'None'
        ET.SubElement(self.after.find('gpu'), 'gpu_recovery_action').text = 'Reset'
        self.assert_issue('gpu_recovery_action', '恢复', 'Recovery')

    def test_strict_snapshot_requires_dram_uncorrectable_not_just_any_ecc_counter(self):
        for tree in (self.before, self.after):
            self.change(tree, 'ecc_errors/aggregate/dram_uncorrectable', 'N/A')
        self.issues()
        failures, reviews = hardware_health.evaluate(self.out, strict=True)
        self.assertTrue(any('dram_uncorrectable' in item for item in failures + reviews), failures + reviews)

    def test_inactive_nvlink_is_visible_in_health_review(self):
        (self.out / 'nvlink.txt').write_text('GPU 0: NVIDIA B300 (UUID: GPU-00000000-0000-0000-0000-000000000000)\n    Link 0: <inactive>\n', encoding='utf-8')
        self.assert_issue('Inactive', 'inactive', '未激活')

    def test_nonempty_kernel_log_does_not_hide_partial_journal_access(self):
        (self.out / 'plan.json').write_text(json.dumps(dict(result_contract=2, gpus=8)), encoding='utf-8')
        (self.out / 'collection-status.json').write_text(json.dumps({'kernel-during.log': dict(
            argv=['journalctl', '-k', '--since', '@1', '--no-pager'], returncode=0, error=None)}), encoding='utf-8')
        (self.out / 'kernel-during.log.stderr').write_text('Hint: You are currently not seeing messages from the system.\n', encoding='utf-8')
        self.assert_issue('kernel', '内核', strict=True)


class BoundGpuEvidenceRegressions(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[1] / 'b300-results' / 'purchase-regressions'
        work.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.out = Path(self.tmp.name)
        self.uuid = 'GPU-00000000-0000-0000-0000-000000000001'
        self.total = 260 * 2**30
        raw = ('<nvidia_smi_log><gpu><uuid>' + self.uuid + '</uuid>'
               '<product_name>NVIDIA B300</product_name><pci><pci_bus_id>0000:01:00.0</pci_bus_id></pci>'
               '<fb_memory_usage><total>266240 MiB</total></fb_memory_usage></gpu></nvidia_smi_log>')
        (self.out / 'before.xml').write_text(raw, encoding='utf-8')
        self.plan = dict(result_contract=2, run_id='private-regression-run', gpus=1, required=['stress'],
                         seconds=600, max_gpu_temp=85, memory_fraction=.85, stress_memory_fraction=.65,
                         nccl_iterations=50, matrix=8192)
        self.common = dict(schema_version=2, run_id=self.plan['run_id'], uuid=self.uuid,
                           hardware_snapshot_sha256=hashlib.sha256((self.out / 'before.xml').read_bytes()).hexdigest(),
                           pci_bus_id='0000:01:00.0', capability=[10, 3], total_memory_bytes=self.total,
                           rank=0, world_size=1, model='', status='PASS')

    def tearDown(self):
        self.tmp.cleanup()

    def stress(self):
        return dict(self.common, test='stress', duration_s=601, requested_seconds=600, matrix=8192, dtype='BF16',
                    memory_tested_gib=int(self.total * .65) // memory_probe.CHUNK_BYTES * .25,
                    gemm_iterations=20, gemm_tflops=1, peak_temperature_c=70, memory_patterns=[17, -31])

    def memory(self, free=None):
        free = self.total if free is None else free
        allocated = memory_probe.allocation_budget(self.total, free, .85)
        return dict(self.common, test='memory', dtype='int32', patterns=list(memory_probe.PATTERNS),
                    total_memory_bytes=self.total, free_memory_before_bytes=free,
                    chunk_bytes=memory_probe.CHUNK_BYTES, scratch_bytes=memory_probe.CHUNK_BYTES,
                    reserve_bytes=memory_probe.RESERVE_BYTES, target_allocation_bytes=allocated,
                    allocated_bytes=allocated, verified_bytes_per_pass=allocated, total_verified_bytes=allocated * 3,
                    requested_memory_fraction=.85, coverage_percent=allocated / self.total * 100,
                    memory_tested_gib=allocated / 2**30,
                    free_memory_limited=allocated < int(self.total * .85) // memory_probe.CHUNK_BYTES * memory_probe.CHUNK_BYTES,
                    cpu_reference_positions_per_chunk=3,
                    cpu_reference_checks_per_pass=allocated // memory_probe.CHUNK_BYTES * 3,
                    duration_s=7, peak_temperature_c=70,
                    passes=[dict(pattern=pattern, written_bytes=allocated, verified_bytes=allocated,
                                 cpu_reference_checks=allocated // memory_probe.CHUNK_BYTES * 3,
                                 write_seconds=1, verify_seconds=1, status='PASS') for pattern in memory_probe.PATTERNS])

    def validate(self, result):
        return gpu_evidence.validate_gpu_result(result, self.plan, self.out)

    def test_complete_stress_has_no_review_finding(self):
        self.assertEqual(self.validate(self.stress()), [])

    def test_minimal_pass_status_is_not_execution_evidence(self):
        with self.assertRaises(ValueError):
            self.validate(dict(test='stress', rank=0, status='PASS'))

    def test_each_binding_field_is_required_and_checked(self):
        for field, value in [('schema_version', 1), ('run_id', 'old-run'), ('hardware_snapshot_sha256', '0' * 64),
                             ('uuid', 'GPU-11111111-1111-1111-1111-111111111111'), ('pci_bus_id', '0000:02:00.0'),
                             ('capability', [10, 0]), ('total_memory_bytes', 80 * 2**30), ('world_size', 8)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(dict(self.stress(), **{field: value}))

    def test_worker_identity_is_bound_to_cuda_identity_probe(self):
        host = self.out / 'identity-host.json'
        host.write_text('{"private_fixture":true}', encoding='utf-8')
        host_hash = hashlib.sha256(host.read_bytes()).hexdigest()
        self.plan['required'].append('identity')
        result = dict(self.stress(), identity_host_sha256=host_hash)
        with self.assertRaises(ValueError):
            self.validate(result)
        probe = dict(uuid=self.uuid, total_memory_bytes=self.total, pci_bus_id='0000:01:00.0',
                     host_sha256=host_hash, status='PASS', capability=[10, 3])
        path = self.out / 'identity-gpu0.json'
        path.write_text(json.dumps(probe), encoding='utf-8')
        self.assertEqual(self.validate(result), [])
        probe['total_memory_bytes'] -= 2**30
        path.write_text(json.dumps(probe), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate(result)

    def test_short_stress_and_nonfinite_metrics_are_rejected(self):
        for field, value in [('duration_s', 60), ('requested_seconds', 60), ('gemm_tflops', float('nan')), ('peak_temperature_c', 90),
                             ('gemm_iterations', 0), ('matrix', 256), ('memory_patterns', [17])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(dict(self.stress(), **{field: value}))

    def test_reduced_stress_memory_coverage_requires_review(self):
        self.assertTrue(self.validate(dict(self.stress(), memory_tested_gib=1)))

    def test_memory_full_requested_coverage_is_accepted(self):
        self.assertEqual(self.validate(self.memory()), [])

    def test_memory_low_free_capacity_is_valid_but_requires_review(self):
        result = self.memory(4 * 2**30)
        memory_probe.validate_evidence(result)
        self.assertTrue(self.validate(result))

    def test_memory_duration_must_cover_actual_phase_timings(self):
        with self.assertRaises(ValueError):
            self.validate(dict(self.memory(), duration_s=1))

    def test_memory_requires_each_chunk_cpu_spot_checks_in_each_pattern(self):
        for kind in ('positions', 'global_count', 'pattern_count', 'missing'):
            result = self.memory()
            if kind == 'positions':
                result['cpu_reference_positions_per_chunk'] = 1
            elif kind == 'global_count':
                result['cpu_reference_checks_per_pass'] -= 3
            elif kind == 'pattern_count':
                result['passes'][1]['cpu_reference_checks'] -= 3
            else:
                result.pop('cpu_reference_checks_per_pass')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(result)

    def test_plan_memory_fraction_cannot_be_lowered_inside_result(self):
        result = self.memory()
        result['requested_memory_fraction'] = .5
        with self.assertRaises(ValueError):
            self.validate(result)

    def test_nccl_requires_complete_workload_and_coherent_metrics(self):
        rows = [dict(size_mib=size, latency_ms=1, algorithm_gbps=size * 2**20 / .001 / 1e9,
                     bus_gbps=0, operation='MAX', sum_correctness='PASS') for size in (16, 64, 256)]
        result = dict(self.common, test='nccl', iterations=50, measurements=rows)
        self.assertEqual(self.validate(result), [])
        for mutate in ('missing_size', 'wrong_iterations', 'wrong_bandwidth', 'missing_correctness'):
            invalid = copy.deepcopy(result)
            if mutate == 'missing_size':
                invalid['measurements'].pop()
            elif mutate == 'wrong_iterations':
                invalid['iterations'] = 3
            elif mutate == 'wrong_bandwidth':
                invalid['measurements'][0]['algorithm_gbps'] *= 2
            else:
                invalid['measurements'][0].pop('sum_correctness')
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.validate(invalid)


class DcgmEntityCoverageRegressions(unittest.TestCase):
    setUp = BoundGpuEvidenceRegressions.setUp
    tearDown = BoundGpuEvidenceRegressions.tearDown

    def dcgm(self):
        return {'entity_groups': [dict(entity_group='GPU', entity_group_id=1,
                                      entities=[dict(entity_id=0, device_id='3182', serial_num='private-serial')])],
                'DCGM Diagnostic': {'test_categories': [dict(category='Private fixture', tests=[
                    dict(name=name, results=[dict(entity_group='GPU', entity_group_id=1, entity_id=0, status='Pass')],
                         test_summary=dict(status='Pass')) for name in sorted(dcgm_evidence.CORE_TESTS)])]}}

    def test_complete_dcgm_entity_and_core_test_coverage(self):
        self.assertEqual(dcgm_evidence.validate_dcgm_result(self.dcgm(), self.plan, self.out), [])

    def test_any_pass_without_entity_coverage_cannot_complete_dcgm(self):
        with self.assertRaises(ValueError):
            dcgm_evidence.validate_dcgm_result({'results': [{'status': 'Pass'}]}, self.plan, self.out)

    def test_dcgm_missing_core_test_or_gpu_result_is_rejected(self):
        for kind in ('test', 'entity', 'result'):
            invalid = self.dcgm()
            if kind == 'test':
                invalid['DCGM Diagnostic']['test_categories'][0]['tests'].pop()
            elif kind == 'entity':
                invalid['entity_groups'][0]['entities'] = []
            else:
                invalid['DCGM Diagnostic']['test_categories'][0]['tests'][0]['results'] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                dcgm_evidence.validate_dcgm_result(invalid, self.plan, self.out)

    def test_dcgm_skip_or_warning_remains_review(self):
        for kind in ('skip', 'warning'):
            invalid = self.dcgm()
            row = invalid['DCGM Diagnostic']['test_categories'][0]['tests'][0]['results'][0]
            if kind == 'skip':
                row['status'] = 'Skip'
            else:
                row['warnings'] = [dict(warning='private warning', error_id=1)]
            with self.subTest(kind=kind):
                self.assertTrue(dcgm_evidence.validate_dcgm_result(invalid, self.plan, self.out))

    def test_dcgm_failed_summary_cannot_be_hidden_behind_pass_entities(self):
        invalid = self.dcgm()
        invalid['DCGM Diagnostic']['test_categories'][0]['tests'][0]['test_summary']['status'] = 'Fail'
        with self.assertRaises(ValueError):
            dcgm_evidence.validate_dcgm_result(invalid, self.plan, self.out)


class ReportBoundEvidenceIntegration(unittest.TestCase):
    setUp = BoundGpuEvidenceRegressions.setUp
    tearDown = BoundGpuEvidenceRegressions.tearDown
    stress = BoundGpuEvidenceRegressions.stress

    def prepare_report(self):
        self.plan['profile'] = 'authenticity'
        (self.out / 'plan.json').write_text(json.dumps(self.plan), encoding='utf-8')
        (self.out / 'after.xml').write_bytes((self.out / 'before.xml').read_bytes())
        (self.out / 'kernel-during.log').write_text('kernel: private fixture\n', encoding='utf-8')
        (self.out / 'stages.tsv').write_text('stress\tPASS\t0\n', encoding='utf-8')

    def test_report_rejects_bare_pass_json_in_new_run_contract(self):
        self.prepare_report()
        (self.out / 'stress-gpu0.json').write_text(json.dumps(dict(test='stress', rank=0, model='', status='PASS')), encoding='utf-8')
        result = report.generate(self.out)
        self.assertEqual(result['overall'], 'FAIL', result)
        self.assertTrue(any('stress-gpu0.json' in item for item in result['failures']), result)

    def test_report_cannot_accept_duplicate_worker_files(self):
        self.prepare_report()
        for name in ('stress-gpu0.json', 'stress-gpu99.json'):
            (self.out / name).write_text(json.dumps(self.stress()), encoding='utf-8')
        result = report.generate(self.out)
        self.assertIn(result['overall'], ('FAIL', 'INCOMPLETE'), result)
        self.assertTrue(any('stress:' in item for item in result['failures'] + result['incomplete']), result)


if __name__ == '__main__':
    unittest.main()
