"""Synthetic evidence only; no GPU measurements or procurement certification."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import authenticity
import dashboard
import gpu_evidence
import hardware_health
import hardware_profiles as hp
import report
import run_authenticity
import test_authenticity as fixture
from test_dashboard import readiness_fixture


class GenericIdentityTests(fixture.IdentityTests):
    # Inherit fixture helpers, but do not duplicate the original test cases.
    def profile_fixture(self, name, cap, count=1, memory='81920 MiB'):
        with patch.object(authenticity.platform, 'system', return_value='Linux'), patch.object(authenticity.platform, 'release', return_value='synthetic-kernel'), patch.object(authenticity.platform, 'machine', return_value='x86_64'), patch.object(authenticity.platform, 'node', return_value='synthetic-host'):
            self.collect_fixture(count=count, capability_name=name, memory=memory)
        for rank in range(count):
            self.update_gpu(rank, name=name, capability=cap, total_memory_bytes=authenticity.memory_bytes(memory))

    def test_known_models_and_capacity_are_not_b300_requirements(self):
        for name, cap, memory in [('NVIDIA H100', [9, 0], '81920 MiB'), ('NVIDIA A100', [8, 0], '40960 MiB'), ('NVIDIA L40S', [8, 9], '49152 MiB'), ('NVIDIA GeForce RTX 4090', [8, 9], '24576 MiB')]:
            with self.subTest(name=name):
                self.profile_fixture(name, cap, memory=memory)
                result = authenticity.evaluate_identity(self.out, 1, 'auto')
                self.assertEqual(result['status'], 'CONSISTENT', result['findings'])
                self.assertFalse(result['purchase_model_verified'])
                self.assertNotIn('memory_outside_conservative_range', self.codes(result))
                self.assertNotIn('unknown_pci_variant', self.codes(result))

    def test_expected_model_contradiction_fails(self):
        self.profile_fixture('NVIDIA H100', [9, 0])
        result = authenticity.evaluate_identity(self.out, 1, 'h200')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('host_name_mismatch', self.codes(result))

    def test_known_generic_architecture_contradiction_fails(self):
        self.profile_fixture('NVIDIA H100', [8, 9])
        self.assertEqual(authenticity.evaluate_identity(self.out, 1, 'auto')['status'], 'FAIL')

    def test_unknown_model_is_review_and_cannot_borrow_b300_identity(self):
        self.profile_fixture('NVIDIA UNKNOWN-SKU', [7, 0])
        result = authenticity.evaluate_identity(self.out, 1, 'auto')
        self.assertEqual(result['status'], 'REVIEW')
        self.assertIn('unlisted_model', self.codes(result))

    def test_sixteen_gpu_mapping_and_missing_card(self):
        self.profile_fixture('NVIDIA H100', [9, 0], count=16)
        self.assertEqual(authenticity.evaluate_identity(self.out, 16, 'auto')['status'], 'CONSISTENT')
        result = authenticity.evaluate_identity(self.out, 17, 'h100')
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('host_gpu_count', self.codes(result))

    def test_t4_fp32_reference_without_bf16(self):
        self.profile_fixture('NVIDIA T4', [7, 5], memory='16384 MiB')
        self.update_gpu(correctness=dict(fp32_cpu_reference=True, integer_memory_pattern=True, bf16_supported=False, bf16_tensor_matrix=None))
        self.assertEqual(authenticity.evaluate_identity(self.out, 1, 'auto')['status'], 'CONSISTENT')

    def test_h100_environment_is_bound_and_validated(self):
        self.profile_fixture('NVIDIA H100', [9, 0])
        host = json.loads((self.out / 'identity-host.json').read_text(encoding='utf-8'))
        raw = authenticity.evidence_bytes(self.out, host['commands']['nvidia-smi-xml']['files']['stdout'])
        (self.out / 'before.xml').write_bytes(raw)
        identity = json.loads((self.out / 'identity-gpu0.json').read_text(encoding='utf-8'))
        result = dict(schema_version=2, run_id='test-only', test='environment', status='PASS', rank=0, world_size=1,
                      uuid=identity['uuid'], pci_bus_id=identity['pci_bus_id'], capability=[9, 0],
                      total_memory_bytes=identity['total_memory_bytes'], hardware_snapshot_sha256=authenticity.sha256(self.out/'before.xml'),
                      identity_host_sha256=authenticity.sha256(self.out/'identity-host.json'), gpu='NVIDIA H100',
                      memory_gib=identity['total_memory_bytes']/2**30, torch='synthetic', cuda='synthetic', nccl=[2, 27, 5], bf16_correctness=True)
        plan=dict(gpus=1, hardware_profile='auto', run_id='test-only', required=['identity'], result_contract=2)
        self.assertEqual(gpu_evidence.validate_gpu_result(result, plan, self.out), [])
        result['capability']=[10, 3]
        with self.assertRaises(ValueError):
            gpu_evidence.validate_gpu_result(result, plan, self.out)


# unittest scans inherited methods. Remove duplicate originals while retaining
# the fixture class' helpers/setup/teardown.
for name in dir(fixture.IdentityTests):
    if name.startswith('test_') and name not in GenericIdentityTests.__dict__:
        setattr(GenericIdentityTests, name, None)

def workspace_temp():
    base=Path(__file__).parent/'.test-tmp'
    base.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


class GenericControlTests(unittest.TestCase):
    def test_nvidia_card_counts_are_adaptive(self):
        with patch.object(dashboard.sys, 'platform', 'linux'):
            for count in (1, 2, 4, 7, 8, 16, 64):
                ready=readiness_fixture(count)
                for gpu in ready['gpus']:
                    gpu.update(name='NVIDIA H100', memory_mib=81920)
                self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP['authenticity'], ready)['available'])
                self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP['compare'], ready)['available'])

    def test_rtx_without_ecc_and_without_nvlink_can_run_hardware(self):
        ready=readiness_fixture(1)
        ready['gpus'][0].update(name='NVIDIA GeForce RTX 4090', ecc='N/A', mig='N/A')
        with patch.object(dashboard.sys, 'platform', 'linux'):
            self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP['authenticity'], ready)['available'])

    def test_cpu_workflow_needs_neither_docker_nor_driver(self):
        ready=readiness_fixture(0)
        ready['tools'].update(docker=False,nvidia_smi=False,dcgmi=False)
        ready['image']['available']=False
        with patch.object(dashboard.sys, 'platform', 'linux'):
            self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP['server'], ready)['available'])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP['authenticity'], ready)['available'])

    def test_mig_ecc_disabled_and_busy_still_block(self):
        for field,value in [('mig','Enabled'),('ecc','Disabled')]:
            ready=readiness_fixture(1)
            ready['gpus'][0].update(name='NVIDIA H100', **{field:value})
            with patch.object(dashboard.sys,'platform','linux'):
                self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP['authenticity'],ready)['available'])

    def test_profile_matching_respects_model_boundaries(self):
        self.assertFalse(hp.matches('h100','NVIDIA H1000'))
        self.assertFalse(hp.matches('l40','NVIDIA L40S'))
        self.assertEqual(hp.identify('NVIDIA L40S'), 'l40s')
        self.assertFalse(hp.public_compatible([dict(name='NVIDIA H100')]*8))
        self.assertTrue(hp.public_compatible([dict(name='NVIDIA B300')]*8))

    def test_request_parameters_are_validated_before_launch(self):
        with workspace_temp() as tmp:
            manager=dashboard.JobManager(Path(tmp),preview=True)
            for field,value in [('hardware_profile','../../bad'),('expected_gpus',True),('expected_gpus',65),('memory_fraction',float('nan')),('max_temp',120)]:
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    manager.start(dict(profile='server',**{field:value}))
            self.assertFalse(manager.jobs)

    def test_command_preserves_reviewed_job_parameters(self):
        with workspace_temp() as tmp:
            manager=dashboard.JobManager(Path(tmp),preview=True)
            manager.active='synthetic'
            manager.jobs['synthetic']=dict(gpus=4,hardware_profile='h100',memory_fraction=.7,max_temp=80)
            with patch.object(manager,'ready',return_value=readiness_fixture(4)):
                cmd=manager.command('authenticity',Path(tmp),7200)
            for option,value in [('--gpus','4'),('--hardware-profile','h100'),('--memory-fraction','0.7'),('--seconds','7200'),('--max-temp','80')]:
                self.assertEqual(cmd[cmd.index(option)+1],value)

    def test_cpu_workflow_never_dispatches_gpu_or_docker(self):
        with workspace_temp() as tmp:
            args=SimpleNamespace(out=Path(tmp),profile='server',gpus=0,hardware_profile='auto',seconds=1,image='unused',memory_fraction=.85,max_temp=85)
            flow=run_authenticity.Workflow(args)
            dispatched=[]
            def execute(label,command,*rest):
                dispatched.append(command)
                flow.record(label,'PASS')
            with patch.object(flow,'execute',side_effect=execute), patch.object(flow,'capture'), patch.object(flow,'stop_container'), patch.object(run_authenticity.signal,'signal'), patch.object(report,'generate',return_value={'overall':'REVIEW'}), patch.object(run_authenticity,'export_evidence'):
                self.assertEqual(flow.run(),2)
            self.assertEqual(len(dispatched),3)
            self.assertTrue(all(command[0] != 'docker' and 'nvidia-smi' not in command for command in dispatched))
            plan=json.loads((Path(tmp)/'plan.json').read_text(encoding='utf-8'))
            self.assertEqual(plan['gpus'],0)
            self.assertNotIn('identity',plan['required'])

    def test_cpu_report_has_no_false_gpu_telemetry_requirement(self):
        with workspace_temp() as tmp:
            out=Path(tmp)
            authenticity.write_json(out/'plan.json',dict(profile='server',gpus=0,hardware_profile='auto',required=[],start_epoch=100,result_contract=2))
            authenticity.write_json(out/'collection-status.json',{'kernel-during.log':dict(argv=['journalctl','-k','--since','@100'],returncode=0,error=None)})
            (out/'kernel-during.log').write_text('kernel: synthetic normal event',encoding='utf-8')
            (out/'kernel-during.log.stderr').write_text('',encoding='utf-8')
            result=report.generate(out)
            self.assertEqual(result['telemetry']['status'],'NOT_APPLICABLE')
            self.assertFalse(result['review'],result['review'])
            self.assertEqual(result['acceptance']['status'],'NOT_ACCEPTED')
            self.assertEqual(result['observed_inventory']['gpus']['status'],'NOT_TESTED')

    def test_absent_consumer_features_do_not_become_fake_failures(self):
        with workspace_temp() as tmp:
            out=Path(tmp)
            from test_authenticity import fixture_xml
            raw=fixture_xml(capability_name='NVIDIA GeForce RTX 4090').replace(b'<current_ecc>Enabled</current_ecc>',b'<current_ecc>N/A</current_ecc>')
            for name in ('before.xml','after.xml'):
                (out/name).write_bytes(raw)
            for name in ('nvlink.txt','nvlink-errors-before.txt','nvlink-errors-after.txt'):
                (out/name).write_text('NVLink: Not Supported',encoding='utf-8')
            failures,review=hardware_health.evaluate(out,strict=True,generic=True)
            self.assertFalse(failures)
            self.assertFalse(review,review)
