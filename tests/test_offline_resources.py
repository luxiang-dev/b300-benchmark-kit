"""Tiny synthetic offline resources; no downloads, installations or GPU results."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle
import dashboard
import offline_manifest as manifest
import offline_resources as offline
from test_kit import workspace_temp
from test_dashboard import readiness_fixture


class OfflineResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=workspace_temp()
        self.root=Path(self.tmp.name).resolve()
        for name in bundle.CODE:
            if name.startswith('images/'):
                continue
            path=self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('synthetic source only',encoding='utf-8')
        self.content=b'tiny synthetic model'
        self.lock=dict(models=[dict(name='tiny',repo='synthetic/only',revision='pinned',files=[dict(path='weights.bin',size=len(self.content),hash_type='sha256',hash=hashlib.sha256(self.content).hexdigest())])])
        for name in ('models.lock.json','deepseek-model.lock.json'):
            (self.root/name).write_text(json.dumps(self.lock),encoding='utf-8')
        (self.root/'models/tiny').mkdir(parents=True)
        (self.root/'models/tiny/weights.bin').write_bytes(self.content)

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self,scope='all',hashes=False):
        return offline.audit(self.root,scope,hashes,inspect_host=False)

    def test_missing_resources_never_report_ready(self):
        result=self.audit()
        self.assertEqual(result['overall'],'INCOMPLETE')
        self.assertFalse(result['offline_ready'])
        self.assertEqual(result['runtime_validation'],'NOT_RUN')
        self.assertIn('Base GPU image archive: missing',result['incomplete'])
        self.assertTrue(any('MLPERF_REPO' in gap for gap in result['incomplete']))
        self.assertEqual(result['acceptance']['status'],'BLOCKED')

    def test_server_scope_needs_no_images_or_models(self):
        (self.root/'models/tiny/weights.bin').unlink()
        result=self.audit('server',True)
        self.assertEqual(result['overall'],'REVIEW')
        self.assertEqual(result['models'],[])
        self.assertFalse(result['offline_ready'])

    def test_presence_does_not_become_hash_verified(self):
        self.assertEqual(self.audit('full')['models'][0]['status'],'PRESENT_UNVERIFIED')

    def test_source_hash_corruption_with_unchanged_size_fails(self):
        (self.root/'models/tiny/weights.bin').write_bytes(b'x'*len(self.content))
        with patch('sys.stdout',new=io.StringIO()):
            result=self.audit('full',True)
        self.assertEqual(result['overall'],'FAIL')
        self.assertIn('source hash mismatch',str(result['failures']))

    def test_model_source_hash_success_still_requires_target_runtime(self):
        with patch('sys.stdout',new=io.StringIO()):
            result=self.audit('full',True)
        self.assertEqual(result['models'][0]['status'],'HASH_VERIFIED')
        self.assertEqual(result['overall'],'INCOMPLETE')
        self.assertFalse(result['offline_ready'])

    def test_missing_weight_and_unsafe_lock_paths_are_reported(self):
        for relative in ('missing.bin','../outside.bin'):
            self.lock['models'][0]['files'][0]['path']=relative
            (self.root/'models.lock.json').write_text(json.dumps(self.lock),encoding='utf-8')
            result=self.audit('full')
            self.assertIn(result['overall'],('INCOMPLETE','FAIL'))
            self.assertTrue(result['models'][0]['missing'] or result['models'][0]['invalid'])

    def test_seal_presence_does_not_execute_untrusted_package_code(self):
        (self.root/'bundle.sha256.json').write_text('{}',encoding='utf-8')
        with patch.object(offline.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','corrupt seal')) as proc:
            result=self.audit('full',True)
        self.assertEqual(result['overall'],'FAIL')
        self.assertEqual(proc.call_args.args[0][1],str(Path(offline.__file__).with_name('bundle.py')))
        self.assertNotEqual(proc.call_args.args[0][1],str(self.root/'bundle.py'))

    def test_malformed_deepseek_lock_keeps_a_failure_record(self):
        (self.root/'deepseek-model.lock.json').write_text('not json',encoding='utf-8')
        self.assertEqual(self.audit()['overall'],'FAIL')

    def test_static_asset_missing_and_stale_snapshot_are_not_readiness(self):
        (self.root/'dashboard/app.js').unlink()
        (self.root/'PACKAGE_STATUS.json').write_text('{"offline_ready":true}',encoding='utf-8')
        self.assertTrue(any('dashboard/app.js' in x for x in self.audit('server')['incomplete']))
        self.assertFalse(self.audit('server')['offline_ready'])

    def test_cli_receipt_and_portable_report_keep_untrusted_text_literal(self):
        output=self.root/'out'
        with patch.object(sys,'argv',['offline_resources.py','--root',str(self.root),'--scope','all','--out',str(output)]),patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(offline.main(),2)
        summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        summary['incomplete']=['<script>bad()</script>']
        rendered=offline.report(summary)
        self.assertNotIn('<script>',rendered)
        self.assertIn('&lt;script&gt;',rendered)
        self.assertTrue((output/'report.html').is_file())

    def test_dashboard_audit_needs_no_gpu_or_docker_and_uses_fixed_command(self):
        ready=readiness_fixture(0)
        ready['tools'].update(docker=False,nvidia_smi=False)
        with patch.object(dashboard.sys,'platform','linux'):
            self.assertTrue(dashboard.profile_eligibility(dashboard.PROFILE_MAP['offline-check'],ready)['available'])
            self.assertFalse(dashboard.profile_eligibility(dashboard.PROFILE_MAP['offline-check'],ready,True)['available'])
        manager=dashboard.JobManager(self.root/'runs',preview=True)
        try:
            manager.active='fixture'
            manager.jobs['fixture']=dict(offline_scope='hardware',verify_hashes=True)
            command=manager.command('offline-check',self.root/'report',600)
            self.assertIn('--verify-hashes',command)
            self.assertEqual(command[command.index('--scope')+1],'hardware')
            self.assertNotIn('bash',command)
        finally:
            manager.active=None
            manager.close()

    def test_dashboard_invalid_audit_arguments_rejected_before_any_work(self):
        manager=dashboard.JobManager(self.root/'runs',preview=True)
        try:
            for field,value in (('offline_scope','../../run'),('offline_scope',[]),('verify_hashes','false'),('verify_hashes',1)):
                with self.assertRaises(ValueError):
                    manager.start(dict(profile='offline-check',**{field:value}))
            self.assertFalse(manager.jobs)
        finally:
            manager.close()


class ExternalManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=workspace_temp()
        self.root=Path(self.tmp.name).resolve()
        (self.root/'empty').mkdir()
        (self.root/'.git').mkdir()
        (self.root/'.git/HEAD').write_text('synthetic only',encoding='utf-8')
        (self.root/'data').write_bytes(b'123')
        self.seal=self.root/'manifest.json'

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_includes_hidden_git_empty_dirs_and_excludes_self(self):
        record=manifest.seal(self.root,self.seal)
        self.assertIn('.git/HEAD',record['entries'])
        self.assertIn('empty',record['entries'])
        self.assertNotIn('manifest.json',record['entries'])
        self.assertEqual(manifest.verify(self.root,self.seal),len(record['entries']))

    def test_same_size_corruption_fails(self):
        manifest.seal(self.root,self.seal)
        (self.root/'data').write_bytes(b'321')
        with self.assertRaisesRegex(ValueError,'mismatch'):
            manifest.verify(self.root,self.seal)

    def test_added_or_removed_resource_fails(self):
        manifest.seal(self.root,self.seal)
        (self.root/'extra').write_text('extra',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'inventory mismatch'):
            manifest.verify(self.root,self.seal)
        (self.root/'extra').unlink()
        (self.root/'empty').rmdir()
        with self.assertRaisesRegex(ValueError,'inventory mismatch'):
            manifest.verify(self.root,self.seal)

    def test_untrusted_manifest_paths_and_shapes_rejected(self):
        for key in ('../outside','/etc/passwd','C:/secret','a\\b'):
            self.seal.write_text(json.dumps(dict(schema=1,entries={key:{}})),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                manifest.verify(self.root,self.seal)
        self.seal.write_text('[]',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Invalid'):
            manifest.verify(self.root,self.seal)

    @unittest.skipUnless(sys.platform=='linux','Requires POSIX link and permission semantics')
    def test_dangling_external_links_recorded_without_following_and_mode_preserved(self):
        (self.root/'link').symlink_to('/missing/container/python')
        (self.root/'data').chmod(0o755)
        record=manifest.seal(self.root,self.seal)
        self.assertEqual(record['entries']['link'],dict(kind='link',target='/missing/container/python'))
        manifest.verify(self.root,self.seal)
        (self.root/'data').chmod(0o644)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            manifest.verify(self.root,self.seal)

    @unittest.skipUnless(sys.platform=='linux','Requires POSIX special files')
    def test_special_device_or_fifo_refused(self):
        os.mkfifo(self.root/'pipe')
        with self.assertRaisesRegex(ValueError,'special'):
            manifest.seal(self.root,self.seal)


if __name__=='__main__':
    unittest.main()
