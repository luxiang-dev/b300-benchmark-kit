"""Ubuntu launcher integration with explicit fake hardware/Docker; no GPU scores."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle
import test_kit as kit

MOCK_TOOL = '''#!/usr/bin/env python3
import datetime,json,os,pathlib,sys,time
tool=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
if tool == 'nvidia-smi':
    if '--query-compute-apps=pid' in args:
        sys.exit(0)
    if any(x.startswith('--query-gpu=') for x in args):
        for i in range(8):
            print(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f') + f',{i},gpu{i},50,500,1000,98')
    elif '-x' in args:
        model=os.environ.get('MOCK_GPU_MODEL','NVIDIA B300')
        print('<nvidia_smi_log><driver_version>580.95.05</driver_version>' + ''.join(
          f'<gpu><uuid>gpu{i}</uuid><product_name>{model}</product_name><mig_mode><current_mig>Disabled</current_mig></mig_mode>'
          '<ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode><ecc_errors><volatile><dram_uncorrectable>0</dram_uncorrectable></volatile></ecc_errors></gpu>'
          for i in range(8)) + '</nvidia_smi_log>')
    else: print('Synthetic hardware inventory, not a GPU measurement')
elif tool == 'journalctl':
    print('kernel: synthetic test-window event')
elif tool == 'docker':
    with open(os.environ['MOCK_RECORD'],'a') as f: f.write(json.dumps(args)+'\\n')
    if args[:2] == ['image','inspect']:
        if '--format' in args:
            print('amd64' if '.Architecture' in args[-1] else 'sha256:'+'0'*64)
        else: print(json.dumps([{'Id':'sha256:'+'0'*64,'Architecture':'amd64'}]))
    elif args and args[0] == 'run':
        time.sleep(3)  # Allows two telemetry samples before the injected failure.
        sys.exit(42)
'''


@unittest.skipUnless(sys.platform == 'linux', 'Requires real Linux process/signal semantics')
class LinuxLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = kit.workspace_temp()
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'kit with spaces'
        self.source.mkdir()
        origin = Path(__file__).resolve().parents[1]
        for name in bundle.CODE:
            if name.startswith('images/'):
                continue
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / name, target)
        (self.source / 'models.lock.json').write_text('{"models":[]}', encoding='utf-8')
        (self.source / 'images').mkdir()
        (self.source / 'images/b300-test.tar').write_bytes(b'SYNTHETIC archive, not a runtime image')
        (self.source / 'images/runtime.inspect.json').write_text(json.dumps([{'Id':'sha256:'+'0'*64,'Architecture':'amd64'}]), encoding='utf-8')
        subprocess.run([sys.executable, str(self.source / 'bundle.py'), 'seal'], check=True, capture_output=True)
        tools = self.root / 'mock-bin'
        tools.mkdir()
        for name in ('nvidia-smi', 'docker', 'journalctl'):
            path = tools / name
            path.write_text(MOCK_TOOL, encoding='utf-8')
            path.chmod(0o755)
        self.record = self.root / 'mock-docker.jsonl'
        self.env = {**os.environ, 'PATH': str(tools)+os.pathsep+os.environ['PATH'], 'MOCK_RECORD': str(self.record)}
        self.out = self.root / 'output with spaces'

    def tearDown(self):
        self.tmp.cleanup()

    def run_mode(self, mode):
        return subprocess.run(['bash', str(self.source / 'run.sh'), mode, str(self.out)],
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_inventory_works_without_gpu_runtime(self):
        proc = self.run_mode('inventory')
        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['overall'], 'INCOMPLETE')
        self.assertTrue((self.out / 'report.html').is_file())

    def test_docker_stage_failure_propagates_and_still_generates_report(self):
        proc = self.run_mode('quick')
        self.assertEqual(proc.returncode, 42, proc.stdout+proc.stderr)
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['overall'], 'FAIL')
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        run = next(call for call in calls if call[0] == 'run')
        self.assertEqual(run[run.index('--network')+1], 'none')
        self.assertEqual(run[run.index('--pull')+1], 'never')
        self.assertTrue(any(call[0] == 'stop' for call in calls))
        self.assertTrue((self.out / 'telemetry-window.json').is_file())

    def test_gb300_is_rejected_before_any_container_load(self):
        self.env['MOCK_GPU_MODEL'] = 'NVIDIA GB300'
        proc = self.run_mode('quick')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('GPU model is not B300', proc.stderr)
        if self.record.exists():
            self.assertFalse(any(json.loads(line)[0] == 'run' for line in self.record.read_text().splitlines()))

    def test_deepseek_failure_propagates_and_container_mounts_are_kept(self):
        repo, scratch = self.root / 'repo', self.root / 'scratch'
        (repo / 'closed/NVIDIA').mkdir(parents=True)
        (repo / '.git').mkdir()
        (repo / 'closed/NVIDIA/Makefile').write_text('# synthetic only', encoding='utf-8')
        scratch.mkdir()
        self.env.update(MLPERF_REPO=str(repo), MLPERF_SCRATCH_PATH=str(scratch), MLPERF_IMAGE='synthetic:test')
        proc = self.run_mode('deepseek')
        self.assertEqual(proc.returncode, 42, proc.stdout+proc.stderr)
        self.assertEqual(json.loads((self.out / 'summary.json').read_text())['overall'], 'FAIL')
        calls = [json.loads(line) for line in self.record.read_text().splitlines()]
        run = next(call for call in calls if call[0] == 'run')
        self.assertIn(str(scratch)+':'+str(scratch), run)
        self.assertEqual(run[run.index('--pull')+1], 'never')
