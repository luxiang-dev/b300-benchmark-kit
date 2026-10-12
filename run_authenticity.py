#!/usr/bin/env python3
"""Local B300 identity/capacity/health workflow; no online downloads or genuine-card verdict."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
import authenticity
import report
from evidence import digest, export_evidence
from telemetry_guard import validate_sample

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class Workflow:
    def __init__(self, args):
        self.args, self.out = args, args.out.resolve()
        self.cancelled = threading.Event()
        self.monitor_done = threading.Event()
        self.monitor = None
        self.current = None
        self.current_lock = threading.Lock()
        self.stage_lock = threading.Lock()
        self.container = 'b300-auth-' + uuid.uuid4().hex[:12]
        self.job_id = os.environ.get('B300_JOB_ID', self.container)
        self.telemetry_start = None
        self.started = time.time()
        self.watchdog_error = None
        self.image_id = None
        self.expected_uuids = None
        self.collection_status = {}

    def record(self, name, status, code=0):
        with self.stage_lock, (self.out / 'stages.tsv').open('a', encoding='utf-8') as stream:
            stream.write(f'{name}\t{status}\t{code}\n')
        print(f'[{name}] {status} ({code})', flush=True)

    def capture(self, args, filename, timeout=20):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
            (self.out / filename).write_text(result.stdout, encoding='utf-8')
            (self.out / (filename + '.stderr')).write_text(result.stderr, encoding='utf-8')
            self.collection_status[filename] = dict(argv=args, returncode=result.returncode, error=None)
            write_json(self.out / 'collection-status.json', self.collection_status)
            return result
        except (OSError, subprocess.TimeoutExpired) as error:
            (self.out / (filename + '.stderr')).write_text(str(error), encoding='utf-8')
            self.collection_status[filename] = dict(argv=args, returncode=None, error=str(error))
            write_json(self.out / 'collection-status.json', self.collection_status)
            return None

    def stop_container(self):
        if shutil.which('docker'):
            try:
                subprocess.run(['docker', 'stop', '--time', '10', self.container], capture_output=True, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def stop_current(self):
        with self.current_lock:
            current = self.current
        if current and current.poll() is None:
            try:
                os.killpg(current.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.stop_container()

    def telemetry(self):
        self.telemetry_start = time.time()
        with (self.out / 'telemetry.csv').open('w', encoding='utf-8', newline='') as stream:
            stream.write('timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n')
            stream.flush()
            while not self.monitor_done.is_set():
                if self.cancelled.is_set():
                    self.stop_current()
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,uuid,temperature.gpu,power.draw,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
                    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
                    samples = [dict(index=int(row[1]), uuid=row[2].strip(), temperature_c=float(row[3]),
                                    power_w=float(row[4]), memory_used_mib=float(row[5]), gpu_util_percent=float(row[6])) for row in rows]
                    validate_sample(samples, self.args.gpus, self.expected_uuids)
                    for row in rows:
                        value = float(row[3])
                        if value >= self.args.max_temp:
                            self.watchdog_error = f'{row[2]} reached {value:g} C (limit {self.args.max_temp} C)'
                            self.cancelled.set()
                    stream.write(result.stdout)
                    stream.flush()
                    if self.watchdog_error:
                        self.stop_current()
                except Exception as error:
                    (self.out / 'telemetry.stderr').write_text(str(error), encoding='utf-8')
                    self.watchdog_error = 'Telemetry failed during the workload: ' + str(error)
                    self.cancelled.set()
                    self.stop_current()
                    break
                self.monitor_done.wait(2)

    def execute(self, label, command, timeout=7200):
        if self.cancelled.is_set():
            raise InterruptedError(self.watchdog_error or 'Operator cancelled the workflow')
        self.record(label, 'RUNNING')
        print('===== ' + label + ' =====', flush=True)
        deadline = time.monotonic() + timeout
        with (self.out / (label + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            with self.current_lock:
                self.current = process
            try:
                while process.poll() is None:
                    if self.cancelled.is_set() or time.monotonic() >= deadline:
                        self.stop_current()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                        raise InterruptedError(self.watchdog_error or ('Cancelled' if self.cancelled.is_set() else 'Stage timeout'))
                    time.sleep(0.25)
                rc = process.returncode
            except Exception:
                self.record(label, 'FAIL', 1)
                raise
            finally:
                with self.current_lock:
                    self.current = None
                self.stop_container()
        self.record(label, 'PASS' if rc == 0 else 'FAIL', rc)
        if rc:
            raise subprocess.CalledProcessError(rc, command)

    def docker(self, command):
        return ['docker', 'run', '--rm', '--pull', 'never', '--name', self.container,
                '--label', 'b300-kit.job=' + self.job_id, '--network', 'none', '--gpus', 'all', '--shm-size=16g',
                '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864', '-e', 'PYTHONUNBUFFERED=1',
                '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                '-v', str(ROOT) + ':/kit:ro', '-v', str(self.out) + ':/out', self.image_id] + command

    def torchrun(self, script, *args):
        return self.docker(['torchrun', '--standalone', '--nproc-per-node=' + str(self.args.gpus),
                            '/kit/' + script] + list(args) + ['--out', '/out', '--gpus', str(self.args.gpus)])

    def preflight(self):
        for tool in ('nvidia-smi', 'python3'):
            if not shutil.which(tool):
                raise RuntimeError('Missing host tool: ' + tool)
        raw = self.capture(['nvidia-smi', '-q', '-x'], 'before.xml')
        if raw is None or raw.returncode:
            raise RuntimeError('Cannot read GPU inventory')
        tree = ET.fromstring(raw.stdout)
        gpus = tree.findall('gpu')
        if len(gpus) != self.args.gpus:
            raise RuntimeError(f'Expected {self.args.gpus} physical GPUs; found {len(gpus)}')
        self.expected_uuids = [gpu.findtext('uuid', '').strip() for gpu in gpus]
        if len(set(self.expected_uuids)) != self.args.gpus or any(not value.startswith('GPU-') for value in self.expected_uuids):
            raise RuntimeError('Physical GPU inventory has missing or duplicate UUIDs')
        for gpu in gpus:
            if not re.search(r'\bB300\b', gpu.findtext('product_name', '')):
                raise RuntimeError('GPU model differs from B300')
            if gpu.findtext('mig_mode/current_mig') not in ('Disabled', 'N/A'):
                raise RuntimeError('Disable MIG before testing full physical GPUs')
            if gpu.findtext('ecc_mode/current_ecc') != 'Enabled':
                raise RuntimeError('ECC must be enabled and readable')
        driver = tree.findtext('driver_version', '0')
        if int(driver.split('.')[0]) < 580:
            raise RuntimeError('Requires a compatible R580+ driver')
        self.capture(['nvidia-smi', 'topo', '-m'], 'topology.txt')
        self.capture(['nvidia-smi', 'nvlink', '-s'], 'nvlink.txt')
        self.capture(['nvidia-smi', 'nvlink', '-e'], 'nvlink-errors-before.txt')
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if busy:
            raise RuntimeError('GPU compute jobs are already running: ' + busy)
        image = subprocess.run(['docker', 'image', 'inspect', self.args.image], capture_output=True,
                               text=True, timeout=20, check=True)
        metadata = json.loads(image.stdout)[0]
        architecture = 'amd64' if os.uname().machine == 'x86_64' else 'arm64' if os.uname().machine == 'aarch64' else None
        if architecture != metadata['Architecture']:
            raise RuntimeError('Local GPU image CPU architecture mismatch')
        self.image_id = metadata['Id']
        write_json(self.out / 'image-used.json', [metadata])
        write_json(self.out / 'execution.json', dict(image_id=self.image_id, command_profile=self.args.profile,
                   expected_gpus=self.args.gpus, code_sha256={p.name: digest(p) for p in sorted(ROOT.glob('*.py'))}))
        self.record('preflight', 'PASS')

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'plan.json').exists():
            raise RuntimeError('Use a fresh result directory')
        required = ['identity']
        if self.args.profile == 'authenticity':
            required += ['server', 'environment', 'memory', 'stress', 'nccl', 'nccl-public', 'dcgm']
        write_json(self.out / 'plan.json', dict(profile=self.args.profile, gpus=self.args.gpus, required=required, max_gpu_temp=self.args.max_temp))
        shutil.copyfile(ROOT / 'public_baselines.json', self.out / 'public-baselines.json')
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.cancelled.set())
        rc = 0
        host_only = False
        try:
            self.record('host-identity', 'RUNNING')
            authenticity.collect_host(self.out)
            self.record('host-identity', 'PASS')
            if self.args.profile == 'identity':
                self.capture(['nvidia-smi', '-q', '-x'], 'before.xml')
                try:
                    self.preflight()
                except Exception as error:
                    self.record('identity-cuda', 'SKIP')
                    (self.out / 'identity-cuda-unavailable.txt').write_text(str(error), encoding='utf-8')
                    identity = authenticity.evaluate_identity(self.out, self.args.gpus)
                    self.record('identity', 'FAIL' if identity['status'] == 'FAIL' else 'PASS', 1 if identity['status'] == 'FAIL' else 0)
                    rc = 1 if identity['status'] == 'FAIL' else 2
                    host_only = True
            else:
                self.preflight()
            if not host_only:
                self.monitor = threading.Thread(target=self.telemetry, daemon=True)
                self.monitor.start()
                self.execute('identity-cuda', self.torchrun('authenticity.py', 'cuda'))
                identity = authenticity.evaluate_identity(self.out, self.args.gpus)
                if identity['status'] == 'FAIL':
                    self.record('identity', 'FAIL', 1)
                    raise RuntimeError('Identity evidence contains contradictions; inspect identity-summary.json')
                self.record('identity', 'PASS')
            if self.args.profile == 'authenticity':
                self.execute('server', [sys.executable, str(ROOT / 'server_probe.py'), 'full', '--out', str(self.out)], 900)
                self.execute('environment', self.torchrun('gpu_test.py', 'environment'))
                self.execute('memory', self.torchrun('memory_probe.py', '--memory-fraction', str(self.args.memory_fraction), '--max-temp', str(self.args.max_temp)))
                self.execute('stress', self.torchrun('gpu_test.py', 'stress', '--seconds', str(self.args.seconds), '--max-temp', str(self.args.max_temp), '--memory-fraction', '0.65'), self.args.seconds + 1800)
                self.execute('nccl', self.torchrun('gpu_test.py', 'nccl', '--iterations', '50'))
                if self.args.gpus == 8:
                    self.execute('nccl-public', self.docker(['python', '/kit/public_benchmark.py', 'run', '--out', '/out', '--timeout', '1800']), 1900)
                else:
                    self.record('nccl-public', 'SKIP')
                if shutil.which('dcgmi'):
                    version = self.capture(['dcgmi', '--version'], 'dcgm-version.txt')
                    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version.stdout if version else '')
                    if match and tuple(map(int, match.groups())) >= (4, 4, 0):
                        self.execute('dcgm', ['dcgmi', 'diag', '-r', '3', '-i', ','.join(map(str, range(self.args.gpus))), '-j'], 10800)
                        shutil.copyfile(self.out / 'dcgm.log', self.out / 'dcgm.json')
                    else:
                        self.record('dcgm', 'SKIP')
                else:
                    self.record('dcgm', 'SKIP')
        except Exception as error:
            rc = 130 if self.cancelled.is_set() and not self.watchdog_error else 1
            self.record('launcher', 'FAIL', rc)
            write_json(self.out / 'workflow-error.json', dict(error=str(error), watchdog=self.watchdog_error,
                       cancelled=self.cancelled.is_set() and not self.watchdog_error))
            print(str(error), file=sys.stderr, flush=True)
        finally:
            self.stop_current()
            self.stop_container()
            self.monitor_done.set()
            if self.monitor:
                self.monitor.join(timeout=12)
            if self.telemetry_start is not None:
                write_json(self.out / 'telemetry-window.json', dict(start_epoch=self.telemetry_start, end_epoch=time.time()))
            self.capture(['nvidia-smi', '-q', '-x'], 'after.xml')
            self.capture(['nvidia-smi', 'nvlink', '-e'], 'nvlink-errors-after.txt')
            self.capture(['journalctl', '-k', '--since', '@' + str(int(self.started)), '--no-pager'], 'kernel-during.log')
            try:
                authenticity.evaluate_identity(self.out, self.args.gpus)
                summary = report.generate(self.out)
                if summary['overall'] == 'FAIL' and rc != 130:
                    rc = 1
                elif not rc:
                    rc = 0 if summary['overall'] == 'PASS' else 1 if summary['overall'] == 'FAIL' else 2
            except Exception as error:
                (self.out / 'report-error.txt').write_text(str(error), encoding='utf-8')
                if rc != 130:
                    rc = 1
            export_evidence(self.out)
        print('Results: ' + str(self.out / 'report.html'), flush=True)
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['identity', 'authenticity'], default='authenticity')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', default='b300-test:1.0')
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--memory-fraction', type=float, default=0.85)
    parser.add_argument('--max-temp', type=int, default=85)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The GPU workflow requires Linux; dashboard --preview can be used on Windows')
    if not 1 <= args.gpus <= 8 or not 1 <= args.seconds <= 86400 or not 0.5 <= args.memory_fraction <= 0.9 or not 1 <= args.max_temp < 120:
        parser.error('Invalid GPU count, duration, memory fraction or temperature limit')
    return Workflow(args).run()


if __name__ == '__main__':
    sys.exit(main())
