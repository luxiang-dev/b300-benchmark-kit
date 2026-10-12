#!/usr/bin/env python3
"""Offline localhost dashboard for B300 evidence workflows; use SSH forwarding remotely."""
import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import tempfile
from urllib.parse import unquote, urlsplit
import uuid
import xml.etree.ElementTree as ET
from evidence import export_evidence
from telemetry_guard import validate_sample
import job_lock

ROOT = Path(__file__).resolve().parent
PROFILES = [
    dict(id='import-image', title='导入离线镜像', description='验证封装哈希后导入本地镜像，不联网拉取。', requires_docker=True),
    dict(id='identity', title='身份交叉核验', description='先采集PCI/序列号；条件满足再核对CUDA身份与实际计算。', requires_gpus=0),
    dict(id='authenticity', title='整机硬件验真与稳定性', description='身份、服务器抽查、约85%HBM、10分钟计算、8卡通信、公测、DCGM。', requires_gpus=8, requires_docker=True, requires_image=True),
    dict(id='quick', title='模型功能快检', description='计算、NCCL、Qwen 0.6B 和 SDXL。', requires_gpus=8, requires_docker=True, requires_image=True, requires_bundle=True),
    dict(id='compare', title='8卡 NCCL 公测对比', description='环境检查及与 Oracle 固定口径一致的 all-reduce。', requires_gpus=8, requires_docker=True, requires_image=True, requires_bundle=True),
    dict(id='full', title='完整功能测试', description='10分钟压力、通信、两个Qwen、SDXL、DCGM；采购验收另行完成。', requires_gpus=8, requires_docker=True, requires_image=True, requires_bundle=True),
    dict(id='burn', title='长时间稳定性', description='默认1小时压力，再执行完整模型及诊断。', requires_gpus=8, requires_docker=True, requires_image=True, requires_bundle=True),
    dict(id='deepseek', title='DeepSeek-R1 / MLPerf', description='独立官方环境的完整 Offline 性能与准确率；需要数小时。', requires_gpus=8, requires_docker=True, requires_deepseek=True),
]
PROFILE_MAP = {p['id']: p for p in PROFILES}
JOB_ID = re.compile(r'^\d{8}-\d{6}-[0-9a-f]{8}$')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def optional_json(path):
    try:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            return None
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def tail(path, size=32000):
    try:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            return ''
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read(size).decode('utf-8', errors='replace')
    except OSError:
        return ''


def run_capture(command, timeout=8):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def readiness(image):
    tools = dict(python3=bool(shutil.which('python3')), docker=bool(shutil.which('docker')),
                 nvidia_smi=bool(shutil.which('nvidia-smi')), dcgmi=bool(shutil.which('dcgmi')))
    gpus, warnings = [], []
    raw = run_capture(['nvidia-smi', '-q', '-x']) if tools['nvidia_smi'] else None
    if raw:
        try:
            tree = ET.fromstring(raw)
            for index, gpu in enumerate(tree.findall('gpu')):
                memory = gpu.findtext('fb_memory_usage/total', '')
                number = re.match(r'([0-9.]+)\s*MiB', memory)
                gpus.append(dict(index=index, name=gpu.findtext('product_name'), uuid=gpu.findtext('uuid'),
                                 memory_mib=float(number[1]) if number else None,
                                 serial=gpu.findtext('serial'), vbios=gpu.findtext('vbios_version'),
                                 mig=gpu.findtext('mig_mode/current_mig'), ecc=gpu.findtext('ecc_mode/current_ecc')))
        except (ValueError, ET.ParseError):
            warnings.append('GPU 清单无法解析；需要读取真实 nvidia-smi XML。')
    metadata = None
    if tools['docker']:
        inspected = run_capture(['docker', 'image', 'inspect', image])
        if inspected:
            try:
                metadata = json.loads(inspected)[0]
            except (ValueError, IndexError, TypeError):
                warnings.append('本地镜像信息无法解析。')
    busy = run_capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']) if tools['nvidia_smi'] else None
    deepseek = {key: bool(os.environ.get(key)) for key in ('MLPERF_REPO', 'MLPERF_IMAGE', 'MLPERF_SCRATCH_PATH')}
    deepseek_ready = all(deepseek.values())
    if deepseek_ready:
        repo, scratch = Path(os.environ['MLPERF_REPO']), Path(os.environ['MLPERF_SCRATCH_PATH'])
        deepseek_ready = (repo / 'closed/NVIDIA/Makefile').is_file() and scratch.is_dir() and bool(run_capture(['docker', 'image', 'inspect', os.environ['MLPERF_IMAGE']]))
    return dict(gpus=gpus, tools=tools, image=dict(available=metadata is not None, id=metadata.get('Id') if metadata else None,
                architecture=metadata.get('Architecture') if metadata else None, tag=image),
                bundle=dict(sealed=(ROOT / 'bundle.sha256.json').is_file(), hardware_sealed=(ROOT / 'hardware.sha256.json').is_file(),
                            archive_available=(ROOT / 'images/b300-test.tar').is_file()),
                deepseek=dict(configured=deepseek_ready, fields=deepseek), busy_pids=busy.splitlines() if busy else [],
                busy_check_available=busy is not None, warnings=warnings)


def profile_eligibility(profile, ready, preview=False):
    reasons = []
    if preview or sys.platform != 'linux':
        reasons.append('预览模式只读；请在 Ubuntu 服务器启动控制服务。')
    if profile['id'] == 'identity' and not ready['tools']['nvidia_smi']:
        reasons.append('需要可读取的 nvidia-smi，才能采集硬件身份。')
    expected = profile.get('requires_gpus')
    if expected:
        if len(ready['gpus']) != expected or any(not re.search(r'\bB300\b', gpu.get('name') or '') for gpu in ready['gpus']):
            reasons.append(f'需要 {expected} 张完整 B300。')
        if any(gpu.get('mig') not in ('Disabled', 'N/A') or gpu.get('ecc') != 'Enabled' for gpu in ready['gpus']):
            reasons.append('MIG 必须关闭，ECC 必须开启并可读取。')
        if ready.get('busy_pids') or not ready.get('busy_check_available'):
            reasons.append('GPU 不空闲，或无法核对计算进程。')
    if profile.get('requires_docker') and not ready['tools']['docker']:
        reasons.append('主机缺少 Docker。')
    if profile.get('requires_image') and not ready['image']['available']:
        reasons.append('本地 GPU 镜像未导入。')
    if profile.get('requires_bundle') and not ready['bundle']['sealed']:
        reasons.append('该模式需要完整模型包及 bundle.sha256.json。')
    if profile.get('requires_deepseek') and not ready['deepseek']['configured']:
        reasons.append('需先准备独立 DeepSeek 源码、scratch、venv、缓存和镜像。')
    if profile['id'] == 'import-image' and (not ready['bundle']['archive_available'] or not (ready['bundle']['sealed'] or ready['bundle']['hardware_sealed'])):
        reasons.append('需准备镜像 tar 和完整包或硬件包的哈希封装。')
    return dict(profile, available=not reasons, unavailable_reason=' '.join(reasons))


class JobManager:
    def __init__(self, results, image='b300-test:1.0', preview=False, max_temp=85):
        self.root = Path(results).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image, self.preview, self.max_temp = image, preview, max_temp
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.jobs, self.active, self.process, self.worker = {}, None, None, None
        self.gpu_lock = None
        self.samples = []
        self._ready, self._ready_time = None, 0
        for path in sorted(self.root.glob('*/job.json')):
            if path.parent.is_symlink() or not path.resolve().is_relative_to(self.root):
                continue
            job = optional_json(path)
            if isinstance(job, dict) and JOB_ID.fullmatch(job.get('id', '')) and job.get('profile') in PROFILE_MAP and path.parent.name == job['id']:
                if job.get('status') in ('RUNNING', 'STOPPING', 'EXPORTING'):
                    job.update(status='INTERRUPTED', error='控制服务中断，先检查本机进程/容器后重新测试。')
                self.jobs[job['id']] = job

    def ready(self, refresh=False):
        if refresh or self._ready is None or time.monotonic() - self._ready_time > 20:
            self._ready = readiness(self.image)
            self._ready_time = time.monotonic()
        return self._ready

    def command(self, profile, out, seconds):
        if profile in ('identity', 'authenticity'):
            return [sys.executable, str(ROOT / 'run_authenticity.py'), '--profile', profile, '--out', str(out),
                    '--image', self.image, '--gpus', '8', '--seconds', str(seconds if profile == 'authenticity' else 600),
                    '--max-temp', str(self.max_temp)]
        if profile == 'import-image':
            # Verify before importing any Docker archive. No shell interpolation.
            verifier = [sys.executable, str(ROOT / 'bundle.py'), 'verify']
            if (ROOT / 'hardware.sha256.json').is_file():
                verifier.append('--hardware-only')
            return verifier
        return ['bash', str(ROOT / 'run.sh'), profile, str(out)]

    def start(self, body):
        if not isinstance(body, dict) or body.get('profile') not in PROFILE_MAP:
            raise ValueError('Unknown profile')
        profile = body['profile']
        seconds = body.get('seconds', 3600 if profile == 'burn' else 600)
        if type(seconds) is not int or not 1 <= seconds <= 86400:
            raise ValueError('seconds must be an integer from 1 to 86400')
        with self.lock:
            if self.active:
                raise RuntimeError('A job is already active')
            ready = self.ready(True)
            eligibility = profile_eligibility(PROFILE_MAP[profile], ready, self.preview)
            if not eligibility['available']:
                raise ValueError(eligibility['unavailable_reason'])
            if profile != 'import-image':
                self.gpu_lock = job_lock.acquire()
            job_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
            out = self.root / job_id
            try:
                out.mkdir()
                job = dict(id=job_id, profile=profile, status='RUNNING', exit_code=None, started_at=utc_now(),
                           ended_at=None, seconds=seconds, cancelled=False, thermal_stop=False,
                           expected_gpu_uuids=[gpu['uuid'] for gpu in ready.get('gpus', [])])
                self.jobs[job_id] = job
                write_json(out / 'job.json', job)
                self.active = job_id
                self.samples = []
                self.worker = threading.Thread(target=self._run, args=(job_id, out), daemon=True)
                self.worker.start()
            except Exception:
                self.active = None
                if self.gpu_lock is not None:
                    os.close(self.gpu_lock)
                    self.gpu_lock = None
                raise
            return job_id

    def _cleanup_containers(self, job_id):
        if not shutil.which('docker'):
            return
        listed = run_capture(['docker', 'ps', '-q', '--filter', 'label=b300-kit.job=' + job_id])
        if listed is None:
            raise RuntimeError('无法核对本次作业的 Docker 容器，停止状态需要人工确认。')
        ids = [line.strip() for line in listed.splitlines() if line.strip()]
        if any(not re.fullmatch(r'[0-9a-f]{12,64}', line) for line in ids):
            raise RuntimeError('本次作业容器清单无法解析，不能确认清理完成。')
        if ids:
            stopped = run_capture(['docker', 'stop', '--time', '10', *ids], timeout=25)
            remaining = run_capture(['docker', 'ps', '-q', '--filter', 'label=b300-kit.job=' + job_id])
            if stopped is None or remaining is None or remaining.strip():
                raise RuntimeError('本次作业容器未确认停止；检查 Docker 与 GPU 进程后再测试。')

    def stop(self, job_id, thermal_reason=None):
        with self.lock:
            if self.active != job_id or job_id not in self.jobs:
                raise ValueError('Job is not active')
            job = self.jobs[job_id]
            if job['status'] == 'EXPORTING':
                raise ValueError('GPU 作业已结束，正在封存证据，请等待导出完成。')
            if job['status'] == 'STOPPING':
                return
            job.update(status='STOPPING', cancelled=thermal_reason is None, thermal_stop=thermal_reason is not None)
            if thermal_reason:
                job['error'] = thermal_reason
            process = self.process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                with self.lock:
                    job['cleanup_error'] = '无法通知启动器停止: ' + str(error)
            threading.Thread(target=self._finish_stop, args=(job_id, process), daemon=True).start()
        # Dispatch both process and Docker cleanup before any result-disk I/O.
        # A full or hung filesystem must not prevent a temperature stop.
        with self.lock:
            try:
                write_json(self.root / job_id / 'job.json', job)
            except OSError as error:
                job['error'] = (job.get('error', '') + '; 无法保存停止状态: ' + str(error)).strip('; ')

    def _finish_stop(self, job_id, process):
        try:
            # A stopped launcher may be blocked collecting diagnostics. Stop
            # owned GPU containers promptly rather than waiting 45 seconds.
            self._cleanup_containers(job_id)
        except Exception as error:
            with self.lock:
                self.jobs[job_id]['cleanup_error'] = str(error)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    with self.lock:
                        self.jobs[job_id]['cleanup_error'] = '启动器强制停止失败: ' + str(error)
        except Exception as error:
            with self.lock:
                self.jobs[job_id]['cleanup_error'] = '启动器退出状态无法确认: ' + str(error)
        finally:
            try:
                self._cleanup_containers(job_id)
            except Exception as error:
                with self.lock:
                    self.jobs[job_id]['cleanup_error'] = str(error)

    def _watch(self, job_id, done):
        while not done.wait(2):
            raw = run_capture(['nvidia-smi', '--query-gpu=uuid,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], timeout=10)
            if raw is None:
                try:
                    self.stop(job_id, '主机遥测读取失败，已停止作业以保留证据。')
                except ValueError:
                    pass
                return
            samples = []
            try:
                for row in csv.reader(raw.splitlines(), skipinitialspace=True):
                    temp, power = float(row[1]), float(row[2])
                    samples.append(dict(timestamp=time.time(), uuid=row[0], temperature_c=temp, power_w=power))
                expected = self.jobs[job_id].get('expected_gpu_uuids') or None
                validate_sample(samples, 8, expected)
                with self.lock:
                    self.samples = (self.samples + samples)[-240:]
                if any(s['temperature_c'] >= self.max_temp for s in samples):
                    self.stop(job_id, f'GPU 温度达到 {self.max_temp}°C，已触发停止。')
                    return
            except (ValueError, IndexError):
                try:
                    self.stop(job_id, '遥测格式或八卡覆盖异常，已停止作业。')
                except ValueError:
                    pass
                return

    def _run(self, job_id, out):
        job = self.jobs[job_id]
        done = threading.Event()
        watcher = None
        cleanup_completed = False
        try:
            environment = {**os.environ, 'B300_JOB_ID': job_id, 'BURN_SECONDS': str(job['seconds']),
                           'MAX_GPU_TEMP': str(self.max_temp), 'PYTHONUNBUFFERED': '1'}
            if self.gpu_lock is not None:
                environment['B300_GPU_LOCK_FD'] = str(self.gpu_lock)
            commands = [self.command(job['profile'], out, job['seconds'])]
            if job['profile'] == 'import-image':
                commands.append(['docker', 'load', '-i', str(ROOT / 'images/b300-test.tar')])
            with (out / 'launcher.log').open('w', encoding='utf-8') as log:
                rc = 0
                for command in commands:
                    with self.lock:
                        if job['cancelled'] or job['thermal_stop']:
                            break
                        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True,
                                                   pass_fds=(self.gpu_lock,) if self.gpu_lock is not None else ())
                        self.process = process
                    if job['profile'] not in ('import-image', 'identity') and watcher is None:
                        watcher = threading.Thread(target=self._watch, args=(job_id, done), daemon=True)
                        watcher.start()
                    rc = process.wait()
                    if rc:
                        break
            done.set()
            if watcher:
                watcher.join(timeout=12)
            self._cleanup_containers(job_id)
            cleanup_completed = True
            summary = optional_json(out / 'summary.json')
            completed_at = utc_now()
            with self.lock:
                # Freeze cancellation and final status in one critical section.
                # A stop between status selection and EXPORTING cannot be
                # overwritten by a stale PASS decision.
                if job['cancelled']:
                    status = 'CANCELLED'
                elif job['thermal_stop']:
                    status = 'FAIL'
                elif summary:
                    status = summary.get('overall', 'FAIL')
                    if (status == 'PASS' and rc != 0) or (status in ('REVIEW', 'INCOMPLETE') and rc not in (0, 2)):
                        status = 'FAIL'
                        job['error'] = f'启动器异常退出 ({rc})；不能仅依赖摘要中的通过状态。'
                else:
                    status = 'PASS' if job['profile'] == 'import-image' and rc == 0 else 'FAIL'
                job.update(status='EXPORTING', exit_code=rc)
                final_record = dict(job, status=status, ended_at=completed_at)
                write_json(out / 'job.json', job)
                write_json(out / 'run-result.json', final_record)
            export_evidence(out)
            with self.lock:
                job.update(status=status, ended_at=completed_at)
        except Exception as error:
            done.set()
            with self.lock:
                job.update(status='CANCELLED' if job['cancelled'] else 'FAIL', ended_at=utc_now(), error=str(error))
        finally:
            done.set()
            if watcher:
                watcher.join(timeout=12)
            cleanup_error = None
            try:
                if self.process and self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError) as error:
                        if isinstance(error, PermissionError):
                            cleanup_error = str(error)
                    self._finish_stop(job_id, self.process)
                if not cleanup_completed:
                    self._cleanup_containers(job_id)
            except Exception as error:
                cleanup_error = str(error)
            with self.lock:
                try:
                    if cleanup_error:
                        job.update(status='FAIL', cleanup_error=cleanup_error, error=cleanup_error)
                    write_json(out / 'job.json', job)
                except OSError as error:
                    job.update(status='FAIL', error='无法保存最终作业状态: ' + str(error))
                finally:
                    self.process = None
                    self.active = None
                    self._ready_time = 0
                    if self.gpu_lock is not None:
                        try:
                            os.close(self.gpu_lock)
                        finally:
                            self.gpu_lock = None

    def job_state(self, job):
        out = self.root / job['id']
        value = dict(job)
        if not out.is_dir() or out.is_symlink() or not out.resolve().is_relative_to(self.root):
            value.update(stages=[], log_tail='', summary=None, identity_summary=None, artifacts=[], telemetry=dict(per_gpu={}, samples=[]))
            return value
        stages = []
        for line in tail(out / 'stages.tsv').splitlines():
            parts = line.split('\t')
            if len(parts) == 3:
                try:
                    stages.append(dict(name=parts[0], status=parts[1], exit_code=int(parts[2])))
                except ValueError:
                    pass
        log_tail = tail(out / 'launcher.log')
        latest = {s['name']: s for s in stages}
        running = next((s['name'] for s in reversed(list(latest.values())) if s['status'] == 'RUNNING'), None)
        if running and re.fullmatch(r'[A-Za-z0-9_.-]+', running):
            log_tail += '\n' + tail(out / (running + '.log'), 16000)
        value.update(stages=stages, log_tail=log_tail, summary=optional_json(out / 'summary.json'),
                     identity_summary=optional_json(out / 'identity-summary.json'))
        value['artifacts'] = [dict(name=p.name, url=f'/runs/{job["id"]}/{p.name}') for p in sorted(out.iterdir())
                              if p.is_file() and not p.is_symlink() and p.name in ('report.html', 'summary.json', 'identity-summary.json', 'identity-host.json',
                                 'server-summary.json', 'observed-inventory.json', 'server-health-before.json', 'server-health-after.json', 'run-result.json', 'evidence.zip', 'evidence-manifest.json', 'public-comparison.json', 'launcher.log', 'topology.txt', 'telemetry.csv')]
        per_gpu = {}
        for sample in self.samples if self.active == job['id'] else []:
            node = per_gpu.setdefault(sample['uuid'], dict(samples=0, peak_temperature_c=sample['temperature_c']))
            node.update(samples=node['samples']+1, latest_temperature_c=sample['temperature_c'], latest_power_w=sample['power_w'],
                        peak_temperature_c=max(node['peak_temperature_c'], sample['temperature_c']))
        value['telemetry'] = dict(per_gpu=per_gpu, samples=self.samples if self.active == job['id'] else [])
        return value

    def state(self, refresh=False):
        ready = self.ready(refresh)
        with self.lock:
            jobs = [self.job_state(job) for job in sorted(self.jobs.values(), key=lambda j: j['started_at'], reverse=True)[:30]]
            active = next((j for j in jobs if j['id'] == self.active), None)
            return dict(platform=sys.platform, demo=self.preview, busy=self.active is not None, active_job=active,
                        jobs=jobs, readiness=ready, profiles=[profile_eligibility(p, ready, self.preview) for p in PROFILES])

    def close(self):
        if self.active:
            try:
                self.stop(self.active)
            except ValueError:
                pass
        if self.worker:
            self.worker.join(timeout=55)


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _host_ok(self):
            port = self.server.server_address[1]
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def _json(self, data, status=200):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)

        def _file(self, path, report_file=False):
            if not path.is_file():
                self._json(dict(error='Not found'), 404)
                return
            self.send_response(200)
            self.send_header('Content-Type', (mimetypes.guess_type(path.name)[0] or 'application/octet-stream') +
                             ('; charset=utf-8' if path.suffix in ('.html', '.css', '.js', '.json', '.txt', '.log') else ''))
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'" if not report_file
                             else "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; sandbox")
            self.end_headers()
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile)

        def do_GET(self):
            if not self._host_ok():
                self._json(dict(error='Invalid Host'), 403)
                return
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if '\x00' in path:
                self._json(dict(error='Invalid path'), 400)
                return
            if path == '/api/session':
                self._json(dict(csrf_token=manager.token))
            elif path == '/api/state':
                self._json(manager.state(refresh=parsed.query == 'refresh=1'))
            elif path in ('/', '/index.html', '/app.js', '/style.css'):
                self._file(ROOT / 'dashboard' / ('index.html' if path == '/' else path[1:]))
            elif path.startswith('/api/jobs/') and path.endswith('/log'):
                key = path.split('/')[3]
                if key not in manager.jobs:
                    self._json(dict(error='Not found'), 404)
                else:
                    self._json(dict(log=tail(manager.root / key / 'launcher.log', 64000)))
            elif path.startswith('/runs/'):
                parts = path.split('/')
                if len(parts) < 4 or parts[2] not in manager.jobs or any(p in ('..', '.') for p in parts[3:]):
                    self._json(dict(error='Not found'), 404)
                    return
                base = manager.root / parts[2]
                if base.is_symlink() or not base.resolve().is_relative_to(manager.root):
                    self._json(dict(error='Invalid path'), 403)
                    return
                candidate = base.joinpath(*parts[3:])
                resolved = candidate.resolve()
                if not resolved.is_relative_to(base.resolve()) or candidate.is_symlink():
                    self._json(dict(error='Invalid path'), 403)
                    return
                self._file(resolved, report_file=resolved.suffix == '.html')
            else:
                self._json(dict(error='Not found'), 404)

        def do_POST(self):
            port = self.server.server_address[1]
            origins = (f'http://127.0.0.1:{port}', f'http://localhost:{port}')
            if not self._host_ok() or self.headers.get('Origin') not in (None, *origins) or not secrets.compare_digest(self.headers.get('X-CSRF-Token', '').encode(), manager.token.encode()):
                self._json(dict(error='Invalid origin or session token'), 403)
                return
            if self.headers.get_content_type() != 'application/json':
                self._json(dict(error='Expected JSON'), 415)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    self._json(dict(error='Invalid request size'), 413)
                    return
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == '/api/jobs':
                    self._json(dict(id=manager.start(body)), 202)
                elif re.fullmatch(r'/api/jobs/[^/]+/stop', path):
                    manager.stop(path.split('/')[3])
                    self._json(dict(status='STOPPING'), 202)
                else:
                    self._json(dict(error='Not found'), 404)
            except (ValueError, TypeError) as error:
                self._json(dict(error=str(error)), 400)
            except RuntimeError as error:
                self._json(dict(error=str(error)), 409)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--results', type=Path, default=ROOT / 'b300-results/dashboard')
    parser.add_argument('--image', default='b300-test:1.0')
    parser.add_argument('--preview', action='store_true', help='Read-only interface preview without GPU execution')
    parser.add_argument('--max-temp', type=int, default=85)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.max_temp < 120:
        parser.error('Invalid port or temperature limit')
    if sys.platform != 'linux' and not args.preview:
        parser.error('Run on Ubuntu/Linux, or use --preview for a read-only preview')
    manager = JobManager(args.results, args.image, args.preview, args.max_temp)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(manager))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    print(f'B300 dashboard: http://127.0.0.1:{args.port} (localhost only)', flush=True)
    print('Run evidence contains device serials and host information; review before sharing.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == '__main__':
    main()
