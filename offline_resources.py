"""Offline resource audit. Never downloads, installs, or certifies a target runtime."""
import argparse
from datetime import datetime, timezone
import json
import html
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import bundle
import download_models
import deepseek_mlperf as ds

SCOPES = ('server', 'hardware', 'full', 'all')
HOST_TOOLS = {
    'server': {'python3':'python3', 'bash':'bash', 'lscpu':'util-linux', 'free':'procps',
               'lsblk':'util-linux', 'df':'coreutils', 'lspci':'pciutils', 'ip':'iproute2',
               'rdma':'iproute2', 'numactl':'numactl', 'sensors':'lm-sensors',
               'dmidecode':'dmidecode', 'smartctl':'smartmontools', 'ipmitool':'ipmitool',
               'journalctl':'systemd'},
    'gpu': {'nvidia-smi':'OEM-approved NVIDIA driver', 'docker':'Docker Engine + CLI + containerd'},
    'deepseek': {'git':'git'},
}
MANUAL_HOST_RESOURCES = [
    'GPU 驱动须匹配 GPU、目标内核及 CPU 架构；离线重装还需安装材料/本地仓库、完整依赖和内核 headers。',
    'GPU 需准备匹配的 NVIDIA Container Toolkit、base、libnvidia-container-tools、libnvidia-container1，并预先配置 Docker。',
    '适用的数据中心卡需兼容 DCGM 与 hostengine；B300 要求诊断阶段，消费卡不假定支持。',
    'NVSwitch 系统需匹配 Fabric Manager/NVLSM 与 OEM 配置，普通 PCIe/消费卡不要求。',
    '正常容器测试无需宿主 CUDA Toolkit/PyTorch；安装驱动时可能需要 DKMS/编译依赖。',
    'Docker、内核日志、SMBIOS、SMART 与本地 BMC 的读取权限需独立核对，找到命令不等于有权限。',
    '无网可使用本机/局域网浏览器；公网映射与 HTTPS 续签仍需网络。',
]


def safe_file(base, relative, extra=()):
    path = download_models.safe_path(Path(base), relative) if not extra else Path(base) / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe resource path')
    resolved = path.resolve()
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in (base, *extra)):
        raise ValueError('Resource link escapes declared package/scratch roots')
    return path


def model_inventory(root, lock_path, directory, hashes=False, scratch=None):
    data = json.loads(Path(lock_path).read_text(encoding='utf-8'))
    result=[]
    for model in data['models']:
        name = model.get('name') or model['path'].removeprefix('build/models/')
        base = Path(directory) / name
        if not base.resolve().is_relative_to(Path(directory).resolve()) and not (scratch and base.resolve().is_relative_to(Path(scratch).resolve())):
            raise ValueError('Unsafe model directory')
        present, missing, invalid = 0, [], []
        for item in model['files']:
            relative = item['path']
            try:
                path = safe_file(base, relative)
                if not path.is_file():
                    missing.append(relative)
                    continue
                if path.stat().st_size != item['size']:
                    invalid.append(relative + ': size mismatch')
                    continue
                present += item['size']
                if hashes:
                    print('Hash check: ' + name + '/' + relative, flush=True)
                    if not download_models.valid_file(path, item):
                        invalid.append(relative + ': source hash mismatch')
            except (OSError, ValueError) as error:
                invalid.append(relative + ': ' + str(error))
        result.append(dict(name=name, repo=model['repo'], revision=model['revision'],
                           expected_bytes=sum(item['size'] for item in model['files']), present_bytes=present,
                           expected_files=len(model['files']), missing=missing, invalid=invalid,
                           status='MISSING' if missing else 'INVALID' if invalid else 'HASH_VERIFIED' if hashes else 'PRESENT_UNVERIFIED'))
    return result


def audit(root, scope='all', hashes=False, repo=None, scratch=None, deepseek_archive=None, deepseek_inspect=None, inspect_host=True):
    if scope not in SCOPES:
        raise ValueError('Unknown offline resource scope')
    root=Path(root).resolve()
    resources, gaps, failures=[], [], []
    def file_entry(label, path, required=True):
        path=Path(path) if path else None
        available=bool(path and path.is_file() and path.stat().st_size)
        entry=dict(name=label,path=str(path) if path else None,status='PRESENT_UNVERIFIED' if available else 'MISSING',bytes=path.stat().st_size if available else None)
        resources.append(entry)
        if required and not available:
            gaps.append(label + ': missing')
        return available
    code=[name for name in bundle.CODE if not name.startswith('images/')]
    for name in code:
        if not (root/name).is_file():
            gaps.append('Source/static asset missing: ' + name)
    resources.append(dict(name='Source and local dashboard assets',expected_files=len(code),status='MISSING' if any(not (root/name).is_file() for name in code) else 'PRESENT_UNVERIFIED'))
    models=[]
    if scope != 'server':
        file_entry('Base GPU image archive',root/'images/b300-test.tar')
        file_entry('Base GPU image metadata',root/'images/runtime.inspect.json')
        seal=root/('hardware.sha256.json' if scope=='hardware' else 'bundle.sha256.json')
        if file_entry('Base package seal',seal) and hashes:
            # Reuse the exact package verifier; it must never resolve new model
            # revisions or treat a mere seal filename as a verified package.
            command=[sys.executable,str(Path(__file__).with_name('bundle.py')),'verify','--root',str(root)]
            if scope=='hardware': command.append('--hardware-only')
            proc=subprocess.run(command,capture_output=True,text=True,timeout=14400,check=False)
            if proc.returncode:
                failures.append('Base package verification failed: ' + (proc.stderr or proc.stdout)[-2000:])
            else:
                resources[-1]['status']='HASH_VERIFIED'
    if scope in ('full','all'):
        try:
            # A successful full seal verification already checked the weights.
            models=model_inventory(root,root/'models.lock.json',root/'models',hashes=hashes and not any(x.get('status')=='HASH_VERIFIED' and x['name']=='Base package seal' for x in resources))
            if hashes and any(x.get('status')=='HASH_VERIFIED' and x['name']=='Base package seal' for x in resources):
                for item in models:
                    if not item['missing'] and not item['invalid']: item['status']='HASH_VERIFIED'
        except (OSError, ValueError, KeyError) as error:
            gaps.append('Base model catalog: ' + str(error))
    deepseek=None
    if scope=='all':
        repo=Path(repo).resolve() if repo else None
        scratch=Path(scratch).resolve() if scratch else None
        try:
            expected=json.loads((root/'deepseek-model.lock.json').read_text(encoding='utf-8'))
            expected_bytes=sum(x['size'] for m in expected['models'] for x in m['files'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            expected_bytes=0
            failures.append('DeepSeek model catalog invalid: '+str(error))
        deepseek=dict(expected_model_bytes=expected_bytes,
                      source_revision=ds.SOURCE_REVISION,mlcommons_revision=ds.MLCOMMONS_REVISION,
                      runtime_check='NOT_RUN',models=[],scope='8 B300 only; independent image/data/evaluator environment')
        file_entry('DeepSeek final image archive',deepseek_archive)
        file_entry('DeepSeek final image metadata',deepseek_inspect)
        if not repo or not scratch:
            gaps.append('Set MLPERF_REPO and MLPERF_SCRATCH_PATH to prepared external directories')
        else:
            workspace=repo/'closed/NVIDIA'
            for name in ('Makefile',str(ds.CONFIG),str(ds.RAW_DATA),'build/data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl',
                         str(ds.DATA/'input_ids_padded.npy'),str(ds.DATA/'input_lens.npy'),'.dsr1-acc-venv/bin/python3',
                         '.dsr1-acc-venv/.requirements_hash','3rdparty/mlc-inference/language/deepseek-r1/eval_accuracy.py',
                         '3rdparty/trtllm', 'build/hf-cache/hub','build/hf-cache/datasets'):
                try:
                    if name=='.dsr1-acc-venv/bin/python3':
                        # venv interpreters link into the container filesystem,
                        # not necessarily into the host's checkout/scratch.
                        link=workspace/name
                        if not (link.is_symlink() or link.is_file()): gaps.append('DeepSeek missing: '+name)
                        continue
                    path=safe_file(workspace,name,extra=(scratch,))
                    if not path.exists(): gaps.append('DeepSeek missing: ' + name)
                except (ValueError,OSError) as error:
                    gaps.append('DeepSeek path invalid: '+name+': '+str(error))
            for folder,revision in ((repo,ds.SOURCE_REVISION),(workspace/'3rdparty/mlc-inference',ds.MLCOMMONS_REVISION)):
                try:
                    observed=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True,timeout=20,check=True).stdout.strip()
                    if observed!=revision: failures.append('DeepSeek source revision mismatch: '+str(folder))
                except (OSError,subprocess.SubprocessError): gaps.append('DeepSeek Git revision unavailable: '+str(folder))
            try:
                deepseek['models']=model_inventory(root,root/'deepseek-model.lock.json',workspace/'build/models',hashes,scratch)
            except (OSError,ValueError,KeyError) as error:
                gaps.append('DeepSeek model inventory: '+str(error))
            gaps.append('DeepSeek dataset/venv/evaluator caches and GPU runtime require verify_offline.sh all on the target; presence is not verification')
    for item in models+(deepseek['models'] if deepseek else []):
        gaps += [item['name']+': missing '+name for name in item['missing']]
        failures += [item['name']+': '+name for name in item['invalid']]
    tools=dict(HOST_TOOLS['server'])
    if scope!='server': tools.update(HOST_TOOLS['gpu'])
    if scope=='all': tools.update(HOST_TOOLS['deepseek'])
    host=[dict(command=name,package=package,status='FOUND' if shutil.which(name) else 'MISSING') for name,package in tools.items()] if inspect_host else []
    if inspect_host:
        gaps += ['Host tool missing: '+item['command']+' ('+item['package']+')' for item in host if item['status']=='MISSING']
        if sys.platform!='linux': gaps.append('Current host is not the Ubuntu/Linux target; host runtime not validated')
    overall='FAIL' if failures else 'INCOMPLETE' if gaps else 'REVIEW'
    result=dict(schema=1,profile='offline-check',scope=scope,overall=overall,offline_ready=False,
                time_utc=datetime.now(timezone.utc).isoformat(),root=str(root),resources=resources,models=models,deepseek=deepseek,
                host_tools=host,host_manual_resources=MANUAL_HOST_RESOURCES,failures=failures,incomplete=gaps,
                review=['Resource audit never substitutes for a target-host disconnected smoke test or actual inference.'],
                runtime_validation='NOT_RUN',stages=[],results=[],coverage_notes=['No downloads, installations or GPU workloads are executed by this audit.'])
    import purchase_acceptance
    result['acceptance']=purchase_acceptance.evaluate(result)
    return result


def report(result):
    """Portable self-contained report; resource paths and findings are untrusted text."""
    esc=lambda value: html.escape(str(value))
    rows=result['resources']+result['models']+(result['deepseek']['models'] if result['deepseek'] else [])
    body=''.join('<tr><td>'+esc(item['name'])+'</td><td>'+esc(item['status'])+'</td><td>'+esc(item.get('path',item.get('revision','')))+'</td></tr>' for item in rows)
    notes=result['failures']+result['incomplete']+result['host_manual_resources']
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>离线资源核对</title><style>body{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;overflow-wrap:anywhere}li{margin:8px}</style><h1>离线资源核对</h1><p>'+esc(result['overall'])+' · 范围 '+esc(result['scope'])+'</p><p>目标机断网运行：未执行。文件/哈希状态不能替代实际运行或采购验收。</p><table><tr><th>资源</th><th>状态</th><th>路径/版本</th></tr>'+body+'</table><h2>缺项和主机准备事项</h2><ul>'+''.join('<li>'+esc(note)+'</li>' for note in notes)+'</ul></html>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--scope',choices=SCOPES,default='all')
    parser.add_argument('--verify-hashes',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deepseek-repo',type=Path,default=os.environ.get('MLPERF_REPO'))
    parser.add_argument('--scratch',type=Path,default=os.environ.get('MLPERF_SCRATCH_PATH'))
    parser.add_argument('--deepseek-archive',type=Path,default=os.environ.get('MLPERF_IMAGE_ARCHIVE'))
    parser.add_argument('--deepseek-inspect',type=Path,default=os.environ.get('MLPERF_IMAGE_INSPECT'))
    args=parser.parse_args()
    result=audit(args.root,args.scope,args.verify_hashes,args.deepseek_repo,args.scratch,args.deepseek_archive,args.deepseek_inspect)
    args.out.mkdir(parents=True,exist_ok=True)
    download_models.write_json(args.out/'summary.json',result)
    (args.out/'report.html').write_text(report(result),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    return 1 if result['overall']=='FAIL' else 2


if __name__=='__main__':
    sys.exit(main())
