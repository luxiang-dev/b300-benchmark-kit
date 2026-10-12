#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-quick}"
if [[ "$profile" == deepseek ]]; then exec bash "$ROOT/run_deepseek_mlperf.sh" "${2:-$PWD/b300-results/$(date +%Y%m%d-%H%M%S)-deepseek}"; fi
if [[ "$profile" == identity || "$profile" == authenticity ]]; then exec python3 "$ROOT/run_authenticity.py" --profile "$profile" --out "${2:-$PWD/b300-results/$(date +%Y%m%d-%H%M%S)-$profile}" --image "${B300_IMAGE:-b300-test:1.0}"; fi
case "$profile" in inventory|quick|full|burn|compare) ;; *) echo "Usage: bash run.sh [inventory|identity|authenticity|quick|full|burn|compare|deepseek] [output-directory]" >&2; exit 2 ;; esac
OUT="${2:-$PWD/b300-results/$(date +%Y%m%d-%H%M%S)-$profile}"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
[[ ! -e "$OUT/plan.json" && ! -e "$OUT/stages.tsv" ]] || { echo "Use a new result directory" >&2; exit 1; }
for c in nvidia-smi python3; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
monitor_pid=""
telemetry_start=""
container_name="b300-test-$$-$(date +%s)"
export ROOT OUT PROFILE="$profile"
python3 - <<'PY'
import json,os
from pathlib import Path
p=os.environ['PROFILE']
required=[] if p=='inventory' else ['environment','stress','nccl','llm-Qwen3-0.6B','image-SDXL-base-1.0']
if p in ('full','burn'): required += ['llm-Qwen3-8B','dcgm']
if p in ('full','burn','compare'): required += ['nccl-public']
if p=='compare': required=['environment','nccl-public']
limit=int(os.environ.get('MAX_GPU_TEMP','85'))
if not 1 <= limit < 120: raise ValueError('MAX_GPU_TEMP must be an integer from 1 to 119')
Path(os.environ['OUT'],'plan.json').write_text(json.dumps({'profile':p,'gpus':8,'required':required,'max_gpu_temp':limit}),encoding='utf-8')
PY
cp -- "$ROOT/public_baselines.json" "$OUT/public-baselines.json"
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/stages.tsv"; }
cleanup() {
    original=$?
    trap - EXIT ERR
    if [[ -n "$monitor_pid" ]]; then
        if ! kill -0 "$monitor_pid" 2>/dev/null; then record telemetry FAIL 1; (( original != 0 )) || original=1; fi
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$telemetry_start" ]]; then
        python3 - "$OUT" "$telemetry_start" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1],'telemetry-window.json').write_text(json.dumps({'start_epoch':float(sys.argv[2]),'end_epoch':time.time()}),encoding='utf-8')
PY
    fi
    if command -v docker >/dev/null; then docker stop --time 10 "$container_name" >/dev/null 2>&1 || true; fi
    nvidia-smi -q -x > "$OUT/after.xml" 2> "$OUT/after.stderr" || true
    journalctl -k --since "@$start_epoch" --no-pager > "$OUT/kernel-during.log" 2> "$OUT/kernel.stderr" || true
    report_rc=0
    python3 "$ROOT/report.py" "$OUT" || report_rc=$?
    echo "Results: $OUT/report.html"
    if (( original == 0 )) && [[ "$profile" != inventory ]]; then original=$report_rc; fi
    exit "$original"
}
start_epoch=$(date +%s)
trap cleanup EXIT
trap 'rc=$?; record launcher FAIL "$rc"' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
{ uname -a; cat /etc/os-release; command -v lscpu >/dev/null && lscpu; } > "$OUT/host.txt" || true
nvidia-smi -q -x > "$OUT/before.xml"
nvidia-smi topo -m > "$OUT/topology.txt" 2>&1 || true
nvidia-smi nvlink -s > "$OUT/nvlink.txt" 2>&1 || true
nvidia-smi > "$OUT/nvidia-smi.txt"
[[ "$profile" != inventory ]] || exit 0
for c in docker timeout; do command -v "$c" >/dev/null || { record preflight FAIL 1; echo "Missing $c" >&2; exit 1; }; done
python3 - <<'PY'
import os,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
gpus=ET.parse(Path(os.environ['OUT'],'before.xml')).getroot().findall('gpu')
assert len(gpus)==8, f'Expected 8 GPUs, found {len(gpus)}'
for g in gpus:
    assert re.search(r'\bB300\b',g.findtext('product_name','')), 'GPU model is not B300'
    assert g.findtext('mig_mode/current_mig','Disabled') in ('Disabled','N/A'), 'Disable MIG before running this physical-GPU suite'
    assert g.findtext('ecc_mode/current_ecc','Enabled') in ('Enabled','N/A'), 'Enable ECC before an acceptance test'
driver=ET.parse(Path(os.environ['OUT'],'before.xml')).getroot().findtext('driver_version','0')
assert int(driver.split('.')[0]) >= 580, f'CUDA 13 package requires driver R580+; found {driver}'
busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
assert not busy, f'GPUs have running compute jobs (PIDs: {busy}); schedule an idle test window'
PY
python3 "$ROOT/bundle.py" verify | tee "$OUT/bundle-verification.log"
image_id=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads(Path(os.environ["ROOT"],"bundle.sha256.json").read_text())["image_id"])')
image_arch=$(python3 -c 'import json,os; from pathlib import Path; print(json.loads(Path(os.environ["ROOT"],"bundle.sha256.json").read_text())["architecture"])')
case "$(uname -m)" in x86_64) host_arch=amd64 ;; aarch64) host_arch=arm64 ;; *) echo "Unsupported server CPU" >&2; exit 1 ;; esac
[[ "$host_arch" == "$image_arch" ]] || { record preflight FAIL 1; echo "Archive CPU architecture mismatch" >&2; exit 1; }
if ! docker image inspect "$image_id" >/dev/null 2>&1; then docker load -i "$ROOT/images/b300-test.tar" | tee "$OUT/image-load.log"; fi
docker image inspect "$image_id" > "$OUT/image-used.json"
record preflight PASS 0
# Background telemetry is retained even if a test fails.
telemetry_start=$(python3 -c 'import time; print(time.time())')
(
    printf 'timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n'
    while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits || exit 1
        sleep 2
    done
) > "$OUT/telemetry.csv" 2> "$OUT/telemetry.stderr" &
monitor_pid=$!
base=(docker run --rm --pull never --name "$container_name" --label "b300-kit.job=${B300_JOB_ID:-cli-$$}" --network none --gpus all --shm-size=16g
      --ulimit memlock=-1 --ulimit stack=67108864
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1
      -e NCCL_DEBUG=INFO -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      -v "$ROOT:/kit:ro" -v "$OUT:/out" "$image_id")
stage() {
    local label="$1"; shift
    record "$label" RUNNING 0
    echo "===== $label ====="
    local rc=0
    # Long loading of eight model replicas is included in this wall-time limit.
    timeout --signal=TERM --kill-after=30s "${STAGE_TIMEOUT:-7200}" "${base[@]}" \
        torchrun --standalone --nproc-per-node=8 /kit/gpu_test.py "$@" 2>&1 | tee "$OUT/$label.log" || rc=$?
    if (( rc == 0 )); then record "$label" PASS 0; else record "$label" FAIL "$rc"; exit "$rc"; fi
}
stage environment environment
public_stage() {
    local rc=0
    record nccl-public RUNNING 0
    timeout --signal=TERM --kill-after=30s "${NCCL_PUBLIC_TIMEOUT:-1800}" "${base[@]}" \
        python /kit/public_benchmark.py run --out /out --timeout "${NCCL_PUBLIC_TIMEOUT:-1800}" \
        2>&1 | tee "$OUT/nccl-public-launcher.log" || rc=$?
    if (( rc == 0 )); then record nccl-public PASS 0; else record nccl-public FAIL "$rc"; exit "$rc"; fi
}
if [[ "$profile" == compare ]]; then
    public_stage
    exit 0
fi
seconds=60
[[ "$profile" != full ]] || seconds=600
[[ "$profile" != burn ]] || seconds="${BURN_SECONDS:-3600}"
stage stress stress --seconds "$seconds" --max-temp "${MAX_GPU_TEMP:-85}" --memory-fraction "${MEMORY_FRACTION:-0.65}"
stage nccl nccl --iterations 50
if [[ "$profile" == full || "$profile" == burn ]]; then public_stage; fi
if [[ "$profile" == quick ]]; then
    stage llm-Qwen3-0.6B llm --model /kit/models/Qwen3-0.6B --iterations 1 --batch 1 --tokens 32
    stage image-SDXL-base-1.0 image --model /kit/models/SDXL-base-1.0 --iterations 1 --steps 4
else
    stage llm-Qwen3-0.6B llm --model /kit/models/Qwen3-0.6B --iterations 3 --batch 4 --tokens 128
    stage llm-Qwen3-8B llm --model /kit/models/Qwen3-8B --iterations 3 --batch 4 --tokens 128
    stage image-SDXL-base-1.0 image --model /kit/models/SDXL-base-1.0 --iterations 3 --steps 20
    if command -v dcgmi >/dev/null; then
        dcgmi --version > "$OUT/dcgm-version.txt" 2>&1 || true
        if ! python3 - <<'PY'
import os,re,sys
from pathlib import Path
text=Path(os.environ['OUT'],'dcgm-version.txt').read_text(errors='replace')
m=re.search(r'(\d+)\.(\d+)\.(\d+)',text)
sys.exit(0 if m and tuple(map(int,m.groups())) >= (4,4,0) else 1)
PY
        then
            record dcgm SKIP 0
            echo "DCGM version could not be verified as 4.4+. Full report will be INCOMPLETE."
            exit 0
        fi
        rc=0
        timeout --signal=TERM --kill-after=30s "${DCGM_TIMEOUT:-10800}" dcgmi diag -r 3 -i 0,1,2,3,4,5,6,7 -j > "$OUT/dcgm.json" 2> "$OUT/dcgm.stderr" || rc=$?
        if (( rc == 0 )); then record dcgm PASS 0; else record dcgm FAIL "$rc"; exit "$rc"; fi
    else
        record dcgm SKIP 0
        echo "DCGM absent: install B300-compatible DCGM 4.4+ on server. Full report will be INCOMPLETE."
    fi
fi
# cleanup captures final hardware evidence and produces both JSON and HTML.
