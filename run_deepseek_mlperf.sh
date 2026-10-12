#!/usr/bin/env bash
# A separately prepared official MLPerf runtime is required. No downloads here.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MLPERF_REPO:?Set MLPERF_REPO to the complete pinned inference_results_v6.0 checkout}"
: "${MLPERF_IMAGE:?Set MLPERF_IMAGE to the locally prepared DeepSeek MLPerf image ID/tag}"
: "${MLPERF_SCRATCH_PATH:?Set MLPERF_SCRATCH_PATH to the prepared model/data scratch directory}"
for c in python3 docker nvidia-smi; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
if [[ -z "${B300_GPU_LOCK_FD:-}" ]]; then exec python3 "$ROOT/job_lock.py" -- bash "$ROOT/run_deepseek_mlperf.sh" "$@"; fi
REPO="$(cd -- "$MLPERF_REPO" && pwd)"
SCRATCH="$(cd -- "$MLPERF_SCRATCH_PATH" && pwd)"
# Upstream Makefiles expand these paths as shell words.
for path in "$REPO" "$SCRATCH"; do
    [[ "$path" != *[[:space:]]* && "$path" != *:* ]] || { echo "Official MLPerf paths must have no whitespace or colon" >&2; exit 1; }
done
[[ -f "$REPO/closed/NVIDIA/Makefile" && -d "$REPO/.git" ]] || { echo "Use the complete official Git checkout (including .git and 3rdparty)" >&2; exit 1; }
OUT="${1:-$PWD/b300-results/$(date +%Y%m%d-%H%M%S)-deepseek}"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
[[ ! -e "$OUT/plan.json" && ! -e "$OUT/stages.tsv" ]] || { echo "Use a new result directory" >&2; exit 1; }
export OUT
python3 - <<'PY'
import json,os
from pathlib import Path
limit=int(os.environ.get('MAX_GPU_TEMP','85'))
if not 1 <= limit < 120: raise ValueError('MAX_GPU_TEMP must be an integer from 1 to 119')
Path(os.environ['OUT'],'plan.json').write_text(json.dumps({'profile':'deepseek','gpus':8,'required':['deepseek-r1-mlperf'],'max_gpu_temp':limit}),encoding='utf-8')
PY
cp -- "$ROOT/public_baselines.json" "$OUT/public-baselines.json"
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/stages.tsv"; }
monitor_pid=""
telemetry_start=""
container_name="b300-deepseek-$$-$(date +%s)"
start_epoch=$(date +%s)
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
    docker stop --time 10 "$container_name" >/dev/null 2>&1 || true
    nvidia-smi -q -x > "$OUT/after.xml" 2> "$OUT/after.stderr" || true
    journalctl -k --since "@$start_epoch" --no-pager > "$OUT/kernel-during.log" 2> "$OUT/kernel.stderr" || true
    report_rc=0
    python3 "$ROOT/report.py" "$OUT" || report_rc=$?
    echo "Results: $OUT/report.html"
    if (( original == 0 )); then original=$report_rc; fi
    exit "$original"
}
trap cleanup EXIT
trap 'rc=$?; record launcher FAIL "$rc"' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-smi -q -x > "$OUT/before.xml"
nvidia-smi topo -m > "$OUT/topology.txt" 2>&1 || true
nvidia-smi nvlink -s > "$OUT/nvlink.txt" 2>&1 || true
{ uname -a; cat /etc/os-release; command -v lscpu >/dev/null && lscpu; } > "$OUT/host.txt" || true
python3 - <<'PY'
import os,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
gpus=ET.parse(Path(os.environ['OUT'],'before.xml')).getroot().findall('gpu')
assert len(gpus)==8, 'Expected 8 GPUs'
for g in gpus:
    assert re.search(r'\bB300\b',g.findtext('product_name','')), 'Requires physical B300 GPUs'
    assert g.findtext('mig_mode/current_mig','Disabled') in ('Disabled','N/A'), 'Disable MIG'
    assert g.findtext('ecc_mode/current_ecc','Enabled') in ('Enabled','N/A'), 'Enable ECC'
driver=ET.parse(Path(os.environ['OUT'],'before.xml')).getroot().findtext('driver_version','0')
assert int(driver.split('.')[0]) >= 580, f'Requires a compatible R580+ driver; found {driver}'
busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
assert not busy, f'GPU compute jobs already running: {busy}'
PY
# Resolve the local image to an immutable ID; Docker is never allowed to pull.
image_id=$(docker image inspect "$MLPERF_IMAGE" --format '{{.Id}}')
image_arch=$(docker image inspect "$image_id" --format '{{.Architecture}}')
case "$(uname -m)" in x86_64) host_arch=amd64 ;; aarch64) host_arch=arm64 ;; *) echo "Unsupported server CPU" >&2; exit 1 ;; esac
[[ "$image_arch" == "$host_arch" ]] || { echo "DeepSeek image CPU architecture mismatch" >&2; exit 1; }
docker image inspect "$image_id" > "$OUT/deepseek-image-used.json"
record preflight PASS 0
telemetry_start=$(python3 -c 'import time; print(time.time())')
(
    printf 'timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n'
    while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits || exit 1
        sleep 2
    done
) > "$OUT/telemetry.csv" 2> "$OUT/telemetry.stderr" &
monitor_pid=$!
rc=0
# Scratch is mounted at its original absolute path to preserve prepared links.
# Keep the complete source repository visible so pinned-revision checks work.
record deepseek-r1-mlperf RUNNING 0
docker run --rm --pull never --name "$container_name" --label "b300-kit.job=${B300_JOB_ID:-cli-$$}" --network none --gpus all \
    --shm-size="${MLPERF_SHM_SIZE:-32g}" --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/bash -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PIP_NO_INDEX=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -e MLPERF_SCRATCH_PATH="$SCRATCH" -v "$ROOT:/kit:ro" -v "$OUT:/out" \
    -v "$REPO:/submission" -v "$REPO/closed/NVIDIA:/work" -v "$SCRATCH:$SCRATCH" \
    -w /submission/closed/NVIDIA "$image_id" \
    -c 'exec python3 /kit/deepseek_mlperf.py run --workspace /submission/closed/NVIDIA --out /out' \
    2>&1 | tee "$OUT/deepseek-launcher.log" || rc=$?
if (( rc == 0 )); then record deepseek-r1-mlperf PASS 0; else record deepseek-r1-mlperf FAIL "$rc"; exit "$rc"; fi
