#!/usr/bin/env bash
# A separately prepared official MLPerf runtime is required. No downloads here.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MLPERF_REPO:?Set MLPERF_REPO to the complete pinned inference_results_v6.0 checkout}"
: "${MLPERF_IMAGE:?Set MLPERF_IMAGE to the locally prepared DeepSeek MLPerf image ID/tag}"
: "${MLPERF_SCRATCH_PATH:?Set MLPERF_SCRATCH_PATH to the prepared model/data scratch directory}"
for c in python3 docker nvidia-smi; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
REPO="$(cd -- "$MLPERF_REPO" && pwd)"
SCRATCH="$(cd -- "$MLPERF_SCRATCH_PATH" && pwd)"
[[ -f "$REPO/closed/NVIDIA/Makefile" && -d "$REPO/.git" ]] || { echo "Use the complete official Git checkout (including .git and 3rdparty)" >&2; exit 1; }
OUT="${1:-$PWD/b300-results/$(date +%Y%m%d-%H%M%S)-deepseek}"
mkdir -p -- "$OUT"
OUT="$(cd -- "$OUT" && pwd)"
[[ ! -e "$OUT/plan.json" && ! -e "$OUT/stages.tsv" ]] || { echo "Use a new result directory" >&2; exit 1; }
export OUT
python3 - <<'PY'
import json,os
from pathlib import Path
Path(os.environ['OUT'],'plan.json').write_text(json.dumps({'profile':'deepseek','gpus':8,'required':['deepseek-r1-mlperf']}),encoding='utf-8')
PY
cp -- "$ROOT/public_baselines.json" "$OUT/public-baselines.json"
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/stages.tsv"; }
monitor_pid=""
container_name="b300-deepseek-$$-$(date +%s)"
start_epoch=$(date +%s)
cleanup() {
    original=$?
    trap - EXIT ERR
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
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
busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
assert not busy, f'GPU compute jobs already running: {busy}'
PY
# Resolve the local image to an immutable ID; Docker is never allowed to pull.
image_id=$(docker image inspect "$MLPERF_IMAGE" --format '{{.Id}}')
docker image inspect "$image_id" > "$OUT/deepseek-image-used.json"
record preflight PASS 0
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
docker run --rm --pull never --name "$container_name" --network none --gpus all \
    --shm-size="${MLPERF_SHM_SIZE:-32g}" --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/bash -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PIP_NO_INDEX=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -e MLPERF_SCRATCH_PATH="$SCRATCH" -v "$ROOT:/kit:ro" -v "$OUT:/out" \
    -v "$REPO:/submission" -v "$REPO/closed/NVIDIA:/work" -v "$SCRATCH:$SCRATCH" \
    -w /submission/closed/NVIDIA "$image_id" \
    -lc 'exec python3 /kit/deepseek_mlperf.py run --workspace /submission/closed/NVIDIA --out /out' \
    2>&1 | tee "$OUT/deepseek-launcher.log" || rc=$?
if (( rc == 0 )); then record deepseek-r1-mlperf PASS 0; else record deepseek-r1-mlperf FAIL "$rc"; exit "$rc"; fi
