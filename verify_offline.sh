#!/usr/bin/env bash
# Target-host rehearsal only: fixed local images; no apt/pip/git downloads.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scope="${1:-all}"
case "$scope" in server|hardware|full|all|deepseek) ;; *) echo 'Usage: bash verify_offline.sh [server|hardware|full|all|deepseek]' >&2; exit 2 ;; esac
[[ "$(uname -s)" == Linux ]] || { echo 'Use the target Ubuntu/Linux host' >&2; exit 2; }
if [[ "$scope" == server ]]; then
    exec python3 "$ROOT/run_authenticity.py" --profile server --out "${2:-$ROOT/b300-results/verify-offline-server-$(date +%Y%m%d-%H%M%S)}"
fi
if [[ -z "${B300_GPU_LOCK_FD:-}" ]]; then exec python3 "$ROOT/job_lock.py" -- bash "$ROOT/verify_offline.sh" "$@"; fi
for tool in python3 docker nvidia-smi; do command -v "$tool" >/dev/null || { echo "Missing $tool" >&2; exit 1; }; done
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "$busy" ]] || { echo 'Require an idle target GPU window for the CUDA smoke test' >&2; exit 1; }
container_name="b300-offline-$$-$(date +%s)"
cleanup() {
    rc=$?
    trap - EXIT
    if ! docker stop --time 10 "$container_name" >/dev/null 2>&1; then
        running=$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)
        if [[ "$running" == true ]]; then
            echo "Cleanup failed: container $container_name is still running" >&2
            [[ "$rc" != 0 ]] || rc=1
        fi
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$scope" != deepseek ]]; then
    options=()
    seal="$ROOT/bundle.sha256.json"
    models=()
    if [[ "$scope" == hardware ]]; then options+=(--hardware-only); seal="$ROOT/hardware.sha256.json"; else models+=(--models); fi
    python3 "$ROOT/bundle.py" verify "${options[@]}"
    image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$seal")
    if ! docker image inspect "$image_id" >/dev/null 2>&1; then docker load -i "$ROOT/images/b300-test.tar"; fi
    docker image inspect "$image_id" >/dev/null
    docker run --rm --pull never --name "$container_name" --label "b300-kit.job=${B300_JOB_ID:-offline-cli-$$}" --network none --gpus all --shm-size=16g \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
      -v "$ROOT:/kit:ro" "$image_id" python /kit/offline_smoke.py "${models[@]}"
fi
if [[ "$scope" == all || "$scope" == deepseek ]]; then
    : "${MLPERF_REPO:?Set the complete pinned official checkout}"
    : "${MLPERF_SCRATCH_PATH:?Set the complete prepared scratch directory}"
    : "${MLPERF_IMAGE:?Set the prepared local DeepSeek image}"
    for path in "$MLPERF_REPO" "$MLPERF_SCRATCH_PATH"; do
        [[ "$path" != *[[:space:]]* && "$path" != *:* ]] || { echo 'Official MLPerf paths must not contain whitespace or colon' >&2; exit 2; }
    done
    if ! docker image inspect "$MLPERF_IMAGE" >/dev/null 2>&1; then
        : "${MLPERF_IMAGE_ARCHIVE:?Set the final local DeepSeek tar archive to import}"
        docker load -i "$MLPERF_IMAGE_ARCHIVE"
    fi
    image_id=$(docker image inspect "$MLPERF_IMAGE" --format '{{.Id}}')
    docker run --rm --pull never --name "$container_name" --label "b300-kit.job=${B300_JOB_ID:-offline-cli-$$}" --network none --gpus all --shm-size=32g \
      --entrypoint python3 \
      -e "MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH" \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
      -v "$MLPERF_REPO:/submission" -v "$MLPERF_REPO/closed/NVIDIA:/work" \
      -v "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" -v "$ROOT:/kit:ro" \
      -w /submission/closed/NVIDIA "$image_id" /kit/deepseek_mlperf.py check --workspace /submission/closed/NVIDIA
fi
echo 'DISCONNECTED PREFLIGHT PASSED. Full hardware/model/DeepSeek runs and OEM acceptance are still separate.'
