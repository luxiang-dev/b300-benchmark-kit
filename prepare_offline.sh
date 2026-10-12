#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-full}"
case "$mode" in full|--hardware-only) ;; *) echo 'Usage: bash prepare_offline.sh [--hardware-only]' >&2; exit 2 ;; esac
for c in docker python3; do command -v "$c" >/dev/null || { echo "Missing $c" >&2; exit 1; }; done
arch="${TARGET_ARCH:-$(uname -m)}"
case "$arch" in x86_64|amd64) platform=linux/amd64; arch=amd64 ;; aarch64|arm64) platform=linux/arm64; arch=arm64 ;; *) echo "Unsupported CPU: $arch" >&2; exit 1 ;; esac
echo "Preparing $platform package. GPU is not required on the preparation host."
if [[ "$mode" != --hardware-only ]]; then python3 download_models.py; fi
mkdir -p images
base_tag="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.10-py3}"
docker pull --platform "$platform" "$base_tag"
# Pin the pulled base by digest, then keep the resulting complete image by ID.
base=$(docker image inspect "$base_tag" --format '{{index .RepoDigests 0}}')
docker build --platform "$platform" --build-arg "BASE_IMAGE=$base" -t b300-test:1.0 .
actual=$(docker image inspect b300-test:1.0 --format '{{.Architecture}}')
[[ "$actual" == "$arch" ]] || { echo "Image architecture mismatch" >&2; exit 1; }
docker run --rm --network none b300-test:1.0 python -c 'import torch,transformers,diffusers,accelerate; print(torch.__version__,torch.version.cuda)'
docker image inspect b300-test:1.0 > images/runtime.inspect.json
docker save -o images/b300-test.tar.part b300-test:1.0
mv -- images/b300-test.tar.part images/b300-test.tar
if [[ "$mode" == --hardware-only ]]; then python3 bundle.py seal --hardware-only; else python3 bundle.py seal; fi
echo "OFFLINE PACKAGE SEALED. Copy this entire directory to your external disk."
