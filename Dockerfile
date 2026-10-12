ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}
LABEL hardware_kit.runtime="nvidia-cuda"
COPY requirements.txt /tmp/b300-requirements.txt
COPY image_constraints.py /tmp/b300-image-constraints.py
# Only unlock the selected app packages; retain the NVIDIA GPU runtime pins.
RUN python /tmp/b300-image-constraints.py /etc/pip/constraint.txt /tmp/b300-requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/b300-requirements.txt \
    && python -m pip check \
    && python -c "import torch, transformers, diffusers, accelerate; assert torch.version.cuda; print(torch.__version__, torch.version.cuda)" \
    && python -m pip freeze > /opt/b300-environment.txt
COPY vendor/nccl-tests-v2.17.6.tar.gz /tmp/nccl-tests-v2.17.6.tar.gz
COPY build_nccl_tests.py /tmp/build_nccl_tests.py
RUN python /tmp/build_nccl_tests.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
WORKDIR /kit
ENTRYPOINT []
CMD ["python", "gpu_test.py", "--help"]
