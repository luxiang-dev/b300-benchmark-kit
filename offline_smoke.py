"""Run only inside a prepared --network none container; no benchmark throughput."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',action='store_true')
    parser.add_argument('--root',type=Path,default=Path('/kit'))
    args=parser.parse_args()
    import torch
    import transformers, diffusers, accelerate, safetensors
    if not torch.version.cuda or not torch.cuda.is_available() or not torch.cuda.device_count():
        raise RuntimeError('The prepared container cannot initialize NVIDIA CUDA on this target')
    devices=[]
    torch.backends.cuda.matmul.allow_tf32=False
    for rank in range(torch.cuda.device_count()):
        torch.cuda.set_device(rank)
        x=torch.ones((128,128),device='cuda',dtype=torch.float32)
        if not torch.equal((x@x).cpu(),torch.full((128,128),128.0)):
            raise RuntimeError('FP32 correctness smoke failed on GPU '+str(rank))
        torch.cuda.synchronize()
        devices.append(dict(index=rank,name=torch.cuda.get_device_name(rank),capability=list(torch.cuda.get_device_capability(rank))))
    # NCCL's binary must be the one built from the vendored, pinned source.
    manifest=Path('/opt/nccl-tests/build-manifest.json')
    binary=Path('/opt/nccl-tests/build/all_reduce_perf')
    record=json.loads(manifest.read_text(encoding='utf-8'))
    from build_nccl_tests import REVISION, SOURCE_SHA256
    if record.get('nccl_tests_revision')!=REVISION or record.get('source_sha256')!=SOURCE_SHA256 or record.get('binary_sha256')!=hashlib.sha256(binary.read_bytes()).hexdigest():
        raise RuntimeError('Prepared nccl-tests binary/source binding failed')
    if args.models:
        for name in ('Qwen3-0.6B','Qwen3-8B'):
            path=args.root/'models'/name
            tok=transformers.AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
            config=transformers.AutoConfig.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
            if not tok.encode('offline smoke') or not config.model_type:
                raise RuntimeError('Offline tokenizer/config loading failed: '+name)
        config=diffusers.DiffusionPipeline.load_config(str(args.root/'models/SDXL-base-1.0'),local_files_only=True)
        if not config.get('_class_name'):
            raise RuntimeError('Offline SDXL pipeline config failed')
    print(json.dumps(dict(test='disconnected-runtime-smoke',status='PASS',torch=torch.__version__,cuda=torch.version.cuda,
                          devices=devices,model_metadata_checked=args.models,
                          boundary='Small CUDA/tokenizer/config checks only; not full inference, HBM stress or procurement acceptance.'),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
