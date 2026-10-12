#!/usr/bin/env python3
"""8-GPU functional tests. Run with torchrun; no online dependencies at runtime."""
import argparse
from datetime import timedelta
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def emit(args, result):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    result.update(test=args.mode, rank=rank, model=args.model, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    suffix = "-" + Path(args.model).name if args.model else ""
    (path / f"{args.mode}{suffix}-gpu{rank}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def setup(args):
    import torch
    import torch.distributed as dist
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() != args.gpus or world != args.gpus:
        raise RuntimeError(f"Expected {args.gpus} visible GPUs/processes, got {torch.cuda.device_count()}/{world}")
    torch.cuda.set_device(rank)
    props = torch.cuda.get_device_properties(rank)
    if not re.search(r"\bB300\b", props.name):
        raise RuntimeError(f"Expected B300, found {props.name}")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device(f"cuda:{rank}"))
    return torch, dist, rank, world


def temperature(rank):
    result = subprocess.run(["nvidia-smi", "-i", str(rank), "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    return float(result.stdout.strip())


def environment(torch, dist, rank, world, args):
    props = torch.cuda.get_device_properties(rank)
    a = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    c = a @ a
    torch.cuda.synchronize()
    if not bool((c == 128).all()):
        raise RuntimeError("BF16 CUDA matrix correctness failed")
    dist.barrier()
    return dict(status="PASS", gpu=props.name, memory_gib=props.total_memory / 2**30,
                capability=list(torch.cuda.get_device_capability(rank)), torch=torch.__version__,
                cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()), world_size=world)


def stress(torch, dist, rank, world, args):
    n = args.matrix
    a = torch.full((n, n), 1 / 16, device="cuda", dtype=torch.bfloat16)
    b = a.clone()
    c = torch.empty_like(a)
    free, total = torch.cuda.mem_get_info()
    # Fill and check much of the HBM, keeping headroom for CUDA kernels.
    budget = min(int(total * args.memory_fraction), free - 2 * 2**30)
    chunk_count = max(0, budget // (256 * 2**20))
    if not chunk_count:
        raise RuntimeError("Insufficient free GPU memory for HBM stress")
    memory = [torch.full((64 * 2**20,), 17, dtype=torch.int32, device="cuda") for _ in range(chunk_count)]
    for block in memory:
        if not bool((block == 17).all()):
            raise RuntimeError("Initial HBM pattern check failed")
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    last_memory = start
    last_temp = start
    value = 17
    gemm_seconds = 0.0
    iterations = 0
    peak_temp = temperature(rank)
    while time.perf_counter() - start < args.seconds:
        t0 = time.perf_counter()
        for _ in range(20):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        gemm_seconds += time.perf_counter() - t0
        iterations += 20
        if not bool((c == n / 256).all()):
            raise RuntimeError("BF16 GEMM output mismatch")
        now = time.perf_counter()
        if now - last_memory >= 5:
            next_value = -31 if value == 17 else 17
            for block in memory:
                if not bool((block == value).all()):
                    raise RuntimeError("HBM retention/pattern mismatch")
                block.fill_(next_value)
                if not bool((block == next_value).all()):
                    raise RuntimeError("HBM write/read mismatch")
            value = next_value
            last_memory = time.perf_counter()
        if now - last_temp >= 5:
            peak_temp = max(peak_temp, temperature(rank))
            if peak_temp >= args.max_temp:
                raise RuntimeError(f"Temperature reached stop threshold {args.max_temp} C")
            last_temp = time.perf_counter()
    for block in memory:
        if not bool((block == value).all()):
            raise RuntimeError("Final HBM check failed")
    peak_temp = max(peak_temp, temperature(rank))
    if peak_temp >= args.max_temp:
        raise RuntimeError(f"Temperature reached stop threshold {args.max_temp} C")
    return dict(status="PASS", duration_s=time.perf_counter() - start, matrix=n, dtype="BF16",
                memory_tested_gib=chunk_count * 0.25, gemm_iterations=iterations,
                gemm_tflops=2 * n ** 3 * iterations / gemm_seconds / 1e12,
                peak_temperature_c=peak_temp, memory_patterns=[17, -31],
                note="Synthetic GEMM only; excludes memory-test time. Not an official peak performance score.")


def nccl(torch, dist, rank, world, args):
    rows = []
    for mib in (16, 64, 256):
        x = torch.empty(mib * 2**20 // 4, device="cuda", dtype=torch.float32)
        for _ in range(5):
            x.fill_(rank + 1)
            dist.all_reduce(x)
        for _ in range(3):
            x.fill_(rank + 1)
            dist.all_reduce(x)
            if not bool((x == world * (world + 1) / 2).all()):
                raise RuntimeError(f"NCCL data corruption at {mib} MiB")
        # Time communication alone: repeated SUM would overflow, so use MAX
        # during measurement, and independent SUM correctness checks above.
        x.fill_(rank + 1)
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000 / args.iterations
        # Use the slowest rank for a conservative collective measurement.
        slowest = torch.tensor(seconds, device="cuda", dtype=torch.float64)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        seconds = slowest.item()
        if not bool((x == world).all()):
            raise RuntimeError("NCCL MAX output mismatch")
        alg = mib * 2**20 / seconds / 1e9
        rows.append(dict(size_mib=mib, latency_ms=seconds * 1000,
                         algorithm_gbps=alg, bus_gbps=alg * 2 * (world - 1) / world,
                         operation="MAX", sum_correctness="PASS"))
    return dict(status="PASS", world_size=world, measurements=rows,
                note="PyTorch NCCL all-reduce; reported bus bandwidth uses 2*(N-1)/N. Not nccl-tests or proof of NVLink routing.")


def llm(torch, dist, rank, world, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    prompt = tok.apply_chat_template([{"role": "user", "content": "请解释人工智能服务器的 GPU、显存和 NVLink 各有什么作用。"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok([prompt] * args.batch, return_tensors="pt", padding=True).to("cuda")
    kwargs = dict(max_new_tokens=args.tokens, min_new_tokens=args.tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model.generate(**inputs, **{**kwargs, "max_new_tokens": 16, "min_new_tokens": 16})
        torch.cuda.synchronize()
        dist.barrier()
        timed = []
        count = 0
        wall_start = time.perf_counter()
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = model.generate(**inputs, **kwargs)
            torch.cuda.synchronize()
            timed.append(time.perf_counter() - t0)
            count += (output.shape[1] - inputs.input_ids.shape[1]) * output.shape[0]
        local_seconds = sum(timed)
        dist.barrier()
        collective_wall = time.perf_counter() - wall_start
        overall = torch.tensor([count], dtype=torch.float64, device="cuda")
        slowest = torch.tensor([collective_wall], dtype=torch.float64, device="cuda")
        dist.all_reduce(overall)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    sample = tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    if not sample.strip():
        raise RuntimeError("Model generated empty text")
    return dict(status="PASS", dtype="BF16", input_tokens_per_request=inputs.input_ids.shape[1],
                batch=args.batch, forced_output_tokens=args.tokens, iterations=args.iterations,
                tokens_per_second=count / local_seconds,
                aggregate_tokens_per_second=overall.item() / slowest.item(),
                latency_s=timed, median_latency_s=statistics.median(timed),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, sample=sample,
                note="8 independent replicas; latency includes prefill and decode. Not tensor parallelism, TTFT or vLLM production benchmarking.")


def image(torch, dist, rank, world, args):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(args.model, local_files_only=True, use_safetensors=True,
                                             variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    prompt = "A small red robot in a modern server room, photographic, detailed, no text"
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        pipe(prompt, num_inference_steps=2, height=1024, width=1024,
             generator=torch.Generator(device="cuda").manual_seed(42))
        torch.cuda.synchronize()
        dist.barrier()
        timed = []
        for i in range(args.iterations):
            t0 = time.perf_counter()
            generated = pipe(prompt, num_inference_steps=args.steps, height=1024, width=1024,
                             generator=torch.Generator(device="cuda").manual_seed(42 + rank + i)).images[0]
            torch.cuda.synchronize()
            timed.append(time.perf_counter() - t0)
            # Non-finite output sometimes produces a black image after conversion.
            import numpy as np
            if float(np.asarray(generated).std()) < 1:
                raise RuntimeError("SDXL image is blank/uniform; inspect VAE/numerical stability")
            generated.save(Path(args.out) / f"sdxl-gpu{rank}-{i}.png")
    return dict(status="PASS", dtype="FP16", resolution=[1024, 1024], steps=args.steps,
                latency_s=timed, images_per_second=len(timed) / sum(timed),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                note="Independent SDXL replicas; timings exclude saving PNG, include pipeline/CPU postprocessing.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["environment", "stress", "nccl", "llm", "image"])
    p.add_argument("--out", default="/out")
    p.add_argument("--model", default="")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--matrix", type=int, default=8192)
    p.add_argument("--memory-fraction", type=float, default=0.65)
    p.add_argument("--max-temp", type=int, default=85)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    if args.seconds < 1 or args.iterations < 1 or args.tokens < 1 or args.batch < 1:
        p.error("seconds/iterations/tokens/batch must be positive")
    if not 0.05 <= args.memory_fraction <= 0.80:
        p.error("--memory-fraction must be 0.05..0.80")
    if args.matrix < 256 or args.matrix > 16384 or args.matrix % 256:
        p.error("--matrix must be 256..16384, divisible by 256")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dist = None
    try:
        torch, dist, rank, world = setup(args)
        result = globals()[args.mode](torch, dist, rank, world, args)
        emit(args, result)
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        emit(args, dict(status="FAIL", error=str(exc), traceback=traceback.format_exc()))
        # torchrun terminates peer ranks on failure; avoid a blocking final barrier.
        return 1


if __name__ == "__main__":
    sys.exit(main())
