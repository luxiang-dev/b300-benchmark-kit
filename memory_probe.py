#!/usr/bin/env python3
"""Active B300 HBM write/read coverage test; launch with torchrun on an idle node.

All allocated chunks remain resident and are written before any are verified.
This checks the measured allocation, not all installed HBM or product provenance.
PyTorch is imported only by the existing GPU setup at execution time.
"""
import argparse
import math
import sys
import time
import traceback

import gpu_test


CHUNK_BYTES = 256 * 2**20
RESERVE_BYTES = 2 * 2**30
WORD_MASK = 2**32 - 1
PATTERNS = ("address", "inverted-address", "lcg")


def allocation_budget(total_bytes, free_bytes, fraction=0.85):
    """Return whole chunks, keeping 2 GiB free in addition to pattern scratch."""
    for name, value in (("total_bytes", total_bytes), ("free_bytes", free_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if total_bytes <= 0 or free_bytes > total_bytes:
        raise ValueError("CUDA memory totals are inconsistent")
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0.50 <= fraction <= 0.90:
        raise ValueError("memory fraction must be finite and within 0.50..0.90")
    # Scratch is separate from tested HBM and is never counted as verified capacity.
    usable = free_bytes - RESERVE_BYTES - CHUNK_BYTES
    budget = min(int(total_bytes * fraction), usable)
    allocated = max(0, budget // CHUNK_BYTES) * CHUNK_BYTES
    if allocated < CHUNK_BYTES:
        raise ValueError("Insufficient free HBM for one test chunk, pattern scratch and 2 GiB reserve")
    return allocated


def pattern_seed(rank, chunk):
    """A distinct 32-bit seed for each supported rank/chunk pair.

    An odd multiplier is a permutation modulo 2**32. Chunk vectors therefore
    differ across GPUs and allocations, unlike a constant fill of every chunk.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 256:
        raise ValueError("rank must be an integer from 0 to 255")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or not 0 <= chunk < 2**24:
        raise ValueError("chunk must be an integer from 0 to 16777215")
    return (((rank << 24) | chunk) * 0x9E3779B1 + 0xA5A5A5A5) & WORD_MASK


def signed_word(value):
    value &= WORD_MASK
    return value if value < 2**31 else value - 2**32


def pattern_word(index, rank, chunk, pattern):
    """CPU reference for a word; used to describe/check the pattern contract."""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < CHUNK_BYTES // 4:
        raise ValueError("index is outside a test chunk")
    seed = pattern_seed(rank, chunk)
    if pattern == "address":
        value = index + seed
    elif pattern == "inverted-address":
        value = ~(index + seed)
    elif pattern == "lcg":
        value = index * 1664525 + (seed ^ 0x6D2B79F5)
    else:
        raise ValueError("Unknown HBM pattern")
    return signed_word(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/out")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--max-temp", type=int, default=85)
    args = parser.parse_args(argv)
    if not 1 <= args.gpus <= 256:
        parser.error("--gpus must be 1..256")
    if not math.isfinite(args.memory_fraction) or not 0.50 <= args.memory_fraction <= 0.90:
        parser.error("--memory-fraction must be finite and 0.50..0.90")
    if not 1 <= args.max_temp < 120:
        parser.error("--max-temp must be 1..119")
    args.mode, args.model = "memory", ""
    return args


def validate_evidence(result):
    """Reject inconsistent PASS records; this is schema validation, not attestation."""
    if not isinstance(result, dict) or result.get("status") != "PASS" or result.get("test") != "memory":
        raise ValueError("Not a passing memory test record")
    if result.get("dtype") != "int32" or result.get("patterns") != list(PATTERNS):
        raise ValueError("Memory dtype or pattern list is inconsistent")
    counts = {}
    for key in ("total_memory_bytes", "free_memory_before_bytes", "chunk_bytes", "reserve_bytes",
                "scratch_bytes", "target_allocation_bytes", "allocated_bytes", "verified_bytes_per_pass",
                "total_verified_bytes"):
        value = result.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        counts[key] = value
    if counts["chunk_bytes"] != CHUNK_BYTES or counts["scratch_bytes"] != CHUNK_BYTES or counts["reserve_bytes"] != RESERVE_BYTES:
        raise ValueError("Memory chunk, scratch or reserve differs from the probe contract")
    fraction = result.get("requested_memory_fraction")
    if type(fraction) not in (int, float):
        raise ValueError("Memory fraction is missing or invalid")
    budget = allocation_budget(counts["total_memory_bytes"], counts["free_memory_before_bytes"], fraction)
    allocated = counts["allocated_bytes"]
    if allocated % CHUNK_BYTES or allocated != counts["target_allocation_bytes"] or allocated != budget:
        raise ValueError("Allocated memory does not match the bounded whole-chunk budget")
    if counts["verified_bytes_per_pass"] != allocated or counts["total_verified_bytes"] != allocated * len(PATTERNS):
        raise ValueError("Verified capacity and repeated-pass byte totals are inconsistent")
    for key, expected in (("coverage_percent", allocated / counts["total_memory_bytes"] * 100),
                          ("memory_tested_gib", allocated / 2**30)):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"{key} does not represent actual capacity coverage")
    limited = allocated < int(counts["total_memory_bytes"] * fraction) // CHUNK_BYTES * CHUNK_BYTES
    if type(result.get("free_memory_limited")) is not bool or result["free_memory_limited"] != limited:
        raise ValueError("Free-memory limitation flag is inconsistent")
    passes = result.get("passes")
    if not isinstance(passes, list) or len(passes) != len(PATTERNS):
        raise ValueError("All three memory pattern passes are required")
    for name, phase in zip(PATTERNS, passes):
        if not isinstance(phase, dict) or phase.get("pattern") != name or phase.get("status") != "PASS":
            raise ValueError("Memory passes are missing, repeated or out of order")
        for key in ("written_bytes", "verified_bytes"):
            if type(phase.get(key)) is not int or phase[key] != allocated:
                raise ValueError("Each pattern must write and verify the whole simultaneous allocation")
        for key in ("write_seconds", "verify_seconds"):
            value = phase.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Memory phase timing is invalid")
    for key in ("duration_s", "peak_temperature_c"):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} is missing or invalid")
    if result["peak_temperature_c"] >= 120:
        raise ValueError("Memory peak temperature is out of range")


class TemperatureGuard:
    def __init__(self, rank, limit):
        self.rank, self.limit = rank, limit
        self.last_check = -math.inf
        self.peak = None

    def check(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_check < 2:
            return
        value = gpu_test.temperature(self.rank)
        if not math.isfinite(value) or not 0 < value < 120:
            raise RuntimeError("Invalid GPU temperature reading")
        self.last_check = time.monotonic()
        self.peak = value if self.peak is None else max(self.peak, value)
        if value >= self.limit:
            raise RuntimeError(f"Temperature reached stop threshold {self.limit} C")


def _fill_expected(torch, scratch, rank, chunk, pattern):
    # Reuse the single index buffer; no second int32 tensor is created per chunk.
    torch.arange(scratch.numel(), dtype=torch.int32, device=scratch.device, out=scratch)
    seed = pattern_seed(rank, chunk)
    if pattern == "lcg":
        scratch.mul_(1664525).add_(signed_word(seed ^ 0x6D2B79F5))
    else:
        scratch.add_(signed_word(seed))
        if pattern == "inverted-address":
            scratch.bitwise_not_()


def run_probe(torch, dist, rank, args, evidence):
    guard = TemperatureGuard(rank, args.max_temp)
    guard.check(force=True)
    free, total = map(int, torch.cuda.mem_get_info())
    allocated = allocation_budget(total, free, args.memory_fraction)
    count = allocated // CHUNK_BYTES
    evidence.update(total_memory_bytes=total, free_memory_before_bytes=free,
                    requested_memory_fraction=args.memory_fraction,
                    dtype="int32", patterns=list(PATTERNS),
                    reserve_bytes=RESERVE_BYTES, scratch_bytes=CHUNK_BYTES,
                    chunk_bytes=CHUNK_BYTES, target_allocation_bytes=allocated,
                    allocated_bytes=0, verified_bytes_per_pass=0,
                    total_verified_bytes=0, coverage_percent=0.0, passes=[],
                    free_memory_limited=allocated < int(total * args.memory_fraction) // CHUNK_BYTES * CHUNK_BYTES,
                    note="Only simultaneously allocated HBM is verified. Repeated passes do not increase capacity coverage; this is not an authenticity certificate.")
    blocks = []
    scratch = torch.empty(CHUNK_BYTES // 4, dtype=torch.int32, device="cuda")
    try:
        for _ in range(count):
            guard.check()
            blocks.append(torch.empty(CHUNK_BYTES // 4, dtype=torch.int32, device="cuda"))
            evidence["allocated_bytes"] = len(blocks) * CHUNK_BYTES
            evidence["coverage_percent"] = evidence["allocated_bytes"] / total * 100
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for pattern in PATTERNS:
            guard.check(force=True)
            phase = dict(pattern=pattern, written_bytes=0, verified_bytes=0)
            evidence["passes"].append(phase)
            t0 = time.perf_counter()
            for index, block in enumerate(blocks):
                guard.check()
                _fill_expected(torch, scratch, rank, index, pattern)
                block.copy_(scratch)
                phase["written_bytes"] += CHUNK_BYTES
            # Every chunk and every rank finishes writing before any verification.
            torch.cuda.synchronize()
            dist.barrier()
            phase["write_seconds"] = time.perf_counter() - t0
            guard.check(force=True)
            t0 = time.perf_counter()
            for index, block in enumerate(blocks):
                guard.check()
                _fill_expected(torch, scratch, rank, index, pattern)
                if not torch.equal(block, scratch):
                    raise RuntimeError(f"HBM write/read mismatch: {pattern}, chunk {index}, rank {rank}")
                phase["verified_bytes"] += CHUNK_BYTES
                evidence["total_verified_bytes"] += CHUNK_BYTES
            torch.cuda.synchronize()
            # Avoid a fast rank overwriting its next pattern while peers still read.
            dist.barrier()
            phase.update(verify_seconds=time.perf_counter() - t0, status="PASS")
            guard.check(force=True)
        evidence.update(verified_bytes_per_pass=allocated, memory_tested_gib=allocated / 2**30,
                        duration_s=time.perf_counter() - start,
                        peak_temperature_c=guard.peak)
        return evidence
    finally:
        if guard.peak is not None:
            evidence["peak_temperature_c"] = guard.peak
        blocks.clear()
        del scratch


def main(argv=None):
    args = parse_args(argv)
    evidence = {}
    try:
        torch, dist, rank, _world = gpu_test.setup(args)
        run_probe(torch, dist, rank, args, evidence)
        gpu_test.emit(args, dict(evidence, status="PASS"))
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        # torchrun terminates peer ranks; no final barrier on a failed rank.
        gpu_test.emit(args, dict(evidence, status="FAIL", error=str(exc), traceback=traceback.format_exc()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
