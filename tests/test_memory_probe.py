"""CPU contract checks; no fabricated GPU memory measurements."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_probe as memory


class MemoryBudgetTests(unittest.TestCase):
    def test_idle_b300_targets_85_percent_in_whole_chunks(self):
        total = 270 * 2**30
        allocated = memory.allocation_budget(total, total)
        self.assertEqual(allocated % memory.CHUNK_BYTES, 0)
        self.assertLessEqual(allocated, int(total * .85))
        self.assertLess(int(total * .85) - allocated, memory.CHUNK_BYTES)
        self.assertGreaterEqual(total - allocated - memory.CHUNK_BYTES, memory.RESERVE_BYTES)

    def test_low_free_capacity_keeps_headroom_and_separate_scratch(self):
        free = 6 * 2**30
        allocated = memory.allocation_budget(270 * 2**30, free, .90)
        self.assertEqual(allocated, free - memory.RESERVE_BYTES - memory.CHUNK_BYTES)
        self.assertGreaterEqual(free - allocated - memory.CHUNK_BYTES, 2 * 2**30)

    def test_insufficient_capacity_is_rejected_without_zero_coverage_pass(self):
        minimum = memory.RESERVE_BYTES + 2 * memory.CHUNK_BYTES
        with self.assertRaisesRegex(ValueError, "Insufficient free HBM"):
            memory.allocation_budget(270 * 2**30, minimum - 1)
        self.assertEqual(memory.allocation_budget(270 * 2**30, minimum), memory.CHUNK_BYTES)

    def test_invalid_counts_and_fractions_are_rejected(self):
        for total, free, fraction in ((0, 0, .85), (100, 101, .85), (-1, 0, .85),
                                      (True, 0, .85), (100, 0.0, .85),
                                      (100, 100, .49), (100, 100, .91),
                                      (100, 100, float("nan")), (100, 100, float("inf")),
                                      (100, 100, True)):
            with self.subTest(total=total, free=free, fraction=fraction), self.assertRaises(ValueError):
                memory.allocation_budget(total, free, fraction)


class PatternContractTests(unittest.TestCase):
    def test_seeds_are_unique_across_eight_gpu_allocations(self):
        # More chunks than a 288 GiB B300 test can allocate, across all eight ranks.
        seeds = [memory.pattern_seed(rank, chunk) for rank in range(8) for chunk in range(1200)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(0 <= seed < 2**32 for seed in seeds))

    def test_inverse_and_lcg_preserve_int32_contract_at_boundaries(self):
        for rank in (0, 7):
            for chunk in (0, 1000):
                for index in (0, 1, memory.CHUNK_BYTES // 4 - 1):
                    address = memory.pattern_word(index, rank, chunk, "address")
                    inverse = memory.pattern_word(index, rank, chunk, "inverted-address")
                    lcg = memory.pattern_word(index, rank, chunk, "lcg")
                    self.assertEqual(inverse, ~address)
                    self.assertTrue(-2**31 <= lcg < 2**31)
                    self.assertNotEqual(address, lcg)

    def test_different_addresses_and_allocations_have_different_patterns(self):
        for pattern in memory.PATTERNS:
            words = [memory.pattern_word(i, 0, 0, pattern) for i in range(100)]
            self.assertEqual(len(words), len(set(words)))
            self.assertNotEqual(memory.pattern_word(0, 0, 0, pattern), memory.pattern_word(0, 7, 0, pattern))
            self.assertNotEqual(memory.pattern_word(0, 0, 0, pattern), memory.pattern_word(0, 0, 1, pattern))

    def test_invalid_seed_domains_do_not_collide_via_silent_wrap(self):
        for rank, chunk in ((-1, 0), (256, 0), (True, 0), (0, -1), (0, 2**24), (0, 1.5)):
            with self.subTest(rank=rank, chunk=chunk), self.assertRaises(ValueError):
                memory.pattern_seed(rank, chunk)
        with self.assertRaises(ValueError):
            memory.pattern_word(0, 0, 0, "unknown")


class MemoryCliTests(unittest.TestCase):
    def test_defaults_match_torchrun_result_contract(self):
        args = memory.parse_args([])
        self.assertEqual((args.gpus, args.mode, args.model), (8, "memory", ""))
        self.assertEqual((args.memory_fraction, args.max_temp), (.85, 85))

    def test_unsafe_arguments_fail_before_gpu_setup(self):
        for argv in (["--gpus", "0"], ["--gpus", "257"], ["--max-temp", "120"],
                     ["--memory-fraction", "nan"], ["--memory-fraction", "inf"],
                     ["--memory-fraction", ".49"], ["--memory-fraction", ".91"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                memory.parse_args(argv)
            self.assertEqual(error.exception.code, 2)

    def test_temperature_guard_stops_before_load_at_limit(self):
        with patch.object(memory.gpu_test, "temperature", return_value=85):
            guard = memory.TemperatureGuard(0, 85)
            with self.assertRaisesRegex(RuntimeError, "stop threshold"):
                guard.check(force=True)
        self.assertEqual(guard.peak, 85)

    def test_unavailable_temperature_is_not_interpreted_as_cool(self):
        for value in (float("nan"), float("inf"), 0, -1, 120):
            with self.subTest(value=value), patch.object(memory.gpu_test, "temperature", return_value=value):
                with self.assertRaisesRegex(RuntimeError, "Invalid GPU temperature"):
                    memory.TemperatureGuard(0, 85).check(force=True)


class MemoryEvidenceTests(unittest.TestCase):
    def fixture(self):
        # Private CPU schema fixture, never exported as an observed GPU result.
        total = 270 * 2**30
        allocated = memory.allocation_budget(total, total)
        return dict(synthetic_only=True, status="PASS", test="memory", dtype="int32",
                    total_memory_bytes=total, free_memory_before_bytes=total,
                    requested_memory_fraction=.85, patterns=list(memory.PATTERNS),
                    chunk_bytes=memory.CHUNK_BYTES, scratch_bytes=memory.CHUNK_BYTES,
                    reserve_bytes=memory.RESERVE_BYTES, target_allocation_bytes=allocated,
                    allocated_bytes=allocated, verified_bytes_per_pass=allocated,
                    total_verified_bytes=allocated * 3, coverage_percent=allocated / total * 100,
                    memory_tested_gib=allocated / 2**30, free_memory_limited=False,
                    duration_s=3, peak_temperature_c=50,
                    passes=[dict(pattern=name, status="PASS", written_bytes=allocated,
                                 verified_bytes=allocated, write_seconds=1, verify_seconds=1)
                            for name in memory.PATTERNS])

    def test_complete_capacity_contract_is_accepted(self):
        self.assertIsNone(memory.validate_evidence(self.fixture()))

    def test_toy_pass_and_zero_capacity_are_rejected(self):
        for result in (dict(status="PASS", test="memory"),
                       dict(self.fixture(), allocated_bytes=0),
                       dict(self.fixture(), allocated_bytes=True)):
            with self.subTest(result=result), self.assertRaises(ValueError):
                memory.validate_evidence(result)

    def test_repeated_pass_bytes_are_never_capacity(self):
        original = self.fixture()
        for key, value in (("coverage_percent", original["coverage_percent"] * 3),
                           ("allocated_bytes", original["total_verified_bytes"]),
                           ("verified_bytes_per_pass", original["total_verified_bytes"]),
                           ("memory_tested_gib", original["memory_tested_gib"] * 3)):
            result = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory.validate_evidence(result)

    def test_partial_or_repeated_patterns_are_rejected(self):
        original = self.fixture()
        variants = []
        variants.append(dict(original, passes=original["passes"][:2]))
        changed = copy.deepcopy(original)
        changed["passes"][2]["pattern"] = "address"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["passes"][1]["verified_bytes"] -= memory.CHUNK_BYTES
        variants.append(changed)
        for result in variants:
            with self.subTest(result=result), self.assertRaises(ValueError):
                memory.validate_evidence(result)

    def test_nonfinite_and_headroom_inconsistent_records_are_rejected(self):
        original = self.fixture()
        for changes in (dict(coverage_percent=float("nan")), dict(peak_temperature_c=float("inf")),
                        dict(requested_memory_fraction=float("nan")),
                        dict(free_memory_before_bytes=memory.RESERVE_BYTES + memory.CHUNK_BYTES),
                        dict(chunk_bytes=1024), dict(reserve_bytes=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                memory.validate_evidence(dict(original, **changes))


if __name__ == "__main__":
    unittest.main()
