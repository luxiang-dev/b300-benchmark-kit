"""Execute the HBM probe with small NumPy tensors, never GPU measurements."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import memory_probe


class Tensor:
    def __init__(self, data, corrupt_copy=False):
        self.data, self.device, self.corrupt_copy = data, 'cuda', corrupt_copy

    def numel(self):
        return self.data.size

    def copy_(self, source):
        self.data[:] = source.data
        if self.corrupt_copy:
            self.data[1] ^= np.int32(1)

    def mul_(self, value):
        np.multiply(self.data, value, out=self.data)
        return self

    def add_(self, value):
        np.add(self.data, value, out=self.data)
        return self

    def bitwise_not_(self):
        np.bitwise_not(self.data, out=self.data)
        return self

    def __getitem__(self, positions):
        return Tensor(self.data[positions.data])

    def cpu(self):
        return self

    def tolist(self):
        return self.data.tolist()


class TorchCpuFake:
    int32, int64 = np.int32, np.int64

    def __init__(self, dishonest_equal=False, corrupt_copy=False, alias_blocks=False):
        self.dishonest_equal, self.corrupt_copy = dishonest_equal, corrupt_copy
        self.alias_blocks, self.empty_calls, self.backing = alias_blocks, 0, None
        self.cuda = SimpleNamespace(mem_get_info=lambda: (4096, 4096), synchronize=lambda: None)

    def empty(self, count, dtype, **_kw):
        self.empty_calls += 1
        if self.alias_blocks and self.empty_calls > 1:
            if self.backing is None:
                self.backing = np.empty(count, dtype=dtype)
            values = self.backing
        else:
            values = np.empty(count, dtype=dtype)
        return Tensor(values, corrupt_copy=self.corrupt_copy)

    def arange(self, count, dtype, out, **_kw):
        out.data[:] = np.arange(count, dtype=dtype)
        return out

    def tensor(self, values, dtype, **_kw):
        return Tensor(np.asarray(values, dtype=dtype))

    def equal(self, left, right):
        return True if self.dishonest_equal else bool(np.array_equal(left.data, right.data))


class MemoryExecutionRegressions(unittest.TestCase):
    def execute(self, corrupt_producer=False, dishonest_equal=False, corrupt_copy=False, alias_blocks=False):
        fake = TorchCpuFake(dishonest_equal, corrupt_copy, alias_blocks)
        original = memory_probe._fill_expected
        def producer(torch, scratch, rank, chunk, pattern):
            original(torch, scratch, rank, chunk, pattern)
            if corrupt_producer:
                scratch.add_(1)
        evidence = {}
        args = SimpleNamespace(max_temp=85, memory_fraction=.85)
        with patch.object(memory_probe, 'CHUNK_BYTES', 64), patch.object(memory_probe, 'RESERVE_BYTES', 128), \
             patch.object(memory_probe, '_fill_expected', side_effect=producer), \
             patch.object(memory_probe.gpu_test, 'temperature', return_value=50):
            memory_probe.run_probe(fake, SimpleNamespace(barrier=lambda: None), 0, args, evidence)
        return evidence

    def test_all_three_patterns_write_verify_and_cpu_spot_check_every_chunk(self):
        evidence = self.execute()
        count = evidence['allocated_bytes'] // evidence['chunk_bytes']
        self.assertEqual(evidence['patterns'], ['address', 'inverted-address', 'lcg'])
        self.assertEqual(evidence['cpu_reference_positions_per_chunk'], 3)
        self.assertEqual(evidence['cpu_reference_checks_per_pass'], count * 3)
        self.assertEqual([phase['cpu_reference_checks'] for phase in evidence['passes']], [count * 3] * 3)
        self.assertEqual([phase['verified_bytes'] for phase in evidence['passes']], [count * 64] * 3)

    def test_independent_cpu_reference_rejects_consistently_faulty_gpu_producer(self):
        with self.assertRaisesRegex(RuntimeError, 'CPU reference mismatch'):
            self.execute(corrupt_producer=True, dishonest_equal=True)

    def test_full_comparison_catches_corruption_outside_the_three_spot_positions(self):
        with self.assertRaisesRegex(RuntimeError, 'HBM write/read mismatch'):
            self.execute(corrupt_copy=True)

    def test_simultaneous_write_before_read_exposes_aliasing_across_chunks(self):
        with self.assertRaisesRegex(RuntimeError, 'HBM write/read mismatch'):
            self.execute(alias_blocks=True)


if __name__ == '__main__':
    unittest.main()
