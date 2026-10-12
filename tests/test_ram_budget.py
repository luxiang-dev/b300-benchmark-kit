"""Host RAM must not hide a smaller systemd/container memory allowance."""
from pathlib import Path
import unittest
from unittest.mock import patch
import server_probe


class RamBudgetTests(unittest.TestCase):
    def test_nested_cgroup_limit_is_respected(self):
        values = {'/proc/meminfo': 'MemAvailable: 8388608 kB\n',
                  '/proc/self/cgroup': '0::/user.slice/session.scope\n',
                  '/sys/fs/cgroup/user.slice/memory.max': '536870912',
                  '/sys/fs/cgroup/user.slice/memory.current': '67108864'}
        def read(path, *args, **kwargs):
            if path.as_posix() in values:
                return values[path.as_posix()]
            raise FileNotFoundError(str(path))
        with patch.object(Path, 'read_text', read):
            self.assertEqual(server_probe.available_ram(), 448 * 2**20)
            self.assertEqual(server_probe.ram_check(256 * 2**20)['status'], 'SKIP')

    def test_unlimited_cgroup_preserves_actual_available_ram(self):
        values = {'/proc/meminfo': 'MemAvailable: 1048576 kB\n', '/proc/self/cgroup': '0::/\n',
                  '/sys/fs/cgroup/memory.max': 'max', '/sys/fs/cgroup/memory.current': '1'}
        def read(path, *args, **kwargs):
            if path.as_posix() in values:
                return values[path.as_posix()]
            raise FileNotFoundError(str(path))
        with patch.object(Path, 'read_text', read):
            self.assertEqual(server_probe.available_ram(), 2**30)
