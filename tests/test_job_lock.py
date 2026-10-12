"""Real Linux advisory-lock inheritance; no GPU work is executed."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import job_lock
import test_kit


@unittest.skipUnless(sys.platform == 'linux', 'Requires actual POSIX flock and inherited descriptors')
class JobLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_kit.workspace_temp()
        self.root = Path(self.temp.name).resolve()
        self.path_patch = patch.object(job_lock, 'lock_path', return_value=self.root / 'b300-kit-gpu.lock')
        self.path_patch.start()
        self.env_patch = patch.dict(os.environ)
        self.env_patch.start()
        os.environ.pop('B300_GPU_LOCK_FD', None)
        self.descriptors = []

    def tearDown(self):
        for descriptor in self.descriptors:
            os.close(descriptor)
        self.env_patch.stop()
        self.path_patch.stop()
        self.temp.cleanup()

    def acquire(self):
        descriptor = job_lock.acquire()
        self.descriptors.append(descriptor)
        return descriptor

    def test_competing_cli_cannot_acquire_an_active_lock(self):
        self.acquire()
        with self.assertRaises(RuntimeError):
            job_lock.acquire()

    def test_inherited_child_reuses_lock_without_releasing_parent(self):
        descriptor = self.acquire()
        env = {**os.environ, 'TMPDIR': str(self.root), 'B300_GPU_LOCK_FD': str(descriptor)}
        result = subprocess.run([sys.executable, '-c', 'import job_lock,os; fd=job_lock.acquire(); os.close(fd)'],
                                cwd=Path(__file__).resolve().parents[1], env=env, pass_fds=(descriptor,), timeout=5)
        self.assertEqual(result.returncode, 0)
        with self.assertRaises(RuntimeError):
            job_lock.acquire()

    def test_foreign_inherited_descriptor_is_rejected(self):
        self.acquire()
        other = os.open(self.root / 'other-file', os.O_CREAT | os.O_RDWR, 0o600)
        self.descriptors.append(other)
        os.environ['B300_GPU_LOCK_FD'] = str(other)
        with self.assertRaises(RuntimeError):
            job_lock.acquire()

    def test_symlink_lock_is_rejected_without_touching_target(self):
        target = self.root / 'private'
        target.write_text('retain', encoding='utf-8')
        (self.root / 'b300-kit-gpu.lock').symlink_to(target)
        with self.assertRaises(RuntimeError):
            job_lock.acquire()
        self.assertEqual(target.read_text(), 'retain')
