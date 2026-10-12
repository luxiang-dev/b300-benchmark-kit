"""One advisory lock shared by dashboard and CLI; an idle maintenance window is still required."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def lock_path():
    return Path(tempfile.gettempdir()) / 'b300-kit-gpu.lock'


def acquire():
    import fcntl
    path = lock_path()
    inherited = os.environ.get('B300_GPU_LOCK_FD')
    if inherited is not None:
        try:
            descriptor = int(inherited)
            metadata, named = os.fstat(descriptor), path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or not stat.S_ISREG(named.st_mode) or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino):
                raise ValueError('Inherited descriptor is not the GPU lock')
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except (OSError, ValueError) as error:
            raise RuntimeError('无法核验继承的整机 GPU 作业锁: ' + str(error)) from error
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    except OSError as error:
        raise RuntimeError('无法取得整机 GPU 作业锁；请在维护窗口使用同一验货运维账号: ' + str(error)) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or os.fstat(descriptor).st_nlink != 1:
            raise RuntimeError('Invalid GPU lock file')
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except (OSError, RuntimeError) as error:
        os.close(descriptor)
        raise RuntimeError('另一个控制器或 CLI 已在使用 GPU，无法取得整机作业锁。') from error


def main():
    if sys.platform != 'linux':
        raise RuntimeError('GPU jobs require Linux')
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise ValueError('Supply a command after --')
    descriptor = acquire()
    try:
        return subprocess.run(command, env={**os.environ, 'B300_GPU_LOCK_FD': str(descriptor)},
                              pass_fds=(descriptor,), check=False).returncode
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    sys.exit(main())
