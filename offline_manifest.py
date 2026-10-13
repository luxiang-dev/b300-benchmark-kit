"""Seal/verify any prepared resource directory. Hashes prove transport integrity only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

from download_models import write_json


def inventory(root, manifest):
    root = Path(root).resolve()
    manifest = Path(manifest).absolute()
    if not root.is_dir():
        raise ValueError('Resource root must be an existing directory')
    entries = {}
    def walk_error(error):
        raise error
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in sorted(dirs + files):
            path = Path(folder) / name
            if path.absolute() == manifest:
                continue
            info = path.lstat()
            key = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                # Record the literal link, never follow it into an undeclared
                # tree. Target validity is checked later in the target runtime.
                entries[key] = dict(kind='link', target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                entries[key] = dict(kind='directory', mode=stat.S_IMODE(info.st_mode))
            elif stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                after = path.stat()
                if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('File changed during hashing: ' + key)
                entries[key] = dict(kind='file', bytes=info.st_size, sha256=digest.hexdigest(), mode=stat.S_IMODE(info.st_mode))
            else:
                raise ValueError('Unsupported special resource: ' + key)
    return entries


def seal(root, manifest):
    result = dict(schema=1, platform=sys.platform, entries=inventory(root, manifest),
                  boundary='Integrity only; no source authenticity, completeness or runtime certification. Links are not followed.')
    write_json(Path(manifest), result)
    return result


def verify(root, manifest):
    record = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if not isinstance(record, dict) or record.get('schema') != 1 or not isinstance(record.get('entries'), dict):
        raise ValueError('Invalid external resource manifest')
    for key in record['entries']:
        path = PurePosixPath(key)
        if not key or path.is_absolute() or '..' in path.parts or '\\' in key or ':' in key:
            raise ValueError('Unsafe manifest path')
    actual = inventory(root, manifest)
    if set(actual) != set(record['entries']):
        raise ValueError('Resource inventory mismatch (added or missing files/directories/links)')
    for key, observed in actual.items():
        expected = record['entries'][key]
        if not isinstance(expected, dict):
            raise ValueError('Invalid manifest entry: ' + key)
        expected = dict(expected)
        observed = dict(observed)
        if sys.platform == 'win32' or record.get('platform') == 'win32':
            expected.pop('mode', None)
            observed.pop('mode', None)
        if observed != expected:
            raise ValueError('Resource mismatch: ' + key)
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    try:
        count = len(seal(args.root, args.manifest)['entries']) if args.action == 'seal' else verify(args.root, args.manifest)
        print(f'{args.action.upper()}: {count} resource entries; integrity only, runtime NOT_RUN')
        return 0
    except (OSError, ValueError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
