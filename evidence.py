"""Seal local run evidence, excluding symlinks and the archive itself."""
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def export_evidence(out):
    out = Path(out).resolve()
    excluded = {'evidence.zip', 'evidence.zip.tmp', 'evidence-manifest.json',
                'evidence-manifest.json.tmp', 'evidence-manifest.json.previous', 'job.json', 'job.json.tmp'}
    controls = [out / name for name in excluded if name not in ('job.json', 'job.json.tmp')]
    if any(path.is_symlink() or (path.exists() and not path.is_file()) for path in controls):
        raise RuntimeError('Evidence seal artifacts must be regular local files')
    def inventory():
        return sorted(p for p in out.rglob('*') if p.is_file() and not p.is_symlink()
                      and p.name not in excluded and p.resolve().is_relative_to(out)
                      and not any(parent.is_symlink() for parent in p.parents if parent != out))
    def snapshot(value):
        # Windows stat/fstat disagree about ctime (creation vs change time).
        # Content hash and nanosecond mtime remain mandatory on both systems.
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_ctime_ns if os.name != 'nt' else 0)
    files = inventory()
    original = {p: snapshot(p.stat()) for p in files}
    manifest = dict(schema=1, purpose='Local run evidence; hashes are not manufacturer certification', files={})
    target = out / 'evidence-manifest.json'
    manifest_temporary = out / 'evidence-manifest.json.tmp'
    manifest_previous = out / 'evidence-manifest.json.previous'
    temporary = out / 'evidence.zip.tmp'
    manifest_published = False
    rollback_failed = False
    had_manifest = target.is_file()
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in files:
                name = path.relative_to(out).as_posix()
                checksum, size = hashlib.sha256(), 0
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
                with os.fdopen(os.open(path, flags), 'rb') as source:
                    before = os.fstat(source.fileno())
                    if not stat.S_ISREG(before.st_mode) or snapshot(before) != original[path]:
                        raise RuntimeError('Evidence changed before sealing: ' + name)
                    with archive.open(name, 'w', force_zip64=True) as destination:
                        for block in iter(lambda: source.read(8 * 2**20), b''):
                            destination.write(block)
                            checksum.update(block)
                            size += len(block)
                    if snapshot(os.fstat(source.fileno())) != original[path] or size != before.st_size:
                        raise RuntimeError('Evidence changed while sealing: ' + name)
                manifest['files'][name] = dict(size=size, sha256=checksum.hexdigest())
            # Independent producers must have finished before sealing. Verify
            # both the inventory and every source after writing all ZIP entries.
            if inventory() != files:
                raise RuntimeError('Evidence file set changed while sealing')
            for path in files:
                name = path.relative_to(out).as_posix()
                if digest(path) != manifest['files'][name]['sha256'] or snapshot(path.stat()) != original[path]:
                    raise RuntimeError('Evidence changed before publication: ' + name)
            if inventory() != files or any(snapshot(p.stat()) != original[p] for p in files):
                raise RuntimeError('Evidence changed during final verification')
            raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
            archive.writestr('evidence-manifest.json', raw)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        with manifest_temporary.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # The ZIP is self-contained and its embedded manifest is authoritative.
        # Retain the old external manifest until both replacement files exist;
        # restore it if ZIP publication fails (for example, a locked archive).
        if had_manifest:
            manifest_previous.write_bytes(target.read_bytes())
        manifest_temporary.replace(target)
        manifest_published = True
        temporary.replace(out / 'evidence.zip')
        return manifest
    except Exception as failure:
        if manifest_published:
            try:
                if had_manifest:
                    manifest_previous.replace(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                rollback_failed = True
                raise RuntimeError('Evidence publication failed; retain evidence-manifest.json.previous for recovery') from failure
        raise
    finally:
        for path in [temporary, manifest_temporary] + ([] if rollback_failed else [manifest_previous]):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # A completed ZIP remains self-contained; cleanup of a stale
                # temporary file must not roll back a successfully sealed ZIP.
                pass
