"""Seal local run evidence, excluding symlinks and the archive itself."""
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def export_evidence(out):
    out = Path(out).resolve()
    excluded = {'evidence.zip', 'evidence.zip.tmp', 'evidence-manifest.json', 'job.json', 'job.json.tmp'}
    files = [p for p in out.rglob('*') if p.is_file() and not p.is_symlink()
             and p.name not in excluded and p.resolve().is_relative_to(out)]
    manifest = dict(schema=1, purpose='Local run evidence; hashes are not manufacturer certification',
                    files={p.relative_to(out).as_posix(): dict(size=p.stat().st_size, sha256=digest(p)) for p in files})
    target = out / 'evidence-manifest.json'
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary = out / 'evidence.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files + [target]:
            archive.write(path, path.relative_to(out).as_posix())
    temporary.replace(out / 'evidence.zip')
    return manifest
