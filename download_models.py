#!/usr/bin/env python3
"""Public safetensors snapshots, pinned commits, resumable downloads; stdlib only."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_path(base, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or not parts.parts or any(p in ("..", ".") or ":" in p or "\\" in p for p in parts.parts):
        raise ValueError(f"Unsafe model path: {name}")
    target = base.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path leaves model directory: {name}")
    return target


def file_hash(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_file(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    return file_hash(path, git=item["hash_type"] == "git-sha1") == item["hash"]


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "b300-usb-kit/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def resolve_models(root, refresh=False):
    lock_path = root / "models.lock.json"
    if lock_path.exists() and not refresh:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    catalog = json.loads((root / "models.json").read_text(encoding="utf-8"))
    lock = {"schema": 1, "models": []}
    for model in catalog:
        metadata = fetch_json(f"https://huggingface.co/api/models/{model['repo']}?blobs=true")
        files = []
        for sibling in metadata["siblings"]:
            name = sibling["rfilename"]
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in model["include"]):
                continue
            safe_path(root / "models" / model["name"], name)
            lfs = sibling.get("lfs")
            files.append({"path": name, "size": sibling["size"],
                          "hash_type": "sha256" if lfs else "git-sha1",
                          "hash": lfs["sha256"] if lfs else sibling["blobId"]})
        if not any(x["path"].endswith(".safetensors") for x in files):
            raise RuntimeError(f"No safe tensor weights found for {model['repo']}")
        lock["models"].append({**model, "revision": metadata["sha"], "files": files})
    write_json(lock_path, lock)
    return lock


def download_one(root, model, item):
    target = safe_path(root / "models" / model["name"], item["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, item):
        print(f"OK {model['name']}/{item['path']}", flush=True)
        return
    part = target.with_name(target.name + ".part")
    url = (f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/"
           + urllib.parse.quote(item["path"], safe="/"))
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > item["size"]:
                part.unlink()
                offset = 0
            if offset < item["size"]:
                headers = {"User-Agent": "b300-usb-kit/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Invalid Content-Range")
                    with part.open("ab" if append else "wb") as stream:
                        for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            stream.write(chunk)
            if not valid_file(part, item):
                if part.exists() and part.stat().st_size >= item["size"]:
                    part.unlink()
                raise RuntimeError(f"Size/hash check failed: {item['path']}")
            part.replace(target)
            print(f"DOWNLOADED {model['name']}/{item['path']}", flush=True)
            return
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"RETRY {model['name']}/{item['path']}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 10))


def verify(root, models):
    failures = []
    for model in models:
        for item in model["files"]:
            path = safe_path(root / "models" / model["name"], item["path"])
            if not valid_file(path, item):
                failures.append(f"{model['name']}/{item['path']}")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--plan", action="store_true", help="Only pin revisions and print exact sizes")
    p.add_argument("--verify", action="store_true", help="Verify locally; never use the network")
    p.add_argument("--refresh", action="store_true", help="Explicitly repin current upstream revisions")
    p.add_argument("--only", nargs="+", help="Names from models.json")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    if args.workers < 1 or args.workers > 8:
        p.error("--workers must be 1..8")
    root = args.root.resolve()
    if args.verify and not (root / "models.lock.json").exists():
        raise RuntimeError("models.lock.json is missing; run --plan online first")
    lock = (json.loads((root / "models.lock.json").read_text(encoding="utf-8"))
            if args.verify else resolve_models(root, args.refresh))
    models = lock["models"]
    if args.only:
        unknown = set(args.only) - {x["name"] for x in models}
        if unknown:
            p.error(f"Unknown models: {sorted(unknown)}")
        models = [x for x in models if x["name"] in args.only]
    for model in models:
        size = sum(x["size"] for x in model["files"])
        print(f"{model['name']}: {size / 2**30:.2f} GiB; revision={model['revision']}", flush=True)
    if args.plan:
        print("Weights not downloaded by --plan.", flush=True)
        return 0
    if not args.verify:
        needed = 0
        for model in models:
            for item in model["files"]:
                target = safe_path(root / "models" / model["name"], item["path"])
                if valid_file(target, item):
                    continue
                part = target.with_name(target.name + ".part")
                offset = part.stat().st_size if part.is_file() else 0
                needed += max(0, item["size"] - offset) if offset <= item["size"] else item["size"]
        if shutil.disk_usage(root).free < needed + 2 * 2**30:
            raise RuntimeError(f"Need {needed / 2**30:.1f} GiB + 2 GiB reserve")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_one, root, model, item) for model in models for item in model["files"]]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    failures = verify(root, models)
    if failures:
        print("MISSING OR CORRUPT:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("All selected model files passed size and source hash verification.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
