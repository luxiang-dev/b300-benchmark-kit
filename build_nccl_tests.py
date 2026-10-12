#!/usr/bin/env python3
"""Build the vendored NVIDIA tests using the image's existing NCCL; never upgrade it."""
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

REVISION = "da0b547b1b9c6e3b1d4c15578087874522ae3761"
SOURCE_SHA256 = "d589c597110da30b52d462269ae0d0381c46811a7b266db09fe865d9aaff31c1"


def header_version(path):
    text = Path(path).read_text()
    values = [int(re.search(r"^#define\s+NCCL_" + name + r"\s+(\d+)", text, re.M)[1])
              for name in ("MAJOR", "MINOR", "PATCH")]
    return tuple(values)


def main():
    import torch
    version = tuple(torch.cuda.nccl.version())
    expected = version[0] * 10000 + version[1] * 100 + version[2]
    # Importing PyTorch loads its dependencies, including the NCCL used by this kit.
    maps = Path("/proc/self/maps").read_text()
    libraries = sorted({line.split()[-1] for line in maps.splitlines()
                        if "/libnccl.so" in line and line.split()[-1].startswith("/")})
    loaded = set(libraries)
    # Some distributions statically link PyTorch's NCCL. Use an existing shared
    # library of exactly the same version in that case, without installing one.
    for pattern in ("/usr/lib/*/libnccl.so*", "/usr/local/cuda/lib64/libnccl.so*",
                    "/usr/local/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so*",
                    "/usr/local/lib/python*/dist-packages/nvidia/nccl/lib/libnccl.so*"):
        libraries.extend(glob.glob(pattern))
    library = None
    for candidate in libraries:
        actual = ctypes.c_int()
        if ctypes.CDLL(candidate).ncclGetVersion(ctypes.byref(actual)) == 0 and actual.value == expected:
            library = Path(candidate).resolve()
            break
    if library is None:
        raise RuntimeError("Missing shared NCCL library matching PyTorch; provide it in the base image")
    candidates = [library.parent.parent / "include/nccl.h", Path("/usr/include/nccl.h"),
                  Path("/usr/local/cuda/include/nccl.h")]
    for base in (Path("/usr/local/lib"), Path("/usr/lib"), Path("/opt/conda/lib")):
        for pattern in ("python*/site-packages/nvidia/nccl/include/nccl.h",
                        "python*/dist-packages/nvidia/nccl/include/nccl.h"):
            candidates.extend(base.glob(pattern))
    header = next((p for p in candidates if p.is_file() and header_version(p) == version), None)
    if header is None:
        raise RuntimeError("Missing NCCL headers matching PyTorch; provide matching headers in the base image")
    stage = Path("/opt/nccl-runtime")
    stage.mkdir(parents=True, exist_ok=True)
    # Include NCCL device API headers together with nccl.h.
    (stage / "include").symlink_to(header.parent.resolve(), target_is_directory=True)
    (stage / "lib").mkdir(exist_ok=True)
    (stage / "lib/libnccl.so").symlink_to(library)
    archive = Path("/tmp/nccl-tests-v2.17.6.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Vendored nccl-tests source checksum mismatch")
    with tarfile.open(archive) as tar:
        tar.extractall("/opt", filter="data")
    source = Path("/opt") / ("nccl-tests-" + REVISION)
    source.rename("/opt/nccl-tests")
    source = Path("/opt/nccl-tests")
    # Keep the default toolkit linker flags and add the exact selected library path.
    with (source / "src/common.mk").open("a") as stream:
        stream.write(f"\nNVLDFLAGS += -Xlinker -rpath -Xlinker {library.parent}\n")
    supported = subprocess.run(["nvcc", "--list-gpu-code"], check=True, capture_output=True, text=True).stdout
    codes = sorted(set(re.findall(r"sm_([0-9]+)\b", supported)), key=int)
    if not codes:
        raise RuntimeError("Cannot enumerate CUDA compiler architectures")
    gencode = " ".join(f"-gencode=arch=compute_{code},code=sm_{code}" for code in codes)
    gencode += f" -gencode=arch=compute_{codes[-1]},code=compute_{codes[-1]}"
    subprocess.run(["make", "-j", str(min(os.cpu_count() or 2, 8)), "MPI=0", f"NCCL_HOME={stage}",
                    "NVCC_GENCODE=" + gencode],
                   cwd=source, check=True)
    binary = source / "build/all_reduce_perf"
    dependencies = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True).stdout
    print(dependencies)
    if "not found" in dependencies:
        raise RuntimeError("nccl-tests has unresolved runtime libraries")
    manifest = dict(compiled_architectures=codes, nccl_tests_revision=REVISION, source_sha256=SOURCE_SHA256, cuda=torch.version.cuda,
                    torch=torch.__version__, nccl_library=str(library), nccl_header=str(header),
                    nccl_selection="pytorch_loaded" if str(library) in loaded else "existing_matching_version",
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (source / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
