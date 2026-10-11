"""Allow only explicitly selected app-package upgrades in NVIDIA's image."""
from pathlib import Path
import re
import sys


def package_name(line):
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else None


def filtered_constraints(constraints, requirements):
    selected = {package_name(line) for line in requirements.splitlines() if package_name(line)}
    protected = {"torch", "torchvision", "torchaudio", "numpy", "triton", "numba", "nvidia-nccl-cu13", "nvidia-cublas-cu13"}
    if selected & protected or any(name.startswith("nvidia-") for name in selected):
        raise ValueError("App requirements must not replace the GPU runtime packages")
    return "\n".join(line for line in constraints.splitlines() if package_name(line) not in selected) + "\n"


if __name__ == "__main__":
    path = Path(sys.argv[1])
    requirements = Path(sys.argv[2]).read_text(encoding="utf-8")
    path.write_text(filtered_constraints(path.read_text(encoding="utf-8"), requirements), encoding="utf-8")
