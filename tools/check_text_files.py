"""Check Linux entry points for CRLF/BOM regressions without loading GPU packages."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = list(root.glob("*.sh")) + list(root.glob("*.py"))
for path in paths:
    raw = path.read_bytes()
    if b"\r\n" in raw or raw.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"Use UTF-8 without BOM and LF line endings: {path.name}")
    raw.decode("utf-8")
print(f"Verified UTF-8/LF for {len(paths)} Linux entry points")
