#!/usr/bin/env python3
"""NVIDIA GPU identity observations, cross-checks and light CUDA correctness probes.

This is evidence of consistency, not vendor authentication. NVIDIA's driver,
CUDA and Linux PCI views share a trust boundary; none proves a serial number is
genuine in a hostile hypervisor/firmware environment. Host collection uses only
the standard library. The CUDA subcommand requires the kit's PyTorch image.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET
import hardware_profiles as hp


SOURCES = {
    "compute_capability": "https://developer.nvidia.com/cuda/gpus",
    "known_pci_variant": "https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/supportedchips.html",
    "memory_specification": "https://resources.nvidia.com/en-us-gpu-resources/blackwell-ultra-datasheet?xs=443990",
    "dgx_specification": "https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html",
    "driver_observations": "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
}
EXPECTED_CAPABILITY = [10, 3]
KNOWN_PCI_VARIANT = {"vendor": "10de", "device": "3182", "subsystem_vendor": "10de", "subsystem_device": "20e6"}
BOUNDARY = (
    "CONSISTENT only means these software observations agree with the selected or recognized GPU profile. "
    "They do not authenticate hardware against forged firmware, a hostile driver or hypervisor. "
    "Confirm the physical board label/serial, OEM invoice and BMC inventory with the vendor; "
    "functional, HBM capacity, ECC and performance tests are separate evidence."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_uuid(value):
    if isinstance(value, bytes) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=value))
    value = str(value or "").strip()
    if value.startswith("GPU-"):
        value = value[4:]
    try:
        return "GPU-" + str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return None


def normalize_bus(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", str(value or "").strip())
    if not match:
        return None
    domain, bus, device, function = (int(item, 16) for item in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def memory_bytes(value):
    """Preserve raw values elsewhere; convert only explicit units, never bare GB."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(KiB|MiB|GiB|KB|MB|GB|B)\s*", str(value or ""))
    if not match:
        return None
    scales = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
              "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    return int(float(match[1]) * scales[match[2]])


def text_at(node, path):
    return node.findtext(path, default="").strip()


def parse_smi_xml(raw):
    root = ET.fromstring(raw)
    if root.tag != "nvidia_smi_log":
        raise ValueError("Unexpected nvidia-smi XML root")
    gpus = []
    for index, node in enumerate(root.findall("gpu")):
        fields = {
            "index": index, "name": text_at(node, "product_name"),
            "brand": text_at(node, "product_brand"), "architecture": text_at(node, "product_architecture"),
            "uuid": text_at(node, "uuid"), "serial": text_at(node, "serial"),
            "vbios_version": text_at(node, "vbios_version"),
            "board_part_number": text_at(node, "board_part_number"),
            "gpu_part_number": text_at(node, "gpu_part_number"),
            "pci_bus_id": text_at(node, "pci/pci_bus_id") or node.get("id", ""),
            "pci_device_id": text_at(node, "pci/pci_device_id"),
            "pci_subsystem_id": text_at(node, "pci/pci_sub_system_id"),
            "memory_total_raw": text_at(node, "fb_memory_usage/total"),
            "memory_reserved_raw": text_at(node, "fb_memory_usage/reserved"),
            "memory_free_raw": text_at(node, "fb_memory_usage/free"),
            "mig_mode": text_at(node, "mig_mode/current_mig"),
            "virtualization_mode": text_at(node, "gpu_virtualization_mode/virtualization_mode"),
            "vgpu_host_mode": text_at(node, "gpu_virtualization_mode/host_vgpu_mode"),
            "ecc_current": text_at(node, "ecc_mode/current_ecc"),
            "compute_mode": text_at(node, "compute_mode"),
            "gsp_firmware_version": text_at(node, "gsp_firmware_version"),
            "inforom_image_version": text_at(node, "inforom_version/img_version"),
            "power_default_limit_raw": text_at(node, "gpu_power_readings/default_power_limit") or text_at(node, "power_readings/default_power_limit"),
        }
        fields["memory_total_bytes"] = memory_bytes(fields["memory_total_raw"])
        fields["memory_reserved_bytes"] = memory_bytes(fields["memory_reserved_raw"])
        gpus.append(fields)
    return {"driver_version": text_at(root, "driver_version"), "cuda_version": text_at(root, "cuda_version"), "gpus": gpus}


def parse_lspci(raw):
    result = {}
    current = None
    for line in raw.splitlines():
        match = re.match(r"^([0-9a-fA-F:.]+)\s+.*\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", line)
        if match:
            current = normalize_bus(match[1])
            if current:
                result[current] = {"vendor": match[2].lower(), "device": match[3].lower()}
        elif line and not line[0].isspace():
            current = None
        elif current:
            subsystem = re.search(r"Subsystem:.*\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", line)
            driver = re.search(r"Kernel driver in use:\s*(\S+)", line)
            if subsystem:
                result[current].update(subsystem_vendor=subsystem[1].lower(), subsystem_device=subsystem[2].lower())
            if driver:
                result[current]["driver"] = driver[1]
    return result


def read_sysfs(bus, root=Path("/sys/bus/pci/devices")):
    bus = normalize_bus(bus)
    result = {"pci_bus_id": bus, "values": {}, "errors": {}}
    if not bus:
        result["errors"]["pci_bus_id"] = "Invalid PCI address"
        return result
    directory = Path(root) / bus
    for field in ("vendor", "device", "subsystem_vendor", "subsystem_device", "class", "revision",
                  "current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
        try:
            result["values"][field] = (directory / field).read_text(encoding="ascii").strip()
        except OSError as error:
            result["errors"][field] = f"{type(error).__name__}: {error}"
    for field in ("driver", "iommu_group", "physfn"):
        path = directory / field
        if path.is_symlink():
            result["values"][field] = path.resolve().name
    return result


def capture_command(out, label, argv, timeout=30):
    """Read-only commands; output is retained even when unsupported or failing."""
    directory = Path(out) / "identity-evidence"
    directory.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        stdout, stderr = completed.stdout, completed.stderr
        status, error = completed.returncode, None
    except subprocess.TimeoutExpired as failure:
        stdout, stderr = failure.stdout or b"", failure.stderr or b""
        status, error = None, "Command timed out"
    except OSError as failure:
        stdout, stderr = b"", b""
        status, error = None, f"{type(failure).__name__}: {failure}"
    files = {}
    for kind, raw in (("stdout", stdout), ("stderr", stderr)):
        path = directory / f"{label}.{kind}.txt"
        path.write_bytes(raw)
        files[kind] = {"path": path.relative_to(out).as_posix(), "sha256": sha256(path)}
    return {"argv": argv, "time_utc": started, "returncode": status, "error": error, "files": files}


def collect_host(out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    commands = {}
    for label, argv in (
        ("nvidia-smi-xml", ["nvidia-smi", "-q", "-x"]),
        ("nvidia-smi-list", ["nvidia-smi", "-L"]),
        ("nvidia-smi-topology", ["nvidia-smi", "topo", "-m"]),
        ("nvidia-smi-nvlink", ["nvidia-smi", "nvlink", "--status"]),
        ("lspci", ["lspci", "-D", "-nn", "-k"]),
        ("virtual-machine", ["systemd-detect-virt", "--vm"]),
        ("container", ["systemd-detect-virt", "--container"]),
    ):
        commands[label] = capture_command(out, label, argv)
    observations, parse_error = {"gpus": []}, None
    try:
        if commands["nvidia-smi-xml"]["returncode"] == 0:
            observations = parse_smi_xml((out / commands["nvidia-smi-xml"]["files"]["stdout"]["path"]).read_bytes())
    except (ET.ParseError, ValueError) as error:
        parse_error = f"{type(error).__name__}: {error}"
    pci = [read_sysfs(gpu["pci_bus_id"]) for gpu in observations["gpus"]]
    evidence = {}
    for label, source in (("os-release", Path("/etc/os-release")), ("nvidia-driver-version", Path("/proc/driver/nvidia/version"))):
        path = out / "identity-evidence" / f"{label}.txt"
        try:
            path.write_bytes(source.read_bytes())
            evidence[label] = {"path": path.relative_to(out).as_posix(), "sha256": sha256(path)}
        except OSError as error:
            evidence[label] = {"error": f"{type(error).__name__}: {error}"}
    pci_path = out / "identity-evidence/pci-sysfs.json"
    write_json(pci_path, pci)
    evidence["pci-sysfs"] = {"path": pci_path.relative_to(out).as_posix(), "sha256": sha256(pci_path)}
    result = {"schema_version": 1, "test": "identity-host", "time_utc": utc_now(),
              "collection_id": str(uuid.uuid4()), "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "hostname": platform.node()},
              "commands": commands, "evidence": evidence, "observations": observations, "parse_error": parse_error,
              "privacy": "Evidence contains hardware serials, UUIDs and host identity. Review before sharing.",
              "boundary": BOUNDARY, "sources": SOURCES}
    write_json(out / "identity-host.json", result)
    return result


def cuda_driver_identity():
    """Query the current CUDA context, avoiding assumptions about visible order."""
    result = {}
    try:
        library = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        get_device = library.cuCtxGetDevice
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_device.restype = ctypes.c_int
        if get_device(ctypes.byref(device)) != 0:
            raise RuntimeError("cuCtxGetDevice failed; no active CUDA context")
        value = (ctypes.c_ubyte * 16)()
        get_uuid = getattr(library, "cuDeviceGetUuid_v2", library.cuDeviceGetUuid)
        get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_uuid.restype = ctypes.c_int
        if get_uuid(ctypes.byref(value), device.value) != 0:
            raise RuntimeError("cuDeviceGetUuid failed")
        result["uuid"] = normalize_uuid(bytes(value))
        bus = ctypes.create_string_buffer(32)
        get_bus = library.cuDeviceGetPCIBusId
        get_bus.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        get_bus.restype = ctypes.c_int
        if get_bus(bus, len(bus), device.value) != 0:
            raise RuntimeError("cuDeviceGetPCIBusId failed")
        result["pci_bus_id"] = bus.value.decode("ascii")
        result["source"] = "CUDA driver API / current CUDA context"
    except (OSError, AttributeError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def collect_cuda(out: Path, expected_gpus: int) -> dict:
    import torch
    out = Path(out)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    result = {"schema_version": 1, "test": "identity-cuda", "rank": rank, "world_size": world,
              "time_utc": utc_now(), "expected_gpus": expected_gpus, "host_sha256": sha256(out / "identity-host.json") if (out / "identity-host.json").is_file() else None,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "boundary": BOUNDARY}
    try:
        if not 1 <= expected_gpus <= hp.MAX_GPUS or world != expected_gpus or torch.cuda.device_count() != expected_gpus:
            raise RuntimeError(f"Expected {expected_gpus} GPUs/workers, got {torch.cuda.device_count()} / {world}")
        torch.cuda.set_device(rank)
        props = torch.cuda.get_device_properties(rank)
        result.update(name=props.name, capability=list(torch.cuda.get_device_capability(rank)),
                      total_memory_bytes=int(props.total_memory), multiprocessors=props.multi_processor_count,
                      torch_uuid=normalize_uuid(getattr(props, "uuid", None)))
        # A CPU int64 reference and GPU FP32 implementation compare exact small
        # integer products; this is correctness evidence rather than a speed score.
        torch.backends.cuda.matmul.allow_tf32 = False
        generator = torch.Generator(device="cpu").manual_seed(4388 + rank)
        left = torch.randint(-8, 9, (128, 128), generator=generator, dtype=torch.int64)
        right = torch.randint(-8, 9, (128, 128), generator=generator, dtype=torch.int64)
        reference = left @ right
        calculated = left.to(device="cuda", dtype=torch.float32) @ right.to(device="cuda", dtype=torch.float32)
        fp32_correct = torch.equal(calculated.cpu().to(torch.float64), reference.to(torch.float64))
        bf16_supported = torch.cuda.get_device_capability(rank)[0] >= 8
        bf16_correct = None
        if bf16_supported:
            bf16 = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
            bf16_correct = bool(((bf16 @ bf16) == 128).all().item())
        pattern = torch.arange(262144, device="cuda", dtype=torch.int32)
        pattern.mul_(17).bitwise_xor_(0x5A5A5A5A)
        expected = (torch.arange(262144, dtype=torch.int32) * 17) ^ 0x5A5A5A5A
        memory_correct = torch.equal(pattern.cpu(), expected)
        torch.cuda.synchronize()
        result["correctness"] = {"fp32_cpu_reference": fp32_correct, "bf16_tensor_matrix": bf16_correct,
                                 "bf16_supported": bf16_supported, "integer_memory_pattern": memory_correct, "memory_tested_bytes": pattern.numel() * pattern.element_size(),
                                 "note": "Light 1 MiB pattern only; full HBM allocation/write/read and sustained load run separately."}
        result["driver_identity"] = cuda_driver_identity()
        result["uuid"] = result["driver_identity"].get("uuid") or result["torch_uuid"]
        result["pci_bus_id"] = result["driver_identity"].get("pci_bus_id")
        result["status"] = "PASS" if all((fp32_correct, bf16_correct if bf16_supported else True, memory_correct)) else "FAIL"
        if result["status"] == "FAIL":
            result["error"] = "CUDA correctness mismatch"
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write_json(out / f"identity-gpu{rank}.json", result)
    return result


def evidence_bytes(out, record):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Missing evidence record")
    root = Path(out).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path escapes output directory")
    raw = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Evidence SHA256 mismatch")
    return raw


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def hex_id(value):
    value = str(value or "").lower().removeprefix("0x")
    return value if re.fullmatch("[0-9a-f]{4}", value) else None


def unpack_pci_id(value):
    value = str(value or "").lower().removeprefix("0x")
    # NVML XML stores the device/subsystem in the high 16 bits, vendor in low.
    return {"device": value[:4], "vendor": value[4:]} if re.fullmatch("[0-9a-f]{8}", value) else {}


def validate_host_shape(host):
    if not isinstance(host, dict) or host.get("schema_version") != 1 or host.get("test") != "identity-host":
        raise ValueError("Unsupported host identity schema")
    if not isinstance(host.get("commands"), dict) or not isinstance(host.get("evidence"), dict):
        raise ValueError("Invalid host command/evidence records")
    for command in host["commands"].values():
        if not isinstance(command, dict) or not isinstance(command.get("files"), dict):
            raise ValueError("Invalid command evidence structure")
        if any(not isinstance(record, dict) for record in command["files"].values()):
            raise ValueError("Invalid command file record")
    if any(not isinstance(record, dict) for record in host["evidence"].values()):
        raise ValueError("Invalid host evidence structure")


def evaluate_identity(out: Path, expected_gpus: int, hardware_profile=None) -> dict:
    if isinstance(expected_gpus, bool) or not isinstance(expected_gpus, int) or not 1 <= expected_gpus <= hp.MAX_GPUS:
        raise ValueError("expected_gpus must be an integer from 1 to 64")
    out = Path(out)
    selected = hp.policy(out, hardware_profile)
    findings, per_gpu, evidence_hashes = [], [], {}
    def add(severity, code, detail, gpu=None):
        item = {"severity": severity, "code": code, "detail": detail}
        if gpu is not None:
            item["gpu"] = gpu
        findings.append(item)
    host, gpus, pci, lspci = {}, [], [], {}
    host_path = out / "identity-host.json"
    try:
        candidate = json.loads(host_path.read_text(encoding="utf-8"))
        validate_host_shape(candidate)
        host = candidate
        evidence_hashes[host_path.name] = sha256(host_path)
        raw = evidence_bytes(out, host["commands"]["nvidia-smi-xml"]["files"]["stdout"])
        if host["commands"]["nvidia-smi-xml"]["returncode"] != 0:
            add("REVIEW", "host_collection_failed", "nvidia-smi XML query did not succeed.")
        else:
            parsed = parse_smi_xml(raw)
            gpus = parsed["gpus"]
            if parsed != host.get("observations"):
                add("FAIL", "host_metadata_mismatch", "Cached host observations do not match the hashed raw XML.")
            if not parsed["driver_version"]:
                add("REVIEW", "driver_version_missing", "NVIDIA driver version was not reported.")
            if len(gpus) != expected_gpus:
                add("FAIL", "host_gpu_count", f"Expected {expected_gpus} physical GPUs, observed {len(gpus)}.")
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        add("REVIEW", "host_evidence_unavailable", f"Host identity evidence cannot be validated: {error}")
    try:
        pci = json.loads(evidence_bytes(out, host["evidence"]["pci-sysfs"]))
        if not isinstance(pci, list) or any(not isinstance(entry, dict) or not isinstance(entry.get("values"), dict) for entry in pci):
            raise ValueError("PCI evidence must be a list")
    except (OSError, ValueError, KeyError, TypeError) as error:
        add("REVIEW", "pci_evidence_unavailable", f"Linux PCI sysfs evidence is unavailable: {error}")
    try:
        command = host["commands"]["lspci"]
        raw = evidence_bytes(out, command["files"]["stdout"])
        if command["returncode"] != 0:
            raise ValueError("lspci did not succeed; install pciutils on the host")
        lspci = parse_lspci(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        add("REVIEW", "lspci_evidence_unavailable", str(error))
    # Validate all retained raw hashes; unsupported optional commands are visible
    # observations rather than automatic GPU failures.
    for label, command in host.get("commands", {}).items():
        for kind, record in command.get("files", {}).items():
            try:
                evidence_bytes(out, record)
                evidence_hashes[record["path"]] = record["sha256"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                add("REVIEW", "raw_evidence_changed", f"{label} {kind}: {error}")
        if label in ("virtual-machine", "container") and command.get("returncode") == 0:
            try:
                value = evidence_bytes(out, command["files"]["stdout"]).decode().strip()
                if value and value != "none":
                    add("REVIEW", "virtual_environment", f"Host collector reported {label}: {value}; physical ownership requires OEM/BMC evidence.")
            except (OSError, ValueError, KeyError, TypeError):
                pass
    for label, record in host.get("evidence", {}).items():
        if "path" not in record:
            continue
        try:
            evidence_bytes(out, record)
            evidence_hashes[record["path"]] = record["sha256"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            add("REVIEW", "raw_evidence_changed", f"{label}: {error}")
    cuda = []
    expected_probe_files = {f"identity-gpu{rank}.json" for rank in range(expected_gpus)}
    extra_probe_files = sorted(path.name for path in out.glob("identity-gpu*.json") if path.name not in expected_probe_files)
    if extra_probe_files:
        add("REVIEW", "unexpected_cuda_probe_files", f"Probe files outside the expected GPU count were found: {extra_probe_files}. Use a fresh output directory.")
    for rank in range(expected_gpus):
        path = out / f"identity-gpu{rank}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("test") != "identity-cuda":
                raise ValueError("Unsupported CUDA identity schema")
            evidence_hashes[path.name] = sha256(path)
            if type(record.get('rank')) is not int or type(record.get('world_size')) is not int or type(record.get('expected_gpus')) is not int or record.get("rank") != rank or record.get("world_size") != expected_gpus or record.get("expected_gpus") != expected_gpus:
                add("FAIL", "cuda_worker_mapping", f"Worker {rank} has inconsistent rank/count metadata.")
            if record.get("host_sha256") != evidence_hashes.get("identity-host.json"):
                add("FAIL", "stale_cuda_evidence", f"Worker {rank} was not collected against this host snapshot.")
            if record.get("status") != "PASS":
                add("FAIL", "cuda_execution_failed", f"Worker {rank}: {record.get('error', 'CUDA probe did not pass')}")
            cuda.append(record)
        except (OSError, ValueError, TypeError) as error:
            add("REVIEW", "cuda_evidence_missing", f"Worker {rank} identity probe is unavailable: {error}")
    for label, records in (("host", gpus), ("cuda", cuda)):
        uuids = [normalize_uuid(item.get("uuid")) for item in records]
        known = [value for value in uuids if value]
        if len(known) != len(set(known)):
            add("FAIL", "duplicate_uuid", f"{label} reports duplicate GPU UUIDs.")
    host_buses = [normalize_bus(gpu.get("pci_bus_id")) for gpu in gpus]
    if len([bus for bus in host_buses if bus]) != len(set(bus for bus in host_buses if bus)):
        add("FAIL", "duplicate_pci_address", "Host reports multiple GPUs at the same PCI address.")
    for gpu in gpus:
        key = normalize_uuid(gpu["uuid"]) or f"host-index-{gpu['index']}"
        bus = normalize_bus(gpu["pci_bus_id"])
        matches = [item for item in cuda if normalize_uuid(item.get("uuid")) == key]
        item = matches[0] if len(matches) == 1 else None
        summary = {"uuid": key, "host_index": gpu["index"], "name": gpu["name"], "pci_bus_id": bus,
                   "serial": gpu["serial"], "vbios_version": gpu["vbios_version"],
                   "memory_total_raw": gpu["memory_total_raw"], "memory_total_bytes": gpu["memory_total_bytes"],
                   "cuda_rank": item.get("rank") if item else None}
        per_gpu.append(summary)
        if not normalize_uuid(gpu["uuid"]):
            add("REVIEW", "host_uuid_invalid", "Host GPU UUID is missing or invalid.", key)
        if not bus:
            add("REVIEW", "host_pci_address_invalid", "Host GPU PCI address is missing or invalid.", key)
        if not hp.matches(selected, gpu["name"]):
            add("FAIL", "host_name_mismatch", f"Expected {selected}; host reports {gpu['name']!r}.", key)
        recognized = hp.identify(gpu["name"])
        summary["recognized_profile"] = recognized
        summary["features"] = {"ecc": "SUPPORTED" if gpu["ecc_current"] in ("Enabled", "Disabled") else "NOT_AVAILABLE", "mig": gpu["mig_mode"]}
        if recognized == "generic":
            add("REVIEW", "unlisted_model", "Model has no built-in architecture rule; verify exact SKU with the OEM.", key)
        if gpu["ecc_current"] == "Disabled":
            add("REVIEW", "ecc_disabled", "Supported ECC is disabled; enable it before procurement testing.", key)
        if gpu["serial"] in ("", "N/A", "Not Supported"):
            add("REVIEW", "serial_unavailable", "Board serial is unavailable; obtain the label/BMC/OEM record.", key)
        if gpu["vbios_version"] in ("", "N/A", "Not Supported"):
            add("REVIEW", "vbios_unavailable", "VBIOS version is unavailable.", key)
        if gpu["mig_mode"] not in ("Disabled", "N/A", "Not Supported"):
            add("REVIEW", "mig_or_unknown_mode", f"MIG mode is {gpu['mig_mode']!r}; this workflow expects full physical GPUs.", key)
        if gpu["virtualization_mode"] not in ("None", "N/A", "Not Supported"):
            add("REVIEW", "vgpu_or_unknown_mode", f"GPU virtualization mode is {gpu['virtualization_mode']!r}.", key)
        pci_matches = [entry for entry in pci if normalize_bus(entry.get("pci_bus_id")) == bus]
        values = pci_matches[0].get("values", {}) if len(pci_matches) == 1 else {}
        ids = {field: hex_id(values.get(field)) for field in KNOWN_PCI_VARIANT}
        summary["pci_ids"] = ids
        if ids["vendor"] and ids["vendor"] != "10de":
            add("FAIL", "pci_vendor_mismatch", f"Linux PCI vendor is {ids['vendor']}, expected NVIDIA 10de.", key)
        if any(not value for value in ids.values()):
            add("REVIEW", "pci_ids_missing", "Linux PCI vendor/device/subsystem evidence is incomplete.", key)
        elif recognized == "b300" and ids != KNOWN_PCI_VARIANT:
            add("REVIEW", "unknown_pci_variant", f"PCI IDs {ids} differ from the documented SXM6 AC variant; confirm this SKU with the OEM.", key)
        if values.get("driver") != "nvidia":
            add("REVIEW", "pci_driver_unknown", f"Linux PCI driver is {values.get('driver')!r}.", key)
        if values.get("physfn"):
            add("REVIEW", "pci_virtual_function", "PCI device is a virtual function; obtain physical inventory evidence.", key)
        for xml_value, names in ((gpu["pci_device_id"], ("vendor", "device")), (gpu["pci_subsystem_id"], ("subsystem_vendor", "subsystem_device"))):
            xml_ids = unpack_pci_id(xml_value)
            if not xml_ids:
                add("REVIEW", "smi_pci_ids_missing", f"nvidia-smi PCI ID {xml_value!r} is unavailable or malformed.", key)
            elif all(ids[field] for field in names) and (xml_ids["vendor"], xml_ids["device"]) != tuple(ids[field] for field in names):
                add("FAIL", "smi_sysfs_pci_mismatch", "nvidia-smi and Linux PCI IDs disagree.", key)
        lspci_ids = lspci.get(bus, {})
        if not lspci_ids:
            add("REVIEW", "lspci_device_missing", "lspci did not expose this PCI device.", key)
        elif any(lspci_ids.get(field) and ids.get(field) and lspci_ids[field] != ids[field] for field in KNOWN_PCI_VARIANT):
            add("FAIL", "lspci_sysfs_mismatch", "lspci and PCI sysfs IDs disagree.", key)
        if not item:
            add("REVIEW", "cuda_uuid_unmatched", "No unique CUDA worker UUID matches this physical GPU.", key)
            continue
        summary.update(cuda_name=item.get("name"), capability=item.get("capability"), cuda_memory_bytes=item.get("total_memory_bytes"))
        if str(item.get("name", "")).removeprefix("NVIDIA ").strip() != gpu["name"].removeprefix("NVIDIA ").strip():
            add("FAIL", "cuda_name_mismatch", f"CUDA reports {item.get('name')!r}.", key)
        expected_capability = hp.capability_for(gpu["name"], selected)
        if not hp.valid_capability(item.get("capability")) or (expected_capability and item.get("capability") != expected_capability):
            add("FAIL", "compute_capability_mismatch", f"Expected architecture {expected_capability}; CUDA reports {item.get('capability')!r}.", key)
        torch_uuid = item.get("torch_uuid")
        if torch_uuid and normalize_uuid(torch_uuid) != key:
            add("FAIL", "torch_driver_uuid_mismatch", "PyTorch and CUDA driver UUIDs disagree.", key)
        cuda_bus = normalize_bus(item.get("pci_bus_id"))
        if not cuda_bus:
            add("REVIEW", "cuda_pci_address_missing", "CUDA PCI address is unavailable.", key)
        elif bus and cuda_bus != bus:
            add("FAIL", "cuda_pci_address_mismatch", "CUDA and host PCI addresses disagree.", key)
        correctness = item.get("correctness", {})
        if not isinstance(correctness, dict):
            correctness = {}
        if any(correctness.get(field) is not True for field in ("fp32_cpu_reference", "integer_memory_pattern")):
            add("FAIL", "cuda_correctness_missing_or_failed", "One or more CUDA correctness checks are missing or failed.", key)
        if hp.valid_capability(item.get("capability")) and item["capability"][0] >= 8 and correctness.get("bf16_tensor_matrix") is not True:
            add("FAIL", "bf16_correctness_failed", "BF16 is expected for this architecture but its correctness check failed or is missing.", key)
        nvml_memory, cuda_memory = gpu["memory_total_bytes"], item.get("total_memory_bytes")
        if not positive_integer(cuda_memory) or not positive_integer(nvml_memory):
            add("REVIEW", "memory_observation_invalid", "CUDA/NVML exposed memory is unavailable or invalid.", key)
        else:
            difference = abs(cuda_memory - nvml_memory)
            tolerance = max((2 * 2**30 if recognized == 'b300' else 256 * 2**20), int(nvml_memory * 0.02))
            if difference > tolerance:
                add("REVIEW", "memory_views_differ", f"CUDA/NVML memory differs by {difference} bytes; check reservation/ECC/MIG and OEM specification.", key)
            if recognized == "b300" and not 230 * 2**30 <= cuda_memory <= 300 * 2**30:
                add("REVIEW", "memory_outside_conservative_range", "Exposed memory lies outside the conservative 230–300 GiB range; compare exact SKU/driver/ECC with the OEM before judging.", key)
    host_uuids = {normalize_uuid(gpu["uuid"]) for gpu in gpus}
    for item in cuda:
        cuda_uuid = normalize_uuid(item.get("uuid"))
        if not cuda_uuid:
            add("REVIEW", "cuda_uuid_unavailable", f"CUDA worker {item.get('rank')} UUID could not be observed.")
        elif cuda_uuid not in host_uuids:
            add("FAIL", "cuda_uuid_not_on_host", f"CUDA worker {item.get('rank')} UUID is absent from the host snapshot.")
    for item in per_gpu:
        relevant = [finding for finding in findings if finding.get("gpu") == item["uuid"]]
        item["status"] = "FAIL" if any(f["severity"] == "FAIL" for f in relevant) else "REVIEW" if any(f["severity"] == "REVIEW" for f in relevant) else "CONSISTENT"
    status = "FAIL" if any(f["severity"] == "FAIL" for f in findings) else "REVIEW" if any(f["severity"] == "REVIEW" for f in findings) else "CONSISTENT"
    result = {"schema_version": 1, "test": "identity-summary", "time_utc": utc_now(), "status": status,
              "hardware_profile": selected, "purchase_model_verified": False,
              "expected_gpus": expected_gpus, "observed_host_gpus": len(gpus), "observed_cuda_workers": len(cuda),
              "findings": findings, "per_gpu": per_gpu, "evidence_sha256": evidence_hashes,
              "boundary": BOUNDARY, "vendor_confirmation": "NOT_PERFORMED",
              "memory_note": "Capacity is observed, not a purchase promise. Verify exact SKU, GB/GiB units, ECC and driver reservations against the written BOM.",
              "sources": SOURCES}
    write_json(out / "identity-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("host", "cuda", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=range(1, hp.MAX_GPUS + 1), default=8)
    args = parser.parse_args()
    if args.phase == "host":
        result = collect_host(args.out)
        print(json.dumps({"phase": "host", "gpus": len(result["observations"]["gpus"]), "output": str(args.out / "identity-host.json")}, ensure_ascii=False))
        return 0
    result = collect_cuda(args.out, args.gpus) if args.phase == "cuda" else evaluate_identity(args.out, args.gpus)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
