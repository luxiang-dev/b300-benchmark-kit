"""CPU fault injection only; fixtures do not represent measured B300 hardware."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import authenticity as identity


def fixture_xml(count=1, capability_name="NVIDIA B300 SXM6 AC", memory="276480 MiB", device="3182", mig="Disabled"):
    entries = []
    for index in range(count):
        entries.append(f"""<gpu id="00000000:{index + 10:02x}:00.0">
<product_name>{capability_name}</product_name><product_brand>NVIDIA</product_brand><product_architecture>Blackwell</product_architecture>
<serial>SYNTHETIC-SERIAL-{index}</serial><uuid>GPU-00000000-0000-0000-0000-{index + 1:012x}</uuid><vbios_version>SYNTHETIC-VBIOS</vbios_version>
<board_part_number>SYNTHETIC-PART</board_part_number><gpu_part_number>SYNTHETIC-PART</gpu_part_number>
<pci><pci_bus_id>00000000:{index + 10:02x}:00.0</pci_bus_id><pci_device_id>0x{device.upper()}10DE</pci_device_id><pci_sub_system_id>0x20E610DE</pci_sub_system_id></pci>
<fb_memory_usage><total>{memory}</total><reserved>1024 MiB</reserved><free>270000 MiB</free></fb_memory_usage>
<mig_mode><current_mig>{mig}</current_mig></mig_mode><gpu_virtualization_mode><virtualization_mode>None</virtualization_mode></gpu_virtualization_mode>
<ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode><compute_mode>Default</compute_mode>
</gpu>""")
    return ("<nvidia_smi_log><driver_version>580.95.05</driver_version><cuda_version>13.0</cuda_version>" + "".join(entries) + "</nvidia_smi_log>").encode()


class IdentityTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / ".test-tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.out = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def collect_fixture(self, count=1, **xml_args):
        raw = fixture_xml(count, **xml_args)
        lspci = "".join(f"0000:{index + 10:02x}:00.0 3D controller [0302]: NVIDIA Corporation Synthetic [10de:{xml_args.get('device', '3182')}]\n"
                       "\tSubsystem: NVIDIA Synthetic [10de:20e6]\n\tKernel driver in use: nvidia\n" for index in range(count)).encode()
        def command(argv, **kwargs):
            if argv == ["nvidia-smi", "-q", "-x"]:
                return subprocess.CompletedProcess(argv, 0, raw, b"")
            if argv[0] == "lspci":
                return subprocess.CompletedProcess(argv, 0, lspci, b"")
            if argv[0] == "systemd-detect-virt":
                return subprocess.CompletedProcess(argv, 1, b"none\n", b"")
            return subprocess.CompletedProcess(argv, 0, b"Synthetic only\n", b"")
        def pci(bus):
            return {"pci_bus_id": identity.normalize_bus(bus), "values": {"vendor": "0x10de", "device": "0x" + xml_args.get("device", "3182"),
                    "subsystem_vendor": "0x10de", "subsystem_device": "0x20e6", "driver": "nvidia"}, "errors": {}}
        with patch.object(identity.subprocess, "run", side_effect=command), patch.object(identity, "read_sysfs", side_effect=pci):
            host = identity.collect_host(self.out)
        for rank, gpu in enumerate(host["observations"]["gpus"]):
            record = {"schema_version": 1, "test": "identity-cuda", "rank": rank, "world_size": count, "expected_gpus": count,
                      "host_sha256": identity.sha256(self.out / "identity-host.json"), "status": "PASS",
                      "name": "NVIDIA B300 SXM6 AC", "uuid": gpu["uuid"], "torch_uuid": gpu["uuid"],
                      "pci_bus_id": gpu["pci_bus_id"], "capability": [10, 3], "total_memory_bytes": gpu["memory_total_bytes"] - 2**30,
                      "correctness": {"fp32_cpu_reference": True, "bf16_tensor_matrix": True, "integer_memory_pattern": True, "memory_tested_bytes": 2**20},
                      "synthetic_only": True}
            identity.write_json(self.out / f"identity-gpu{rank}.json", record)
        return host

    def update_gpu(self, rank=0, **changes):
        path = self.out / f"identity-gpu{rank}.json"
        record = json.loads(path.read_text())
        record.update(changes)
        identity.write_json(path, record)

    def codes(self, result):
        return {item["code"] for item in result["findings"]}

    def test_unit_conversion_never_treats_commercial_gb_as_gib(self):
        self.assertEqual(identity.memory_bytes("270 GB"), 270000000000)
        self.assertEqual(identity.memory_bytes("270 GiB"), 270 * 2**30)
        self.assertEqual(identity.memory_bytes("276480 MiB"), 270 * 2**30)
        self.assertIsNone(identity.memory_bytes("270"))
        self.assertIsNone(identity.memory_bytes("N/A"))

    def test_uuid_bus_and_lspci_parsing(self):
        self.assertEqual(identity.normalize_bus("00000000:AB:00.0"), "0000:ab:00.0")
        self.assertIsNone(identity.normalize_bus("../../etc/passwd"))
        value = "00000000-0000-0000-0000-000000000001"
        self.assertEqual(identity.normalize_uuid(value), "GPU-" + value)
        self.assertEqual(identity.normalize_uuid(bytes.fromhex(value.replace("-", ""))), "GPU-" + value)
        self.assertIsNone(identity.normalize_uuid("MIG-GPU-" + value + "/1/0"))
        self.assertEqual(identity.parse_lspci("0000:0a:00.0 3D controller [0302]: NVIDIA [10de:3182]\n\tSubsystem: Device [10de:20e6]\n\tKernel driver in use: nvidia\n")["0000:0a:00.0"]["device"], "3182")

    def test_consistent_fixture_is_not_vendor_authentication(self):
        host = self.collect_fixture()
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "CONSISTENT", result["findings"])
        self.assertEqual(result["vendor_confirmation"], "NOT_PERFORMED")
        self.assertIn("do not authenticate", result["boundary"])
        self.assertIn("identity-gpu0.json", result["evidence_sha256"])
        self.assertEqual(host["observations"]["gpus"][0]["board_part_number"], "SYNTHETIC-PART")

    def test_workers_match_uuid_even_if_cuda_order_changes(self):
        self.collect_fixture(count=2)
        paths = [self.out / f"identity-gpu{rank}.json" for rank in range(2)]
        records = [json.loads(path.read_text()) for path in paths]
        for rank, record in enumerate(reversed(records)):
            record["rank"] = rank
            identity.write_json(paths[rank], record)
        result = identity.evaluate_identity(self.out, 2)
        self.assertEqual(result["status"], "CONSISTENT", result["findings"])
        self.assertEqual([item["cuda_rank"] for item in result["per_gpu"]], [1, 0])

    def test_wrong_compute_capability_fails(self):
        self.collect_fixture()
        self.update_gpu(capability=[10, 0])
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("compute_capability_mismatch", self.codes(result))

    def test_cuda_uuid_and_bus_contradictions_fail(self):
        self.collect_fixture()
        self.update_gpu(pci_bus_id="0000:fe:00.0")
        self.assertIn("cuda_pci_address_mismatch", self.codes(identity.evaluate_identity(self.out, 1)))
        self.update_gpu(uuid="GPU-00000000-0000-0000-0000-000000000099")
        self.assertIn("cuda_uuid_not_on_host", self.codes(identity.evaluate_identity(self.out, 1)))

    def test_unknown_oem_pci_variant_needs_review_not_false_fake_verdict(self):
        self.collect_fixture(device="3183")
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("unknown_pci_variant", self.codes(result))

    def test_mig_and_partial_capacity_need_review(self):
        self.collect_fixture(mig="Enabled", memory="34560 MiB")
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("mig_or_unknown_mode", self.codes(result))
        self.assertIn("memory_outside_conservative_range", self.codes(result))

    def test_commercial_memory_unit_and_driver_reserve_can_be_consistent(self):
        self.collect_fixture(memory="270 GB")
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "CONSISTENT", result["findings"])

    def test_missing_gpu_probe_is_review(self):
        self.collect_fixture()
        (self.out / "identity-gpu0.json").unlink()
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("cuda_evidence_missing", self.codes(result))

    def test_unavailable_cuda_uuid_is_incomplete_evidence(self):
        self.collect_fixture()
        self.update_gpu(uuid=None, torch_uuid=None)
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("cuda_uuid_unavailable", self.codes(result))

    def test_host_only_wrong_name_fails_before_cuda_execution(self):
        self.collect_fixture(capability_name="NVIDIA H100")
        (self.out / "identity-gpu0.json").unlink()
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("host_name_mismatch", self.codes(result))

    def test_stale_cuda_probe_and_incomplete_correctness_fail(self):
        self.collect_fixture()
        self.update_gpu(host_sha256="0" * 64, correctness={"fp32_cpu_reference": True})
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("stale_cuda_evidence", self.codes(result))
        self.assertIn("cuda_correctness_missing_or_failed", self.codes(result))

    def test_hash_tampering_and_path_escape_are_detected(self):
        host = self.collect_fixture()
        record = host["commands"]["nvidia-smi-xml"]["files"]["stdout"]
        (self.out / record["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            identity.evidence_bytes(self.out, record)
        with self.assertRaisesRegex(ValueError, "escapes"):
            identity.evidence_bytes(self.out, {"path": "../escaped", "sha256": "0" * 64})
        result = identity.evaluate_identity(self.out, 1)
        self.assertNotEqual(result["status"], "CONSISTENT")
        self.assertIn("raw_evidence_changed", self.codes(result))

    def test_sysfs_and_xml_pci_id_disagreement_fails(self):
        host = self.collect_fixture()
        pci_record = host["evidence"]["pci-sysfs"]
        path = self.out / pci_record["path"]
        pci = json.loads(path.read_text())
        pci[0]["values"]["device"] = "0x2330"
        identity.write_json(path, pci)
        pci_record["sha256"] = identity.sha256(path)
        identity.write_json(self.out / "identity-host.json", host)
        self.update_gpu(host_sha256=identity.sha256(self.out / "identity-host.json"))
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("smi_sysfs_pci_mismatch", self.codes(result))
        self.assertIn("lspci_sysfs_mismatch", self.codes(result))

    def test_host_count_and_duplicate_worker_uuid_fail(self):
        self.collect_fixture(count=2)
        first = json.loads((self.out / "identity-gpu0.json").read_text())
        self.update_gpu(1, uuid=first["uuid"])
        result = identity.evaluate_identity(self.out, 2)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("duplicate_uuid", self.codes(result))
        result = identity.evaluate_identity(self.out, 1)
        self.assertIn("host_gpu_count", self.codes(result))

    def test_extra_old_probe_prevents_consistent_verdict(self):
        self.collect_fixture()
        identity.write_json(self.out / "identity-gpu7.json", {"synthetic_only": True})
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIn("unexpected_cuda_probe_files", self.codes(result))

    def test_collector_command_failure_is_retained_and_no_positive_verdict(self):
        with patch.object(identity.subprocess, "run", side_effect=FileNotFoundError("synthetic missing command")):
            host = identity.collect_host(self.out)
        self.assertIn("FileNotFoundError", host["commands"]["nvidia-smi-xml"]["error"])
        result = identity.evaluate_identity(self.out, 1)
        self.assertEqual(result["status"], "REVIEW")

    def test_malformed_evidence_is_review_instead_of_crashing(self):
        identity.write_json(self.out / "identity-host.json", ["invalid"])
        identity.write_json(self.out / "identity-gpu0.json", ["invalid"])
        self.assertEqual(identity.evaluate_identity(self.out, 1)["status"], "REVIEW")
        self.collect_fixture()
        self.update_gpu(correctness=[True, True, True])
        self.assertIn("cuda_correctness_missing_or_failed", self.codes(identity.evaluate_identity(self.out, 1)))
        for count in (0, 65, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                identity.evaluate_identity(self.out, count)


if __name__ == "__main__":
    unittest.main()
