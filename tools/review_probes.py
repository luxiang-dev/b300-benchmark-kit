"""CPU review probes only. All model, source, GPU, and benchmark data are synthetic."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

KIT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(KIT), str(KIT / "tests")]
import deepseek_mlperf as ds
import deepseek_assets
import report
import test_kit
import test_deepseek_mlperf as test_ds


def report_probes():
    fixture = test_kit.ReportTests()
    fixture.setUp()
    try:
        (fixture.out / "telemetry.csv").unlink()
        missing = report.generate(fixture.out)["overall"]
        (fixture.out / "telemetry.csv").write_text(
            "timestamp,index,uuid,temperature_c,power_w,memory_used_mib,gpu_util_percent\n"
            "now,0,gpu0,50,500,1000,98\n", encoding="utf-8")
        partial = report.generate(fixture.out)["overall"]
        return {"missing_telemetry": missing, "one_of_eight_gpus_telemetry": partial}
    finally:
        fixture.tearDown()


def response_probe():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        out = Path(tmp)
        result, _, acc = test_ds.save_evidence(out)
        raw = acc / "mlperf_log_accuracy.json"
        ds.write_json(raw, [{"qsl_idx": i, "data": "01"} for i in range(4388)])
        result["evidence_sha256"][raw.relative_to(out).as_posix()] = ds.sha(raw)
        ds.write_json(out / "deepseek-mlperf.json", result)
        accepted = ds.load_evidence(out) is not None
        return {"one_byte_instead_of_int32_responses_accepted": accepted}


def config_environment_probe():
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    process = SimpleNamespace(poll=lambda: None)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp, \
         patch.dict(ds.os.environ, {"CONFIG_DIR": "/synthetic/external-configs"}), \
         patch.object(ds.urllib.request, "urlopen", side_effect=[URLError("not started"), Response()]), \
         patch.object(ds.subprocess, "Popen", return_value=process) as launch, \
         patch.object(ds.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
         patch.object(ds, "stop"):
        ds.run_phase(Path(tmp), Path(tmp), "PerformanceOnly", 10, 10)
        return {"unchecked_config_dir_inherited_by_server": launch.call_args.kwargs["env"].get("CONFIG_DIR"),
                "command_pins_config_dir": any("--config_dir" in x for x in launch.call_args.args[0])}


def dataset_probe():
    import numpy as np
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        root = Path(tmp)
        workspace = root / "submission"
        required = ["Makefile", str(ds.CONFIG), "code/deepseek-r1/tensorrt/generation_config.json",
                    "3rdparty/mlc-inference/loadgen/mlperf.conf",
                    "3rdparty/mlc-inference/language/deepseek-r1/eval_accuracy.py",
                    "3rdparty/mlc-inference/language/deepseek-r1/submodules/prm800k/setup.py",
                    "3rdparty/mlc-inference/tools/submission/submission_checker/constants.py",
                    "code/deepseek-r1/tensorrt/requirements.accuracy.txt",
                    str(ds.ORIGINAL / "config.json"), str(ds.ORIGINAL / "tokenizer.json"),
                    str(ds.ORIGINAL / "tokenizer_config.json"), str(ds.MODEL / "hf_quant_config.json"),
                    "build/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"]
        for name in required:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
        ds.write_json(workspace / ds.MODEL / "config.json",
                      {"model_type": "deepseek_v3", "num_hidden_layers": 61, "n_routed_experts": 256})
        ds.write_json(workspace / ds.MODEL / "model.safetensors.index.json",
                      {"weight_map": {"synthetic": "synthetic.safetensors"}})
        (workspace / ds.MODEL / "synthetic.safetensors").write_bytes(b"synthetic")
        (workspace / "3rdparty/trtllm").mkdir()
        (workspace / ds.DATA).mkdir(parents=True)
        # Deliberately invalid shape, dtype, lengths and token content.
        np.save(workspace / ds.DATA / "input_ids_padded.npy", np.full((4388, 1), np.nan))
        np.save(workspace / ds.DATA / "input_lens.npy", np.full(4388, -1, dtype=np.int64))
        venv = root / "synthetic-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python3").write_bytes(b"synthetic")
        (venv / ".requirements_hash").write_text(
            ds.sha(workspace / "code/deepseek-r1/tensorrt/requirements.accuracy.txt"))
        fake_code = ModuleType("code")
        fake_code.__path__ = []
        common = ModuleType("code.common")
        common.__path__ = []
        mlcommons = ModuleType("code.common.mlcommons")
        mlcommons.loadgen = SimpleNamespace(_vstr="v6.0", model_config={"accuracy-target": {"deepseek-r1": []}})
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(device_count=lambda: 8, get_device_name=lambda i: "NVIDIA B300")
        torch.version = SimpleNamespace(cuda="synthetic")
        torch.__version__ = "synthetic"
        def path_factory(value):
            return venv if str(value) == "/work/.dsr1-acc-venv" else Path(value)
        original_sys_path = sys.path[:]
        try:
            with patch.object(ds, "Path", side_effect=path_factory), \
                 patch.object(deepseek_assets, "verify", return_value=[]), \
                 patch.object(ds.subprocess, "check_output", side_effect=[ds.SOURCE_REVISION, ds.MLCOMMONS_REVISION]), \
                 patch.object(ds.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                 patch.dict(sys.modules, {"code": fake_code, "code.common": common,
                                         "code.common.mlcommons": mlcommons, "torch": torch}), \
                 patch("builtins.print"), \
                 patch.object(ds.importlib, "import_module", return_value=SimpleNamespace(__version__="synthetic")):
                ds.check(workspace)
                return {"nan_float64_inputs_4388_by_1_and_negative_lengths_accepted": True,
                        "note": "External assets, Git and GPU conditions mocked; actual dataset validation executed."}
        finally:
            sys.path[:] = original_sys_path


if __name__ == "__main__":
    observations = {"synthetic_only": True, "report": report_probes(),
                    "responses": response_probe(), "dataset": dataset_probe(),
                    "configuration": config_environment_probe()}
    target = Path(__file__).with_name("probe_results.json")
    target.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(observations, indent=2))
