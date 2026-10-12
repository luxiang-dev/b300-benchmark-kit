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
        try:
            accepted = ds.load_evidence(out) is not None
        except ValueError:
            accepted = False
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
         patch.object(ds, "run_harness_process"), \
         patch.object(ds, "stop"):
        ds.run_phase(Path(tmp), Path(tmp), "PerformanceOnly", 10, 10)
        return {"unchecked_config_dir_inherited_by_server": launch.call_args.kwargs["env"].get("CONFIG_DIR"),
                "command_pins_config_dir": any("--config_dir" in x for x in launch.call_args.args[0])}


def dataset_probe():
    import numpy as np
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        workspace = Path(tmp)
        (workspace / ds.DATA).mkdir(parents=True)
        np.save(workspace / ds.DATA / 'input_ids_padded.npy', np.full((4388, 1), np.nan))
        np.save(workspace / ds.DATA / 'input_lens.npy', np.full(4388, -1, dtype=np.int64))
        try:
            ds.validate_dataset(workspace, 129280)
            accepted = True
        except RuntimeError:
            accepted = False
        return {'nan_float64_inputs_4388_by_1_and_negative_lengths_accepted': accepted,
                'note': 'Actual dataset validator executed on explicitly synthetic malformed arrays.'}


if __name__ == "__main__":
    observations = {"synthetic_only": True, "report": report_probes(),
                    "responses": response_probe(), "dataset": dataset_probe(),
                    "configuration": config_environment_probe()}
    target = Path(__file__).with_name("probe_results.json")
    target.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(observations, indent=2))
