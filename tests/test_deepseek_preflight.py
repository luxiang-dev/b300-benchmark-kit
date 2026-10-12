"""CPU regressions for malformed data and offline tokenizer dependency preparation."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepseek_mlperf as ds
import prepare_deepseek_accuracy as prep
import test_kit as kit


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = kit.workspace_temp()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_token_output_format_and_summary_consistency(self):
        valid = [{"qsl_idx": i, "data": struct.pack('<ii', 1, 2).hex()} for i in range(3)]
        accuracy = dict(tokens_per_sample=2)
        ds.validate_responses(valid, accuracy, samples=3)
        for data in ('01', '0100000000', struct.pack('<i', -1).hex(), struct.pack('<i', 129280).hex()):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ds.validate_responses([{**e, 'data': data} for e in valid], accuracy, samples=3)
        with self.assertRaisesRegex(ValueError, 'lengths disagree'):
            ds.validate_responses(valid, dict(tokens_per_sample=3), samples=3)
        with self.assertRaises(ValueError):
            ds.validate_responses([{**e, 'qsl_idx': True} for e in valid], accuracy, samples=3)

    def test_real_dataset_validation_rejects_corruption_and_matches_raw(self):
        (self.root / ds.DATA).mkdir(parents=True)
        (self.root / ds.RAW_DATA).parent.mkdir(parents=True)
        inputs = np.ones((4388, 3140), dtype=np.int32)
        inputs[:, 0] = 42
        lengths = np.ones(4388, dtype=np.int32)
        pd.DataFrame({'tok_input': [[42] for _ in range(4388)], 'tok_input_len': lengths}).to_pickle(self.root / ds.RAW_DATA)
        np.save(self.root / ds.DATA / 'input_ids_padded.npy', inputs)
        np.save(self.root / ds.DATA / 'input_lens.npy', lengths)
        ds.validate_dataset(self.root, 129280)
        inputs[0, 0] = 43
        np.save(self.root / ds.DATA / 'input_ids_padded.npy', inputs)
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            ds.validate_dataset(self.root, 129280)
        for malformed in (np.full((4388, 1), np.nan), np.ones((4388, 3140), dtype=np.int64)):
            np.save(self.root / ds.DATA / 'input_ids_padded.npy', malformed)
            with self.assertRaisesRegex(RuntimeError, 'int32 shapes'):
                ds.validate_dataset(self.root, 129280)
        np.save(self.root / ds.DATA / 'input_ids_padded.npy', inputs)
        np.save(self.root / ds.DATA / 'input_lens.npy', np.full(4388, -1, dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, 'input lengths'):
            ds.validate_dataset(self.root, 129280)

    def test_offline_hub_cache_created_from_verified_local_files(self):
        source = self.root / 'build/models/original'
        source.mkdir(parents=True)
        files = []
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            data = b'{"synthetic_only":true}'
            (source / name).write_bytes(data)
            files.append(dict(path=name, size=len(data), hash_type='sha256', hash=hashlib.sha256(data).hexdigest()))
        model = dict(repo='deepseek-ai/DeepSeek-R1', name='original', revision=ds.TOKENIZER_REVISION, files=files)
        with patch.object(prep.deepseek_assets, 'catalog', return_value=(self.root / 'build', [model])):
            destination = prep.prepare(self.root)
            self.assertEqual(destination.parent.name, 'snapshots')
            self.assertEqual(destination.name, ds.TOKENIZER_REVISION)
            prep.verify_cache(self.root)
            self.assertEqual((destination / 'tokenizer.json').read_bytes(), (source / 'tokenizer.json').read_bytes())
            (source / 'tokenizer.json').write_bytes(b'bad')
            with self.assertRaises(RuntimeError):
                prep.prepare(self.root)

    def test_accuracy_preflight_uses_offline_evaluator_environment(self):
        venv = self.root / 'venv'
        (venv / 'bin').mkdir(parents=True)
        (venv / 'bin/python3').write_bytes(b'synthetic executable')
        requirements = self.root / 'code/deepseek-r1/tensorrt/requirements.accuracy.txt'
        requirements.parent.mkdir(parents=True)
        requirements.write_text('synthetic requirements', encoding='utf-8')
        (venv / '.requirements_hash').write_text(ds.sha(requirements), encoding='utf-8')
        with patch.dict(os.environ, {'TRANSFORMERS_CACHE': '/untrusted-cache'}), patch.object(ds.subprocess, 'run') as run:
            ds.accuracy_environment(self.root, venv)
        env = run.call_args.kwargs['env']
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['HF_HUB_CACHE'], str(self.root / 'build/hf-cache/hub'))
        self.assertNotIn('TRANSFORMERS_CACHE', env)
        self.assertIn("deepseek-ai/DeepSeek-R1", run.call_args.args[0][2])

    def test_both_phases_pin_config_and_reject_inherited_make_overrides(self):
        from types import SimpleNamespace
        from urllib.error import URLError
        process = SimpleNamespace(poll=lambda: None)
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
        with patch.dict(os.environ, {'CONFIG_DIR': '/wrong', 'MAKEFLAGS': 'RUN_ARGS=--precision=fp8'}), \
             patch.object(ds.urllib.request, 'urlopen', side_effect=[URLError('not started'), Response()]), \
             patch.object(ds.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(ds, 'run_harness_process'), patch.object(ds, 'stop'):
            ds.run_phase(self.root, self.root, 'PerformanceOnly', 10, 10)
        self.assertEqual(launch.call_args.kwargs['env']['CONFIG_DIR'], str(self.root / 'configs'))
        self.assertNotIn('MAKEFLAGS', launch.call_args.kwargs['env'])
        self.assertIn('--config_dir=configs', launch.call_args.args[0][-1])

    def test_harness_timeout_and_failure_always_clean_process_group(self):
        from unittest.mock import Mock
        for result in (subprocess.TimeoutExpired('harness', 1), 42):
            proc = Mock()
            if isinstance(result, Exception):
                proc.wait.side_effect = result
            else:
                proc.wait.return_value = result
            with patch.object(ds.subprocess, 'Popen', return_value=proc) as launch, patch.object(ds, 'stop') as stopped:
                with self.assertRaises((subprocess.TimeoutExpired, subprocess.CalledProcessError)):
                    ds.run_harness_process(self.root, 'PerformanceOnly', {}, None, 1)
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            stopped.assert_called_once_with(proc)
