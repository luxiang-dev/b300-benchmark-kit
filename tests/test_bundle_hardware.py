"""Offline package regressions with tiny synthetic archives, never real GPU images."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle
from test_kit import workspace_temp


class HardwareBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = workspace_temp()
        self.root = Path(self.temp.name)
        for name in bundle.CODE:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic package fixture only\n', encoding='utf-8')
        (self.root / 'images/runtime.inspect.json').write_text(json.dumps([dict(Id='sha256:synthetic', Architecture='amd64')]), encoding='utf-8')
        (self.root / 'models.lock.json').write_text(json.dumps(dict(models=[dict(name='missing-synthetic-weight', files=[])])), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def run_bundle(self, action, hardware=True):
        argv = ['bundle.py', action, '--root', str(self.root)] + (['--hardware-only'] if hardware else [])
        with patch.object(sys, 'argv', argv):
            return bundle.main()

    def test_hardware_seal_does_not_require_or_verify_model_weights(self):
        with patch.object(bundle, 'verify', side_effect=AssertionError('Hardware route must not load model weights')):
            self.assertEqual(self.run_bundle('seal'), 0)
            self.assertEqual(self.run_bundle('verify'), 0)
        seal = json.loads((self.root / 'hardware.sha256.json').read_text(encoding='utf-8'))
        self.assertTrue(seal['hardware_only'])
        self.assertEqual(set(seal['files']), set(bundle.CODE))
        self.assertFalse((self.root / 'bundle.sha256.json').exists())

    def test_mutated_dashboard_cannot_verify(self):
        self.run_bundle('seal')
        (self.root / 'dashboard/app.js').write_text('changed fixture', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            self.run_bundle('verify')

    def test_mismatched_seal_type_and_missing_inventory_are_rejected(self):
        self.run_bundle('seal')
        path = self.root / 'hardware.sha256.json'
        seal = json.loads(path.read_text(encoding='utf-8'))
        seal['hardware_only'] = False
        path.write_text(json.dumps(seal), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Seal type mismatch'):
            self.run_bundle('verify')
        seal['hardware_only'] = True
        seal['files'].pop('dashboard/app.js')
        path.write_text(json.dumps(seal), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'inventory mismatch'):
            self.run_bundle('verify')

    def test_full_bundle_still_requires_model_verification(self):
        with patch.object(bundle, 'verify', return_value=['missing-synthetic-weight']) as verify:
            with self.assertRaisesRegex(RuntimeError, 'Model files missing'):
                self.run_bundle('seal', hardware=False)
            verify.assert_called_once()
        with patch.object(bundle, 'verify', return_value=[]):
            self.assertEqual(self.run_bundle('seal', hardware=False), 0)
            self.assertEqual(self.run_bundle('verify', hardware=False), 0)


if __name__ == '__main__':
    unittest.main()
