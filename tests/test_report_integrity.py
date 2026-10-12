"""Report aggregation must not hide nonzero exits or unavailable log access."""
import unittest
import test_kit
import report


class StageIntegrityTests(unittest.TestCase):
    # Reuse the eight-card private fixture, without publishing fake results.
    setUp = test_kit.ReportTests.setUp
    tearDown = test_kit.ReportTests.tearDown
    save = test_kit.ReportTests.save
    snapshot = test_kit.ReportTests.snapshot
    telemetry = test_kit.ReportTests.telemetry

    def test_unknown_or_nonzero_pass_stage_cannot_pass(self):
        for row in ('environment\tPASS\t7\n', 'environment\tUNKNOWN\t0\n', 'malformed-row\n'):
            with self.subTest(row=row):
                (self.out / 'stages.tsv').write_text(row, encoding='utf-8')
                self.assertEqual(report.generate(self.out)['overall'], 'FAIL')

    def test_successful_empty_journal_is_distinguished_from_unavailable(self):
        (self.out / 'kernel-during.log').write_text('-- No entries --\n', encoding='utf-8')
        self.assertEqual(report.generate(self.out)['overall'], 'REVIEW')
        self.save('collection-status.json', {'kernel-during.log': dict(argv=['journalctl', '-k', '--since', '@1'], returncode=0, error=None)})
        (self.out / 'kernel-during.log.stderr').write_text('', encoding='utf-8')
        self.assertEqual(report.generate(self.out)['overall'], 'PASS')
        (self.out / 'kernel-during.log.stderr').write_text('Permission/access warning', encoding='utf-8')
        self.assertEqual(report.generate(self.out)['overall'], 'REVIEW')


if __name__ == '__main__':
    unittest.main()
