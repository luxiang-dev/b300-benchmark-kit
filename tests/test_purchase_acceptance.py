"""A technical green result must never become an automatic buying decision."""
import unittest
import purchase_acceptance


class AcceptancePolicyTests(unittest.TestCase):
    def test_all_technical_checks_pass_still_requires_purchase_evidence(self):
        result = purchase_acceptance.evaluate(dict(overall='PASS', failures=[], incomplete=[], review=[]))
        self.assertEqual(result['status'], 'NOT_ACCEPTED')
        self.assertEqual(result['final_acceptance'], 'NOT_PERFORMED')
        self.assertGreaterEqual(len(result['pending_checks']), 9)
        self.assertTrue(all(item['status'] == 'PENDING' for item in result['pending_checks']))

    def test_failure_review_or_incomplete_blocks_acceptance(self):
        for state in ('FAIL', 'REVIEW', 'INCOMPLETE', 'UNKNOWN'):
            with self.subTest(state=state):
                self.assertEqual(purchase_acceptance.evaluate(dict(overall=state))['status'], 'BLOCKED')

    def test_supplied_accepted_field_cannot_clear_independent_policy(self):
        result = purchase_acceptance.evaluate(dict(overall='PASS', acceptance=dict(status='ACCEPTED', pending_checks=[])))
        self.assertEqual(result['status'], 'NOT_ACCEPTED')
        self.assertTrue(result['pending_checks'])
