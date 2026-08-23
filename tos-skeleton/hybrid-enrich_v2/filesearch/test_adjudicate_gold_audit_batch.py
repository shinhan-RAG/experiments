import unittest

from adjudicate_gold_audit_batch import indexed_decisions, indexed_decisions_relaxed


class IndexedDecisionsTests(unittest.TestCase):
    def test_requires_exact_qid_order(self):
        audit = {"decisions": [{"qid": "q1"}, {"qid": "q2"}]}
        self.assertEqual(set(indexed_decisions(audit, ["q1", "q2"], "a")), {"q1", "q2"})
        with self.assertRaisesRegex(ValueError, "qid/order mismatch"):
            indexed_decisions(audit, ["q2", "q1"], "a")

    def test_rejects_duplicate_qids(self):
        audit = {"decisions": [{"qid": "q1"}, {"qid": "q1"}]}
        with self.assertRaises(ValueError):
            indexed_decisions(audit, ["q1", "q1"], "a")

    def test_relaxed_index_records_duplicate_and_missing(self):
        audit = {"decisions": [
            {"qid": "q1", "decision": "pass"},
            {"qid": "q1", "decision": "fix"},
            {"qid": "qx", "decision": "pass"},
        ]}
        indexed, errors = indexed_decisions_relaxed(audit, ["q1", "q2"], "a")
        self.assertEqual(indexed["q1"]["decision"], "pass")
        self.assertEqual(indexed["q2"]["decision"], "invalid")
        self.assertIn("q1", errors)
        self.assertIn("q2", errors)
        self.assertTrue(any(key.startswith("<extra:") for key in errors))


if __name__ == "__main__":
    unittest.main()
