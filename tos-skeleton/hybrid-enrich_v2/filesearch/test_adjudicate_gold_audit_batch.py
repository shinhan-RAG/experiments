import unittest

from adjudicate_gold_audit_batch import (indexed_decisions, indexed_decisions_relaxed,
                                         validate_decision)


def op(**overrides):
    base = {"op": "add_required_group", "group": None, "jo": None, "c0": None,
            "c1": None, "members": [], "evidence_role": None}
    return base | overrides


class AddRequiredGroupValidationTests(unittest.TestCase):
    """Adjudication reuses the finalize validator, so the new op must behave
    identically when a cross-audit decision is checked for selectability."""

    def setUp(self):
        self.gold = {"q": {"qid": "q", "groups": [
            {"key": "abcde", "members": [{"jo": "j1", "c0": 0, "c1": 5}]}]}}
        self.jo = {"j1": {"element_id": "j1", "char_start": 0, "char_end": 10,
                          "text": "abcdefghij"},
                   "j2": {"element_id": "j2", "char_start": 20, "char_end": 30,
                          "text": "klmnopqrst"}}

    def error_of(self, decision):
        try:
            validate_decision(decision, self.gold, self.jo)
        except (KeyError, TypeError, ValueError) as exc:
            return str(exc)
        return ""

    def test_valid_add_required_group_is_selectable(self):
        decision = {"qid": "q", "decision": "fix", "ops": [op(
            members=[{"jo": "j2", "c0": 20, "c1": 25, "evidence_role": "direct"}])]}
        self.assertEqual(self.error_of(decision), "")

    def test_empty_members_is_not_selectable(self):
        decision = {"qid": "q", "decision": "fix", "ops": [op()]}
        self.assertIn("empty member operation", self.error_of(decision))

    def test_duplicate_key_is_not_selectable(self):
        decision = {"qid": "q", "decision": "fix", "ops": [op(
            members=[{"jo": "j1", "c0": 0, "c1": 5, "evidence_role": "direct"}])]}
        self.assertIn("duplicate required group key", self.error_of(decision))

    def test_exclude_row_cannot_carry_the_op(self):
        decision = {"qid": "q", "decision": "exclude", "ops": [op(
            members=[{"jo": "j2", "c0": 20, "c1": 25, "evidence_role": "direct"}])]}
        self.assertIn("must have no operations", self.error_of(decision))


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
