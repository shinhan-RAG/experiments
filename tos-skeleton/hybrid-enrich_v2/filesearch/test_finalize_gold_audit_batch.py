import json
import tempfile
import unittest
from pathlib import Path

from finalize_gold_audit_batch import validate_decision


class FinalizeGoldAuditTest(unittest.TestCase):
    def setUp(self):
        self.gold = {
            "q": {"qid": "q", "groups": [{"members": [{"jo": "j1", "c0": 0, "c1": 5}]}]}
        }
        self.jo = {"j1": {"element_id": "j1", "char_start": 0, "char_end": 10},
                   "j2": {"element_id": "j2", "char_start": 20, "char_end": 30}}

    def test_valid_replace_span(self):
        decision = {"qid": "q", "decision": "fix", "ops": [{
            "op": "replace_span", "group": 0, "jo": "j1", "c0": 1, "c1": 9,
            "members": [], "evidence_role": "direct",
        }]}
        validate_decision(decision, self.gold, self.jo)

    def test_rejects_outside_span(self):
        decision = {"qid": "q", "decision": "fix", "ops": [{
            "op": "replace_span", "group": 0, "jo": "j1", "c0": 1, "c1": 11,
            "members": [], "evidence_role": "direct",
        }]}
        with self.assertRaisesRegex(ValueError, "span outside JO"):
            validate_decision(decision, self.gold, self.jo)

    def test_rejects_duplicate_final_groups(self):
        gold = {"q": {"qid": "q", "groups": [
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
            {"members": [{"jo": "j2", "c0": 20, "c1": 25}]},
        ]}}
        decision = {"qid": "q", "decision": "fix", "ops": [{
            "op": "replace_group_members", "group": 1, "jo": None, "c0": None, "c1": None,
            "members": [{"jo": "j1", "c0": 0, "c1": 5, "evidence_role": "direct"}],
            "evidence_role": None,
        }]}
        with self.assertRaisesRegex(ValueError, "duplicate final required groups"):
            validate_decision(decision, gold, self.jo)

    def test_allows_independent_spans_in_the_same_jo(self):
        gold = {"q": {"qid": "q", "groups": [
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
            {"members": [{"jo": "j1", "c0": 5, "c1": 10}]},
        ]}}
        decision = {"qid": "q", "decision": "pass", "ops": []}
        validate_decision(decision, gold, self.jo)

    def test_pass_cannot_mutate(self):
        decision = {"qid": "q", "decision": "pass", "ops": [{"op": "replace_span"}]}
        with self.assertRaisesRegex(ValueError, "must have no operations"):
            validate_decision(decision, self.gold, self.jo)

    def test_exclude_can_quarantine_preexisting_duplicate_groups(self):
        gold = {"q": {"qid": "q", "groups": [
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
        ]}}
        decision = {"qid": "q", "decision": "exclude", "ops": []}
        validate_decision(decision, gold, self.jo)

    def test_pass_still_rejects_preexisting_duplicate_groups(self):
        gold = {"q": {"qid": "q", "groups": [
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
            {"members": [{"jo": "j1", "c0": 0, "c1": 5}]},
        ]}}
        decision = {"qid": "q", "decision": "pass", "ops": []}
        with self.assertRaisesRegex(ValueError, "duplicate final required groups"):
            validate_decision(decision, gold, self.jo)


if __name__ == "__main__":
    unittest.main()
