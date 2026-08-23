import unittest

from select_gold_audit_batch import is_reviewed, question_gate


class ReviewedStatusTests(unittest.TestCase):
    def test_recognizes_both_review_markers(self):
        self.assertTrue(is_reviewed({"status": "retrieval_blind_completeness_reviewed"}))
        self.assertTrue(is_reviewed({"completeness_review": {"decision": "pass"}}))
        self.assertFalse(is_reviewed({"status": "scoped_lsh_gold"}))

    def test_classification_membership_gate_matches_tool_policy(self):
        self.assertTrue(question_gate("뇌경색 보장되는 진단 코드는?",
                                      "classification_membership"))
        self.assertTrue(question_gate("양성뇌종양이 포함되나요?",
                                      "classification_membership"))
        self.assertFalse(question_gate("수술이 포함되나요?",
                                       "classification_membership"))
        self.assertFalse(question_gate("보험금은 얼마인가요?",
                                       "classification_membership"))

    def test_unknown_question_gate_fails_closed(self):
        with self.assertRaises(ValueError):
            question_gate("질문", "unknown")


if __name__ == "__main__":
    unittest.main()
