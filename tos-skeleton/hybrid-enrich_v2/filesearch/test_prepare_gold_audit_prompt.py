import json
import tempfile
import unittest
from pathlib import Path

from prepare_gold_audit_prompt import render


class RenderGoldAuditPromptTests(unittest.TestCase):
    def test_primary_and_review_modes_freeze_count_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            qids = Path(directory) / "qids.json"
            qids.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
            primary = render(str(qids), "gold.jsonl", "jo.jsonl")
            review = render(str(qids), "gold.jsonl", "jo.jsonl", "first.json")
        self.assertIn("배치 qid 2개", primary)
        self.assertNotIn("first.json", primary)
        self.assertIn("first.json", review)
        self.assertIn("반박 가능한 참고", review)

    def test_explicit_official_split_is_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            qids = Path(directory) / "qids.json"
            qids.write_text(json.dumps(["q1"]), encoding="utf-8")
            prompt = render(str(qids), "gold.jsonl", "jo.jsonl",
                            official="official_test.csv")
        self.assertIn("`official_test.csv`", prompt)
        self.assertNotIn("정답셋_348_train", prompt)


if __name__ == "__main__":
    unittest.main()
