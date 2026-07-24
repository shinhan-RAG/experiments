import json
import tempfile
import unittest
from pathlib import Path

from src.eval.dataset_eda import analyze_dataset, render_markdown


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class DatasetEdaTests(unittest.TestCase):
    def test_reports_shape_gold_structure_and_lexical_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            write_jsonl(raw_dir / "corpus.jsonl", [
                {"_id": "d1", "title": "보험", "text": "사망보험금 지급 조건"},
                {"_id": "d2", "title": "finance", "text": "bond market duration"},
                {"_id": "d3", "title": "", "text": "unrelated evidence"},
            ])
            write_jsonl(raw_dir / "queries.jsonl", [
                {"_id": "q1", "text": "사망보험금 지급 조건"},
                {"_id": "q2", "text": "unseen phrase"},
            ])
            write_jsonl(raw_dir / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "d1", "score": 1},
                {"query-id": "q1", "corpus-id": "d3", "score": 1},
                {"query-id": "q2", "corpus-id": "d2", "score": 1},
            ])

            report = analyze_dataset("fixture", raw_dir)
            self.assertEqual(report["corpus"]["document_count"], 3)
            self.assertEqual(report["queries"]["query_count"], 2)
            self.assertEqual(report["relevance"]["multi_gold_query_rate"], 0.5)
            self.assertEqual(
                report["lexical_alignment"]["zero_token_overlap_query_rate"], 0.5
            )
            self.assertEqual(
                report["lexical_alignment"]["exact_normalized_query_in_gold_rate"],
                0.5,
            )
            self.assertIn("fixture EDA", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
