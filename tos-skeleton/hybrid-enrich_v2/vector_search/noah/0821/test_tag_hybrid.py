import unittest
import json
from pathlib import Path

from tag_hybrid import build_sparse, build_view, sparse_search, tokens


class TagHybridUnitTests(unittest.TestCase):
    def test_sparse_prefers_matching_semantic_tag(self):
        views = ["보험료 납입면제 조건", "암 진단 보험금 지급", "계약 해지 환급금"]
        hits = sparse_search(build_sparse(views), "보험료 면제", top_k=3)
        self.assertEqual(hits[0][0], 0)

    def test_view_contains_tags_locator_and_source(self):
        tag = {
            "contract_key": "암특약", "subject_key": ["암진단금"], "role": ["payment_trigger"],
            "qualifier": ["90일"], "reference": [], "schema_tag": "paragraph",
            "locator": {"article": "제3조", "article_title": "보험금 지급사유", "table_headers": [], "row_keys": [], "section": "제2관"},
            "search_text": "암 진단 지급",
        }
        view = build_view(tag, {"text": "피보험자가 암으로 진단확정되었을 때 지급합니다."})
        for expected in ("암특약", "암진단금", "보험금 지급 사유 조건", "제3조", "진단확정"):
            self.assertIn(expected, view)

    def test_korean_bigrams_support_unspaced_query(self):
        self.assertIn("납입", tokens("납입면제"))

    def test_no_arm_embeds_or_dense_searches_tags(self):
        arms = json.loads((Path(__file__).resolve().parent / "arms.json").read_text(encoding="utf-8"))
        for arm in arms.values():
            self.assertNotIn("dense", arm.get("tag_weights", {}))


if __name__ == "__main__":
    unittest.main()
