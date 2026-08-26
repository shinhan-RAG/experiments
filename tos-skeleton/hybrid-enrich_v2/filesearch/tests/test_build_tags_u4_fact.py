import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import unittest

from build_tags_u4_fact import (
    appendix_context,
    benefit_aliases,
    document_keys,
    evidence_anchors,
    explicit_table_refs,
    fact_roles,
    table_keys,
)


class FactTagBuilderTest(unittest.TestCase):
    def test_claim_document_enumeration_is_preserved(self):
        row = {
            "element_type": "paragraph",
            "text": '사고증명서는 "사망진단서, 장해진단서, 진료기록부"를 말합니다.',
        }
        anchors = evidence_anchors(row)
        self.assertIn("장해진단서", anchors)
        self.assertIn("claim_procedure", fact_roles(row["text"]))

    def test_answer_sentence_with_generic_case_word_is_not_filtered(self):
        sentence = "보험안내자료 내용이 약관과 다른 경우 계약자에게 유리한 내용으로 봅니다."
        anchors = evidence_anchors({"element_type": "paragraph", "text": sentence})
        self.assertIn(sentence, anchors)

    def test_benefit_name_has_conservative_surface_aliases(self):
        aliases = benefit_aliases("암진단급여금은 최초 1회 지급합니다.")
        self.assertIn("암진단금", aliases)
        self.assertIn("암진단보험금", aliases)

    def test_explicit_external_table_reference_is_normalized(self):
        self.assertEqual(["별첨:2:표:12-1"],
                         explicit_table_refs("별첨2 [표 12-1] 참조"))

    def test_table_namespaces_never_collapse_same_number(self):
        self.assertEqual(
            ["부표:1", "별첨:2:표:1", "표:1"],
            table_keys("부표1 / 별첨2 [표1] / 표1"),
        )
        self.assertEqual(["부표:2-2"], explicit_table_refs("<부표2-2> 참조"))

    def test_appendix_heading_flows_to_adjacent_table_not_reference_line(self):
        rows = [
            {"element_id": "e0", "element_type": "paragraph", "line_start": 1,
             "line_end": 2, "text": "표12-1\n허혈심장질환 분류표"},
            {"element_id": "e1", "element_type": "table", "line_start": 3,
             "line_end": 5, "text": "| 협심증 | I20 |"},
            {"element_id": "e2", "element_type": "paragraph", "line_start": 20,
             "line_end": 20, "text": "별첨2 [표 6-1] 참조"},
        ]
        contexts = appendix_context(rows)
        self.assertEqual("표:12-1", contexts["e1"])
        self.assertNotEqual("별첨:2:표:6-1", contexts.get("e2"))

    def test_appendix_context_stops_before_unrelated_later_table(self):
        rows = [
            {"element_id": "heading", "element_type": "paragraph", "line_start": 1,
             "line_end": 2, "text": "표3 질병분류표"},
            {"element_id": "related", "element_type": "table", "line_start": 3,
             "line_end": 6, "text": "| 질병 | 코드 |"},
            {"element_id": "article", "element_type": "paragraph", "line_start": 7,
             "line_end": 10, "text": "제8조 약관 규정의 준용"},
            {"element_id": "unrelated", "element_type": "table", "line_start": 11,
             "line_end": 14, "text": "| 구분 | 진단서 |"},
        ]
        contexts = appendix_context(rows)
        self.assertEqual("표:3", contexts["related"])
        self.assertNotIn("unrelated", contexts)

    def test_document_key_changes_at_source_marker(self):
        rows = [
            {"text": "<!-- 원본: /corpus/doc-a/page.md -->"},
            {"text": "본문"},
            {"text": "<!-- 원본: /corpus/doc-b/page.md -->"},
            {"text": "본문"},
        ]
        keys = document_keys(rows)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[2], keys[3])
        self.assertNotEqual(keys[0], keys[2])
        self.assertTrue(all(key.startswith("doc:") for key in keys))

    def test_appendix_context_never_crosses_document_boundary(self):
        rows = [
            {"element_id": "h", "element_type": "paragraph", "line_start": 1,
             "line_end": 2, "text": "표2-1 분류표"},
            {"element_id": "other-doc-table", "element_type": "table", "line_start": 3,
             "line_end": 5, "text": "| unrelated |"},
        ]
        contexts = appendix_context(rows, ["doc-a", "doc-b"])
        self.assertNotIn("other-doc-table", contexts)


if __name__ == "__main__":
    unittest.main()
