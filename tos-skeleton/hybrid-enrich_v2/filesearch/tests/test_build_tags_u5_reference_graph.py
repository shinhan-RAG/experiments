import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import unittest

from build_tags_u5_reference_graph import (
    build_reference_aliases,
    build_table_catalog,
    declared_and_referenced_keys,
    explicit_redirect_pairs,
    toc_like_reference_list,
)


class ReferenceGraphBuilderTest(unittest.TestCase):
    def test_namespace_parser_separates_declaration_and_reference(self):
        declared, refs = declared_and_referenced_keys(
            "부표2-2\n허혈심장질환 분류표\n별첨2 [표 12-1] 참조")
        self.assertEqual(["부표:2-2"], declared)
        self.assertEqual(["별첨:2:표:12-1"], refs)
        declared, refs = declared_and_referenced_keys(
            '이 질병은 <부표2-2> "질병 분류표"에 해당하는 항목을 말합니다.')
        self.assertEqual([], declared)
        self.assertEqual(["부표:2-2"], refs)

    def test_reference_toc_is_not_a_substantive_target(self):
        self.assertTrue(toc_like_reference_list(
            "| [별첨2] 표 12-1. 허혈심장질환 분류표 | 3,295 |\n"
            "| [별첨2] 표 12-2. 급성심근경색증 분류표 | 3,297 |\n"
            "| [별첨2] 표 12-3. 특정급성심근경색증 분류표 | 3,299 |"))
        self.assertFalse(toc_like_reference_list(
            "[별첨2] 표 12-1. 허혈심장질환 분류표\n"
            "| 대상 질병 명 | 분류 코드 |\n| 협심증 | I20 |"))

    def test_two_hop_path_crosses_only_explicit_bridge(self):
        elements = [
            {"element_id": "e1", "contract_scope": "특약A"},
            {"element_id": "e2", "contract_scope": "특약A"},
            {"element_id": "e3", "contract_scope": "파서오염특약"},
        ]
        tags = [
            {"element_id": "e1", "document_key": "doc1", "contract_key": "특약A"},
            {"element_id": "e2", "document_key": "doc1", "contract_key": "특약A"},
            {"element_id": "e3", "document_key": "doc1", "contract_key": "파서오염특약"},
        ]
        jo_rows = [
            {"element_id": "j1", "members": ["e1"],
             "text": '"허혈심장질환"은 <부표2-2> 참조'},
            {"element_id": "j2", "members": ["e2"],
             "text": "부표2-2\n허혈심장질환 분류표\n별첨2 [표 12-1] 참조"},
            {"element_id": "j3", "members": ["e3"],
             "text": "[별첨2] 표 12-1. 허혈심장질환 분류표"},
        ]
        aliases, terms, provenance, graph = build_reference_aliases(
            elements, tags, jo_rows)
        self.assertEqual(["특약A"], aliases["j3"])
        self.assertIn("허혈심장질환", terms["j3"])
        self.assertTrue(any(path["source"] == "j1" and path["target"] == "j3"
                            and path["hops"] == 2 for path in provenance["j3"]))
        self.assertEqual([], graph["ambiguous"])

    def test_global_appendix_does_not_chain_into_another_table(self):
        elements = [
            {"element_id": "e1", "contract_scope": "특약A"},
            {"element_id": "e2", "contract_scope": "파서오염특약"},
            {"element_id": "e3", "contract_scope": "파서오염특약"},
        ]
        tags = [
            {"element_id": "e1", "document_key": "doc1", "contract_key": "특약A"},
            {"element_id": "e2", "document_key": "doc1", "contract_key": "파서오염특약"},
            {"element_id": "e3", "document_key": "doc1", "contract_key": "파서오염특약"},
        ]
        jo_rows = [
            {"element_id": "j1", "members": ["e1"],
             "text": "이 수술은 별첨2 [표 89] 참조"},
            {"element_id": "j2", "members": ["e2"],
             "text": "[별첨2] 표 89. 심장질환 수술표\n다른 내용은 <표3> 참조"},
            {"element_id": "j3", "members": ["e3"],
             "text": "표3\n장해분류표"},
        ]
        _, _, _, graph = build_reference_aliases(elements, tags, jo_rows)
        self.assertTrue(any(path["source"] == "j1" and path["target"] == "j2"
                            and path["hops"] == 1 for path in graph["paths"]))
        self.assertFalse(any(path["source"] == "j1" and path["target"] == "j3"
                             and path["hops"] == 2 for path in graph["paths"]))

    def test_multiple_local_schedules_do_not_cross_product_appendices(self):
        bridge_text = (
            "부표2-1\n뇌혈관질환 분류표\n별첨2 [표 11-1] 참조\n"
            "부표2-2\n허혈심장질환 분류표\n별첨2 [표 12-1] 참조"
        )
        self.assertEqual(
            {"부표:2-1": "별첨:2:표:11-1", "부표:2-2": "별첨:2:표:12-1"},
            explicit_redirect_pairs(bridge_text),
        )
        elements = [
            {"element_id": "e1", "contract_scope": "특약A"},
            {"element_id": "e2", "contract_scope": "특약A"},
            {"element_id": "e3", "contract_scope": "특약A"},
            {"element_id": "e4", "contract_scope": "문서부록"},
            {"element_id": "e5", "contract_scope": "문서부록"},
        ]
        tags = [
            {"element_id": row["element_id"], "document_key": "doc1",
             "contract_key": row["contract_scope"]}
            for row in elements
        ]
        jo_rows = [
            {"element_id": "j1", "members": ["e1"],
             "text": "뇌혈관질환은 <부표2-1> 참조"},
            {"element_id": "j2", "members": ["e2"],
             "text": "허혈심장질환은 <부표2-2> 참조"},
            {"element_id": "j3", "members": ["e3"], "text": bridge_text},
            {"element_id": "j4", "members": ["e4"],
             "text": "[별첨2] 표 11-1. 뇌혈관질환 분류표"},
            {"element_id": "j5", "members": ["e5"],
             "text": "[별첨2] 표 12-1. 허혈심장질환 분류표"},
        ]
        _, _, provenance, _ = build_reference_aliases(elements, tags, jo_rows)
        paths = {(row["source"], row["target"], tuple(row["keys"]))
                 for rows in provenance.values() for row in rows if row["hops"] == 2}
        self.assertIn(("j1", "j4", ("부표:2-1", "별첨:2:표:11-1")), paths)
        self.assertIn(("j2", "j5", ("부표:2-2", "별첨:2:표:12-1")), paths)
        self.assertNotIn(("j1", "j5", ("부표:2-1", "별첨:2:표:12-1")), paths)
        self.assertNotIn(("j2", "j4", ("부표:2-2", "별첨:2:표:11-1")), paths)

    def test_ambiguous_redirect_for_one_schedule_fails_closed(self):
        self.assertEqual({}, explicit_redirect_pairs(
            "부표2-2\n별첨2 [표 12-1] 참조\n별첨2 [표 12-2] 참조"))

    def test_redirect_does_not_reuse_stale_local_after_new_section(self):
        self.assertEqual({}, explicit_redirect_pairs(
            "부표2-2\n허혈심장질환 분류표\n"
            "제3조 보험금 지급사유\n별첨2 [표 12-1] 참조"))
        self.assertEqual({}, explicit_redirect_pairs(
            "부표2-1 및 부표2-2는 별첨2 [표 12-1] 참조"))

    def test_ambiguous_targets_never_use_unique_redirector_heuristic(self):
        elements = [
            {"element_id": "e1", "contract_scope": "특약A"},
            {"element_id": "e2", "contract_scope": "특약A"},
            {"element_id": "e3", "contract_scope": "특약A"},
        ]
        tags = [{"element_id": row["element_id"], "document_key": "doc1",
                 "contract_key": "특약A"} for row in elements]
        jo_rows = [
            {"element_id": "j1", "members": ["e1"], "text": "<부표2-1> 참조"},
            {"element_id": "j2", "members": ["e2"],
             "text": "부표2-1\n지급기준표\n별첨2 [표 1] 참조"},
            {"element_id": "j3", "members": ["e3"], "text": "부표2-1\n다른 지급기준표"},
        ]
        aliases, _, _, graph = build_reference_aliases(elements, tags, jo_rows)
        self.assertNotIn("j2", aliases)
        self.assertNotIn("j3", aliases)
        self.assertTrue(any(row["source"] == "j1" for row in graph["ambiguous"]))

    def test_table_catalog_preserves_cross_jo_region_until_next_heading(self):
        elements = [
            {"element_id": "e1", "text": "[별첨2] 표 24. 1~5종 수술분류표"},
            {"element_id": "e2", "text": "| 수술명 | 수술종류 |"},
            {"element_id": "e3", "text": "## II. 악성신생물 치료 목적의 수술"},
            {"element_id": "e4", "text": "표26-1 | 재해골절 분류표"},
            {"element_id": "e5", "text": "| 골절 | S02 |"},
        ]
        tags = [{"element_id": row["element_id"], "document_key": "doc1"}
                for row in elements]
        member_to_jo = {"e1": "j1", "e2": "j1", "e3": "j2",
                        "e4": "j3", "e5": "j3"}
        catalog, jo_documents = build_table_catalog(elements, tags, member_to_jo)
        surgery = next(row for row in catalog if row["key"] == "별첨:2:표:24")
        self.assertEqual(["j1", "j2"], [row["jo"] for row in surgery["variants"]])
        self.assertEqual("e3", surgery["end_element"])
        fracture = next(row for row in catalog if row["key"] == "별첨:2:표:26-1")
        self.assertEqual(["j3"], [row["jo"] for row in fracture["variants"]])
        self.assertEqual({"j1": "doc1", "j2": "doc1", "j3": "doc1"}, jo_documents)

    def test_every_literal_table_heading_closes_previous_region(self):
        elements = [
            {"element_id": "e1", "text": "[별첨2] 표 24. 1~5종 수술분류표\n수술 항목 A"},
            {"element_id": "e2", "text": "표25. 보험가입 안내\n가입 문구"},
            {"element_id": "e3", "text": "표26. 골절 분류표\n골절 S02"},
        ]
        tags = [{"element_id": row["element_id"], "document_key": "doc1"}
                for row in elements]
        catalog, _ = build_table_catalog(
            elements, tags, {"e1": "j1", "e2": "j2", "e3": "j3"})
        surgery = next(row for row in catalog if row["key"] == "별첨:2:표:24")
        self.assertEqual("e1", surgery["end_element"])
        self.assertNotIn("가입 문구", surgery["variants"][0]["preview"])

    def test_same_element_next_heading_is_line_level_boundary(self):
        elements = [{"element_id": "e1", "text": (
            "[별첨2] 표 10-1. 고액암 분류표\nC40~C41\n"
            "표10-2. 일반암 분류표\nC00~C97") }]
        tags = [{"element_id": "e1", "document_key": "doc1"}]
        catalog, _ = build_table_catalog(elements, tags, {"e1": "j1"})
        high = next(row for row in catalog if row["key"] == "별첨:2:표:10-1")
        self.assertIn("C40~C41", high["variants"][0]["preview"])
        self.assertNotIn("C00~C97", high["variants"][0]["preview"])

    def test_noncontiguous_duplicate_title_fails_closed(self):
        elements = [
            {"element_id": "e1", "text": "[별첨2] 표 1. 암 분류표\nC00"},
            {"element_id": "e2", "text": "표2. 재해 분류표\nS00"},
            {"element_id": "e3", "text": "표3. 암 분류표\nC01"},
        ]
        tags = [{"element_id": row["element_id"], "document_key": "doc1"}
                for row in elements]
        catalog, _ = build_table_catalog(
            elements, tags, {"e1": "j1", "e2": "j2", "e3": "j3"})
        self.assertFalse(any(row["title_surface"] == "암분류표" for row in catalog))


if __name__ == "__main__":
    unittest.main()
