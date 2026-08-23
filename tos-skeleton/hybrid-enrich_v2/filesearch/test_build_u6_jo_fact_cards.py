import unittest

from build_u6_jo_fact_cards import anchor_blocks, build


def element(eid, c0, text):
    return {"element_id": eid, "element_type": "paragraph", "title": "",
            "text": text, "char_start": c0, "char_end": c0 + len(text),
            "line_start": c0, "line_end": c0, "contract_scope": "계약A"}


def tag(eid, anchors=None, roles=None):
    return {"element_id": eid, "schema_version": "v", "schema_tag": "paragraph",
            "contract_key": "계약A", "subject_key": [], "role": roles or [],
            "locator": {}, "qualifier": [], "reference": [], "search_text": "",
            "document_key": "doc:1", "evidence_anchor": anchors or []}


class BuildFactCardTest(unittest.TestCase):
    def test_bundles_fragmented_list_into_one_parent_card(self):
        elements = [element("e00000", 0, "정의"), element("e00001", 10, "수술"),
                    element("e00002", 20, "연간 1회")]
        tags = [tag("e00000", ["주요치료 정의"], ["definition"]),
                tag("e00001", ["1. 수술"]), tag("e00002", ["2. 연간 1회"], ["limit_frequency"])]
        jo = [{"element_id": "j00000", "title": "지급사유", "text": "",
               "char_start": 0, "char_end": 30, "contract_scope": "계약A",
               "members": ["e00000", "e00001", "e00002"]}]
        out_e, out_t, out_j, count = build(elements, tags, jo)
        self.assertEqual(1, count)
        self.assertEqual(["e00000", "e00001", "e00002", "e00003"], out_j[0]["members"])
        self.assertEqual("j00000", out_t[-1]["parent_jo"])
        self.assertIn("1. 수술", out_t[-1]["evidence_anchor"][0])
        self.assertIn("2. 연간 1회", out_e[-1]["text"])
        self.assertEqual(["e00000", "e00001", "e00002"], out_e[-1]["source_element_ids"])
        self.assertEqual("j00000", out_e[-1]["parent_jo"])

    def test_does_not_duplicate_atomic_element(self):
        elements = [element("e00000", 0, "한 문장")]
        tags = [tag("e00000", ["한 문장"])]
        jo = [{"element_id": "j00000", "title": "", "text": "한 문장",
               "char_start": 0, "char_end": 4, "contract_scope": "계약A",
               "members": ["e00000"]}]
        out_e, out_t, out_j, count = build(elements, tags, jo)
        self.assertEqual(0, count)
        self.assertEqual(elements, out_e)
        self.assertEqual(tags, out_t)

    def test_soft_budget_never_truncates_source_anchor(self):
        tags = {"e00000": tag("e00000", ["A" * 20]),
                "e00001": tag("e00001", ["B" * 20])}
        blocks = anchor_blocks(["e00000", "e00001"], tags, max_chars=10)
        self.assertEqual([(["e00000"], ["A" * 20]), (["e00001"], ["B" * 20])], blocks)


if __name__ == "__main__":
    unittest.main()
