#!/usr/bin/env python3
import unittest

from agent_tools import group_items_by_jo, preserve_baseline_jo_prefix


class GroupedCandidateTest(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {"id": "e1", "jo": "j1", "preview": "first", "src": "tag"},
            {"id": "e2", "jo": "j2", "preview": "second"},
            {"id": "c1", "jo": "j1", "preview": "decisive", "src": "meta"},
            {"id": "e3", "jo": "", "preview": "unmapped"},
        ]

    def test_all_members_previews_and_raw_ranks_are_preserved(self):
        grouped = group_items_by_jo(self.raw)
        flattened = sorted(
            (variant for group in grouped for variant in group["evidence_variants"]),
            key=lambda item: item["raw_rank"])
        self.assertEqual([item["id"] for item in self.raw], [item["id"] for item in flattened])
        self.assertEqual([1, 2, 3, 4], [item["raw_rank"] for item in flattened])
        self.assertEqual([item["preview"] for item in self.raw],
                         [item["preview"] for item in flattened])
        self.assertEqual([item.get("score", 0) for item in self.raw],
                         [item["score"] for item in flattened])
        self.assertEqual("tag", flattened[0]["src"])
        self.assertEqual("j1", grouped[0]["id"])
        self.assertEqual(["j1", "j2", ""], [item["jo"] for item in grouped])
        self.assertNotIn("preview", grouped[0])

    def test_cap_is_explicit_ablation_only(self):
        grouped = group_items_by_jo(self.raw, max_evidence=1)
        self.assertEqual(1, len(grouped[0]["evidence_variants"]))

    def test_preserve_complete_raw_prefix_for_unique_jo_gate(self):
        def pair(eid):
            return ({"element_id": eid}, 1.0)
        baseline = [pair("e1"), pair("e2"), pair("e3"), pair("e4")]
        challenger = [pair("e1"), pair("x1"), pair("e2"), pair("e3"), pair("e4")]
        mapping = {"e1": "j1", "e2": "j1", "e3": "j2", "e4": "j3", "x1": "jx"}
        merged = preserve_baseline_jo_prefix(baseline, challenger, mapping, 2)
        self.assertEqual(["e1", "e2", "e3"],
                         [element["element_id"] for element, _ in merged[:3]])


if __name__ == "__main__":
    unittest.main()
