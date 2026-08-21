import json
import tempfile
import unittest
from pathlib import Path

import agent_tools as vector_tools


HERE = Path(__file__).resolve().parent


class ExperimentArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))

    def test_cache_fingerprint_depends_on_content_not_mtime(self):
        for tools in (vector_tools,):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "dependency.txt"
                path.write_text("first", encoding="utf-8")
                before, _ = tools.dependency_fingerprint([path])
                original_mtime = path.stat().st_mtime
                path.write_text("second", encoding="utf-8")
                path.touch()
                # mtime의 구체값과 무관하게 내용 변경은 반드시 cache key를 바꾼다.
                self.assertNotEqual(before, tools.dependency_fingerprint([path])[0])
                self.assertIsInstance(original_mtime, float)

    def test_c26_fact_arm_changes_only_tag_ranking_bundle(self):
        base = self.arms["c26_scope_base_rank_view_hybrid"]
        candidate = self.arms["c26_scope_fact_tag_ensemble_hybrid"]
        for field in ("elements", "jo", "ranker", "profile", "router", "facet",
                      "meta", "meta_view", "search_only", "fallback", "score_view"):
            self.assertEqual(base[field], candidate[field])
        self.assertEqual("tags_u4_fact_rules.jsonl", candidate["tags"])
        self.assertEqual("quota", candidate["tag_ensemble"])
        self.assertEqual(5, candidate["preserve_jo_top"])

    def test_unique_jo_quota_is_deterministic_and_jo_diverse(self):
        def item(eid):
            return ({"element_id": eid}, 1.0)
        rankings = [
            [item("e1"), item("e2"), item("e3")],
            [item("e4"), item("e5"), item("e6")],
        ]
        member_to_jo = {"e1": "j1", "e2": "j1", "e3": "j2",
                        "e4": "j3", "e5": "j4", "e6": "j4"}
        ranked = vector_tools.unique_jo_quota(rankings, member_to_jo, [2, 1], limit=4)
        self.assertEqual(["e1", "e3", "e4", "e5"],
                         [element["element_id"] for element, _ in ranked])
        self.assertEqual(4, len({member_to_jo[element["element_id"]]
                                for element, _ in ranked}))

    def test_c26_preserves_complete_prefix_through_fifth_unique_jo(self):
        def item(eid):
            return ({"element_id": eid}, 1.0)
        baseline = [item(eid) for eid in ("e1", "e2", "e3", "e4", "e5", "e6", "e7")]
        challenger = [item(eid) for eid in ("x1", "e7", "e1", "e2", "e3")]
        mapping = {"e1": "j1", "e2": "j1", "e3": "j2", "e4": "j3",
                   "e5": "j4", "e6": "j5", "e7": "j6", "x1": "jx"}
        ranked = vector_tools.preserve_baseline_jo_prefix(
            baseline, challenger, mapping, n_jo=5)
        self.assertEqual(["e1", "e2", "e3", "e4", "e5", "e6"],
                         [element["element_id"] for element, _ in ranked[:6]])
        self.assertEqual("x1", ranked[6][0]["element_id"])

    def test_c27_differs_from_c26_only_by_safety_prefix(self):
        c26 = dict(self.arms["c26_scope_fact_tag_ensemble_hybrid"])
        c27 = dict(self.arms["c27_scope_fact_tag_safe10_hybrid"])
        self.assertEqual(5, c26.pop("preserve_jo_top"))
        self.assertEqual(10, c27.pop("preserve_jo_top"))
        self.assertEqual(c26, c27)

    def test_c28_adds_only_the_evidence_unit_quota_ranker(self):
        c26 = dict(self.arms["c26_scope_fact_tag_ensemble_hybrid"])
        c28 = dict(self.arms["c28_scope_fact_unit_ensemble_hybrid"])
        self.assertEqual(3, c28.pop("ensemble_unit_quota"))
        self.assertEqual("membership_or_code", c28.pop("ensemble_unit_gate"))
        unit_weights = c28.pop("ensemble_unit_sfw")
        self.assertEqual(c26, c28)
        self.assertEqual(1.0, unit_weights["evidence_unit"])
        self.assertEqual(8, unit_weights["_evidence_unit_rare"])
        self.assertTrue(all(value == 0.0 for key, value in unit_weights.items()
                            if key not in {"evidence_unit", "_evidence_unit_rare"}))

    def test_evidence_unit_gate_is_generic_and_conditional(self):
        self.assertTrue(vector_tools.evidence_unit_gate(
            "이 질병도 보장 범위에 포함되나요?", "membership_or_code"))
        self.assertTrue(vector_tools.evidence_unit_gate(
            "I20.90 질병코드가 해당하나요?", "membership_or_code"))
        self.assertFalse(vector_tools.evidence_unit_gate(
            "안내자료와 약관이 다르면 무엇이 우선하나요?", "membership_or_code"))

    def test_membership_support_plan_uses_direct_identity_surface(self):
        class Index:
            fields = [
                {"identity_full": "[기본]소액암진단특약(갱신형)"},
                {"identity_full": "[소액암진단형]납입보조특약"},
                {"identity_full": "다른특약"},
            ]

        class Search:
            def ensure_structured(self):
                return Index()

        plan = vector_tools.membership_support_plan(
            Search(), "상품에 소액암이 자궁경부 제자리암도 포함되나요?")
        self.assertEqual("소액암", plan["category"])
        self.assertEqual("자궁경부 제자리암", plan["member"])
        self.assertEqual(["[기본]소액암진단특약(갱신형)"], plan["slots"]["identity"])

    def test_membership_support_plan_does_not_fire_for_unrelated_question(self):
        class Search:
            def ensure_structured(self):
                raise AssertionError("identity index should not be read")

        self.assertIsNone(vector_tools.membership_support_plan(
            Search(), "수술급여금은 연간 몇 회인가요?"))


if __name__ == "__main__":
    unittest.main()
