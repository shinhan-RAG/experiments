import copy
import json
import tempfile
import unittest
from pathlib import Path

import agent_tools as vector_tools
import host_agent_runner


HERE = Path(__file__).resolve().parent


class ExperimentArmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))

    def test_host_runner_hybrid_search_provenance_path_exists(self):
        self.assertTrue((host_agent_runner.VS / "hybrid_search.py").is_file())

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

    def test_c37_changes_only_semantic_fact_card_dataset(self):
        baseline = dict(self.arms["c29_membership_dual_evidence_hybrid"])
        candidate = dict(self.arms["c37_jo_fact_card_hybrid"])
        self.assertEqual("elements_u3.jsonl", baseline.pop("elements"))
        self.assertEqual("tags_u4_fact_rules.jsonl", baseline.pop("tags"))
        self.assertEqual("elements_u3jo.jsonl", baseline.pop("jo"))
        self.assertEqual("elements_u6_fact_cards.jsonl", candidate.pop("elements"))
        self.assertEqual("tags_u6_fact_cards.jsonl", candidate.pop("tags"))
        self.assertEqual("elements_u6_fact_cards_jo.jsonl", candidate.pop("jo"))
        self.assertEqual(baseline, candidate)

    def test_c38_keeps_c29_and_adds_only_fact_card_overlay(self):
        baseline = dict(self.arms["c29_membership_dual_evidence_hybrid"])
        candidate = dict(self.arms["c38_jo_fact_card_additive_hybrid"])
        overlay = candidate.pop("fact_card_overlay")
        self.assertEqual(baseline, candidate)
        self.assertEqual("elements_u6_fact_cards_only.jsonl", overlay["elements"])
        self.assertEqual("tags_u6_fact_cards_only.jsonl", overlay["tags"])
        self.assertEqual(5, overlay["limit"])

    def test_fact_card_surface_gate_requires_concrete_support(self):
        rejected = vector_tools.fact_card_query_support(
            ["소액암이", "자궁경부", "제자리암도", "포함되나요"],
            "다른 암 특약의 지급 한도와 보장 조건")
        self.assertFalse(rejected["allowed"])
        accepted = vector_tools.fact_card_query_support(
            ["소액암이", "자궁경부", "제자리암도", "포함되나요"],
            "소액암에서 자궁경부 제자리암 D06을 포함")
        self.assertTrue(accepted["allowed"])
        strong = vector_tools.fact_card_query_support(
            ["사고증명서", "알려줘"], "제출할 사고증명서")
        self.assertTrue(strong["allowed"])

    def test_c39_differs_from_c38_only_by_fail_closed_surface_gate(self):
        ungated = dict(self.arms["c38_jo_fact_card_additive_hybrid"])
        gated = dict(self.arms["c39_jo_fact_card_surface_gated_hybrid"])
        ungated_overlay = dict(ungated.pop("fact_card_overlay"))
        gated_overlay = dict(gated.pop("fact_card_overlay"))
        self.assertEqual(ungated, gated)
        self.assertFalse(ungated_overlay.pop("surface_gate"))
        self.assertTrue(gated_overlay.pop("surface_gate"))
        self.assertEqual(2, gated_overlay.pop("min_token_matches"))
        self.assertEqual(5, gated_overlay.pop("strong_token_chars"))
        self.assertEqual(ungated_overlay, gated_overlay)

    def test_c40_differs_from_c29_only_by_nested_reference_bundle(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(self.arms["c40_nested_reference_bundle_hybrid"])
        bundle = challenger.pop("reference_bundle")
        self.assertEqual(base, challenger)
        self.assertEqual("tags_u5_reference_graph_stats.json", bundle["stats"])
        self.assertEqual("membership_or_code", bundle["gate"])
        self.assertEqual(2, bundle["max_hops"])

    def test_nested_reference_bundle_keeps_top_level_order(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "e2", "jo": "j2"}]
        before = copy.deepcopy(items)
        paths = [{"source": "j1", "target": "j3", "keys": ["부표:1"], "hops": 1}]
        jo_index = {"j3": {"element_id": "j3", "contract_scope": "특약",
                           "title": "부표", "text": "질병 코드"}}
        linked = vector_tools.reference_bundles_for_items(
            items, paths, jo_index, ["질병"], lambda text, _: text[:20])
        self.assertEqual(["e1", "e2"], [row["id"] for row in items])
        self.assertEqual(before[1], items[1])
        self.assertEqual("j3", items[0]["linked_evidence"][0]["id"])
        self.assertEqual(1, len(linked))

    def test_reference_bundle_gate_is_fail_closed(self):
        self.assertTrue(vector_tools.reference_bundle_enabled("허혈심장질환 분류코드는?"))
        self.assertTrue(vector_tools.reference_bundle_enabled("D06도 포함되나요?"))
        self.assertFalse(vector_tools.reference_bundle_enabled("보험금은 얼마인가요?"))

    def test_reference_projection_deduplicates_and_requires_query_support(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "e2", "jo": "j2"}]
        paths = [
            {"source": "j1", "target": "j3", "keys": ["부표:2"], "hops": 1},
            {"source": "j2", "target": "j3", "keys": ["부표:2"], "hops": 1},
            {"source": "j1", "target": "j4", "keys": ["별첨:1"], "hops": 1},
        ]
        jo_index = {
            "j3": {"element_id": "j3", "text": "허혈심장질환 분류표 I20 I25"},
            "j4": {"element_id": "j4", "text": "보장 대상이 되는 재해분류표"},
        }
        projected = vector_tools.reference_projection_for_items(
            items, paths, jo_index,
            ["허혈심장질환의", "질병분류코드는", "무엇인가요"],
            raw_query="허혈심장질환 질병분류코드는?", limit=2)
        self.assertEqual(["j3"], [row["jo"] for row in projected])
        self.assertEqual([1, 2], projected[0]["source_ranks"])
        self.assertEqual(["j1", "j2"], projected[0]["source_jos"])

    def test_reference_projection_does_not_repeat_existing_target(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "e3", "jo": "j3"}]
        paths = [{"source": "j1", "target": "j3", "keys": ["부표:2"], "hops": 1}]
        jo_index = {"j3": {"element_id": "j3", "text": "허혈심장질환 분류표"}}
        self.assertEqual([], vector_tools.reference_projection_for_items(
            items, paths, jo_index, ["허혈심장질환"],
            raw_query="허혈심장질환 코드는?", limit=2))

    def test_reference_projection_gate_rejects_action_membership(self):
        self.assertTrue(vector_tools.reference_projection_enabled(
            "뇌경색도 뇌혈관질환에 해당되나요?"))
        self.assertTrue(vector_tools.reference_projection_enabled(
            "심장질환 수술코드는 무엇인가요?"))
        self.assertFalse(vector_tools.reference_projection_enabled(
            "주요치료에는 어떤 치료가 포함되나요?"))

    def test_reference_projection_requires_explicit_quoted_anchor(self):
        items = [{"id": "e1", "jo": "j1"}]
        paths = [{"source": "j1", "target": "j2", "keys": ["표:3"], "hops": 1}]
        jo_index = {"j2": {"element_id": "j2", "text": "다른 심장질환 분류표"}}
        self.assertEqual([], vector_tools.reference_projection_for_items(
            items, paths, jo_index, ["심장질환", "분류표"],
            raw_query="'본인일부부담금 산정특례 심장질환의 수술' 분류표", limit=2))

    def test_c44_differs_from_c29_only_by_reference_projection(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(self.arms["c44_reference_projection_hybrid"])
        projection = challenger.pop("reference_projection")
        self.assertEqual(base, challenger)
        self.assertEqual("tags_u5_reference_graph_stats.json", projection["stats"])
        self.assertEqual("classification_membership", projection["gate"])
        self.assertEqual(2, projection["limit"])
        self.assertEqual(2, projection["max_hops"])

    def test_scenario_intent_is_activity_agnostic_and_fail_closed(self):
        diving = vector_tools.scenario_intent_query(
            "스쿠버다이빙으로 인한 후유장해도 보상 돼?")
        hiking = vector_tools.scenario_intent_query(
            "등산하다가 생긴 후유장해도 보장되나요?")
        self.assertEqual("accident_coverage", diving["intent"])
        self.assertEqual(diving["query"], hiking["query"])
        self.assertNotIn("스쿠버", diving["query"])
        self.assertIsNone(vector_tools.scenario_intent_query("수술급여금은 얼마인가요?"))
        self.assertIsNone(vector_tools.scenario_intent_query(
            "암진단으로 급여금을 받은 뒤 다른 진단도 보장 가능한가요?"))

    def test_scenario_intent_normalizes_unseen_colloquial_forms(self):
        injured = vector_tools.scenario_intent_query(
            "취미 활동 중에 다쳐도 보험금 받을 수 있어?")
        fracture = vector_tools.scenario_intent_query(
            "운동하다가 골절됐는데 보장되나요?")
        accident = vector_tools.scenario_intent_query(
            "여행 중 사고가 나면 보험 돼?")
        self.assertEqual("accident_coverage", injured["intent"])
        self.assertIn("상해", injured["query"])
        self.assertIn("골절", fracture["query"])
        self.assertIn("재해", accident["query"])
        self.assertNotIn("취미", injured["query"])
        self.assertNotIn("운동", fracture["query"])
        self.assertIsNone(vector_tools.scenario_intent_query(
            "여행 중 병원에 가면 보험료는 얼마야?"))
        self.assertIsNone(vector_tools.scenario_intent_query(
            "계약 실효 상태에서 유지중 발생한 사고도 보장 가능해?"))

    def test_c41_differs_from_c29_only_by_scenario_intent_bundle(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(self.arms["c41_scenario_intent_bundle_hybrid"])
        bundle = challenger.pop("intent_bundle")
        self.assertEqual(base, challenger)
        self.assertEqual(1, bundle["limit"])
        self.assertEqual("core", bundle["profile"])

    def test_c42_differs_from_c41_only_by_prepend_mode(self):
        evidence_only = copy.deepcopy(self.arms["c41_scenario_intent_bundle_hybrid"])
        prepended = copy.deepcopy(self.arms["c42_scenario_intent_prepend_hybrid"])
        self.assertNotIn("mode", evidence_only["intent_bundle"])
        self.assertEqual("prepend_dedup", prepended["intent_bundle"].pop("mode"))
        self.assertEqual(evidence_only, prepended)

    def test_claim_roles_require_multiple_independent_roles(self):
        roles = vector_tools.requested_claim_roles(
            "지급 요건과 연간 몇 회인지 알려줘", {}, limit=3)
        self.assertIn("payment_trigger", roles)
        self.assertIn("limit_frequency", roles)
        self.assertEqual([], vector_tools.requested_claim_roles(
            "보험금 지급 요건은?", {}, limit=3))

    def test_c43_differs_from_c29_only_by_claim_bundle(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(self.arms["c43_claim_role_bundle_hybrid"])
        bundle = challenger.pop("claim_bundle")
        self.assertEqual(base, challenger)
        self.assertEqual(3, bundle["max_roles"])
        self.assertEqual(2, bundle["per_role"])

    def test_c45_differs_from_c29_only_by_strict_explicit_bracket(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(self.arms["c45_strict_explicit_bracket_hybrid"])
        self.assertTrue(challenger.pop("strict_explicit_bracket"))
        self.assertEqual(base, challenger)

    def test_explicit_contract_scopes_rejects_inferred_or_contextual_contracts(self):
        contracts = [
            "(간편)[기본]일반암진단특약(무배당, 갱신형)",
            "(간편)[삭감없음용]일반암진단특약(무배당, 갱신형)",
            "다른특약(무배당, 갱신형)",
        ]
        explicit = vector_tools.explicit_contract_scopes(
            "(간편)[기본]일반암진단특약의 지급 요건과 횟수는?",
            {"contract": contracts[:1]})
        self.assertEqual(contracts[:1], explicit)
        self.assertEqual([], vector_tools.explicit_contract_scopes(
            "암진단급여금의 지급 요건과 횟수는?", {"contract": contracts}))

    def test_claim_role_injection_preserves_prefix_and_moves_without_duplicates(self):
        items = [{"id": f"e{i}", "jo": f"j{i}"} for i in range(1, 9)]
        groups = [
            {"role": "payment_trigger", "items": [{"id": "j7", "jo": "j7"}]},
            {"role": "limit_frequency", "items": [{"id": "j9", "jo": "j9"}]},
        ]
        merged, promoted = vector_tools.inject_claim_role_items(
            items, groups, after=5, limit=2)
        self.assertEqual(["j1", "j2", "j3", "j4", "j5"],
                         [row["jo"] for row in merged[:5]])
        self.assertEqual(["j7", "j9"], [row["jo"] for row in promoted])
        self.assertEqual(1, sum(row["jo"] == "j7" for row in merged))

    def test_c46_adds_only_explicit_claim_slot_fusion_to_c45(self):
        base = copy.deepcopy(self.arms["c45_strict_explicit_bracket_hybrid"])
        challenger = copy.deepcopy(self.arms["c46_explicit_claim_slots_hybrid"])
        bundle = challenger.pop("claim_bundle")
        self.assertEqual(base, challenger)
        self.assertTrue(bundle["explicit_identity_only"])
        self.assertEqual(5, bundle["inject_after"])
        self.assertEqual(1, bundle["per_role"])

    def test_reference_evidence_selects_exact_table_region_without_mutating_page(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "e2", "jo": "j2"}]
        before = copy.deepcopy(items)
        paths = [{"source": "j1", "bridge": "jbridge", "target": "j3",
                  "keys": ["부표:2-2", "별첨:2:표:12-1"], "hops": 2}]
        jo_index = {
            "j3": {"element_id": "j3", "members": ["ehead", "erow", "enext"],
                   "contract_scope": "공용 별첨", "title": ""},
        }
        element_index = {
            "ehead": {"element_id": "ehead",
                      "text": "[별첨2] 표 12-1 허혈심장질환 분류표"},
            "erow": {"element_id": "erow",
                     "text": "협심증 I20 급성심근경색증 I21 만성 허혈심장병 I25"},
            "enext": {"element_id": "enext", "text": "판정 시점에 관한 주석"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index,
            ["허혈심장질환", "질병분류코드"],
            raw_query="허혈심장질환 질병분류코드는?", limit=2)
        self.assertEqual(before, items)
        self.assertEqual(["j3"], [row["jo"] for row in linked])
        self.assertEqual(["ehead", "erow", "enext"], linked[0]["region_member_ids"])
        self.assertIn("협심증 I20", linked[0]["preview"])

    def test_reference_evidence_can_explain_an_existing_deep_result(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "erow", "jo": "j3"}]
        paths = [{"source": "j1", "target": "j3",
                  "keys": ["별첨:2:표:12-1"], "hops": 1}]
        jo_index = {"j3": {"element_id": "j3", "members": ["ehead", "erow"]}}
        element_index = {
            "ehead": {"element_id": "ehead", "text": "표12-1 허혈심장질환 분류표"},
            "erow": {"element_id": "erow", "text": "협심증 I20 만성 허혈심장병 I25"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index, ["허혈심장질환", "코드"],
            raw_query="허혈심장질환 코드는?", limit=1)
        self.assertEqual("j3", linked[0]["jo"])
        self.assertEqual(2, linked[0]["base_result_rank"])

    def test_reference_evidence_rejects_heading_without_requested_code_row(self):
        items = [{"id": "e1", "jo": "j1"}]
        paths = [{"source": "j1", "target": "j2",
                  "keys": ["별첨:2:표:12-1"], "hops": 1}]
        jo_index = {"j2": {"element_id": "j2", "members": ["ehead"]}}
        element_index = {"ehead": {"element_id": "ehead",
                                    "text": "[별첨2] 표12-1 허혈심장질환 분류표"}}
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index, ["허혈심장질환", "코드"],
            raw_query="허혈심장질환 코드는?", limit=2)
        self.assertEqual([], linked)

    def test_reference_evidence_crosses_ocr_noise_to_reach_code_row(self):
        items = [{"id": "e1", "jo": "j1"}, {"id": "ecode", "jo": "j2"}]
        before = copy.deepcopy(items)
        paths = [{"source": "j1", "target": "j2",
                  "keys": ["별첨:2:표:42"], "hops": 1}]
        jo_index = {"j2": {"element_id": "j2",
                            "members": ["elabel", "ehead", "enoise", "edef", "ecode"]}}
        element_index = {
            "elabel": {"element_id": "elabel", "text": "[별첨2] 표39\n표42"},
            "ehead": {"element_id": "ehead", "text": "대상포진 분류표"},
            "enoise": {"element_id": "enoise", "text": "unrelated OCR extraction noise"},
            "edef": {"element_id": "edef", "text": "대상포진으로 분류되는 대상질병"},
            "ecode": {"element_id": "ecode", "text": "대상포진 B02"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index, ["대상포진", "질병분류코드"],
            raw_query="대상포진 질병분류 코드는?", region_members=8)
        self.assertEqual(before, items)
        self.assertEqual("j2", linked[0]["jo"])
        self.assertIn("ecode", linked[0]["region_member_ids"])
        self.assertIn("B02", linked[0]["preview"])

    def test_strict_reference_region_stops_before_mixed_next_label(self):
        items = [{"id": "esource", "jo": "jsource"}]
        paths = [{"source": "jsource", "target": "jtarget",
                  "keys": ["별첨:2:표:39"], "hops": 1}]
        jo_index = {
            "jtarget": {"element_id": "jtarget",
                        "members": ["e39", "erow", "enote", "emixed", "e42"],
                        "title": "대상포진 분류표"},
        }
        element_index = {
            "e39": {"element_id": "e39", "text": "표39 특정 류마티스 관절염 분류표"},
            "erow": {"element_id": "erow", "text": "류마티스관절염 M05 M06"},
            "enote": {"element_id": "enote", "text": "분류코드 판정 주석"},
            "emixed": {"element_id": "emixed", "text": "표39 반복\n표42"},
            "e42": {"element_id": "e42", "text": "대상포진 B02"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index, ["류마티스", "분류코드"],
            raw_query="류마티스 분류코드는?", region_members=8,
            stop_on_mixed_labels=True, sanitize_region_title=True)
        self.assertEqual(["e39", "erow", "enote"], linked[0]["region_member_ids"])
        self.assertNotIn("대상포진", linked[0]["preview"])
        self.assertEqual("참조표 39", linked[0]["jo_title"])

    def test_reference_evidence_surfaces_direct_existing_table_region(self):
        items = [{"id": "edeep", "jo": "j2"}]
        before = copy.deepcopy(items)
        jo_index = {"j2": {"element_id": "j2",
                            "members": ["elabel", "enoise", "edef", "etable"]}}
        element_index = {
            "elabel": {"element_id": "elabel", "text": "표6-1 제자리의 신생물 분류표"},
            "enoise": {"element_id": "enoise", "text": "OCR noise"},
            "edef": {"element_id": "edef", "text": "제자리암 보장대상 질병"},
            "etable": {"element_id": "etable", "text": "분류코드 D00 D01 D06 D09"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, [], jo_index, element_index, ["제자리암", "보장대상", "분류코드"],
            raw_query="제자리암의 보장대상 질병분류 코드는?", region_members=8)
        self.assertEqual(before, items)
        self.assertEqual("j2", linked[0]["jo"])
        self.assertEqual(0, linked[0]["hops"])
        self.assertEqual("existing_hybrid_region", linked[0]["provenance"])
        self.assertEqual(1, linked[0]["base_result_rank"])

    def test_reference_evidence_does_not_direct_scan_non_reference_question(self):
        items = [{"id": "e1", "jo": "j1"}]
        jo_index = {"j1": {"element_id": "j1", "members": ["etable"]}}
        element_index = {"etable": {"element_id": "etable",
                                      "text": "표1 수술급여금 연간 1회 C00"}}
        linked = vector_tools.reference_evidence_for_items(
            items, [], jo_index, element_index, ["수술급여금", "연간", "1회"],
            raw_query="수술급여금은 연간 몇 회인가요?", region_members=8)
        self.assertEqual([], linked)

    def test_reference_evidence_rejects_longer_disease_name_as_direct_match(self):
        items = [{"id": "e1", "jo": "j1"}]
        jo_index = {"j1": {"element_id": "j1", "members": ["etable"]}}
        element_index = {"etable": {"element_id": "etable",
                                      "text": "표127 대상포진수막염 B02.1"}}
        linked = vector_tools.reference_evidence_for_items(
            items, [], jo_index, element_index, ["대상포진", "질병분류", "코드"],
            raw_query="대상포진 질병분류 코드는?", region_members=8)
        self.assertEqual([], linked)

    def test_reference_graph_region_precedes_direct_existing_region(self):
        items = [{"id": "edirect", "jo": "jdirect"},
                 {"id": "esource", "jo": "jsource"}]
        paths = [{"source": "jsource", "target": "jgraph",
                  "keys": ["별첨:2:표:42"], "hops": 1}]
        jo_index = {
            "jdirect": {"element_id": "jdirect", "members": ["edirect"]},
            "jgraph": {"element_id": "jgraph", "members": ["egraph"]},
        }
        element_index = {
            "edirect": {"element_id": "edirect", "text": "표127 대상포진 B02.1"},
            "egraph": {"element_id": "egraph", "text": "표42 대상포진 B02"},
        }
        linked = vector_tools.reference_evidence_for_items(
            items, paths, jo_index, element_index, ["대상포진", "코드"],
            raw_query="대상포진 코드는?", region_members=8)
        self.assertEqual("jgraph", linked[0]["jo"])
        self.assertEqual("reference_graph", linked[0]["provenance"])

    def test_reference_evidence_gate_accepts_generic_scenario(self):
        self.assertTrue(vector_tools.reference_evidence_enabled(
            "스쿠버다이빙으로 인한 후유장해도 보상 돼?"))
        self.assertTrue(vector_tools.reference_evidence_enabled(
            "허혈심장질환 분류코드는?"))
        self.assertFalse(vector_tools.reference_evidence_enabled(
            "수술급여금은 얼마인가요?"))

    def test_exact_table_catalog_is_document_scoped_and_preserves_all_jos(self):
        catalog = [
            {"doc": "doc1", "key": "별첨:2:표:24", "title": "1~5종 수술분류표",
             "title_surface": "15종수술분류표", "variants": [
                 {"jo": "j1", "element_ids": ["e1"], "preview": "일반 수술"},
                 {"jo": "j2", "element_ids": ["e2"], "preview": "악성신생물 수술"}]},
            {"doc": "doc2", "key": "별첨:2:표:24", "title": "1~5종 수술분류표",
             "title_surface": "15종수술분류표", "variants": [
                 {"jo": "j9", "element_ids": ["e9"], "preview": "다른 문서"}]},
        ]
        self.assertEqual([], vector_tools.table_catalog_evidence(
            "통합건강원 1~5종 수술분류표", catalog, ["doc1", "doc2"]))
        items = vector_tools.table_catalog_evidence(
            "통합건강원 1~5종 수술분류표", catalog, ["doc1"])
        self.assertEqual(["j1", "j2"], [row["jo"] for row in items])
        self.assertTrue(all(row["source_document"] == "doc1" for row in items))
        self.assertEqual([], vector_tools.table_catalog_evidence(
            "수술 급여금은 얼마야?", catalog, ["doc1"]))

    def test_exact_table_catalog_fails_closed_on_variant_overflow(self):
        catalog = [{"doc": "doc1", "key": "별첨:2:표:10-1",
                    "title": "고액암 분류표", "title_surface": "고액암분류표",
                    "variants": [
                        {"jo": f"j{i}", "element_ids": [f"e{i}"], "preview": "근거"}
                        for i in range(1, 7)]}]
        self.assertEqual([], vector_tools.table_catalog_evidence(
            "고액암 분류표", catalog, ["doc1"], max_variants=5))

    def test_c47_adds_only_non_reranking_reference_evidence_to_c45(self):
        base = copy.deepcopy(self.arms["c45_strict_explicit_bracket_hybrid"])
        challenger = copy.deepcopy(self.arms["c47_reference_region_hybrid"])
        evidence = challenger.pop("reference_evidence")
        self.assertEqual(base, challenger)
        self.assertEqual("reference_or_scenario", evidence["gate"])
        self.assertEqual(40, evidence["max_source_rank"])
        self.assertEqual(2, evidence["max_hops"])
        self.assertEqual(2, evidence["limit"])
        self.assertEqual(8, evidence["region_members"])

    def test_c48_adds_only_strict_region_boundary_and_title_to_c47(self):
        base = copy.deepcopy(self.arms["c47_reference_region_hybrid"])
        challenger = copy.deepcopy(self.arms["c48_strict_reference_region_hybrid"])
        base_evidence = base.pop("reference_evidence")
        strict_evidence = challenger.pop("reference_evidence")
        self.assertEqual(base, challenger)
        self.assertTrue(strict_evidence.pop("stop_on_mixed_labels"))
        self.assertTrue(strict_evidence.pop("sanitize_region_title"))
        self.assertEqual(base_evidence, strict_evidence)

    def test_c49_differs_from_c48_only_by_precise_graph_artifact(self):
        base = copy.deepcopy(self.arms["c48_strict_reference_region_hybrid"])
        challenger = copy.deepcopy(self.arms["c49_precise_reference_graph_hybrid"])
        base_stats = base["reference_evidence"].pop("stats")
        challenger_stats = challenger["reference_evidence"].pop("stats")
        self.assertEqual(base, challenger)
        self.assertEqual("tags_u5_reference_graph_stats.json", base_stats)
        self.assertEqual("tags_u5_reference_graph_v12_stats.json", challenger_stats)

    def test_c50_adds_only_colloquial_intent_bundle_to_c49(self):
        base = copy.deepcopy(self.arms["c49_precise_reference_graph_hybrid"])
        challenger = copy.deepcopy(
            self.arms["c50_precise_reference_colloquial_hybrid"])
        bundle = challenger.pop("intent_bundle")
        self.assertEqual(base, challenger)
        self.assertEqual("prepend_dedup", bundle["mode"])
        self.assertEqual(1, bundle["limit"])

    def test_c51_adds_only_exact_title_catalog_to_c50(self):
        base = copy.deepcopy(
            self.arms["c50_precise_reference_colloquial_hybrid"])
        challenger = copy.deepcopy(
            self.arms["c51_exact_table_catalog_hybrid"])
        base_stats = base["reference_evidence"].pop("stats")
        challenger_stats = challenger["reference_evidence"].pop("stats")
        self.assertTrue(challenger["reference_evidence"].pop("catalog_title_lookup"))
        self.assertEqual(5, challenger["reference_evidence"].pop(
            "catalog_max_variants"))
        self.assertEqual(base, challenger)
        self.assertEqual("tags_u5_reference_graph_v12_stats.json", base_stats)
        self.assertEqual("tags_u5_reference_graph_v13_stats.json", challenger_stats)

    def test_c52_differs_from_c51_only_by_safe_catalog_artifact(self):
        base = copy.deepcopy(self.arms["c51_exact_table_catalog_hybrid"])
        challenger = copy.deepcopy(self.arms["c52_safe_reference_catalog_hybrid"])
        base_stats = base["reference_evidence"].pop("stats")
        challenger_stats = challenger["reference_evidence"].pop("stats")
        self.assertEqual(base, challenger)
        self.assertEqual("tags_u5_reference_graph_v13_stats.json", base_stats)
        self.assertEqual("tags_u5_reference_graph_v14_stats.json", challenger_stats)

    def test_fallback_status_distinguishes_success_from_silent_failure(self):
        source = (HERE / "agent_tools.py").read_text(encoding="utf-8")
        self.assertIn('fb_status = "ok:no_new_items"', source)
        self.assertIn('fb_status = f"ok:{fb_used}"', source)
        self.assertIn('"fallback_status": fb_status', source)
        self.assertNotIn("Path(meta_python).resolve()", source)

    # --- R1: identity alias expansion / reference follow (기본 off) ---

    def test_r1_arm_differs_from_c29_only_by_two_new_options(self):
        base = copy.deepcopy(self.arms["c29_membership_dual_evidence_hybrid"])
        challenger = copy.deepcopy(
            self.arms["r1_identity_expand_reference_follow_hybrid"])
        expand = challenger.pop("identity_expand")
        follow = challenger.pop("reference_follow")
        self.assertEqual(base, challenger)
        self.assertNotIn("identity_expand", base)
        self.assertNotIn("reference_follow", base)
        self.assertEqual(4, expand["min_variant_len"])
        self.assertEqual("tags_u5_reference_graph_v14_stats.json", follow["stats"])
        self.assertEqual(5, follow["quota"])

    def test_reference_follow_gate_is_fail_closed(self):
        self.assertTrue(vector_tools.reference_follow_enabled(
            "납입면제 대상 질병의 질병분류코드는 무엇인가요?"))
        self.assertTrue(vector_tools.reference_follow_enabled(
            "청구시 필요한 구비서류는 무엇인가요?"))
        self.assertFalse(vector_tools.reference_follow_enabled(
            "뇌혈관질환진단특약에서 뇌출혈만 보장되나요"))
        self.assertTrue(vector_tools.reference_follow_enabled("아무 질문", gate=""))
        with self.assertRaises(ValueError):
            vector_tools.reference_follow_enabled("질문", gate="unknown")

    def test_shared_segments_come_from_the_contract_name_set(self):
        cores = ["첫날부터입원", "첫날부터상급종합병원입원", "암직접치료상급종합병원통원"]
        shared = vector_tools.contract_shared_segments(cores, min_contracts=2)
        self.assertIn("상급종합병원", shared)
        self.assertIn("첫날부터", shared)
        self.assertNotIn("암직접치료", shared)
        variants = vector_tools.identity_variants("첫날부터상급종합병원입원", shared)
        self.assertIn("첫날부터입원", variants)

    def test_identity_expansion_needs_a_distinctive_alias_after_exact_route(self):
        class _Router:
            contracts = ["(간편)[기본]뇌혈관질환진단특약(무배당, 갱신형)",
                         "(간편)[기본]허혈심장질환진단특약(무배당, 갱신형)",
                         "(간편)보험료납입면제특약(무배당, 갱신형)",
                         "(간편)보험료납입보조특약(무배당, 갱신형)"]

            @staticmethod
            def _lcs(a, b):
                from clm_search import Router
                return Router._lcs(a, b)

        router = _Router()
        # 라우터가 완전 일치로 뇌혈관질환진단특약을 확정한 질문: 비변별 조각("질환진단")으로
        # 다른 특약(허혈심장질환진단특약)을 끌어오지 않는다.
        added, _ = vector_tools.identity_expansion_candidates(
            router, "뇌혈관질환진단특약에서 뇌출혈만 보장되나요",
            {"contract": ["(간편)[기본]뇌혈관질환진단특약(무배당, 갱신형)"]}, {})
        self.assertEqual([], added)
        # 라우터가 아무 특약도 못 잡은 질문: 문서에서 도출한 별칭으로 회수한다.
        added, audit = vector_tools.identity_expansion_candidates(
            router, "통합건강 납입면제 가입해야지 면제되나요", {}, {})
        self.assertEqual(["(간편)보험료납입면제특약(무배당, 갱신형)"], added)
        self.assertEqual("납입면제", audit["via"][added[0]])
        # 발화 조건이 없으면 아무것도 더하지 않는다(기존 결과 보존).
        self.assertEqual(([], {}), vector_tools.identity_expansion_candidates(
            router, "청약 철회는 언제까지 되나요", {}, {}))

    def test_reference_follow_uses_routed_contract_edges_only(self):
        class _Search:
            E = [{"element_id": "e1"}, {"element_id": "e2"}]
            _jo = [{"element_id": "j2", "members": ["e1"]},
                   {"element_id": "j3", "members": ["e2"]}]
            _eidx = {"e1": 0, "e2": 1}

        stats = {"edge_ledger": [
            {"source": "j1", "target": "j2", "hops": 1, "keys": ["표:12-1"],
             "source_contracts": ["(간편)보험료납입면제특약(무배당, 갱신형)"]},
            {"source": "j9", "target": "j3", "hops": 1, "keys": ["표:1"],
             "source_contracts": ["(간편)정기특약(무배당, 갱신형)"]},
        ]}
        ranked, audit = vector_tools.reference_follow_ranking(
            _Search(), {"contract": ["(간편)보험료납입면제특약(무배당, 갱신형)"]}, stats)
        self.assertEqual(["e1"], [element["element_id"] for element, _ in ranked])
        self.assertEqual(["j2"], [row["jo"] for row in audit["targets"]])
        # 특약이 라우팅되지 않으면 fail-closed.
        self.assertEqual(([], {}), vector_tools.reference_follow_ranking(
            _Search(), {}, stats))

    def test_r1_options_are_wired_behind_arm_flags(self):
        source = (HERE / "agent_tools.py").read_text(encoding="utf-8")
        self.assertIn('if arm.get("identity_expand"):', source)
        self.assertIn('if arm.get("reference_follow"):', source)
        self.assertIn('rankings.append(follow_ranked)', source)

    # --- R6: intent role sub-query (기본 off) ---

    def test_r6_arm_differs_from_r1v_only_by_intent_role(self):
        base = copy.deepcopy(self.arms["r1v_verify_identity_reference_hybrid"])
        challenger = copy.deepcopy(self.arms["r6_intent_role_hybrid"])
        intent = challenger.pop("intent_role")
        self.assertEqual(base, challenger)
        self.assertNotIn("intent_role", base)
        self.assertEqual({"per_role_quota": 4, "total_quota": 8, "max_roles": 2}, intent)

    def test_intent_role_gate_is_fail_closed(self):
        # 특약을 라우팅한 질문은 role 규칙에 매치해도 발화하지 않는다.
        self.assertEqual([], vector_tools.intent_role_gate(
            "보험금은 어떻게 청구하나요",
            {"contract": ["(간편)[기본]뇌혈관질환진단특약(무배당, 갱신형)"]}))
        self.assertEqual([], vector_tools.intent_role_gate(
            "보험금은 어떻게 청구하나요", {"identity": ["(간편)정기특약(무배당, 갱신형)"]}))
        # 특약이 없어도 role 규칙에 하나도 매치하지 않으면 발화하지 않는다.
        self.assertEqual([], vector_tools.intent_role_gate("암이재발하면또받을수있나요?", {}))
        # 두 조건이 모두 성립할 때만 발화하고, max_roles 로 상한을 둔다.
        self.assertEqual(["claim_procedure"], vector_tools.intent_role_gate(
            "보험금은 어떻게 청구하나요", {}))
        roles = vector_tools.intent_role_gate(
            "보장개시 전에 진단되면 특약이 소멸되나요", {}, max_roles=2)
        self.assertEqual(["timing_period", "contract_lifecycle"], roles)
        self.assertEqual(1, len(vector_tools.intent_role_gate(
            "보장개시 전에 진단되면 특약이 소멸되나요", {}, max_roles=1)))
        self.assertEqual([], vector_tools.intent_role_gate(
            "보험금은 어떻게 청구하나요", {}, max_roles=0))

    def test_intent_role_vocabulary_is_derived_from_jo_titles(self):
        class _Search:
            _jo = [{"element_id": "j1", "title": "제2-2조 보험금의 지급사유"},
                   {"element_id": "j2", "title": "제2-2조 보험금의 지급사유"},
                   {"element_id": "j3", "title": "제2-1조 특약의 보장개시"},
                   {"element_id": "j4", "title": "제2-1조 특약의 보장개시"},
                   {"element_id": "j5", "title": "제2-9조의2 “특약의 무효”에 대한 특칙"},
                   {"element_id": "j6", "title": "제2-9조의2 “특약의 무효”에 대한 특칙"},
                   {"element_id": "j7", "title": "제2-3조 단발성 제목"}]

        search = _Search()
        vocabulary = vector_tools.role_title_vocabulary(search, top_k=4, min_count=2)
        self.assertEqual(["보험금의 지급사유"], vocabulary["payment_trigger"])
        self.assertEqual(["특약의 보장개시"], vocabulary["timing_period"])
        self.assertEqual(["특약의 무효"], vocabulary["contract_lifecycle"])
        # 조제목이 없는 role 은 어휘가 없다(하드코딩된 기본 어휘 금지).
        self.assertNotIn("claim_procedure", vocabulary)
        # 위 어휘는 전부 위 제목에서 나온 것이며 코드에는 제목 리터럴이 없다.
        source = (HERE / "agent_tools.py").read_text(encoding="utf-8")
        for phrase in ("보험금의 지급사유", "특약의 보장개시", "특약의 무효",
                       "보험금의 청구"):
            self.assertNotIn(phrase, source)
        # 같은 검색 인스턴스에서 재계산 없이 캐시된 동일 객체를 돌려준다.
        self.assertIs(vocabulary, vector_tools.role_title_vocabulary(
            search, top_k=4, min_count=2))

    def test_intent_role_subquery_uses_question_tokens_plus_role_vocabulary(self):
        captured = {}

        class _Search:
            def match_table(self, slots, tokens):
                return None, [0]

            def rank_structured(self, slots, tokens, raw_query, weights=None,
                                profile="core", limit=400, lexical_counts=None):
                captured["slots"] = slots
                captured["tokens"] = tokens
                captured["query"] = raw_query
                return [({"element_id": "e1"}, 1.0)]

        rankings, audit = vector_tools.intent_role_rankings(
            _Search(), "보험금은 어떻게 청구하나요", {}, ["보험금", "청구"],
            ["claim_procedure"], {"claim_procedure": ["보험금 등의 청구"]})
        self.assertEqual(1, len(rankings))
        self.assertEqual(["claim_procedure"], [row["role"] for row in audit])
        self.assertEqual(["보험금 등의 청구"], audit[0]["vocabulary"])
        self.assertEqual(["claim_procedure"], captured["slots"]["role"])
        # 원 질문 토큰이 보존되고 조제목 어휘 토큰이 더해진다.
        self.assertEqual(["보험금", "청구"], captured["tokens"][:2])
        self.assertIn("보험금은 어떻게 청구하나요", captured["query"])
        self.assertIn("보험금 등의 청구", captured["query"])
        # 어휘가 없는 role 은 보조 질의를 만들지 않는다.
        self.assertEqual(([], []), vector_tools.intent_role_rankings(
            _Search(), "보험금은 어떻게 청구하나요", {}, ["보험금"],
            ["claim_procedure"], {}))

    def test_r6_option_is_wired_behind_the_arm_flag_in_tail_quota(self):
        source = (HERE / "agent_tools.py").read_text(encoding="utf-8")
        self.assertIn('if arm.get("intent_role"):', source)
        self.assertIn('intent_role_subquery', source)
        # tail quota: 기존 quota 앙상블 뒤에 붙고 unique_jo_quota 를 재사용한다.
        self.assertLess(source.index('if arm.get("intent_role"):'),
                        source.index("res = unique_jo_quota("))
        self.assertLess(source.index('if arm.get("reference_follow"):'),
                        source.index('if arm.get("intent_role"):'))


if __name__ == "__main__":
    unittest.main()
