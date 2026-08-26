import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import unittest

from schema_adapter import adapt_tag
from clm_search import Router, SlotSearch
from structured_search import StructuredTagIndex, fact_query_expansion, variants_of


class StructuredSearchTest(unittest.TestCase):
    def setUp(self):
        self.elements = [
            {"contract_scope": "암특약(무배당, 갱신형)"},
            {"contract_scope": "암특약(무배당, 해약환급금 미지급형)"},
            {"contract_scope": "뇌특약(무배당, 갱신형)"},
        ]
        self.tags = [
            {"contract_key": self.elements[0]["contract_scope"], "subject_key": ["표적항암"], "role": ["payment_trigger"],
             "locator": {"article": "제3조", "article_title": "보험금 지급사유"}, "schema_tag": "text"},
            {"contract_key": self.elements[1]["contract_scope"], "subject_key": ["표적항암"], "role": ["payment_trigger"],
             "locator": {"article": "제3조", "article_title": "보험금 지급사유"}, "schema_tag": "text"},
            {"contract_key": self.elements[2]["contract_scope"], "subject_key": ["뇌경색"], "role": ["code_reference"],
             "locator": {"article": "제8조", "article_title": "분류표", "table_headers": ["I63.9"]},
             "reference": ["제3조"], "schema_tag": "table"},
        ]
        self.idx = StructuredTagIndex(self.elements, self.tags)

    def test_variant_is_preserved(self):
        self.assertIn("갱신형", variants_of(self.elements[0]["contract_scope"]))
        ranked = self.idx.rank("암특약 갱신형 표적항암", {"contract": ["암특약"]})
        self.assertEqual(0, ranked[0][0])

    def test_table_and_article_are_searchable(self):
        ranked = self.idx.rank("I63.9 제8조 분류표", {"schema": ["table"]})
        self.assertEqual(2, ranked[0][0])

    def test_base_identity_keeps_variants_recallable(self):
        ranked = self.idx.rank("암특약 표적항암", {"contract": ["암특약"]})
        self.assertEqual({0, 1}, {ranked[0][0], ranked[1][0]})

    def test_unknown_schema_field_is_kept_as_low_weight_extra(self):
        tags = [
            {"element_id": "a", "novel_domain_key": "알파고유식별어"},
            {"element_id": "b", "novel_domain_key": "베타고유식별어"},
            {"element_id": "c", "novel_domain_key": "감마고유식별어"},
        ]
        idx = StructuredTagIndex(self.elements, tags)
        self.assertEqual(1, idx.rank("베타고유식별어", {}, limit=1)[0][0])
        self.assertEqual(3, idx.adapter_stats["documents_with_unknown"])

    def test_portable_business_document_keys_map_to_common_axes(self):
        row = adapt_tag({
            "document_type": "사업방법서", "product_name": "연금보험",
            "breadcrumb": ["10. 보험료에 관한 사항"], "conditions": ["월납"],
        })
        self.assertEqual(["연금보험"], row["identity"])
        self.assertEqual(["10. 보험료에 관한 사항"], row["locator"])
        self.assertEqual(["월납"], row["constraint"])
        self.assertEqual(["사업방법서"], row["structure"])

    def test_fact_card_axes_separate_container_and_evidence(self):
        row = adapt_tag({
            "document_key": "통합상품약관",
            "contract_key": "진단특약",
            "evidence_anchor": ["최초 1회의 진단확정에 한함"],
            "answer_values": ["최초 1회"],
            "parent_jo": "j00001",
        })
        self.assertEqual(["통합상품약관"], row["container"])
        self.assertEqual(["진단특약"], row["identity"])
        self.assertEqual(["최초 1회의 진단확정에 한함", "최초 1회", "j00001"],
                         row["evidence"])
        self.assertEqual([], row["relation"])

    def test_fact_weights_are_opt_in_and_rank_answer_kernel(self):
        elements = [{"contract_scope": "범용문서"}] * 2
        tags = [
            {"evidence_anchor": ["보험안내자료 내용이 약관과 다르면 계약자에게 유리"]},
            {"locator": {"article_title": "보험안내자료 제공"}},
        ]
        idx = StructuredTagIndex(elements, tags)
        base = idx.score("보험안내자료와 약관 내용이 다르면", {})
        enabled = idx.score("보험안내자료와 약관 내용이 다르면", {},
                            weights={"evidence": 2.4})
        self.assertEqual(0.0, enabled[1] - base[1])
        self.assertGreater(enabled[0] - base[0], 0.0)

    def test_zero_weight_fact_axis_does_not_change_legacy_idf(self):
        elements = [{"contract_scope": "범용문서"}] * 2
        legacy = [{"subject_key": ["직접근거"]}, {"subject_key": ["다른근거"]}]
        enriched = [dict(legacy[0], evidence_anchor=["직접근거"] * 5), dict(legacy[1])]
        legacy_scores = StructuredTagIndex(elements, legacy).score("직접근거", {})
        enriched_scores = StructuredTagIndex(elements, enriched).score("직접근거", {})
        self.assertEqual(legacy_scores, enriched_scores)

    def test_evidence_unit_scores_best_row_without_long_table_penalty(self):
        elements = [{"contract_scope": "범용문서"}] * 2
        tags = [
            {"evidence_anchor": [
                "무관한 표 행 " * 80,
                "자궁경부의 제자리암종 | D06",
                "다른 무관한 표 행 " * 80,
            ]},
            {"evidence_anchor": ["자궁경부 안내와 제자리암 일반 설명"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        base = idx.score("자궁경부 제자리암 D06", {}, weights={"evidence": 0.0})
        enabled = idx.score("자궁경부 제자리암 D06", {},
                            weights={"evidence": 0.0, "evidence_unit": 2.4})
        self.assertEqual(base, [0.0, 0.0])
        self.assertGreater(enabled[0], enabled[1])
        self.assertTrue(idx.last_evidence_unit["enabled"])

    def test_evidence_unit_rare_terms_ignore_generic_question_boilerplate(self):
        elements = [{"contract_scope": "범용문서"}] * 2
        tags = [
            {"evidence_anchor": ["자궁경부의 제자리암종 | D06"]},
            {"evidence_anchor": ["보험에서 암이 포함되는 경우를 안내합니다"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        ranked = idx.rank(
            "이 보험에서 자궁경부 제자리암도 포함되나요", {},
            weights={"evidence_unit": 1.0, "_evidence_unit_rare": 6}, limit=2)
        self.assertEqual(0, ranked[0][0])
        self.assertIn("자궁", idx.last_evidence_unit["selected_terms"])

    def test_fact_query_expands_benefit_surface_and_implicit_frequency(self):
        expanded = fact_query_expansion("암진단금은 몇 번 받을 수 있나요?", {})
        self.assertIn("암진단급여금", expanded)
        self.assertIn("limit_frequency", expanded)

    def test_hierarchy_treats_document_identity_as_container_not_section(self):
        elements = [
            {"contract_scope": "통합상품"},
            {"contract_scope": "진단특약"},
        ]
        tags = [
            {"document_key": "통합상품", "contract_key": "통합상품"},
            {"document_key": "통합상품", "contract_key": "진단특약",
             "evidence_anchor": ["질문 직접 근거"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        base = idx.score("통합상품 질문 직접 근거", {"contract": ["통합상품"]},
                         weights={"evidence": 2.4})
        hierarchical = idx.score(
            "통합상품 질문 직접 근거", {"contract": ["통합상품"]},
            weights={"evidence": 2.4, "_hierarchy": 1.0})
        self.assertGreater(base[0] - hierarchical[0], 0.0)
        self.assertGreater(hierarchical[1], hierarchical[0])

    def test_nested_locator_allows_more_specific_table_axis(self):
        row = adapt_tag({"locator": {"article": "제8조", "table_headers": ["질병코드"]}})
        self.assertEqual(["제8조"], row["locator"])
        self.assertEqual(["질병코드"], row["table"])

    def test_portfolio_decomposes_multiple_roles_without_llm(self):
        slots = {"contract": ["암특약(무배당, 갱신형)"],
                 "role": ["payment_trigger", "claim_procedure"],
                 "subject": ["표적항암"]}
        seed = self.idx.rank("암특약 지급과 청구", slots, profile="core", limit=3)
        specs = self.idx.build_portfolio_specs(
            "암특약 지급과 청구", slots, ["표적항암", "지급", "청구"], "role", seed)
        self.assertEqual(["role:payment_trigger", "role:claim_procedure"],
                         [spec["phase"] for spec in specs])

    def test_evidence_slots_infer_implicit_multi_role_question(self):
        slots = {"contract": ["암특약(무배당, 갱신형)"], "subject": ["암진단급여금"]}
        seed = self.idx.rank("어떤 진단을 받아야 하며 여러 번 지급되나요", slots,
                             profile="core", limit=3)
        specs = self.idx.build_portfolio_specs(
            "어떤 진단을 받아야 하며 여러 번 지급되나요", slots,
            ["진단", "여러", "지급"], "evidence", seed)
        phases = [spec["phase"] for spec in specs]
        self.assertIn("evidence:payment_trigger", phases)
        self.assertIn("evidence:limit_frequency", phases)
        self.assertIn("evidence:criteria_rule", phases)
        self.assertTrue(all(spec["slots"]["contract"] == slots["contract"] for spec in specs))

    def test_evidence_slots_do_not_fire_for_single_role(self):
        slots = {"contract": ["암특약"], "role": ["payment_trigger"]}
        seed = self.idx.rank("보험금 지급사유", slots, profile="core", limit=3)
        self.assertEqual([], self.idx.build_portfolio_specs(
            "보험금 지급사유", slots, ["보험금", "지급사유"], "evidence", seed))

    def test_evidence_roles_cover_count_word_and_membership_question(self):
        base_count_roles = self.idx.infer_evidence_roles(
            "어떤 경우에 지급되며 몇 회까지 받을 수 있나요", {})
        self.assertEqual([], base_count_roles)
        count_roles = self.idx.infer_evidence_roles(
            "어떤 경우에 지급되며 몇 회까지 받을 수 있나요", {}, expanded=True)
        self.assertIn("payment_trigger", count_roles)
        self.assertIn("limit_frequency", count_roles)

        base_membership_roles = self.idx.infer_evidence_roles(
            "자궁경부 제자리암도 포함되나요", {})
        self.assertEqual([], base_membership_roles)
        membership_roles = self.idx.infer_evidence_roles(
            "자궁경부 제자리암도 포함되나요", {}, expanded=True)
        self.assertIn("criteria_rule", membership_roles)
        self.assertIn("code_reference", membership_roles)

    def test_evidence_role_bonus_prefers_multi_role_answer(self):
        elements = [{"contract_scope": "범용문서"}] * 3
        tags = [
            {"role": ["definition", "code_reference"]},
            {"role": ["payment_trigger", "limit_frequency", "criteria_rule"]},
            {"role": ["payment_trigger"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        slots = {"role": ["limit_frequency"]}
        base = idx.score("급여금 지급 요건과 연간 횟수", slots,
                         weights={"_evidence_role": 0.0})
        boosted = idx.score("급여금 지급 요건과 연간 횟수", slots,
                            weights={"_evidence_role": 1.0})
        self.assertGreater(boosted[1] - base[1], boosted[2] - base[2])
        self.assertGreater(boosted[2] - base[2], boosted[0] - base[0])
        self.assertEqual(3, idx.last_evidence_role["max_matched_roles"])

    def test_evidence_role_bonus_does_not_fire_for_single_role_question(self):
        idx = StructuredTagIndex(
            [{"contract_scope": "범용문서"}],
            [{"role": ["payment_trigger"]}],
        )
        base = idx.score("보험금 지급사유", {"role": ["payment_trigger"]},
                         weights={"_evidence_role": 0.0})
        boosted = idx.score("보험금 지급사유", {"role": ["payment_trigger"]},
                            weights={"_evidence_role": 1.0})
        self.assertEqual(base, boosted)

    def test_axis_coordination_bonus_prefers_multi_axis_match(self):
        ranked = self.idx.rank(
            "암특약 표적항암 지급", {"contract": ["암특약"],
                                   "subject": ["표적항암"], "role": ["payment_trigger"]},
            weights={"_axis": 1.0, "_coord": 2.0}, profile="core", limit=3)
        self.assertIn(ranked[0][0], {0, 1})
        self.assertNotEqual(2, ranked[0][0])

    def test_exact_identity_bonus_prefers_explicit_bracketed_qualifier(self):
        elements = [
            {"contract_scope": "(간편)[기본]심장질환진단특약(무배당, 갱신형)"},
            {"contract_scope": "(간편)[삭감없음용]심장질환진단특약(무배당, 갱신형)"},
        ]
        tags = [
            {"contract_key": row["contract_scope"], "subject_key": ["진단급여금"],
             "role": ["payment_trigger"]} for row in elements
        ]
        idx = StructuredTagIndex(elements, tags)
        routed = {"contract": [row["contract_scope"] for row in elements],
                  "subject": ["진단급여금"]}
        ranked = idx.rank("(간편)[기본]심장질환진단특약 진단급여금", routed,
                          weights={"_exact_identity": 4.0}, profile="core")
        self.assertEqual(0, ranked[0][0])

    def test_router_strict_explicit_bracket_is_generic_and_opt_in(self):
        contracts = [
            "(간편)[기본]고액암진단특약(무배당, 갱신형)",
            "(간편)[삭감없음용]고액암진단특약(무배당, 갱신형)",
            "(간편)[임의판본X]고액암진단특약(무배당, 갱신형)",
        ]
        router = Router(contracts)
        baseline, _ = router.route("[기본]고액암진단특약 지급사유")
        self.assertGreater(len(baseline["contract"]), 1)
        router.strict_explicit_bracket = True
        basic, _ = router.route("[기본]고액암진단특약 지급사유")
        self.assertEqual([contracts[0]], basic["contract"])
        comparison, _ = router.route(
            "[기본]과 [삭감없음용] 고액암진단특약 지급사유 비교")
        self.assertEqual({contracts[0], contracts[1]}, set(comparison["contract"]))
        arbitrary, _ = router.route("[임의판본X]고액암진단특약 지급사유")
        self.assertEqual([contracts[2]], arbitrary["contract"])

    def test_exact_identity_bonus_rejects_nested_or_inferred_identity(self):
        elements = [
            {"contract_scope": "암진단특약(무배당)"},
            {"contract_scope": "재진단암진단특약(무배당)"},
        ]
        tags = [{"contract_key": row["contract_scope"], "subject_key": ["지급사유"]}
                for row in elements]
        idx = StructuredTagIndex(elements, tags)
        routed = {"contract": [row["contract_scope"] for row in elements]}
        baseline = idx.score("재진단암진단특약 지급사유", routed,
                             weights={"_exact_identity": 0.0})
        boosted = idx.score("재진단암진단특약 지급사유", routed,
                            weights={"_exact_identity": 4.0})
        self.assertEqual(baseline[0], boosted[0])
        self.assertGreater(boosted[1], baseline[1])

        inferred = {"contract": [elements[0]["contract_scope"]],
                    "_conf": {"contract": 0.75}}
        inferred_base = idx.score("암진단특약 지급사유", inferred,
                                  weights={"_exact_identity": 0.0})
        inferred_boost = idx.score("암진단특약 지급사유", inferred,
                                   weights={"_exact_identity": 4.0})
        self.assertEqual(inferred_base, inferred_boost)

    def test_exact_identity_bonus_accepts_explicit_base_name_without_plan_label(self):
        elements = [
            {"contract_scope": "(간편)[50%이상장해형]6대질병장해특약(무배당, 갱신형)"},
            {"contract_scope": "(간편)[50%장해납입면제형]보험료납입면제특약(무배당, 갱신형)"},
        ]
        tags = [{"contract_key": row["contract_scope"], "role": ["premium_waiver"]}
                for row in elements]
        idx = StructuredTagIndex(elements, tags)
        routed = {"contract": [row["contract_scope"] for row in elements],
                  "role": ["premium_waiver"]}
        base = idx.score("6대 질병 장해 특약 납입면제 이후 갱신시 보험료 납입면제",
                         routed, weights={"_exact_identity": 0.0})
        boosted = idx.score("6대 질병 장해 특약 납입면제 이후 갱신시 보험료 납입면제",
                            routed, weights={"_exact_identity": 4.0})
        self.assertGreater(boosted[0], base[0])
        self.assertEqual(base[1], boosted[1])

    def test_exact_identity_bonus_does_not_promote_function_without_contract_suffix(self):
        elements = [{"contract_scope": "보험료납입면제특약(무배당, 갱신형)"}]
        tags = [{"contract_key": elements[0]["contract_scope"],
                 "role": ["premium_waiver"]}]
        idx = StructuredTagIndex(elements, tags)
        routed = {"contract": [elements[0]["contract_scope"]],
                  "role": ["premium_waiver"]}
        base = idx.score("갱신하면 보험료 납입면제 되나요", routed,
                         weights={"_exact_identity": 0.0})
        boosted = idx.score("갱신하면 보험료 납입면제 되나요", routed,
                            weights={"_exact_identity": 4.0})
        self.assertEqual(base, boosted)

    def test_locator_coverage_prefers_direct_semantic_heading_match(self):
        elements = [{"contract_scope": ""}, {"contract_scope": ""}]
        tags = [
            {"locator": {"article_title": "계약 전 알릴 의무 위반의 효과"}},
            {"locator": {"article_title": "계약 전 알릴 의무"}},
        ]
        idx = StructuredTagIndex(elements, tags)
        base = idx.score("알릴의무 위반했을 때 규정", {},
                         weights={"_locator_coverage": 0.0})
        boosted = idx.score("알릴의무 위반했을 때 규정", {},
                            weights={"_locator_coverage": 4.0})
        self.assertGreater(boosted[0] - base[0], boosted[1] - base[1])
        self.assertGreater(idx.last_locator_coverage["boosted_documents"], 0)

    def test_locator_coverage_rejects_one_generic_fragment(self):
        idx = StructuredTagIndex(
            [{"contract_scope": ""}],
            [{"locator": {"article_title": "보험금 지급사유"}}],
        )
        base = idx.score("지급 되나요", {}, weights={"_locator_coverage": 0.0})
        boosted = idx.score("지급 되나요", {}, weights={"_locator_coverage": 4.0})
        self.assertEqual(base, boosted)
        self.assertEqual(0, idx.last_locator_coverage["boosted_documents"])

    def test_sequence_uses_only_explicit_routed_axes(self):
        slots = {"contract": ["암특약"], "subject": ["표적항암"],
                 "role": ["payment_trigger"], "qualifier": ["연간 1회"]}
        seed = self.idx.rank("암특약 표적항암 지급", slots, profile="core", limit=3)
        specs = self.idx.build_portfolio_specs(
            "암특약 표적항암 지급", slots, ["표적항암", "지급"], "sequence", seed)
        self.assertEqual(["sequence:identity_topic", "sequence:function:payment_trigger"],
                         [spec["phase"] for spec in specs])
        self.assertTrue(all("identity" in spec["slots"] for spec in specs))
        self.assertFalse(any("claim_procedure" in spec["query"] for spec in specs))

    def test_hybrid_gate_uses_axis_coverage_not_candidate_count(self):
        search = SlotSearch.__new__(SlotSearch)
        search.rows = [
            {"contract": ["암특약"], "subject": ["표적항암"],
             "role": ["payment_trigger"], "qualifier": [], "schema": []},
            {"contract": ["뇌특약"], "subject": ["뇌경색"],
             "role": ["definition"], "qualifier": [], "schema": []},
        ]
        search._eidx = {"e0": 0, "e1": 1}
        good = search.hybrid_fallback_gate(
            {"contract": ["암특약"], "subject": ["표적항암"]},
            [({"element_id": "e0"}, 10.0)] * 400)
        bad = search.hybrid_fallback_gate(
            {"contract": ["암특약"], "subject": ["표적항암"]},
            [({"element_id": "e1"}, 10.0)] * 400)
        self.assertFalse(good["fallback"])
        self.assertTrue(bad["fallback"])
        self.assertIn("identity_miss", bad["reasons"])

    def test_portfolio_relax_removes_initial_identity(self):
        slots = {"contract": ["암특약"], "role": ["payment_trigger"]}
        seed = self.idx.rank("암특약 보험금", slots, profile="core", limit=3)
        spec = self.idx.build_portfolio_specs(
            "암특약 보험금", slots, ["보험금"], "relax", seed)[0]
        self.assertEqual("identity_relax", spec["phase"])
        self.assertNotIn("contract", spec["slots"])

    def test_gated_relax_requires_router_identity_overload(self):
        two = {"contract": ["암특약", "뇌특약"], "role": ["definition"]}
        seed = self.idx.rank("질병 정의", two, profile="core", limit=3)
        self.assertEqual([], self.idx.build_portfolio_specs(
            "질병 정의", two, ["질병", "정의"], "relax_gated", seed))

        three = {"contract": ["암특약", "뇌특약", "순환계특약"],
                 "role": ["definition"]}
        specs = self.idx.build_portfolio_specs(
            "질병 정의", three, ["질병", "정의"], "relax_gated", seed)
        self.assertEqual(["identity_relax"], [spec["phase"] for spec in specs])
        self.assertIn("3개 이상", specs[0]["reason"])

    def test_portfolio_preserves_seed_quota(self):
        slots = {"contract": ["암특약"], "role": ["payment_trigger", "claim_procedure"]}
        seed = self.idx.rank("암특약 표적항암", slots, profile="core", limit=3)
        portfolio = self.idx.rank_portfolio(
            "암특약 표적항암", slots, ["표적항암"], profile="core", mode="all",
            limit=3, window=3, seed_quota=1)
        self.assertEqual(seed[0][0], portfolio[0][0])
        self.assertEqual("seed", self.idx.last_portfolio["trace"][0]["phase"])

    def test_rrf_evidence_preserves_single_role_ranking_exactly(self):
        slots = {"contract": ["암특약"], "role": ["payment_trigger"]}
        seed = self.idx.rank("암특약 보험금 지급사유", slots, profile="core", limit=3)
        portfolio = self.idx.rank_portfolio(
            "암특약 보험금 지급사유", slots, ["암특약", "보험금", "지급사유"],
            profile="core", mode="rrf_evidence", limit=3, window=3)
        self.assertEqual(seed, portfolio)
        self.assertEqual([], self.idx.last_portfolio["specs"])

    def test_rrf_evidence_exposes_each_requested_role_with_identity_guard(self):
        elements = [
            {"contract_scope": "암특약"},
            {"contract_scope": "암특약"},
            {"contract_scope": "암특약"},
            {"contract_scope": "무관특약"},
        ]
        tags = [
            {"contract_key": "암특약", "subject_key": ["암"],
             "role": ["definition"]},
            {"contract_key": "암특약", "subject_key": ["암"],
             "role": ["payment_trigger"]},
            {"contract_key": "암특약", "subject_key": ["암"],
             "role": ["limit_frequency"]},
            {"contract_key": "무관특약", "subject_key": ["암"],
             "role": ["limit_frequency"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        slots = {"contract": ["암특약"], "subject": ["암"]}
        ranked = idx.rank_portfolio(
            "암특약 보험금 지급 요건과 연간 횟수", slots,
            ["암특약", "보험금", "지급", "연간", "횟수"],
            profile="core", mode="rrf_evidence", limit=4, window=4,
            coverage_seed=1, coverage_per_role=1)
        phases = [row["phase"] for row in idx.last_portfolio["trace"][:4]]
        self.assertIn("coverage:function:payment_trigger", phases)
        self.assertIn("coverage:function:limit_frequency", phases)
        self.assertNotEqual(3, ranked[1][0])

    def test_rrf_axes_decomposes_explicit_variant_comparison(self):
        elements = [
            {"contract_scope": "[기본]암진단특약(무배당, 갱신형)"},
            {"contract_scope": "[삭감없음용]암진단특약(무배당, 갱신형)"},
            {"contract_scope": "[기본]무관특약(무배당, 갱신형)"},
        ]
        tags = [
            {"contract_key": elements[0]["contract_scope"], "role": ["timing_period"]},
            {"contract_key": elements[1]["contract_scope"], "role": ["timing_period"]},
            {"contract_key": elements[2]["contract_scope"], "role": ["timing_period"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        slots = {"contract": [elements[0]["contract_scope"], elements[1]["contract_scope"]],
                 "role": ["timing_period"]}
        ranked = idx.rank_portfolio(
            "암진단특약 기본과 삭감없음용 보장개시일", slots,
            ["암진단특약", "기본", "삭감없음용", "보장개시일"],
            profile="core", mode="rrf_axes", limit=4, window=4,
            coverage_seed=0, coverage_per_role=1)
        # 앞선 function anchor가 이미 한 판본을 골랐더라도 두 판본 element가
        # 모두 결과 앞쪽에 있어야 한다. 중복 카드를 만들지는 않는다.
        self.assertEqual({0, 1}, {idx for idx, _ in ranked[:2]})

    def test_rrf_safe_axes_preserves_seed_prefix_and_rejects_distractor(self):
        elements = [
            {"contract_scope": "[기본]암진단특약(무배당, 갱신형)"},
            {"contract_scope": "[삭감없음용]암진단특약(무배당, 갱신형)"},
            {"contract_scope": "[기본]암진단특약(무배당, 갱신형)"},
            {"contract_scope": "[삭감없음용]암진단특약(무배당, 갱신형)"},
        ]
        tags = [
            {"contract_key": elements[0]["contract_scope"],
             "role": ["contract_lifecycle"],
             "locator": {"article_title": "특약내용 변경과 감액"}},
            {"contract_key": elements[1]["contract_scope"],
             "role": ["contract_lifecycle"],
             "locator": {"article_title": "특약내용 변경과 감액"}},
            {"contract_key": elements[2]["contract_scope"],
             "role": ["timing_period", "payment_trigger"],
             "subject_key": ["암진단급여금"], "qualifier": ["보장개시일"]},
            {"contract_key": elements[3]["contract_scope"],
             "role": ["timing_period", "payment_trigger"],
             "subject_key": ["암진단급여금"], "qualifier": ["보장개시일"]},
        ]
        idx = StructuredTagIndex(elements, tags)
        slots = {"contract": [elements[0]["contract_scope"], elements[1]["contract_scope"]],
                 "role": ["timing_period"], "subject": ["암진단급여금"]}
        seed = idx.rank("암진단특약 기본과 삭감없음용 보장개시일", slots,
                        profile="core", limit=4)
        ranked = idx.rank_portfolio(
            "암진단특약 기본과 삭감없음용 보장개시일", slots,
            ["암진단특약", "기본", "삭감없음용", "보장개시일"],
            profile="core", mode="rrf_safe_axes", limit=4, window=4,
            coverage_seed=1, coverage_per_role=1)
        self.assertEqual(seed[0][0], ranked[0][0])
        # answer-bearing timing 후보는 앞쪽에 들어오고 감액 distractor만으로
        # variant coverage가 충족됐다고 간주하지 않는다.
        self.assertTrue({2, 3}.intersection(idx for idx, _ in ranked[:3]))
        coverage = [row for row in idx.last_portfolio["trace"]
                    if row["phase"].startswith("coverage:")]
        self.assertTrue(all(row["index"] not in {0, 1} for row in coverage))

    def test_context_backtrack_requires_context_gate(self):
        slots = {"contract": ["암특약(무배당, 갱신형)"],
                 "subject": ["암수술급여금"]}
        self.assertIsNone(Router.context_backtrack(
            "암특약의 암수술급여금 지급 대상", slots, ["암특약", "암수술급여금"]))

    def test_context_backtrack_removes_product_form_false_intent(self):
        identity = "통합보장보험(무배당, 해약환급금 미지급형)"
        plan = Router.context_backtrack(
            "통합보장보험(무배당, 해약환급금 미지급형) 보험으로 MRI 촬영시 환급이 가능한가요",
            {"contract": [identity], "role": ["contract_lifecycle"],
             "subject": ["해약환급금"]},
            ["통합보장보험", "무배당", "해약환급금", "미지급형", "MRI", "촬영시", "환급이"])
        self.assertIn("collection_context", plan["reasons"])
        self.assertNotIn("subject", plan["slots"])
        self.assertNotIn("role", plan["slots"])
        self.assertIn("MRI", plan["tokens"])


if __name__ == "__main__":
    unittest.main()
