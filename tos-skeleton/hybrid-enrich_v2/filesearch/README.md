# filesearch

Semantic Tag 기반 파일서치. 문서 250212판, element 우주 **u3**(항 단위 32,046) 색인, 조 단위(**u3jo** 7,612) 채점.

**최종 채택 조합**: arm `r1v_verify_identity_reference_hybrid` + Gold v22 (train 213, agent ITT R@5 **.9147**).
실행·재현·필수 파일 SHA는 `../vector_search/noah/0819/FINAL_STACK_REPRO.md`,
전체 판정 이력은 `STRUCTURED_SEARCH_EXPERIMENT.md`(실험 대장) 참조.

## 파이프라인 (전부 규칙, 결정론)
```
build_universe.py --doc <250212.md> --clean          # element 우주
stabilize_universe_ids.py                            # id 안정화 (u3 stable)
build_jo_universe.py                                 # 조 단위 우주 (u3jo)
tag_rules.py                                         # 8슬롯 태그 (규칙)
build_tags_u4_fact.py                                # U4 fact 태그 (최종 arm 입력)
build_tags_u5_reference_graph.py                     # U5 참조 그래프 (reference_follow 입력, stats만 사용)
qtag_llm.py --model haiku                            # 질의 태그 (llm 라우터 arm용, 동결 산출물)
stats.py --a A.jsonl --b B.jsonl --key R@5           # paired 검정 (부호검정/BCa/Holm)
```
에이전트 실측 러너·arm 정의·결정론 진단(eval_det)은 `../vector_search/noah/0819/` (host_agent_runner.py, arms.json).

## gold (v22, 213문항)
- 기반: lsh train QA셋 gold element 텍스트를 원문에 char span으로 앵커링. test149는 봉인(열람·사용 금지).
- 감사 체인: retrieval-blind 2단 감사(1차 + 독립 교차검토 + validator + adjudication) `audits/` + manifest.
  ledger 적용 스크립트: `apply_gold_completeness_ledger.py`, `apply_spec3_occurrence_expansion.py` (v21 결정론 확장), `adjudicate_gold_audit_batch.py` 등.
- 현행: `out/gold_train_scoped_u3_reviewed_overlay_v22.jsonl` (sha f75805c8…). 1행 1문항, `groups`=[AND], 그룹 내 `members`=[OR].

## 채점 (scoring.py + units.py)
- fractional evidence-group Recall@K = Top-K가 덮은 group / 전체 group (OR은 1개면 충족). Success@K, sufficient@K, MRR@10.
- 제출 id는 e*(항), j*(조), c*(청크) 모두 허용 — units.py가 조 단위로 사상. 어느 검색기든 상위 K id 리스트만 있으면 채점 가능.
- 보고 수치는 에이전트 실측(ITT, reps≥2)만 쓴다. 결정론 수치는 선별·진단용.

## 검색기 (최종 arm 구성 요소)
- CLM 슬롯 채널(clm_search.py): 후보 = 슬롯 합집합 ∪ 어휘 채널, 규칙 라우터(특약명·역할·대상어·값), contract 소프트 부스트.
- U4 fact 태그 BM25F + CLM quota 앙상블(agent_tools.py, arms.json의 `r1v_…` 참조), identity_expand + reference_follow(U5, fail-closed).
- Meta V9 hybrid(bm25+dense RRF, `../vector_search/hybrid_search.py`) 강제 결합. reranker·학습 정렬 없음.
