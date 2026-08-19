# filesearch — Semantic Tag 파일서치 검색기 (hybrid-enrich_v2 의 우리 담당분)

## 파이프라인 (원문·QA 는 ../../hybrid-enrich/noah_qaset 참조, out/ 은 비커밋)
1. `build_universe.py --doc <250212.md> --clean --name elements_u2.jsonl` → u2 (32,366) · `build_jo_universe.py --elements out/elements_u2.jsonl --out out/elements_u2jo.jsonl` → 조 단위 (7,599)
2. 태그: `tag_rules.py`(NEW 8슬롯 규칙) · `tag_old.py`(OLD 스키마 이식)
3. gold(span): `map_gold_spans.py --doc … --qa <train csv>` · `map_lsh_gold.py`(lsh v4)
4. 라우터: 규칙(내장) + `qtag_llm.py`(질문+특약사전만 입력, haiku 1회, 캐시 동결)
5. 결정론 선별: `eval_det.py --qtags out/qtags_haiku.jsonl --qmode llm --arms "clm:lex=count:w=contract=2;and;…"` (조 map-back 채점, R@K 곡선)
6. 에이전트 실측: `agent_tools.py`(search/read/submit) + `agent_runner.py`(claude) / `agent_runner_oai.py`(OpenAI 호환: vLLM Qwen·LiteLLM 등). `INSTANCE_README.md` 참조.

## 검색기 스펙(v2ds 등록 스펙 재구현 + 고도화)
- 후보 = 슬롯 합집합 ∪ 어휘 채널 / 점수 = 맞은 슬롯 수(가중 가능) + 질문 토큰 수 / 동점 = 문서순
- 라우터 = 규칙(특약명·역할·대상어·값) ∪ 동결 LLM qtags ∪ 에이전트 지정 슬롯 · contract 소프트 부스트(w2)
- 금지: BM25·임베딩·reranker·학습 정렬 (임베딩 채널은 별도 도구 `vsearch` — 결합 arm 에서만)

## 문서
`PREREG_v2.md`(사전 등록) · `RESULT_*_20260818.txt`(결정론·진단·결합 선별 결과) · `_diag/`, `_vector_rnd/`(보고 arm 아님)
