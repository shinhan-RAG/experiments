# hybrid-enrich_v2 — 실험 베이스라인 (2026-08-19)

`tos-skeleton/hybrid-enrich`(dev, PR #49 시점)를 그대로 복사한 뒤 **Semantic Tag 파일서치 검색기(`filesearch/`)를 넣은 것**이다.
회의 결정(2026-08-19): 평가 프로세스·Metadata 검색은 고정하고 **Semantic Tag 검색기만 바꿔 비교**한다. Metadata 검색(임베딩 채널)은 동료가 추가 예정.

## 무엇이 다른가
| 영역 | hybrid-enrich (원본) | hybrid-enrich_v2 |
|---|---|---|
| 원본 코드 | seyeong·koda-design 작성 스크립트 전부 | **그대로 복사(수정 없음)** — 원본 계보 유지용. 우리 실험은 `filesearch/` 만 사용 |
| 우주(element) | `build_elements.py`(260507판) | `filesearch/build_universe.py --clean` — 같은 분할 규칙을 **250212판**에 적용 + 형식 버그 수정(`$$` 산식 한 줄·페이지 노이즈·경계 병합) → **u2 32,366** / 채점 `build_jo_universe.py` → **u2jo 7,599** |
| Semantic Tag | `build_enrichment_v2.py`(OLD) · `build_element_fields_v4.py`(NEW 규칙) | `filesearch/tag_old.py`(OLD 이식) · `filesearch/tag_rules.py`(NEW 8슬롯 규칙) · LLM 슬롯 채움은 후속 |
| 파일서치 검색기 | `slot_filesearch.py`(AND) | `filesearch/clm_search.py` — CLM(슬롯 합집합+어휘 count) + 규칙/LLM qtags 라우터(`qtag_llm.py`) + contract 소프트 부스트. AND 는 대조 모드로 포함 |
| 에이전트 도구 | `tools.py`(vector/grep/read) | `filesearch/agent_tools.py`(search/read/submit, S5 예산, 전량 로그) + 러너 2종(claude / OpenAI 호환) |
| 채점 | 위치 기반 R@K | `filesearch/scoring.py` — gold **char span**(우주 독립) 대비 fractional evidence-group R@K·S@K·MRR (비율 채점). gold: `map_gold_spans.py`(QA 출처→span) · `map_lsh_gold.py`(lsh v4 gold→span) |

## 데이터 경로
- 원문·QA 는 커밋하지 않는다. `../hybrid-enrich/noah_qaset/`(250212 md · train csv)를 그대로 참조. test149 는 사용 금지.
- 산출물은 `filesearch/out/`(gitignore). 재생성 순서는 `filesearch/README.md`.

## 우리가 쓰지 않는 것 (정리 대상, 원 저자 파일은 삭제하지 않음)
- 원본 스크립트 전부(`build_*.py`, `run_*.py`, `eval_*.py`, `tools.py`, `codex_retrieval_tool.py`, `slot_filesearch.py` 등)는 계보 보존용 복사본. `filesearch/`는 이 중 `slot_filesearch.py`(fuzzy 매칭 규칙)·`build_element_fields_v4.py`(정규식)·`aliases.json` 세 파일만 import 한다.
- `filesearch/_vector_rnd/`(임베딩 채널 R&D: BGE-m3-ko 색인·Qdrant 업서트·결정론 결합 평가)와 `filesearch/_diag/`(BM25 진단)는 **보고 arm 이 아니며** 선별·진단 기록용이다.
