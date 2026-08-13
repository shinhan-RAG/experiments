# hybrid-enrich 산출물 재생성 파이프라인

PR #22에서 추가한 검색 도구(`codex_retrieval_tool.py`, `slot_filesearch.py`)가 읽는
`out/*.jsonl` 산출물은 용량 문제로 `.gitignore` 처리되어 저장소에 없습니다.
이 문서는 그 산출물을 만드는 스크립트 실행 순서를 정리합니다.

## 전제

- 원본 문서 경로가 각 스크립트 상단 `DOC` 상수에 **로컬 절대경로로 하드코딩**되어 있습니다.
  다른 환경에서 실행하려면 해당 상수를 자기 경로로 바꿔야 합니다.
  (`판매약관_신한(간편가입)통합건강보험 원(ONE)..._260507.md`, `정답셋_359_최종_v4.jsonl`)
- 모든 입출력은 `hybrid-enrich/out/` 아래에 생성됩니다.
- 임베딩(`embed.py`)에는 로컬 임베딩 서버가 필요합니다.

## 1. 기본 파싱 · 태깅

| 순서 | 스크립트 | 주요 출력 |
| --- | --- | --- |
| 1 | `build_elements.py` | `elements.jsonl` |
| 2 | `build_chunks.py` | `chunks.jsonl`, `chunk_stats.json` |
| 3 | `build_element_tags.py` | `element_tags.jsonl` |
| 4 | `build_enrichment_v2.py` | `element_tags_v2.jsonl` (OLD 태그 스키마) |
| 5 | `build_views_v2.py` / `build_views_v3.py` | `grep_tag_v*.jsonl`, `chunk_metadata_v3.jsonl`, `vec_*_texts.jsonl` |
| 6 | `build_element_fields_v4.py` | `element_semantic_fields_v4.jsonl` (NEW 필드 스키마 v4) |
| 7 | `embed.py` | `vec_*.npy`, `vec_*_ids.json` |

## 2. 구조 복구(repair) 계열

구조적으로 의심스러운 proxy element만 재분할하고, 그 자식들의 태그만 다시 만들어
기존 태그와 이어붙이는 흐름입니다.

| 순서 | 스크립트 | 입력 | 출력 |
| --- | --- | --- | --- |
| 1 | `repair_suspect_elements.py` | `elements.jsonl` | `elements_repaired_v1.jsonl`, `elements_repaired_delta_v1.jsonl`, `element_repair_manifest_v1.json` |
| 2 | 태그 생성(에이전트) | `elements_repaired_v1.jsonl` | `element_tags_old_repaired_full_v1.jsonl`, `element_tags_new_repaired_full_v1.jsonl` (재분할된 자식분) |
| 3 | `remap_train350_repaired.py` | `train350_gold.jsonl`, `elements_repaired_v1.jsonl` | `train350_gold_repaired_v1.jsonl` |
| 4 | `finalize_repaired_tags.py` | 위 전부 + `element_tags_v2.jsonl`, `element_semantic_fields_v4.jsonl` | `element_tags_{old,new}_repaired_{full,delta}_v1.jsonl`, `grep_{base,tag_old,tag_new}_repaired_v1.jsonl`, `repaired_tag_variants_audit.json` |
| 5 | `audit_repaired_structure.py`, `audit_retrieval_input_quality.py` | — | `repaired_structure_audit.json`, `retrieval_input_quality_audit.json` |

> **주의**: `finalize_repaired_tags.py`는 2단계에서 만든
> `element_tags_*_repaired_full_v1.jsonl`을 읽어 미변경 element의 기존 태그와 스플라이스한 뒤
> **같은 파일명으로 덮어씁니다.** 재실행 시 2단계 산출물을 먼저 백업해 두세요.

최종 `element_tags_new_repaired_full_v1.jsonl` / `element_tags_old_repaired_full_v1.jsonl`은
각각 61,170행(= `elements_repaired_v1.jsonl` 전체 행수)이며,
행 수와 `element_id` 순서가 서로 정렬되어 있습니다.

- NEW 스키마 키: `element_id, schema_version, schema_tag, contract_key, subject_key, role, locator, qualifier, reference, field_sources, search_text`
- OLD 스키마 키: `element_id, element_type, contract_scope, topic, article, semantic_role, values, conditions, aliases, is_toc`

## 3. 검색 A/B (PR #22 도구가 소비하는 구간)

| 순서 | 스크립트 | 출력 |
| --- | --- | --- |
| 1 | `preflight_retriever_ab25.py` | `preflight_retriever_ab25.json`, `retriever_ab25_manifest.json` |
| 2 | `run_retriever_ab25.py` 또는 `run_codex_retriever_ab25.py` | `retrieval_retriever_ab25.jsonl`, 세션 로그 |
| 3 | `eval_retriever_ab25.py` | `eval_retriever_ab25.json`, `eval_retriever_ab25_details.jsonl` |

`preflight`는 gold 350문항, 25문항 batch, vector/tag ID 정합성을 검사합니다.
여기서 실패하면 2·3단계를 돌리지 마세요.

## 4. QA 데이터셋 계열 (검색 도구와 별개)

`build_train350_gold.py`, `split_qa503_train_test.py`,
`build_document_coverage_qa_v2.py`, `expand_document_coverage_qa_v3.py`,
`build_type_coverage_supplement.py`, `build_complexity_difficulty_supplement.py`,
`review_supplement_qa.py`, `manual_review_document_coverage_v3.py`,
`eda_question_type_coverage.py`, `eda_train_test_vs_document.py`

## 산출물 전달

`out/element_tags*.jsonl`은 `.gitignore` 대상입니다(파일당 20~70MB).
외부 공유가 필요하면 압축해서 별도로 전달하세요.
