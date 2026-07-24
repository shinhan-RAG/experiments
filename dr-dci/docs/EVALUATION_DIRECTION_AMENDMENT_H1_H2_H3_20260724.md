# Peter 실험 평가 방향 Amendment — H1 텍스트, H2 요소, H3 법률, H4 교집합

기록일: 2026-07-24
상태: **사전 설계 감사 기록 — 모든 taxonomy/@tag A/B·Agent·Part 1~3 실행 미실행**

## 1. 변경 이유와 사전성

이 amendment는 MIRACL-ko taxonomy artifact 생성, taxonomy A/B, element-tag A/B, Agent 실행, Peter Part 1~3 및 AIHub 법률 primary 실행이 모두 **미실행**인 상태에서 기록한다. 따라서 관측된 treatment 결과를 보고 성공 기준을 바꾸거나 유리한 데이터셋을 사후 선택한 변경이 아니다.

기존 MIRACL-ko fixture와 lexical smoke는 이미 확보된 **title/text 기반 passage retrieval의 text-only baseline**으로 보존한다. 이 baseline은 taxonomy 효과, element-tag 효과, Agentic 효과, 법률/보험 전이를 보여 주지 않는다. MIRACL taxonomy generator/KT 후보 계약도 삭제하거나 재해석하지 않으며, 실제 모델 실행은 이 amendment 이후 별도 승인 전까지 보류한다.

변경의 근거는 데이터와 현행 `@el` 경로의 표현 범위다.

- MIRACL-ko corpus schema는 `corpus_id`, `article_id`, `passage_index`, `title`, `text`만 제공하는 passage 계약이다. 표, 그림, 페이지 좌표 또는 parser element는 없다.
- 현행 `scripts/build_tags.py`는 parser 산출물을 읽지 않고 `text`를 빈 줄로 나누며, 각 분할의 앞 200자만 LLM 입력으로 쓴다.
- 따라서 MIRACL text-only 결과로 표·목록·그림·문서 좌표 기반 element navigation을 주장할 수 없다.

이 문서는 기존 [AIHub 법률 primary · MIRACL-ko secondary 평가 계획](LEGAL_PRIMARY_MIRACL_SECONDARY_EVALUATION_PLAN_20260724.md)을 역사 기록으로 보존한다. **향후 평가 위계와 최종 성공 해석에 한해서는 이 amendment가 해당 문서를 supersede한다.** 다만 기존 Legal Gate L0, corpus/qrel 정합성, source-parent qrel의 해석 경계, 라이선스·PII·held-out 계약은 계속 유효하며 이 amendment로 완화하거나 대체하지 않는다. 기존 fixture·lexical 결과·taxonomy contract·KT 후보의 원문이나 수치도 수정하지 않는다.

## 2. 사전 등록한 가설별 평가 위계

| 가설 | 평가 대상과 단위 | 확인하려는 제한된 질문 | 성공 근거로 사용할 수 없는 것 | 현재 상태 |
|---|---|---|---|---|
| **H1 taxonomy** | MIRACL-ko, text-only **passage** retrieval | 고정 taxonomy soft boost가 일반 한국어 passage text retrieval에서 baseline 대비 효과가 있는가 | 표/그림/목록 인식, parent-document 회수, 법률·보험 전이 | fixture와 lexical baseline만 있음; taxonomy artifact/A-B/Agent 미실행 |
| **H2a structural type** | 장문 복합 문서의 parser-derived **element** retrieval | 동일 parser element universe에서 `structural_type` exact typed filter가 filter 없는 control보다 element/parent gold 회수를 개선하는가 | semantic role의 추가 효과, 평문 줄바꿈 분할, 추정된 table/list/figure, MIRACL passage 결과 | Gate D0 및 별도 protocol 미완료; retrieval-only A/B 미실행 |
| **H2b semantic role** | H2a와 같은 parser-derived **element** retrieval | H2a의 structural filter 조건을 고정한 뒤 `semantic_role` exact filter를 추가하면 element/parent gold 회수가 더 개선되는가 | structural type과 semantic role을 처음부터 함께 활성화한 복합 효과, Agent 효과 | H2a protocol/결과 검증 및 별도 승인 전; retrieval-only A/B 미실행 |
| **H3 legal transfer** | AIHub 한국어 법률의 text-only **parent document** retrieval | 고정된 방법이 source-parent gold 판례의 parent-document 회수를 개선하는가 | 전반적인 법률 relevance, MIRACL 성공으로 법률 실패를 상쇄하는 주장 | 기존 Legal Gate L0와 primary protocol 미완료; 실행 미실행 |
| **H4 target-intersection** | 법률/보험 도메인의 표·이미지·도형을 포함한 장문 복합 문서; **element와 parent document를 함께 평가** | target-domain 복합 문서에서 parser-derived element navigation과 parent-document 회수가 함께 확인되는가 | 일반 복합 문서 H2와 text-only 법률 H3의 별도 성공을 제품 성능으로 조합하는 주장 | D0/H4 데이터 후보 선정·EDA 및 protocol 미완료; 실행 미실행 |
| **최종 제품 주장** | H2a, H2b, H3, H4의 사전 등록 결과 | H2a/H2b의 구조·의미 효과, H3의 법률 parent 효과, H4의 법률/보험 복합문서 교집합 효과가 각각 승인된 계약에서 확인되는가 | H1 단독 성공, H2와 H3의 별도 성공만, 데이터셋 점수 평균·합산, 물리 문서 11만 건 운영 검증 | 미실행 |

판정은 데이터셋과 단위를 합산하지 않는다. 특히 H1 성공은 H2a/H2b/H3/H4 실패를 구제하지 않고, H3는 현재 AIHub qrel의 source-parent 연결 범위를 넘어 “법률 전반의 관련성”으로 해석하지 않는다. H2와 H3의 별도 성공은 필요하지만 충분하지 않다. **최종 제품 수준의 주장은 H4 target-intersection이 사전 승인된 protocol에서 통과할 때만 가능하다.**

### 최종 제품 판정표

| H2a/H2b | H3 | H4 target-intersection | 허용되는 해석 |
|---|---|---|---|
| 사전 기준 통과 | 사전 기준 통과 | 사전 기준 통과 | 법률/보험 복합 문서에서 element와 parent 수준의 제한된 제품 가설을 지지한다. |
| 사전 기준 통과 | 사전 기준 통과 | 미실행 또는 미달 | H2/H3의 개별 근거만 기록한다. 법률/보험 복합 문서 제품 성능은 주장하지 않는다. |
| 하나 이상 미달 | 어떤 결과든 | 어떤 결과든 | 미달한 가설을 그대로 기록하며, 다른 데이터셋 점수로 보정하거나 합산하지 않는다. |
| 어떤 결과든 | 어떤 결과든 | 미실행 | H4가 없으므로 최종 제품 주장은 보류한다. |

H1의 결과는 위 표의 어느 행도 바꾸지 않는다. H1은 H4 전의 일반-domain text-only 방법 근거일 뿐이다.

## 3. 기존 산출물 영향도

| 기존 산출물 또는 경로 | 보존되는 역할 | 이 amendment 이후 금지되는 해석·용도 | 변경 여부 |
|---|---|---|---|
| `docs/MIRACL_KO_PRETEST_20260723.md`, 20K/50K/110K fixture | H1의 Korean text passage 데이터 준비와 controlled distractor scale 입력 | 장문 복합 문서, 표·그림, parent-document 또는 법률/보험 검증 | 무변경 |
| `docs/MIRACL_KO_LEXICAL_SMOKE_20260723.md` 및 revalidation | H1 전의 standalone lexical plumbing/text-only baseline | taxonomy 효과, @tag 효과, Agentic RAG 효과, 신한 운영 성능 | 무변경·재실행 금지 |
| MIRACL taxonomy generation contract, exact plan 후보, KT candidate/bundle 설계 | H1 taxonomy artifact 생성의 **후보 실행 계약** | 승인·실행 완료, H2/H4 element artifact, A/B 결과 | 무변경·실행 보류; D0는 별도 승인된 H1 실행을 자동 차단하거나 H2/H4 증거로 승격하지 않음 |
| `docs/LEGAL_PRIMARY_MIRACL_SECONDARY_EVALUATION_PLAN_20260724.md` | AIHub legal primary/MIRACL secondary의 역사적 위계와 계속 유효한 Legal Gate L0·정합성·license/PII 계약 | H2/H4가 이미 구현됐거나 MIRACL focused runner가 존재한다는 주장, 또는 이 amendment 이전의 최종 제품 판정 위계 | 무변경; 향후 평가 위계·최종 성공 해석은 이 amendment가 supersede |
| TREC-COVID focused Part 1·2 및 historical 결과 | 기존 document-level 역사 기록과 TREC 전용 실행 계약 | H1/H2a/H2b/H3/H4의 새 성공 근거 또는 신한 운영 성능 | 무변경 |
| `scripts/build_tags.py`와 기존 tag 파일 형식 | 현행 구현의 감사 대상·프로토타입 | parser-derived element treatment, D0 통과 artifact, H2 실험 입력 | 코드 변경 없음; H2에 사용 금지 |
| H4 대상 법률/보험 복합문서 corpus·parser output·gold | 아직 없는 D0/H4 후보 선정·EDA의 대상 | H2 또는 H3의 결과를 해당 data 없이 H4/제품 증거로 표현 | 아직 미선정·미생성 |

## 4. `@el` 현행 코드 결함과 미검증 범위

다음은 실행 결과가 아니라 정적 코드 대조 결과다. 이 amendment에서는 수리·실행하지 않는다.

| 항목 | 코드 근거 | 영향 | 현 상태 |
|---|---|---|---|
| Parser 부재 | `scripts/build_tags.py:54-84`는 `doc["text"]`를 `\n{2,}`로만 분할하고 `idx`, `text`, `doc_id`만 만든다. | page, bbox, 원본 파일, 실제 structural type을 잃는다. | H2 입력으로 부적합 |
| 축약된 의미 입력 | `scripts/build_tags.py:96-100`은 각 element의 앞 200자만 전달한다. | 긴 표·캡션·목록·문맥을 근거로 한 태그를 검증할 수 없다. | 미검증 |
| 구조 추정 | `scripts/build_tags.py:16-46`은 `paragraph/table/list/figure`를 모델 출력으로 요구하지만 parser-derived 구조 필드를 주지 않는다. | table/list/figure는 실제 구조가 아닌 텍스트 추정이 된다. | H2 주장 불가 |
| 무음 fallback | `scripts/build_tags.py:112-131`은 `None`, JSON parse 오류, invalid type/sub를 paragraph/summary 또는 기본 sub로 바꾼다. | API·parse·schema 실패가 Other/unknown 또는 오류로 드러나지 않아 treatment 품질을 측정할 수 없다. | fail-loud 계약 미구현 |
| tag namespace 불일치 | A 방식 출력은 `@el:{type}/{sub}` (`scripts/build_tags.py:125-132`)이나 Agent는 `@el:definition` 등의 sub-only 문자열을 안내한다 (`src/agent/dci_agent.py:59-62`). | 예를 들어 `@el:definition`은 `@el:paragraph/definition`의 substring도 아니므로 기대한 exact element를 찾지 못한다. | consumer 계약 미구현 |
| 문자열 포함 filter | `Workspace.grep()`은 `tag_filter in elem["tag"]`를 사용한다 (`src/agent/workspace.py:39-50`). | namespace, field, equality, unknown policy가 없는 substring matching이라 typed filter가 아니다. | H2 A/B에 사용 금지 |
| 최초 retrieval 복구 불가 | `PullRetriever`는 tags를 index/pull 입력으로 사용하지 않으며 (`src/agent/retriever.py:66-165`), tag는 pull 뒤 `Document.tags`로 workspace에 주입된다 (`src/agent/dci_agent.py:432-463`). | workspace grep은 이미 pull된 후보만 탐색하므로 최초 retrieval miss를 회복할 수 없다. | retrieval 효과 미검증 |

그러므로 현행 @tag 경로에는 다음 결과가 **없다**: parser fidelity, tag accuracy, typed-filter precision/recall, tag-control 대비 retrieval delta, first-pull recovery, Agent answer delta, 표·그림·목록 처리 정확도. 현행 fallback과 문자열 포함 매칭은 H2 실험의 control/treatment로 사용하지 않는다.

## 5. Gate D0 — H2/H4 parser-derived element 입력·평가 계약

Gate D0는 H2와 H4를 위한 모델 호출 전 데이터·계약 gate다. 아래 항목이 immutable manifest와 validator로 확인되기 전에는 H2/H4 tag treatment 또는 H2/H4 Agent 실행을 승인하지 않는다. D0는 별도로 승인된 H1 MIRACL text taxonomy 실행을 자동 차단하지 않으며, H1 결과를 H2/H4 증거로 승격하지도 않는다.

### 5.1 원본·parser·식별자 계약

| 범주 | 필수 계약 |
|---|---|
| 원본 | 원본 PDF/PPTX/DOCX/image와 byte SHA-256, 파일 형식, 라이선스·내부 사용 범위, source revision/획득 시각을 보존한다. 추출 텍스트만 있는 corpus는 H2의 구조 평가 입력으로 승인하지 않는다. |
| parent | `parent_doc_id`, revision/source URI 또는 안전한 source reference, document hash, 문서 순서와 parent-level text reference를 고정한다. |
| element | `element_id`는 parent 안에서 안정·고유해야 하며, `parent_doc_id`, `element_order`, `page`, `bbox`, original source reference를 가진다. `bbox`는 좌표계, page width/height, 회전/단위를 함께 기록한다. |
| parser provenance | parser 이름·버전·컨테이너/코드 hash·설정·OCR 엔진·입력 hash·실행 시각·실패/제외 element를 manifest에 기록한다. 재실행 시 동일 parser input에서 ID와 좌표가 결정론적으로 재현돼야 한다. |

### 5.2 구조와 의미를 분리한 typed element schema

`structural_type`과 `semantic_role`은 서로 다른 field이며 한 값을 다른 field의 대체값으로 쓰지 않는다.

```json
{
  "parent_doc_id": "opaque-parent-id",
  "element_id": "opaque-element-id",
  "element_order": 17,
  "page": 4,
  "bbox": {"coordinate_space": "page-points-v1", "x0": 72.0, "y0": 144.0, "x1": 520.0, "y1": 318.0},
  "structural_type": "table",
  "semantic_role": "eligibility_condition",
  "text": "parser-extracted text",
  "source_ref": {"file_sha256": "...", "page": 4, "object_ref": "..."}
}
```

- `structural_type`은 parser 증거가 있는 closed enum(예: heading, paragraph, list, list_item, table, table_cell, figure, caption, footnote)만 허용한다.
- `semantic_role`은 별도 closed schema와 provenance를 가지며, 구조 type을 LLM이 추정해 채우는 방식과 혼합하지 않는다.
- `unknown`은 명시적 상태·사유·provenance를 가진 element assignment일 뿐, parse/API/schema 실패의 무음 fallback이 아니다. 실패는 retry 정책을 거친 뒤 fail-loud로 남긴다.

### 5.3 표·목록·그림의 최소 구조

| 구조 | 최소 보존 정보 |
|---|---|
| table | `table_id`, parent/page/bbox, cell text, `row_index`, `column_index`, header 여부, row/column span, table-caption 및 source reference. 표 전체와 각 cell의 관계가 역참조 가능해야 한다. |
| list | `list_id`, item order/level, parent list와 list item 관계, marker와 text, page/bbox. |
| image/figure | `figure_id`, page/bbox, immutable original/crop reference, caption element ID·text, OCR text와 OCR engine/version/provenance. caption/OCR가 없음을 임의 텍스트로 대체하지 않는다. |

### 5.4 시각 정보 주장 경계

- OCR, caption, page/bbox 및 원본 figure reference만 검색·필터·표시에 쓰는 결과는 **figure navigation** 또는 **text-mediated retrieval**로만 표기한다.
- pixel/image embedding 또는 VLM 입력이 없고, 사람 검수된 visual gold가 없으면 figure/diagram의 내용 이해, 도형 추론, 이미지 질의응답을 주장하지 않는다.
- visual gold는 caption text와 별개로 figure가 질의의 시각적 근거인지 검수한 qrel/annotation이어야 한다. parser가 추출한 bbox나 OCR text 자체를 visual understanding 정답으로 사용하지 않는다.

### 5.5 독립 gold·qrel·leakage·license 계약

- parser 산출물은 구조 정답이 아니다. 별도 검수 gold에 structural type, page/bbox, table row/column/span, figure-caption 연결을 기록하고 parser fidelity를 독립적으로 측정한다. structural type exact agreement, bbox overlap/IoU, table relation/cell-span agreement, caption-link agreement의 분모·제외 규칙은 retrieval qrel과 분리해 사전 고정한다.
- parent qrel과 element qrel도 parser fidelity gold와 별도 파일·단위·분모로 관리한다. 각 retrieval qrel은 `qid`와 `parent_doc_id` 또는 `element_id`를 명시하며, element positive가 parent positive로 연결되는 규칙은 별도 필드와 검증으로 고정한다. parser output을 그대로 retrieval gold로 복사하거나, parser 오류를 정답으로 승인하지 않는다.
- 하나의 질의가 여러 parent 또는 여러 element gold를 가질 수 있다. parent와 element recall/Hit/nDCG를 합산하거나 이름만 바꾸어 재사용하지 않는다.
- taxonomy/tag generator, query tagger, retrieval filter 및 Agent 입력은 query, qrel, relevance, gold/evidence ID, answer, rubric, source-parent 연결 label을 읽지 않는다. 이 금지는 API signature, input hash manifest, negative test로 검사한다.
- source 원본, parser 산출물, OCR, qrel, license/PII 검사를 분리한다. PII/사용 제한/재배포 제한이 있는 record는 raw 포함 여부와 결과 공개 범위를 실행 전 결정한다.

### 5.6 H2a/H2b 정확한 typed-filter와 인과 분리

H2의 arm은 동일한 parent/element corpus, parser output, text, element IDs, retrieval backend/model, query 집합, qrel, top-k, candidate budget, seed, latency 측정 및 raw-row 계약을 쓴다. parsing, chunking, prompt, query rewrite, pull 수, score boost, rerank, Agent 조건을 단계 사이에 함께 바꾸지 않는다.

typed filter는 문자열 포함·prefix match가 아니라 canonical object의 field-level exact equality만 쓴다. `unknown`은 eligible이 아니며 `@el:` 문자열·display label·substring으로 비교하지 않는다. query-side filter는 gold/qrel을 보지 않는 고정 규칙 또는 독립적으로 고정한 annotation으로만 만들고 source·version·hash를 기록한다.

| 단계 | Control | Treatment | 유일한 변경점 | 금지 |
|---|---|---|---|---|
| **H2a structural type** | 동일 parser element candidate universe에서 typed filter 비활성화; tag label을 prompt/index/score에 노출하지 않음 | 같은 candidate universe에서 `structural_type`만 exact filter; `semantic_role=not_applicable` | `structural_type` typed-filter 활성화 | semantic role, score boost, rerank, prompt, Agent 동시 변경 |
| **H2b semantic role** | H2a의 `structural_type` exact filter를 같은 값·같은 query set·같은 config로 유지 | H2a 조건에 `semantic_role` exact filter 하나를 추가 | `semantic_role` typed-filter 추가 | structural filter 값을 새로 바꾸기, H2a와 H2b를 처음부터 한 composite treatment로 만들기, Agent 동시 변경 |

H2b는 H2a의 data/config/raw-row 검증이 끝나고 별도 승인된 뒤에만 수행한다. H2a의 effect 결과를 보고 filter 값·query subset·metric을 재튜닝하지 않는다. H2a 또는 H2b의 retrieval-only 결과는 별도 Agent 평가 없이 Agent navigation·answer 성능으로 표현하지 않는다.

H2a filter 예시는 다음과 같다.

```json
{
  "filter_contract_version": "element-filter-v1",
  "namespace": "parser-element-v1",
  "structural_type": "table",
  "semantic_role": "not_applicable",
  "operator": "structural_type_exact"
}
```

H2b는 위 object의 `structural_type`을 그대로 유지하고 `semantic_role`을 closed enum 값으로 채우며 `operator: "all_exact"`을 사용한다. 최초 retrieval miss 회복을 주장하려면 이 filter가 workspace grep이 아니라 승인된 retrieval-stage candidate selection에 적용되고 control과 같은 per-query candidate row로 측정돼야 한다. 그렇지 않으면 결과는 pulled-workspace inspection 효과로만 표기한다.

### 5.7 D0 통과 기준과 산출물

1. 원본·parser·element·qrel·license manifest의 SHA와 row count가 서로 대조된다.
2. orphan/duplicate/conflicting parent·element ID, invalid page/bbox, missing parent link, qrel orphan은 fail-loud 한다.
3. table/list/figure/caption/OCR 관계와 parser provenance가 샘플과 전체 집계에서 검증되고, parser fidelity gold와 retrieval qrel의 독립성이 검토된다.
4. leakage 검사, H2a/H2b typed-filter exact-match 검사, tagless-control isolation 검사가 negative fixture로 통과한다.
5. parent·element raw rows, parser fidelity metric, retrieval metric, paired comparison 입력, 실패·제외 query 수와 provenance가 결과 전에 schema로 고정된다.

통과 산출물은 source/hash manifest, parser manifest, normalized parent/element records, independent parser-fidelity gold, parent/element retrieval-qrel manifests, leakage/license report, H2a/H2b typed-filter schema, control/treatment config, synthetic fixture 및 validator다. D0 통과는 tag 품질이나 retrieval 개선을 뜻하지 않는다.

## 6. 다음 실행 순서와 승인 gate

| 순서 | 승인 전 작업 | 다음 단계로 가기 위한 조건 | 금지되는 자동 진행 |
|---|---|---|---|
| 0 | 본 amendment와 기존 결과를 동결 | 문서 검토 완료 | 모델 실행·KT bundle 재생성 |
| 1 | H1 candidate execution은 별도 재승인까지 보류 | 기존 immutable model/runtime/plan/approval 입력과 H1 실행 승인 | D0가 H1을 자동 차단하는 것, H1 결과를 H2/H3/H4 성공으로 해석 |
| 2 | **Gate D0/H4 후보 선정·EDA**: 법률/보험 복합 원본, parser, independent gold, qrel, leakage/license 계약 준비 | D0 validator와 독립 검토 승인 | taxonomy/@tag 생성 또는 retrieval 실행, 후보 결과를 H4 효과로 표현 |
| 3 | H2a structural-type retrieval-only protocol 고정 | H2a 단일변수, element/parent metric, parser-fidelity metric, minimum practical effect, raw-row/CI 계약 승인 | semantic role·rerank·prompt·Agent를 동시 추가 |
| 4 | H2a retrieval-only control 대 treatment 1회 | 결과 validator·paired 분석·provenance audit 통과 | H2a 결과를 보고 filter/query/metric 재튜닝, Agent 자동 실행 |
| 5 | H2b semantic-role retrieval-only protocol 및 1회 실행 | H2a의 data/config/raw-row 검증 완료와 H2b 별도 승인; H2a structural filter 고정 | H2a와 H2b를 복합 treatment로 재해석, Agent 자동 실행 |
| 6 | H2 Agent protocol (선택적 후속) | H2 retrieval-only가 사전 승인한 실질 기준을 통과하고, Agent budget/prompt/judge가 별도 승인 | retrieval-only 성공을 Agent navigation·answer 성능으로 표현 |
| 7 | H3 Legal Gate L0와 parent-document primary protocol 승인 | held-out query/exclusion hash, 법률 metric/effect criterion, license/PII 계약 승인 | MIRACL/H2 결과로 H3 gate 우회 |
| 8 | H3 primary 1회 실행 | H3 결과 validator·paired 분석·provenance audit 통과 | H3 결과만으로 H4/복합문서 제품 주장 |
| 9 | H4 target-intersection protocol 및 1회 평가 | D0 통과, 법률/보험 복합문서의 element+parent qrel, H4 effect criterion, independent parser-fidelity audit 승인 | H2/H3 결과를 H4 대신 사용, 시각 정보만으로 image understanding 주장 |
| 10 | 최종 제품 판정 | H2a/H2b, H3, H4의 독립적 사전 계약 결과를 최종 판정표대로 해석 | 데이터셋 점수 합산, 사후 성공 기준 변경 |

현재의 다음 허용 작업은 **Gate D0/H4에 맞는 법률/보험 복합 문서 데이터 후보 선정과 EDA**뿐이다. taxonomy artifact, taxonomy/@tag A/B, Agent, Peter Part 1~3, AIHub legal primary, KT bundle 재생성과 모델/API 호출은 모두 **미실행·미승인** 상태로 유지한다.

## 7. 현재 완료 상태의 정확한 표현

- **보존 완료:** MIRACL-ko text-only passage fixture와 standalone lexical plumbing baseline, 기존 taxonomy execution contract 및 역사 문서.
- **이번 amendment 완료:** H1/H2a/H2b/H3/H4 평가 위계, 현행 @tag 코드의 제한, D0 독립 gold·시각 주장 경계와 후속 승인 순서의 사전 기록.
- **완료가 아님:** taxonomy artifact 생성, taxonomy 효과, @tag 정확도/효과, parser fidelity, Agent 효과, Part 1~3, AIHub legal primary, H4 target-intersection, 법률/보험 복합문서 최종 제품 검증.

이 구분을 유지함으로써 현재의 text-only baseline을 삭제하지 않으면서도, parser-derived evidence가 없는 MIRACL passage 결과가 구조 요소·법률/보험 제품 주장으로 확장되는 것을 막는다.
