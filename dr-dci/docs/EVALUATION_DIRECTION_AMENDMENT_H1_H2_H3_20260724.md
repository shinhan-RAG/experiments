# Peter 실험 평가 방향 Amendment — H1 텍스트, H2 요소, H3 법률

기록일: 2026-07-24
상태: **사전 설계 감사 기록 — 모든 taxonomy/@tag A/B·Agent·Part 1~3 실행 미실행**

## 1. 변경 이유와 사전성

이 amendment는 MIRACL-ko taxonomy artifact 생성, taxonomy A/B, element-tag A/B, Agent 실행, Peter Part 1~3 및 AIHub 법률 primary 실행이 모두 **미실행**인 상태에서 기록한다. 따라서 관측된 treatment 결과를 보고 성공 기준을 바꾸거나 유리한 데이터셋을 사후 선택한 변경이 아니다.

기존 MIRACL-ko fixture와 lexical smoke는 이미 확보된 **title/text 기반 passage retrieval의 text-only baseline**으로 보존한다. 이 baseline은 taxonomy 효과, element-tag 효과, Agentic 효과, 법률/보험 전이를 보여 주지 않는다. MIRACL taxonomy generator/KT 후보 계약도 삭제하거나 재해석하지 않으며, 실제 모델 실행은 이 amendment 이후 별도 승인 전까지 보류한다.

변경의 근거는 데이터와 현행 `@el` 경로의 표현 범위다.

- MIRACL-ko corpus schema는 `corpus_id`, `article_id`, `passage_index`, `title`, `text`만 제공하는 passage 계약이다. 표, 그림, 페이지 좌표 또는 parser element는 없다.
- 현행 `scripts/build_tags.py`는 parser 산출물을 읽지 않고 `text`를 빈 줄로 나누며, 각 분할의 앞 200자만 LLM 입력으로 쓴다.
- 따라서 MIRACL text-only 결과로 표·목록·그림·문서 좌표 기반 element navigation을 주장할 수 없다.

이 문서는 기존 [AIHub 법률 primary · MIRACL-ko secondary 평가 계획](LEGAL_PRIMARY_MIRACL_SECONDARY_EVALUATION_PLAN_20260724.md)을 역사 기록으로 보존한다. 아래 위계는 앞으로의 실행과 해석을 위한 amendment이며, 기존 fixture·lexical 결과·taxonomy contract·KT 후보의 원문이나 수치를 수정하지 않는다.

## 2. 사전 등록한 가설별 평가 위계

| 가설 | 평가 대상과 단위 | 확인하려는 제한된 질문 | 성공 근거로 사용할 수 없는 것 | 현재 상태 |
|---|---|---|---|---|
| **H1 taxonomy** | MIRACL-ko, text-only **passage** retrieval | 고정 taxonomy soft boost가 일반 한국어 passage text retrieval에서 baseline 대비 효과가 있는가 | 표/그림/목록 인식, parent-document 회수, 법률·보험 전이 | fixture와 lexical baseline만 있음; taxonomy artifact/A-B/Agent 미실행 |
| **H2 element tags** | 장문 복합 문서의 parser-derived **element** retrieval 및 element-to-parent navigation | 실제 구조·의미가 있는 element tag와 typed filter가 tag 없는 control보다 element/parent gold 회수를 개선하는가 | 평문 줄바꿈 분할, 추정된 table/list/figure, MIRACL passage 결과 | Gate D0 및 별도 protocol 미완료; @tag A/B·Agent 미실행 |
| **H3 legal transfer** | AIHub 한국어 법률의 **parent document** retrieval | 고정된 방법이 source-parent gold 판례의 parent-document 회수를 개선하는가 | 전반적인 법률 relevance, MIRACL 성공으로 법률 실패를 상쇄하는 주장 | 기존 Legal Gate L0와 primary protocol 미완료; 실행 미실행 |
| **최종 제품 주장** | 법률/보험의 장문 복합 문서 | H2의 parser-derived element navigation과 H3의 target-domain parent-document 효과가 모두 확인되는가 | H1 단독 성공, 한 데이터셋의 점수 평균·합산, 물리 문서 11만 건 운영 검증 | 미실행 |

판정은 데이터셋과 단위를 합산하지 않는다. 특히 H1 성공은 H2 또는 H3 실패를 구제하지 않고, H3는 현재 AIHub qrel의 source-parent 연결 범위를 넘어 “법률 전반의 관련성”으로 해석하지 않는다. 최종 제품 수준의 주장은 H2와 H3가 각각 사전 승인된 protocol에서 확인된 뒤에만 가능하다.

## 3. 기존 산출물 영향도

| 기존 산출물 또는 경로 | 보존되는 역할 | 이 amendment 이후 금지되는 해석·용도 | 변경 여부 |
|---|---|---|---|
| `docs/MIRACL_KO_PRETEST_20260723.md`, 20K/50K/110K fixture | H1의 Korean text passage 데이터 준비와 controlled distractor scale 입력 | 장문 복합 문서, 표·그림, parent-document 또는 법률/보험 검증 | 무변경 |
| `docs/MIRACL_KO_LEXICAL_SMOKE_20260723.md` 및 revalidation | H1 전의 standalone lexical plumbing/text-only baseline | taxonomy 효과, @tag 효과, Agentic RAG 효과, 신한 운영 성능 | 무변경·재실행 금지 |
| MIRACL taxonomy generation contract, exact plan 후보, KT candidate/bundle 설계 | H1 taxonomy artifact 생성의 **후보 실행 계약** | 승인·실행 완료, H2 element tag artifact, A/B 결과 | 무변경·실행 보류 |
| `docs/LEGAL_PRIMARY_MIRACL_SECONDARY_EVALUATION_PLAN_20260724.md` | AIHub legal primary/MIRACL secondary의 역사적 평가 위계와 Legal Gate L0 | H2가 이미 구현됐거나 MIRACL focused runner가 존재한다는 주장 | 무변경; 이 amendment가 향후 H1/H2/H3 해석을 보완 |
| TREC-COVID focused Part 1·2 및 historical 결과 | 기존 document-level 역사 기록과 TREC 전용 실행 계약 | H1/H2/H3의 새 성공 근거 또는 신한 운영 성능 | 무변경 |
| `scripts/build_tags.py`와 기존 tag 파일 형식 | 현행 구현의 감사 대상·프로토타입 | parser-derived element treatment, D0 통과 artifact, H2 실험 입력 | 코드 변경 없음; H2에 사용 금지 |

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

## 5. Gate D0 — parser-derived element 입력·평가 계약

Gate D0는 모델 호출이나 taxonomy 생성 전의 데이터·계약 gate다. 아래 항목이 immutable manifest와 validator로 확인되기 전에는 H2 tag treatment 또는 H2 Agent 실행을 승인하지 않는다.

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

### 5.4 qrel·leakage·license 계약

- parent qrel과 element qrel을 별도 파일·단위·분모로 관리한다. 각 qrel은 `qid`와 `parent_doc_id` 또는 `element_id`를 명시하며, element positive가 parent positive로 연결되는 규칙은 별도 필드와 검증으로 고정한다.
- 하나의 질의가 여러 parent 또는 여러 element gold를 가질 수 있다. parent와 element recall/Hit/nDCG를 합산하거나 이름만 바꾸어 재사용하지 않는다.
- taxonomy/tag generator, query tagger, retrieval filter 및 Agent 입력은 query, qrel, relevance, gold/evidence ID, answer, rubric, source-parent 연결 label을 읽지 않는다. 이 금지는 API signature, input hash manifest, negative test로 검사한다.
- source 원본, parser 산출물, OCR, qrel, license/PII 검사를 분리한다. PII/사용 제한/재배포 제한이 있는 record는 raw 포함 여부와 결과 공개 범위를 실행 전 결정한다.

### 5.5 정확한 typed-filter와 단일변수 A/B

H2의 첫 비교는 **tag 없는 control 대 parser-derived typed-tag treatment 한 쌍**이다. 두 arm은 동일한 parent/element corpus, parser output, text, element IDs, retrieval backend/model, query 집합, qrel, top-k, candidate budget, seed, latency 측정과 answer/judge 조건을 쓴다. treatment 외의 parsing, chunking, prompt, query rewrite, pull 수를 함께 바꾸지 않는다.

typed filter는 문자열 포함·prefix match가 아니라 다음 canonical object의 field-level exact equality만 쓴다.

```json
{
  "filter_contract_version": "element-filter-v1",
  "namespace": "parser-element-v1",
  "structural_type": "table",
  "semantic_role": "eligibility_condition",
  "operator": "all_exact"
}
```

- 각 field는 closed enum 또는 explicit `not_applicable`이어야 한다. `unknown`은 boost/filter eligible이 아니며, `@el:` 문자열·display label·substring으로 비교하지 않는다.
- query-side typed filter는 gold/qrel을 보지 않는 고정 규칙 또는 독립적으로 고정된 annotation으로만 만들고, source·version·hash를 기록한다.
- control은 동일 element candidate universe에서 typed filter를 **비활성화**하고 tag label을 prompt/index/score에 노출하지 않는다. treatment의 유일한 추가 작동점은 사전 고정한 typed filter 또는 tag-based eligibility이며, score boost·rerank·Agent prompt 변경을 동시에 추가하지 않는다.
- 최초 retrieval miss 회복을 주장하려면 tag가 workspace grep만이 아니라 승인된 retrieval-stage candidate selection에 적용돼야 하며, control과 같은 raw per-query candidate row에서 이를 측정해야 한다. 그렇지 않으면 결과는 pulled-workspace inspection 효과로만 표기한다.

### 5.6 D0 통과 기준과 산출물

1. 원본·parser·element·qrel·license manifest의 SHA와 row count가 서로 대조된다.
2. orphan/duplicate/conflicting parent·element ID, invalid page/bbox, missing parent link, qrel orphan은 fail-loud 한다.
3. table/list/figure/caption/OCR 관계와 parser provenance가 샘플과 전체 집계에서 검증된다.
4. leakage 검사, typed-filter exact-match 검사, tagless-control isolation 검사가 negative fixture로 통과한다.
5. parent·element raw rows, metric definition, paired comparison 입력, 실패·제외 query 수와 provenance가 결과 전에 schema로 고정된다.

통과 산출물은 source/hash manifest, parser manifest, normalized parent/element records, parent/element qrel manifests, leakage/license report, typed-filter schema, control/treatment config, synthetic fixture 및 validator다. D0 통과는 tag 품질이나 retrieval 개선을 뜻하지 않는다.

## 6. 다음 실행 순서와 승인 gate

| 순서 | 승인 전 작업 | 다음 단계로 가기 위한 조건 | 금지되는 자동 진행 |
|---|---|---|---|
| 0 | 본 amendment와 기존 결과를 동결 | 문서 검토 완료 | 모델 실행·KT bundle 재생성 |
| 1 | H1 candidate execution은 별도 재승인까지 보류 | 기존 immutable model/runtime/plan/approval 입력과 실행 승인 | H1 결과를 H2/H3 성공으로 해석 |
| 2 | **Gate D0** 원본·parser·qrel·leakage/license 계약 준비 | D0 validator와 독립 검토 승인 | taxonomy/@tag 생성 또는 retrieval 실행 |
| 3 | H2 control/treatment protocol과 typed-filter contract 고정 | 단일변수, parent/element metric, minimum practical effect, raw-row/CI 계약 승인 | 여러 tag·prompt·rerank·Agent 변경의 동시 실행 |
| 4 | H2 retrieval-only control 대 treatment 1회 | 결과 validator·paired 분석·provenance audit 통과 | H2 실패 후 조건 재튜닝 또는 Agent 자동 실행 |
| 5 | H3 Legal Gate L0와 parent-document primary protocol 승인 | held-out query/exclusion hash, 법률 metric/effect criterion, license/PII 계약 승인 | MIRACL/H2 결과로 H3 gate 우회 |
| 6 | H3 primary 1회 실행 및 최종 제품 판정 | H2와 H3의 독립적 사전 계약 결과를 위계대로 해석 | 데이터셋 점수 합산, 사후 성공 기준 변경 |

현재 허용되는 작업은 Gate D0의 비실행 설계·데이터 계약 감사뿐이다. taxonomy artifact, taxonomy/@tag A/B, Agent, Peter Part 1~3, AIHub legal primary, KT bundle 재생성과 모델/API 호출은 모두 **미실행·미승인** 상태로 유지한다.

## 7. 현재 완료 상태의 정확한 표현

- **보존 완료:** MIRACL-ko text-only passage fixture와 standalone lexical plumbing baseline, 기존 taxonomy execution contract 및 역사 문서.
- **이번 amendment 완료:** H1/H2/H3 평가 위계, 현행 @tag 코드의 제한, Gate D0와 후속 승인 순서의 사전 기록.
- **완료가 아님:** taxonomy artifact 생성, taxonomy 효과, @tag 정확도/효과, Agent 효과, Part 1~3, AIHub legal primary, 법률/보험 복합문서 최종 제품 검증.

이 구분을 유지함으로써 현재의 text-only baseline을 삭제하지 않으면서도, parser-derived evidence가 없는 MIRACL passage 결과가 구조 요소·법률/보험 제품 주장으로 확장되는 것을 막는다.
