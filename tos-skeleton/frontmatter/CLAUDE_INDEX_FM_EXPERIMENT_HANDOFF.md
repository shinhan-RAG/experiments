# Claude 핸드오프: Index -> Frontmatter 문서 선택 실험

## Claude에게 내리는 작업 지시

이 문서를 읽은 뒤 추가 설계 토론으로 멈추지 말고, 아래 명세대로 실험 runner를 구현하고 smoke test까지 실행하라.

목표는 Collection의 9,417개 문서 중 사용자 질문에 맞는 `document_id` 하나를 선택하는 것이다. 답변 생성과 Chunk/Element 검색은 하지 않는다.

## 변경하면 안 되는 검색 흐름

```text
질문
  -> Collection Index에서 아직 보지 않은 후보 5개 선택
  -> 그 5개 문서의 Frontmatter만 읽기
  -> 적합 문서가 있으면 1개 선택
  -> 없으면 Index로 돌아가 다음 후보 5개 선택
  -> 해당 5개의 Frontmatter만 읽기
  -> 선택하거나 최대 10개 배치 후 not_found
```

DR-DCI, BM25, 외부 임베딩 모델, 외부 reranker 실험이 아니다. **Claude 자신이 Index와 Frontmatter를 읽고 반복적으로 문서를 선택하는 에이전트**다.

## 작업 위치와 준비된 파일

```text
/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter
```

입력 artifact:

```text
out/collection_index.json
out/doc_frontmatter.jsonl
out/document_search_qa_500_candidate_v1.jsonl
```

QA 설명과 검수 파일:

```text
DOCUMENT_SEARCH_QA_500_CANDIDATE.md
out/document_search_qa_500_candidate_v1_manifest.json
out/document_search_qa_500_candidate_v1_review.csv
```

구현된 cascade 제어 규칙과 테스트:

```text
structured_document_search.py
test_structured_document_search.py
```

먼저 실행:

```bash
cd /Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter
python3 -m unittest -v test_structured_document_search.py
```

## 사용 금지

다음은 잘못된 평면 BM25 실험이다. 본 실험에 사용하지 마라.

```text
eval_document_search_qa_500.py
out/document_search_qa_500_bm25_eval_v1.json
out/document_search_qa_500_bm25_details_v1.jsonl
DOCUMENT_SEARCH_QA_500_BM25_RESULT.md
```

QA 500건은 유지하고 위 검색 결과만 무효로 취급한다.

## A/B/C/D 조건

| Arm | 접근 가능 artifact | 실행 방식 |
|---|---|---|
| A | 파일명과 경로 | 파일명만 보고 선택 |
| B | 파일명과 경로 + Index | Index만 보고 선택. Frontmatter 접근 금지 |
| C | 파일명과 경로 + Frontmatter | Frontmatter만 보고 선택. Index 접근 금지 |
| D | 파일명과 경로 + Index + Frontmatter | Index 후보 5개 -> 해당 FM 확인 -> 없으면 다음 5개 |

격리는 지시문으로만 처리하지 마라. Arm별 입력 디렉터리 또는 artifact access guard로 허용되지 않은 파일을 읽을 수 없게 만들어라.

- A에서 Index/FM 접근 시 error
- B에서 FM 접근 시 error
- C에서 Index 접근 시 error
- D에서 현재 Index 후보 배치 밖의 FM 접근 시 error

## Claude 실행 통제

QA 한 건과 Arm 한 개를 **새 Claude 세션**에서 실행한다. 한 세션에서 여러 Arm을 처리하면 이전 Arm의 artifact가 컨텍스트에 남으므로 금지한다.

고정하고 결과에 기록:

- Claude 모델 식별자
- sampling 설정
- system prompt hash
- tool allowlist
- max turns
- `candidate_batch_size=5`
- `max_batches=10`

별도의 LLM을 검색 내부에서 호출하지 않는다. 실험을 수행하는 Claude가 유일한 판단 모델이다.

## Claude에 제공할 입력

```json
{
  "qid": "mixed-0001",
  "query": "사용자 질문 원문"
}
```

다음 필드는 평가 runner만 읽을 수 있다. Claude 검색 세션에 전달하지 마라.

```text
gold_document_id
gold_source_path
constraints
evidence
review_status
```

## Index 후보 선정

Index 필드를 구조대로 읽는다.

- `product_name`: 상품 조건
- `title`, `file_name`: 문서 식별 단서
- `doc_type`: 문서 종류
- `effective_date`, `version`: 날짜와 개정 조건
- `is_representative`: 최신 정본 조건
- `flags`, `kind_hints`: 추가 상품 조건
- `document_id`: Frontmatter 조회 키

Index 전체를 대화 컨텍스트에 넣지 말고 `jq`, `rg` 또는 제한된 helper로 필요한 엔트리만 조회한다.

후보 우선순위:

1. 질문에 명시된 상품명
2. 질문에 명시된 문서 종류
3. 질문에 명시된 날짜, 연도, 버전, 최신 여부
4. title과 질문의 내용 단서
5. 아직 확인하지 않은 문서

매 배치에서 후보 `document_id` 5개와 선정 이유를 로그로 남긴다. Index 단계에서 최종 정답으로 확정하지 않는다.

## Frontmatter 판정

D에서는 현재 후보 배치의 `document_id`에 해당하는 Frontmatter만 읽는다. 전체 Frontmatter 검색으로 Index 단계를 우회하지 않는다.

각 후보에 대해 기록:

```json
{
  "document_id": "doc-...",
  "relevant": true,
  "matched_fields": ["헤딩", "특약구성"],
  "matched_values": ["보험금 지급사유"],
  "reason": "질문의 내용 조건을 포함"
}
```

선택 기준:

- `identity`: Index의 상품, 문서종류, 날짜, 정본 조건을 모두 충족하면 선택 가능
- `content`: FM에서 질문과 연결되는 필드와 값이 있어야 선택 가능
- `mixed`: Index 식별 조건과 FM 내용 조건을 모두 충족해야 선택 가능
- 근거가 없거나 후보 간 차이를 설명할 수 없으면 추측하지 말고 다음 Index 배치로 이동

`없음`은 gold를 보고 판단하지 않는다. 현재 후보 FM에서 질문을 지지하는 필드와 값을 제시하지 못하는 경우다.

## 최종 출력 JSON

각 세션은 설명문 없이 JSON 하나만 출력한다.

```json
{
  "qid": "mixed-0001",
  "arm": "D",
  "status": "selected",
  "selected_document_id": "doc-...",
  "index_batches": [],
  "frontmatter_checks": [],
  "inspected_document_ids": [],
  "final_reason": "",
  "turns": 0,
  "duration_ms": 0
}
```

허용 status: `selected`, `not_found`, `error`.

- B의 `frontmatter_checks`는 반드시 빈 배열
- C의 `index_batches`는 반드시 빈 배열

## 신규 구현 파일

기존 파일을 덮어쓰지 말고 다음을 작성한다.

```text
build_claude_arm_views.py
claude_document_selector_prompt.md
run_claude_document_selection.py
validate_claude_document_selection.py
eval_claude_document_selection.py
```

출력:

```text
out/claude_docselect_smoke.jsonl
out/claude_docselect_dev.jsonl
out/claude_docselect_test.jsonl
out/claude_docselect_eval.json
CLAUDE_DOCUMENT_SELECTION_RESULT.md
```

Runner 요구사항:

1. `--split smoke|dev|test` 지원
2. `--arms A,B,C,D` 지원
3. 문항 x Arm별 독립 Claude 세션
4. 실패 후 재시작 가능한 증분 JSONL
5. 중복 `(qid, arm)` 실행 방지
6. stdout, stderr, artifact access log 저장
7. JSON schema 실패 시 `error` 기록
8. 검색 종료 후에만 gold와 결합

## 첫 실행

처음부터 2,000회를 실행하지 마라.

- `identity` 3건
- `content` 3건
- `mixed` 3건
- 총 9문항 x 4 Arm = 36회

Smoke 검증:

- artifact 격리 위반 0건
- 모든 응답 JSON schema 통과
- D에서 첫 배치 실패 후 다음 배치로 이동하는 사례 확인
- D의 FM 조회 문서가 해당 Index 배치의 부분집합
- gold, constraints, evidence가 Claude 입력에 포함되지 않음

Smoke 통과 후 dev 100건 x 4 Arm을 실행한다. Dev에서 prompt와 batch 설정을 확정한 뒤 test 400건 x 4 Arm을 한 번 실행한다.

## 평가

주 지표:

- Accuracy@1: `selected_document_id == gold_document_id`

보조 지표:

- `not_found` 비율
- D의 평균 배치 수
- D의 평균 FM 확인 문서 수
- 첫 배치 성공률
- 다음 배치 복구율
- artifact 격리 위반율
- 평균 turns와 latency
- `identity`, `content`, `mixed` 유형별 Accuracy@1

효과 비교:

- B-A: Index 효과
- C-A: Frontmatter 효과
- D-max(B,C): Index -> FM 순차 탐색의 추가 효과

`content`, `mixed`는 candidate gold이므로 사람 검수 전 결과에는 `preliminary`를 표시한다.

## 즉시 수행 순서

```text
1. 기존 cascade 단위 테스트
2. Arm별 artifact view와 access guard 구현
3. Claude 단일 문항 D 실행
4. 첫 후보 FM 부적합 시 다음 Index 배치 이동 로그 확인
5. 9문항 x 4 Arm smoke 실행
6. smoke 검증 결과를 보고하고 dev 실행 전 중지
```

구현 도중 BM25, DR-DCI, vector pipeline으로 바꾸지 마라. 핵심 독립변수는 Claude가 Index와 Frontmatter를 계층적으로 탐색할 수 있는지 여부다.
