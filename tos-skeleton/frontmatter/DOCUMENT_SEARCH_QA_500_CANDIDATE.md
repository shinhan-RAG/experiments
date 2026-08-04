# 문서 선택 검색 QA 500 후보셋

## 목적

동일한 사용자 질문에 대해 적재 정보만 바꾸었을 때 컬렉션 내 정답 문서를 얼마나 잘 선택하는지 비교한다.

| 조건 | 파일명/원문 | Collection Index JSON | Document Frontmatter |
|---|---:|---:|---:|
| A | O | X | X |
| B | O | O | X |
| C | O | X | O |
| D | O | O | O |

네 조건은 같은 500개 질문, 같은 검색기 설정, 같은 후보 문서 집합을 사용해야 한다. Index와 Frontmatter의 표현을 이후에 변경해도 이 QA는 고정해서 재사용한다.

## 파일

- `out/document_search_qa_500_candidate_v1.jsonl`: 평가 입력 500건
- `out/document_search_qa_500_candidate_v1_manifest.json`: 건수, 해시, 생성 제약
- `out/document_search_qa_500_candidate_v1_review.csv`: 사람 검수용 시트
- `build_document_search_qa_500.py`: 재현 가능한 생성기

## 구성

| 유형 | 건수 | 질문 신호 |
|---|---:|---|
| identity | 150 | 상품명 + 연도 + 문서 종류 |
| content | 200 | 원문 본문의 조항/주제 |
| mixed | 150 | 상품명 + 연도 + 문서 종류 + 본문 주제 |

`dev` 100건과 `test` 400건으로 고정 분할했다. 모든 문항은 단일 `gold_document_id`와 원문 근거를 가진다. 500개 정답 경로와 ID가 현재 `collection_index.json`에 존재하고 일치하는 것도 확인했다.

## 평가 지표

- 주 지표: `Accuracy@1` (`Hit@1`)
- 보조 지표: `Hit@5`, `MRR`
- 비교값: `B-A`(Index 효과), `C-A`(Frontmatter 효과), `D-A`(결합 효과), `D-max(B,C)`(결합 추가 효과)
- 전체 점수와 함께 `identity`, `content`, `mixed` 유형별 점수를 각각 보고한다.

## 현재 상태와 제한

이 파일은 **후보셋**이다. Index나 Frontmatter에서 질문 또는 정답을 만들지 않고 원본 Markdown의 경로, 파일명, 제목과 본문만 사용했다. 따라서 특정 적재 스키마에 종속되지는 않는다.

`identity`는 동일 상품·문서 종류·연도 조합이 원문 집합에서 하나인 항목으로 제한했다. `content`와 `mixed`는 지정된 원문에 실제 근거가 있으나, 같은 질문에 답할 수 있는 다른 문서가 없는지 전수 판정하지 않았다. `review.csv`에서 `gold_source_path`와 근거를 검토하고 대체 정답이 발견되면 단일 정답을 유지할 수 있도록 질문 조건을 보강하거나 해당 문항을 교체해야 한다.

이 검수가 끝나기 전의 점수는 최종 성능이 아니라 예비 비교 결과로 표시한다.
