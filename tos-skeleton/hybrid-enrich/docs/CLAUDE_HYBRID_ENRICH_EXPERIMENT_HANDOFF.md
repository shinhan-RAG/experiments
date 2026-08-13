# Claude 핸드오프: 에이전틱 하이브리드 검색 + 메타데이터/시멘틱 태그 실험

## 1. 목적

이 문서를 읽은 뒤 추가 설계 토론으로 멈추지 말고, 아래 명세대로 구현하고 smoke까지 실행하라.

검색 대상은 판매약관 한 문서로 고정한다. 문서 선택도 답변 생성도 하지 않는다. 평가하는 것은 **Claude 에이전트가 grep 검색과 벡터 검색 두 도구를 조합해, 질문의 근거가 담긴 gold 청크를 top-10 안에 찾아오는 능력**이다.

증명할 것은 세 가지다.

1. 청크에 **메타데이터**(청크 단위, 벡터 색인 강화)를 부착하면 baseline 대비 Recall@5가 **+10pt 이상** 오른다
2. 엘리먼트에 **시멘틱 태그**(엘리먼트 단위, grep 뷰 강화)를 부착하면 baseline 대비 Recall@5가 **+10pt 이상** 오른다
3. **둘 다** 부착하면 각각 단독보다 더 오른다

용어는 2026-08-03 적재 설계 회의 정의를 따른다: 메타데이터 = 벡터서치용 청크 단위 부가 정보, 시멘틱 태그 = 파일서치용 엘리먼트 단위 구조 정보. 이 실험 결과는 2026-08-07 적재 설계 문서의 직접 근거가 된다.

## 2. 입력 artifact

작업 위치:

```text
/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/hybrid-enrich
```

입력:

```text
# QA 359건 (신뢰도=확정 303건만 사용)
/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl

# 검색 대상 단일 문서 (249,047 lines)
/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md

# 청크 스키마 참고용 (재사용 금지, 문서 버전이 다름)
/Users/seyoung/Downloads/0804_QAset/shinhan-tos/corpus.jsonl
```

490셋의 corpus, QA, qrels, smoke_baseline 수치는 이번 실험과 비교·재사용하지 않는다.

## 3. 검색 단위: 엘리먼트 기반 청크

검색 단위는 하나다. 트랙을 나누지 않는다.

1. 문서를 구조 **엘리먼트**로 deterministic하게 분할한다: heading / paragraph / table / formula / list. 엘리먼트 경계·ID·line 범위는 전 Arm 동일.
2. 각 엘리먼트를 **청크**로 만든다. 600자 이하 엘리먼트는 그대로 1청크. 600자 초과 엘리먼트는 recursive splitter(chunk_size 600자, overlap 100자, separators `["\n\n", "\n", ". ", " ", ""]`)로 분할한다.
3. 모든 청크는 `element_id`를 보존한다. 시멘틱 태그는 엘리먼트 단위로 생성해 소속 청크 전체에 상속한다. 메타데이터는 청크 단위로 생성한다.

청크 스키마:

```json
{
  "chunk_id": "...",
  "element_id": "...",
  "element_type": "paragraph|table|formula|list|heading",
  "section_path": ["제2편", "..."],
  "text": "raw 원문",
  "line_start": 1,
  "line_end": 10,
  "char_start": 0,
  "char_end": 600
}
```

청킹 설정은 실험 시작 전에 고정하며 검색 성능을 보고 변경하지 않는다. 변경이 필요하면 결과를 폐기하고 새 version으로 재실행한다.

**알려진 한계 (수정하지 말고 기록만)**: 초대형 표(최대 682라인)에서 유래한 청크와 직렬화 enrichment를 합치면 임베딩 모델 max 길이를 넘을 수 있다. 초과 시 앞에서부터 max까지 절단하되, 절단 규칙은 전 Arm 동일하게 적용하고 절단 발생 청크 수를 결과에 기록한다.

## 4. Gold 매핑: quote 우선, line은 보조

359셋 `출처[].line`은 약 20%가 실제 문서 위치와 불일치한다(불량 line 값 53건 포함, 사전 검증 완료). 따라서 **line을 믿지 말고 quote 텍스트로 gold를 만든다.**

절차 (QA당, 출처당):

1. quote를 정규화(NFC, CRLF→LF, 연속 공백 1칸 축약)한 뒤 문서 전문에서 검색
2. 매치 1곳 → 그 구간과 겹치는 모든 청크가 gold
3. 매치 여러 곳 → `line`과 가장 가까운 매치를 채택
4. 매치 0곳 → quote 앞 80자로 재시도, 그래도 없으면 해당 출처는 `source_failed`
5. QA의 모든 출처가 실패한 경우만 `mapping_failed`로 제외 (사전 검증 기준 12건 예상, 291건 사용 가능)

gold는 any-hit으로 평가한다(gold 청크 중 하나라도 top-k에 들면 hit). 출처별 매핑 결과(quote, 채택 위치, 겹친 chunk_ids)를 `out/gold_mapping_audit.jsonl`에 남긴다.

## 5. QA 표본: 100건

매핑 성공 집합(~291건)에서 **100건**을 seed 고정 층화 추출한다(사업구분 × gold element_type). 나머지 ~191건은 이번에 실행하지 않고 확인용 held-out으로 남겨둔다.

```text
smoke: 12건 (100건 중에서, element_type별 최소 3건)
main: 100건 전체
seed: 20260804
```

`out/split_manifest.json`에 qid 목록을 먼저 저장한 뒤 enrichment 개발을 시작한다.

## 6. Enrichment

### 6.1 메타데이터 (청크 단위 — 벡터 색인에만 반영)

질문 파일을 절대 읽지 않고 원문·청크·문서 구조만으로 생성한다. 원문에 없는 사실 생성 금지.

```json
{
  "contract_scope": "주계약 또는 정확한 특약명",
  "section_path": ["제2편", "제3관", "제12조 보험금의 지급사유"],
  "topic": ["보험료 납입면제"],
  "entities": ["피보험자", "보험료납입면제대상계약"],
  "event_or_condition": ["암 진단확정", "장해지급률 50% 이상"],
  "benefit_or_action": ["차회 이후 보험료 납입 면제"],
  "aliases": ["납입면제", "납면"],
  "summary": "1문장"
}
```

`aliases`는 QA와 무관하게 작성한 보험 도메인 공통 사전을 쓰고 사전 파일 hash를 기록한다.

### 6.2 시멘틱 태그 (엘리먼트 단위 — grep 뷰에만 반영)

동일하게 질문 파일 접근 금지. 짧은 canonical phrase로 만들고, 원문/구조에서 근거를 못 찾는 태그는 버린다.

```json
{
  "element_type": "table",
  "contract_scope": "표적항암약물허가치료특약",
  "section_path": ["제2편", "제2-2조"],
  "semantic_role": ["보험금 지급기준"],
  "table_title": "...", "column_headers": [], "row_labels": [],
  "formula_subject": "...", "article_label": "..."
}
```

구조별 필드: table→`table_title/column_headers/row_labels`, formula→`formula_subject/variables`, list→`list_title/item_topics`, paragraph→`article_label/semantic_role`.

생성 프로세스는 QA 경로 접근이 차단된 별도 실행으로 하고 접근 로그를 남긴다. generator 모델·프롬프트 hash·schema version을 기록한다.

## 7. Arm 설계: 경로 분리가 핵심

| Arm | 벡터 색인 텍스트 (임베딩 입력) | grep 뷰 텍스트 (rg 대상) |
|---|---|---|
| `BASE` | 청크 text | 청크 text |
| `META` | **metadata 직렬화 + text** | 청크 text |
| `TAG` | 청크 text | **semantic tag 직렬화 + text** |
| `BOTH` | metadata + text | tag + text |

META에서 grep 뷰에 메타데이터를 노출하지 않고, TAG에서 벡터 색인에 태그를 넣지 않는다. "메타데이터=벡터서치용 / 시멘틱 태그=파일서치용"이라는 적재 설계 가정을 그대로 검증하기 위함이다.

격리는 지시문이 아니라 물리적으로: Arm별 벡터 인덱스 디렉터리와 grep 대상 JSONL을 분리하고, 도구가 자기 Arm 것만 접근하게 guard를 건다. `read_chunk` 반환과 gold 판정은 전 Arm에서 raw text만 사용한다.

## 8. 에이전트 실행

에이전트에게 도구 3개만 준다. **모든 Arm에서 도구 구성은 동일하다.** 다른 것은 색인/뷰 내용뿐이다.

```text
vector_search(query, top_k<=10) -> [{chunk_id, score, preview 200자}]
grep_search(pattern, max_hits<=20) -> [{chunk_id, matched_line}]   # rg 고정 옵션
read_chunk(chunk_id) -> raw 청크 전문
```

- 도구 호출 합계 최대 8회. 초과 시 그 시점 후보로 강제 종료
- 에이전트 입력은 `{qid, question}` 뿐. gold, 출처, quote, 정답 절대 미포함
- QA 1건 × Arm 1개 = 새 Claude 세션. 한 세션 다중 Arm 금지
- 고정·기록: 모델 식별자, sampling, system prompt hash, tool allowlist, max_tool_calls=8, 임베딩 모델명·max 길이
- 검색 내부에서 별도 LLM 호출 금지. 실험 수행 Claude가 유일한 판단 모델

세션 출력 (설명문 없이 JSON 하나):

```json
{
  "qid": "1",
  "arm": "BOTH",
  "status": "ranked",
  "ranked_chunk_ids": ["최대 10개, 순위순"],
  "tool_calls": [{"tool": "grep_search", "arg": "납입면제", "hits": 12}],
  "final_reason": "",
  "turns": 0,
  "duration_ms": 0
}
```

허용 status: `ranked`, `not_found`, `error`.

## 9. 실행 규모와 Gate

총 실행: smoke 12건 × 4 Arm = 48세션 → main 100건 × 4 Arm = 400세션.

### Smoke 검증 (성능 판단 아님)

- Arm 격리 위반 0건 (META의 grep 결과에 metadata 문자열 미노출 등)
- 전 응답 JSON schema 통과, 도구 상한 작동
- gold/quote가 에이전트 입력에 미포함
- enrichment 생성 로그에 QA 접근 0건

### Acceptance gate (main 100건)

```text
G1: META Recall@5 >= BASE Recall@5 + 10pt
G2: TAG  Recall@5 >= BASE Recall@5 + 10pt
G3: BOTH Recall@5 >  max(META, TAG) Recall@5
```

gate 미달 시 test를 늘리지 말고 **실패 분석 → enrichment 스키마·직렬화·프롬프트만 개선 → 재실행**한다. 청킹 설정, 도구 명세, QA/gold, 검색 상한은 iteration 중 변경 금지. iteration마다 schema version, prompt hash, 전체 지표, qid별 rank 변화, regression 사례를 기록한다.

정직성 원칙: gate는 설계 개선의 목표이지 결과 조작의 면허가 아니다. 개선 iteration을 거쳐도 미달이면 미달로 보고한다. gold를 보고 태그를 고치는 행위(qid별 실패 사례에서 quote를 태그에 역주입)는 leakage이며 금지한다. 실패 분석은 "어떤 유형의 질문이 어떤 도구 경로에서 실패했는가" 수준까지만 허용한다.

## 10. 평가 지표

주 지표 (any-hit): Recall@1 / Recall@5 / Recall@10, MRR@10

보조 지표:

- element_type별, 사업구분별 Recall@5
- Arm별 도구 프로파일: grep/vector 호출 비율, 첫 호출 도구, gold를 처음 노출시킨 도구
- BASE miss → enriched hit 수, BASE hit → enriched miss 수 (regression)
- `not_found`율, 평균 도구 호출 수, 평균 turns/latency, mapping_failed 수
- qid 단위 paired bootstrap 10,000회 95% CI (META-BASE, TAG-BASE, BOTH-max)

효과 비교: META−BASE(메타데이터 효과, 벡터 경로), TAG−BASE(태그 효과, grep 경로), BOTH−max(META,TAG)(결합 추가 효과).

## 11. 구현 파일

```text
build_elements_and_chunks.py    # 엘리먼트 분할 + 600자 청킹
map_gold_chunks.py              # quote 우선 gold 매핑 + audit
build_qa_sample.py              # 확정 303 → 매핑 성공 → 층화 100건
build_chunk_metadata.py         # QA 접근 차단
build_element_semantic_tags.py  # QA 접근 차단
build_arm_views.py              # Arm별 벡터 인덱스 + grep JSONL + guard
agent_tools.py                  # 도구 3종
claude_retrieval_prompt.md
run_claude_retrieval.py         # --split smoke|main --arms BASE,META,TAG,BOTH
                                # 증분 JSONL, (qid,arm) 중복 방지, 재시작 가능
eval_retrieval.py               # paired metrics, bootstrap, failure slices
```

출력:

```text
out/chunks.jsonl
out/qa100_gold.jsonl
out/gold_mapping_audit.jsonl
out/split_manifest.json
out/chunk_metadata.jsonl
out/element_semantic_tags.jsonl
out/retrieval_{split}_{arm}.jsonl
out/eval_main.json
out/experiment_manifest.json
CLAUDE_HYBRID_ENRICH_RESULT.md
```

## 12. 즉시 수행 순서

```text
1. 엘리먼트 분할 + 청킹, 청크 통계 보고 (개수, element_type 분포, 절단 발생 수)
2. quote 우선 gold 매핑 → mapping_failed 건수와 audit 보고
3. 100건 층화 샘플 + manifest 저장
4. enrichment 생성 (QA 차단 확인)
5. Arm별 인덱스/뷰 + 격리 guard 구현
6. 단일 문항 BOTH 실행, 도구 로그 확인
7. smoke 12건 × 4 Arm 실행
8. smoke 검증 결과를 보고하고 main 실행 전 중지
```

## 13. 금지 사항

- QA 질문·정답·quote를 보고 metadata/tag를 생성하거나 수정하지 않는다
- Arm 간 청크/엘리먼트 경계를 다르게 하지 않는다
- enrichment 문자열을 gold 판정이나 read_chunk 반환에 포함하지 않는다
- reranker, 외부 LLM judge, query rewrite 고정 파이프라인, 문서 선택 단계를 추가하지 않는다
- line 번호를 gold 매핑의 1차 근거로 쓰지 않는다 (quote 우선, line은 다중 매치 선별용)
- 490셋 수치와 직접 비교하지 않는다

최종적으로 답할 질문: **메타데이터(벡터 경로)와 시멘틱 태그(grep 경로)가 각각 baseline 대비 Recall@5를 10pt 이상 올리고, 둘의 결합이 단독보다 더 오르는가?**
