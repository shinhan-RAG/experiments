# Part 1·2 집중 재검증 계획

## 1. 목적

Peter의 기존 Part 1·2 결과를 삭제하거나 덮어쓰지 않고, 가장 중요한 주장만 좁혀 재검증한다.

- Part 1: taxonomy가 DR-DCI의 검색 워크스페이스 품질을 실제로 높이는가.
- Part 2: taxonomy의 효과와 DR-DCI의 검색 품질이 20K→50K→110K 문서 확장에서 유지되는가.
- Part 5의 dense 대 `hybrid_rrf` 결과는 동결하고 이번 비교에 다시 포함하지 않는다.
- tags, prefix, metadata 조합은 위 두 질문이 끝나기 전에는 추가하지 않는다.

본 계획은 기능이 많을수록 좋다는 가정이 아니라, 한 번에 한 변수만 바꾸고 질의별 paired 결과로 판단한다.

## 2. 근거 등급

근거는 다음 순서와 표기로 사용한다.

1. 학회 게재 논문 또는 연구기관의 원 논문
2. arXiv 원문. 동료평가 완료 여부를 별도로 표기한다.
3. Google Research 등 전문기관의 공식 기술 문서와 연구진 설명
4. 현재 저장소 코드와 실측 결과

공식 블로그의 제품 수치와 arXiv 사전논문 결과를 독립적으로 재현된 일반 법칙처럼 표현하지 않는다.

## 3. 기존 결과에서 확인한 사실

### 3.1 Part 1

PDF에는 baseline Gold Recall `0.0392`, taxonomy-only `0.0484`, stack-all `0.0505`가 기록돼 있다. 그러나 현재 코드에는 다음 해석 공백이 있다.

1. `taxonomy_only`와 `stack_tax`는 이름만 다르고 treatment 설정이 완전히 같다.
2. `stack_all`과 `stack_tax`의 차이는 `0.0004`이지만 질의별 paired 신뢰구간이 없다. 따라서 `stack_all`을 통계적으로 확인된 최적 설정이라고 부를 근거가 부족하다.
3. taxonomy는 agent가 `taxonomy_filter`를 실제 호출할 때만 pull score에 영향을 준다. 기존 결과에는 taxonomy-filter 사용 횟수가 기록되지 않아 효과의 작동 경로를 확인할 수 없었다.
4. 로컬 워크스페이스에는 taxonomy, tags, prefix, metadata 산출물이 없다. 이전 로더는 파일이 없어도 `None`으로 계속 진행하므로 arm 이름과 실제 treatment가 달라질 수 있었다.

### 3.2 Part 2

20K·50K·110K subset은 현재 데이터에서 다음 조건을 만족한다.

- 각각 20,000·50,000·110,000개의 고유 문서를 포함한다.
- 작은 subset이 큰 subset의 부분집합이다.
- 세 subset 모두 positive gold 문서 17,537개를 전부 포함한다.

이는 같은 질문과 evidence를 유지하고 distractor만 추가하는 통제된 규모 확장 방식과 부합한다. DR-DCI 논문도 동일 질문과 gold evidence를 고정한 채 distractor를 추가해 100K→10M 확장을 평가했다. 단, 이 논문은 2026년 arXiv 사전논문이므로 이번 저장소에서 별도 재현이 필요하다.

현재 Part 2에는 다음 공백이 있다.

1. Part 1에서 통계적으로 선택되지 않은 `stack_all`을 `best_config`로 하드코딩했다.
2. 규모별 taxonomy·prefix·metadata를 각각 생성하면 공유 문서의 augmentation 값까지 바뀔 수 있다. 이 경우 corpus size만 바뀐 실험이 아니다.
3. PDF는 Hybrid에 동일 전처리를 적용했다고 설명하지만 현재 `run_hybrid()`는 raw title/text만 색인한다. 따라서 기존 DR-DCI 대 Hybrid 절대값은 전처리까지 통제된 비교로 표현하면 안 된다.
4. 집계 평균만 있고 같은 질의의 20K→50K→110K 변화에 대한 paired 신뢰구간이 없다.

## 4. 논문에서 차용하는 범위

### 4.1 DR-DCI

DR-DCI는 retriever를 최종 evidence 인터페이스가 아니라 agent가 워크스페이스를 확장하는 수단으로 사용한다. 논문은 동일한 dynamic pull 인터페이스에서 dense와 BM25를 별도로 비교하고, corpus scaling에서는 질문과 gold evidence를 고정한 채 distractor만 증가시킨다.

이번 실험은 이 중 **통제된 distractor scaling**과 **bounded workspace의 품질·비용 동시 측정**만 차용한다. Peter 구현이 논문의 공식 재현이라고 주장하지 않는다.

### 4.2 Direct Corpus Interaction

DCI 논문은 고정 top-k가 exact lexical constraint, sparse clue conjunction, local context verification을 일찍 제거할 수 있다고 문제를 제기한다. 이번 Part 1은 taxonomy가 이 문제를 해결한다고 미리 가정하지 않는다. taxonomy가 agent의 후보 발견을 실제로 돕는지 실험 대상으로 둔다.

### 4.3 Google Research의 sufficient context

Google Research는 검색된 context가 질문을 답하기에 충분한지와 모델이 충분한 context를 활용했는지를 분리해 평가할 필요가 있다고 제안한다. 이번 단계에서는 별도 sufficient-context agent를 추가하지 않는다. 대신 answer accuracy만 보지 않고 workspace gold recall을 함께 측정해 검색 실패와 답변 실패를 구분한다.

## 5. 좁힌 실험

### 5.1 Part 1: taxonomy 단일 변수

비교군은 두 개뿐이다.

| Arm | 설정 |
|---|---|
| Control | DR-DCI baseline, augmentation 없음 |
| Treatment | baseline + taxonomy |

고정 변수:

- TREC-COVID 20K subset
- 같은 50개 질의와 qrels
- 같은 embedding, agent LLM, prompt, max turns, pull Top-K, workspace cap
- 같은 실행 코드와 seed

주 평가:

- 질의별 workspace Gold Recall의 `taxonomy - baseline` paired bootstrap 95% CI

보조 평가:

- answer accuracy와 judge 유효 표본 수
- pull 수, latency, prompt/completion token
- taxonomy-filter가 적용된 pull 횟수
- API가 제공하는 경우 `system_fingerprint`
- efficiency는 보조 지표로만 사용한다. recall을 pull 수로 나눈 파생값이므로 단독 품질 지표로 해석하지 않는다.

진행 기준:

- Gold Recall delta의 95% CI가 0보다 클 때 taxonomy의 긍정 효과 후보로 본다.
- CI가 0을 포함하면 불확실로 판정하고, `stack_all` 확대 대신 질의별 승패와 taxonomy 사용 여부만 분석한다.
- CI가 0보다 작으면 taxonomy 확대 실험을 중단한다.

1회 실행의 paired bootstrap은 질의 간 변동만 반영하고 LLM 실행 간 변동은 반영하지 않는다. OpenAI Chat Completions의 `seed`도 공식 문서상 best-effort이며 결정성이 보장되지 않는다. 따라서 1회 실행은 screening으로 취급하고, 긍정 신호가 확인될 때만 같은 두 arm을 반복 실행해 확인한다. 반복 횟수는 비용 승인 전에 확정하며, 서로 다른 backend fingerprint가 섞이면 별도로 표기한다.

### 5.2 Part 2: 같은 메커니즘의 규모 확장

Part 1을 통과한 경우에만 taxonomy arm을 확장한다.

| Scale | Control | Treatment |
|---|---|---|
| 20K | baseline | taxonomy-only |
| 50K | baseline | taxonomy-only |
| 110K | baseline | taxonomy-only |

주 평가:

- 각 규모의 `taxonomy - baseline` paired Gold Recall delta
- 각 arm의 `50K - 20K`, `110K - 20K` paired Gold Recall delta

운영 평가:

- 평균/p50/p95 latency
- pull 수, workspace 문서 수, token 사용량
- taxonomy-filter pull 사용률

공유 문서의 taxonomy 값은 규모별로 같아야 한다. 110K 정본에서 50K·20K subset을 필터링해 파생하는 방식을 우선한다. 별도 생성본을 사용하면 사전검사가 공유 문서의 값 변경을 차단한다.

### 5.3 선행 실험: retrieval-only distractor scale probe

augmentation 산출물과 무관하게 지금 실행 가능한 유일한 실측이며, 5.1·5.2의
agent 실험보다 먼저 수행한다.

| 고정 요소 | 값 |
|---|---|
| 질의 | 동일한 TREC-COVID 50개 (positive gold 보유 질의만 분모) |
| corpus | nested 20K / 50K / 110K (gold 17,537 전량 공통) |
| retriever | 동일 dense embedding·동일 파라미터 |
| 변경 변수 | distractor 수만 |

지표: 주 지표 `nDCG@10`(graded, pytrec_eval/BEIR 관례의 선형 gain·log2 할인),
`Precision@20`. 보조 지표 `Recall@5/20`, `Hit@5/10`, latency. 통계는 같은
질의를 짝지은 scale 쌍별 paired bootstrap 95% CI.

지표 선택 이유: 질의당 positive gold 중앙값이 478이라 Recall@20 상한이
약 0.042로 낮고 Hit@5/10은 포화된다. nDCG@10은 TREC-COVID의 graded label
(1/2)을 활용해 상위 순위 품질을 보존하고, P@20은 상위 20개 후보의 순도를
직접 잰다.

해석 규칙: 지표가 scale에 평탄하면 "원문 질의 1회 dense ranking이 주된
병목"이라는 설명이 약화되고, 하락하면 검색단 degradation의 존재가 지지된다.
평탄한데 agent 성능이 하락하면 후단(질의 재작성, 반복 pull, workspace 처리)
을 별도 점검할 근거가 된다. 이 probe는 병목을 국소화할 뿐 단독으로 원인을
확정하지 않는다.

```bash
python run_experiment.py --part 2 --scale-probe
# H200 인스턴스 일괄 실행: bash scripts/run_scale_probe_gte.sh
```

#### 실측 결과 (2026-07-22, H200·gte-Qwen2-1.5B-instruct)

`results/part2_scale_probe/20260722_011134.json` (인스턴스 보관, 정책상
미추적). preflight `ready`, 분모 50/50 질의, seed 42.

| 지표 | 20K | 50K | 110K | 110K−20K Δ | 95% CI |
|---|---:|---:|---:|---:|---|
| nDCG@10 | 0.823 | 0.754 | 0.630 | **−0.194** | **[−0.245, −0.145]** |
| Precision@20 | 0.853 | 0.778 | 0.651 | **−0.202** | **[−0.240, −0.164]** |
| Recall@20 | 0.0442 | 0.0399 | 0.0329 | **−0.011** | **[−0.015, −0.008]** |
| Hit@5 / Hit@10 | 1.0 | 1.0 | 1.0 | 0 | 포화(예측대로) |
| latency (s) | 0.118 | 0.256 | 0.528 | +0.410 | [+0.406, +0.414] |

세 scale 쌍(20↔50, 20↔110, 50↔110) 모두에서 주 지표 CI 전체가 0 아래다.
질의별 분해는 단조성을 보인다: 상위 scale에서 nDCG@10이 좋아진 질의는
**0개**(20K→110K에서 40개 악화·10개 동률, P@20은 45개 악화·5개 동률)다.
단, gold가 전량 20K에 포함되고 distractor만 추가되는 설계에서 개선 부재
자체는 자연스러우므로, 판정 근거는 개선 질의 수가 아니라 하락 폭과
paired CI다.

판정 (5.3의 사전 명시 규칙 적용):

- **검색단 degradation의 존재가 지지된다.** distractor 5.5배(gold 밀도
  2.39%→0.43%) 희석에서 상위 순위 품질이 상대 기준 약 −24% 하락했다.
  Part 2에서 관측된 agent 수준 저하에는 최소한 검색단 성분이 실재한다.
- 단, 절대 수준은 유지된다: 110K에서도 P@20 0.651은 무작위 기대치의 약
  145배다(질의당 positive gold 평균 493.5 기준 기대치 ≈ 0.0045). 검색이
  무너진 것이 아니라 순도가 희석되는 양상이다.
- 이 결과는 병목의 국소화이며 agent 후단의 기여를 배제하지 않는다.
  agent 수준 저하량과의 정량 비교는 Gate 2·3(taxonomy agent 실험)에서만
  가능하다.
- Gate 2·3과의 연결: taxonomy soft boost는 저하가 관측된 검색 계층에
  작용하는 처치이므로 검증 우선순위를 갖는다. 실제 개선 여부는 원본
  산출물 확보 후의 Gate 2·3(H1b)에서만 판정한다.

한계: TREC-COVID의 심층 qrels 구조(질의당 gold 수백 개)에 특화된 관찰이며,
gold가 희소한 코퍼스(FiQA류·실무 문서)에 그대로 일반화하지 않는다. n=50,
임베딩은 자가 서빙 gte-Qwen2 단일 모델이고, CI는 질의 간 변동만 반영할 뿐
distractor 표본·임베딩 모델 변동은 반영하지 않는다(단일 nested 표본 실측).
원시 결과 JSON은 현재 인스턴스에만 보관돼 있으므로, 결과 bundle과 SHA-256
해시를 회수·보존해야 재현 검증이 완결된다.

## 6. 구현된 보호 장치

- `scripts/audit_part12.py`: 모델 호출 전 subset·gold·augmentation 계약 검사
- `src/eval/part12_contracts.py`: duplicate arm, nested subset, gold 보존, 산출물 존재·coverage, 공유 문서 augmentation 일치 검사
- `compare_result_rows()`: 집계 평균이 아니라 query ID를 맞춘 paired bootstrap 비교
- `taxonomy_filtered_pulls`: taxonomy가 실제로 사용됐는지 계측
- `system_fingerprint`: 제공되는 backend 변경 식별자를 결과에 보존
- `python run_experiment.py --part 1 --focused`: baseline 대 taxonomy-only만 실행
- `python run_experiment.py --part 2 --focused`: baseline·taxonomy-only의 20K·50K·110K 확장만 실행
- `python run_experiment.py --part 2 --scale-probe`: agent/LLM 없는 dense retrieval scale probe (nDCG@10·P@20 주 지표, scale 쌍별 paired CI)

기존 Part 1·2·5 결과 디렉터리는 수정하지 않는다. 집중 실험은 별도 결과 디렉터리에 저장한다.

## 7. 현재 실행 판정

현재 로컬 사전검사 결과:

- subset/gold 계약: 통과
- duplicate arm 경고: `taxonomy_only == stack_tax`
- taxonomy 20K 산출물: 없음
- Peter의 augmentation 저장소는 현재 인증 없이 조회되지 않음
- 원 생성기는 `Qwen/Qwen3-8B`를 `localhost:8100`에서 호출하지만 현재 해당 endpoint가 없음
- retrieval-only scale probe: **실측 완료** (5.3 결과 참조 — 검색단 degradation 지지)
- Part 1 집중 실행: 차단

따라서 지금 숫자를 새로 만들 수는 없다. Peter가 사용한 taxonomy artifact를 원 실행 환경에서 가져오거나 동일 모델·prompt·subset 계약으로 복원한 뒤 다음 순서로 진행한다. 환경에 존재하는 다른 API key로 taxonomy를 새로 생성하면 생성 모델까지 바뀌므로 원 결과의 단일변수 재검증이 아니다.

```bash
# 산출물 없이 지금 실행 가능 (임베딩 endpoint만 필요)
python run_experiment.py --part 2 --scale-probe

# taxonomy artifact 확보 후
python scripts/audit_part12.py --size 20000
python run_experiment.py --part 1 --focused

# Part 1 판정 후에만 실행
python scripts/audit_part12.py
python run_experiment.py --part 2 --focused
```

## 8. 중단선

이번 단계에서는 다음을 하지 않는다.

- tags A/B/C 재비교
- prefix·metadata 조합 탐색
- 새로운 retriever 또는 fusion 추가
- 별도 sufficient-context agent 추가
- 새로운 데이터셋 추가
- Part 4·5 재실행

Part 1 taxonomy 단일 효과와 Part 2 scale robustness를 확인한 뒤 다음 한 가지 변수만 선택한다.

## 9. 근거 자료

- Zhuofeng Li et al., *Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction*, arXiv:2605.05242, 2026. https://arxiv.org/abs/2605.05242
- *DR-DCI: Scaling Direct Corpus Interaction via Dynamic Workspace Expansion*, arXiv:2606.14885v1, 2026. https://arxiv.org/html/2606.14885v1
- Hailey Joren et al., *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems*, Google Research, 2025. https://research.google/pubs/sufficient-context-a-new-lens-on-retrieval-augmented-generation-systems/
- Cyrus Rashtchian and Da-Cheng Juan, *Unlocking dependable responses with Gemini Enterprise Agent Platform's Agentic RAG*, Google Research, 2026. https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/
- OpenAI, *Chat Completions API Reference* (`seed`는 best-effort이며 determinism은 보장되지 않음). https://platform.openai.com/docs/api-reference/chat
