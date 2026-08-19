# Hybrid Enrichment v2 실행 결과 및 감사 보고

## 결론

100문항 × 4 Arm, 총 400개 v2 세션은 완료됐다. 그러나 사전 정의한 세 gate는 모두 통과하지 못했다.

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BASE | 0.22 | 0.36 | 0.44 | 0.285 |
| META | 0.34 | 0.44 | 0.47 | 0.379 |
| TAG | 0.28 | 0.42 | 0.49 | 0.343 |
| BOTH | 0.32 | 0.39 | 0.42 | 0.355 |

- G1 `META >= BASE + 10pt`: **실패** (`+8pt`)
- G2 `TAG >= BASE + 10pt`: **실패** (`+6pt`)
- G3 `BOTH > max(META, TAG)`: **실패** (`-5pt vs META`, `-3pt vs TAG`)

현재 결과는 metadata와 tag에 긍정적인 신호가 있음을 보여주지만, enrichment의 인과 효과를 확정하기에는 실행 간 에이전트 변동이 너무 크다.

## Paired 비교

qid 단위 paired bootstrap 10,000회 결과:

| 비교 | Recall@5 차이 | 95% CI | MRR@10 차이 | 95% CI |
|---|---:|---:|---:|---:|
| META − BASE | +0.08 | [0.00, 0.16] | +0.094 | [0.027, 0.164] |
| TAG − BASE | +0.06 | [-0.04, 0.16] | +0.058 | [-0.023, 0.137] |
| BOTH − BASE | +0.03 | [-0.06, 0.12] | +0.069 | [-0.003, 0.144] |
| BOTH − META | -0.05 | [-0.14, 0.04] | -0.024 | [-0.110, 0.063] |
| BOTH − TAG | -0.03 | [-0.13, 0.07] | +0.011 | [-0.070, 0.096] |

META는 Recall@5 CI 하한이 정확히 0이고 MRR@10은 양의 구간이다. TAG와 BOTH는 두 지표 모두 0을 포함한다. 따라서 `META가 유망하다`는 해석은 가능하지만 `TAG와 BOTH가 baseline보다 개선됐다`고 확정할 수 없다.

Recall@5 paired 전환:

- META: baseline miss를 hit로 바꾼 13건, baseline hit를 miss로 바꾼 5건
- TAG: 개선 17건, 회귀 11건
- BOTH: 개선 12건, 회귀 9건

## 오류 민감도

Arm별 비정상 종료는 다음과 같다.

| Arm | ranked | error | not_found |
|---|---:|---:|---:|
| BASE | 96 | 3 | 1 |
| META | 92 | 5 | 3 |
| TAG | 90 | 6 | 4 |
| BOTH | 93 | 3 | 4 |

모든 Arm이 정상 ranked인 공통 81문항만 계산하면 Recall@5는 BASE 0.358, META 0.469, TAG 0.457, BOTH 0.432다. 오류를 제외해도 순서는 같지만 이 분석은 사후 complete-case 분석이므로 주 결과를 대체하지 않는다.

## Element type별 Recall@5

| Type | n | BASE | META | TAG | BOTH |
|---|---:|---:|---:|---:|---:|
| formula | 14 | 0.357 | 0.643 | 0.357 | 0.286 |
| table | 57 | 0.386 | 0.439 | 0.439 | 0.439 |
| text | 29 | 0.310 | 0.345 | 0.414 | 0.345 |

표본은 작지만 META의 개선은 formula에, TAG의 개선은 text에 집중되는 패턴이다. 이는 사전 가설이 아니라 사후 관찰이므로 다음 실험의 층화 가설로만 사용한다.

## 에이전트 도구 행동

| Arm | 평균 호출 | vector 사용 세션 | keyword 사용 세션 | 첫 호출 vector | 첫 호출 keyword |
|---|---:|---:|---:|---:|---:|
| BASE | 6.30 | 84 | 100 | 50 | 49 |
| META | 6.22 | 75 | 97 | 40 | 59 |
| TAG | 6.19 | 75 | 99 | 40 | 60 |
| BOTH | 6.55 | 81 | 99 | 44 | 56 |

BOTH가 후보 다양성 부족 때문에 일찍 수렴했다는 증거는 없다. BOTH의 평균 호출 수와 grep/read 호출 수는 오히려 가장 많다. 현재 로그에는 호출 인자만 있고 각 호출이 반환한 candidate ID와 score가 없어서 다음은 복원할 수 없다.

- gold가 처음 노출된 도구
- vector 결과가 후속 grep query에 미친 영향
- 도구별 candidate union과 최종 선택 사이의 손실

## 중요한 식별 한계

### 1. 독립 Claude 세션 변동이 관측 효과보다 크다

같은 100문항의 이전 실행에서 BASE Recall@5는 0.53이었고 v2 BASE는 0.36이다. baseline 재실행만으로 `-17pt`가 변했으며, 이는 META `+8pt`와 TAG `+6pt`보다 크다.

또한 META와 BASE에서 keyword 뷰는 동일하지만, vector를 호출하지 않은 META 세션도 독립 세션 특성 때문에 BASE와 다른 결과를 냈다. 따라서 현재 차이는 다음이 섞인 total effect다.

```text
enrichment 효과 + 독립 에이전트 실행 변동 + 오류 변동
```

### 2. 최종 순위가 전부 element다

400세션의 최종 `ranked_chunk_ids`에 들어간 ID는 모두 `e*` element였고 vector가 반환하는 `c*` chunk는 0건이었다. metadata가 도움이 됐다면 vector 결과가 후속 검색 행동을 바꾼 간접 효과다. 현재 로그만으로 이 경로를 검증할 수 없다.

### 3. v2의 100문항은 held-out test가 아니다

같은 100문항의 v1 결과를 보고 v2 enrichment를 만들고 동일 100문항을 재평가했다. 따라서 이 집합은 dev이며 일반화 성능을 주장할 수 없다. 매핑 성공 후 남겨둔 약 191문항이 실제 held-out 후보다.

### 4. 실행 통제와 로그 누락

- Claude sampling 설정이 결과 manifest에 기록되지 않았다.
- 도구 반환 결과가 저장되지 않아 행동 경로 분석이 불가능하다.
- smoke manifest는 12개 entry지만 중복 qid가 있어 고유 문항은 10개다.
- 모델과 prompt hash는 400세션에서 동일했고 `(qid, arm)` 중복은 0건이다.

## 판정

이 실험에서 정직하게 말할 수 있는 결론은 다음이다.

> dev 100문항의 단일 실행에서 META와 TAG는 BASE보다 각각 Recall@5 +8pt, +6pt의 긍정 신호를 보였다. 그러나 +10pt gate는 통과하지 못했고, TAG의 paired CI는 0을 포함한다. BOTH는 단독 Arm보다 낮았다. 동일 BASE 재실행 변동이 17pt이므로 현재 결과만으로 enrichment 효과를 확정할 수 없다.

## 다음 실행 권고

현재 400세션을 다시 돌려 결과가 좋은 iteration을 선택하지 않는다. 다음은 남은 held-out 문항에서 수행한다.

1. 도구가 반환한 candidate ID, score, preview를 runner가 authoritative log로 저장한다.
2. parse error를 줄이기 위해 Claude 최종 응답을 JSON schema 또는 structured output으로 강제한다.
3. sampling 설정을 명시하고 동일 qid/arm을 최소 3회 반복한다. 비용이 부담되면 held-out에서 층화 60문항을 뽑아 먼저 반복한다.
4. Arm 순서를 qid별로 균형화하고 실행 시간대를 교차 배치한다.
5. v2 enrichment와 prompt를 동결한 뒤 held-out을 한 번 평가한다.
6. 주 분석은 반복 평균의 paired 차이, 보조 분석은 세션별 분산과 도구 경로 매개 분석으로 한다.

감사 원본은 `out/eval_v2_audit.json`, 재현 코드는 `eval_v2_audit.py`에 있다.
