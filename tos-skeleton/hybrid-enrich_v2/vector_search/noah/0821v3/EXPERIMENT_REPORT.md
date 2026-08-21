# 0821v3 C29 dual-tool 전체 train 실험 보고서

실행일: 2026-08-21  
대상: evaluable train 281문항  
결론: **실행 규약과 인프라는 정상 동작했지만, 현재 explicit dual-tool agent 방식을 기존 C29보다 우수한 방식으로 채택할 근거는 없다.** R@5 차이는 불확실하고, R@10과 suff@10은 기존 host-controlled C29보다 낮았다.

## 1. 실험 설정

| 항목 | 설정 |
|---|---|
| 모델 | Codex CLI `gpt-5.6-luna` |
| reasoning effort | `medium` |
| worker | 4 |
| 반복 | 1 rep |
| QA/Gold | `gold_train_scoped_u3_reviewed_overlay_v1.jsonl`, 281문항 |
| semantic 도구 | C29 CLM + structured BM25F fact + membership evidence quota |
| metadata 도구 | V9 BM25 + BGE-m3-ko dense, RRF(k=60) |
| 도구 규약 | 모든 문항에서 `search`와 `msearch`를 각각 최소 1회 사용 후 submit |
| 채점 단위 | U3 JO, Gold evidence-group fractional Recall 및 sufficiency |

모델에는 Gold를 제공하지 않았으며, Gold는 호스트 채점에만 사용했다. Dense query encoding은 한 번 로드한 로컬 BGE-m3-ko embedding 서버를 4 worker가 공유했다.

고정 데이터 SHA256(LF 정규화):

- `elements_u3.jsonl`: `9d13e8d61781a4d39593795b0a30b33d41da7e1134335f04056e3d35540f5a49`
- `elements_u3jo.jsonl`: `44d856004c473e186880cbe749c70d701d1ae25f528c6128cc575a004e8d980c`
- `tags_u4_fact_rules.jsonl`: `19df8e21d9f9287394cf9d02924d5d0e2c84be7e9bf3ae9fca4f02aaacd18104`
- Gold: `d8038733c364f6441dd3d8bfc9bd3ac5bccb1b640b8458f344a96291be0936af`
- QID: `63d35e9f132392a659c34fd004bb60e550bcbbba38c766dd6acc074fe78752ea`

## 2. 최종 결과

| n | R@1 | R@5 | R@10 | suff@5 | suff@10 | RR@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 281 | .3932 | **.6068** | .6174 | .5836 | .5943 | .5083 |

QID bootstrap 95% CI:

| 지표 | 95% CI |
|---|---:|
| R@5 | [.5504, .6625] |
| R@10 | [.5611, .6720] |
| suff@5 | [.5267, .6406] |
| suff@10 | [.5374, .6512] |

R@5 분포는 만점 164문항, 부분점수 14문항, 0점 103문항이다. 빈 제출은 없었고 평균 제출 수는 2.95개였다.

## 3. Gold 품질층

| 품질층 | n | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|---:|
| retrieval-blind reviewed | 29 | .5862 | **.9483** | .9828 | .9310 | .9655 |
| scoped, completeness 미검수 | 252 | .3710 | **.5675** | .5754 | .5437 | .5516 |

29문항 reviewed 층과 나머지 252문항 사이의 격차가 매우 크다. 전체 점수에는 검색 실패뿐 아니라 scoped Gold의 completeness 미검수 영향이 섞여 있으므로 `.6068`을 순수 검색기 품질로 해석하면 안 된다.

## 4. 문항 난이도별 결과

| task type | n | R@5 | R@10 | suff@5 | suff@10 | 평균 제출 수 |
|---|---:|---:|---:|---:|---:|---:|
| single_lookup | 172 | .7151 | .7151 | .7151 | .7151 | 2.40 |
| multi_evidence | 102 | .4363 | .4657 | .3824 | .4118 | 3.80 |
| comparison | 4 | .2500 | .2500 | .0000 | .0000 | 4.75 |
| exhaustive_list | 2 | .5000 | .5000 | .5000 | .5000 | 4.00 |
| document_global | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.00 |

Gold group 수로 보면 단일 group 238문항의 R@5/suff@5는 `.6471/.6471`이지만, 2개 이상 group 43문항은 `.3837/.2326`이다. 다중근거의 모든 근거를 찾아 제출하는 문제가 가장 큰 병목이다.

## 5. 검색 노출과 agent 선택

각 채널의 반환 ID를 JO로 사상해 Gold group 노출을 다시 계산했다.

| 후보 범위 | fractional recall | Gold 0개 노출 qid | 모든 group 충분 노출 |
|---|---:|---:|---:|
| 최초 semantic 40 | .7788 | 56 | 213/281 (.7580) |
| 최초 metadata 40 | .7651 | 58 | 207/281 (.7367) |
| 최초 두 채널 union | .8743 | 28 | 238/281 (.8470) |
| 모든 재검색 union | .8968 | 22 | 244/281 (.8683) |

두 채널은 상호 보완적이다. 최초 단일 채널에서는 Gold가 하나도 노출되지 않은 문항이 각각 56/58개였지만 union에서는 28개로 감소했다.

반면 모든 검색 union에서 Gold가 충분히 노출됐는데도 최종 submit이 suff@10을 충족하지 못한 문항이 78개였다. 모든 검색 union 자체가 불충분한 문항은 37개였다. 즉 다음 개선의 우선순위는 단순 후보 확대보다 다음 두 부분을 분리하는 것이다.

1. 노출된 정답을 고르는 agent 선택/제출 정책
2. 여전히 충분 노출되지 않는 37문항의 cross-reference·multi-evidence 회수

## 6. 기존 C29 결과와 paired 비교

비교 대상은 동일 281 QID/Gold, Luna-medium의 기존 host-controlled C29 결과다. 다만 기존 실행은 `search` 내부에서 tag+meta를 강제 결합하고 max-actions=4를 사용했으며, 이번 실행은 `search`와 `msearch`를 별도 도구로 노출했다. 따라서 검색기 단독 A/B가 아니라 **전체 agent protocol 비교**다.

| 지표 | 기존 C29 | 이번 실행 | Δ | paired bootstrap 95% CI | 승/패/동률 |
|---|---:|---:|---:|---:|---:|
| R@1 | .3867 | .3932 | +.0065 | [-.0439, +.0563] | 30/30/221 |
| R@5 | .6314 | .6068 | -.0246 | [-.0810, +.0323] | 34/44/203 |
| R@10 | .6874 | .6174 | **-.0700** | **[-.1293, -.0125]** | 30/54/197 |
| suff@5 | .6050 | .5836 | -.0214 | [-.0783, +.0356] | 31/37/213 |
| suff@10 | .6584 | .5943 | **-.0641** | **[-.1246, -.0071]** | 29/47/205 |

R@5와 suff@5 차이는 CI가 0을 포함한다. 그러나 R@10과 suff@10은 이번 protocol이 낮다. 이번 agent의 평균 제출 수는 2.95개로 기존 6.20개보다 훨씬 적었다. 명시적 두 도구가 후보 노출을 넓혔지만 모델이 더 보수적으로 제출하면서 Top-10 이득을 사용하지 못한 것이 주요 가설이다.

## 7. 실행 무결성과 장애 처리

- 최종 결과: 281 rows / 281 unique QID
- fatal error: 0
- 필수 도구 위반: 0
- empty submit: 0
- 도구 호출: search 328, msearch 299, read 1,299, submit 281
- 도구 timestamp 기준 wall time: 약 4,076초(67.9분, retry 포함 범위)
- 평균 문항 elapsed: 56.44초

최초 실행에서 `v3-offline-0465` 한 셀의 `calls.jsonl`에 빈 줄이 생겨 submit 시 JSON parse가 실패했다. 실패 시도는 `result_failed_attempt1.json`으로 보존하고, submit 전 blank-line sanitation과 순차 도구 호출 지시를 추가한 뒤 동일 모델·데이터·arm으로 셀 전체를 재실행했다. clean retry는 오류 없이 완료됐다.

주의: resume 과정에서 최상위 `manifest.json`이 retry 시점의 runner/wrapper SHA로 갱신됐다. 280개 최초 성공 셀과 1개 retry 셀은 검색 arm과 데이터가 같지만, retry 셀에만 위 로그 위생 패치와 순차 호출 문구가 적용됐다. 실패 원자료는 보존돼 있으나, 이 차이는 엄밀한 단일 바이너리 실행으로 해석할 때의 provenance 제한이다.

## 8. 판정과 다음 실험

이번 결과로 확인된 것은 다음과 같다.

- C29 semantic과 V9 hybrid metadata를 별도 도구로 제공하는 환경은 4 worker에서 안정적으로 실행 가능하다.
- 두 채널 union은 단일 채널보다 Gold 노출을 크게 개선한다.
- 넓어진 후보가 최종 제출 성능으로 이어지지 않았으며, 기존 host-controlled C29 대비 R@10/suff@10은 하락했다.
- 따라서 현재 explicit dual-tool protocol은 production 또는 새 기준선으로 승격하지 않는다.

다음 실험 우선순위:

1. 두 채널을 agent에 80개로 따로 노출하지 말고 host에서 JO 중복 제거 후 RRF/quota로 40개에 융합한다.
2. multi-evidence 문항에서 역할별 최소 1개 제출을 유도하되, 단일근거 문항 과제출을 막는 gate를 함께 둔다.
3. 모든 union에서 충분 노출된 244문항 중 선택 실패 78문항을 대상으로 read/submit 정책만 고정 비교한다.
4. 후보 불충분 37문항은 선택 실험에서 분리하고 cross-reference/graph 회수 실험으로 다룬다.
5. 252개 completeness 미검수 Gold를 검색 결과 비노출 상태에서 감사하고, 별도 holdout 및 2회 이상 rep로 재검증한다.

## 9. 산출물

- `out/c29_dual_tool_train281_luna_medium_w4/manifest.json`
- `out/c29_dual_tool_train281_luna_medium_w4/results.jsonl`
- `out/c29_dual_tool_train281_luna_medium_w4/summary.json`
- `out/c29_dual_tool_train281_luna_medium_w4/analysis.json`
- `out/c29_dual_tool_train281_luna_medium_w4/sessions/`
- 분석 재현: `python analyze_results.py`

