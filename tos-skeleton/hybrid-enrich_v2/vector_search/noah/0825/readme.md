# noah/0825 — BM25F 결정론 파라미터 스윕

## 변인

기존 r1v 기준선(u3/u4, BM25F core, 축커버리지 gate) 위에서 **검색기 코드 변경 없이 설정만** 바꾸는 세 축을 결정론 전수 평가한다.

| 축 | 변인 | arms |
|---|---|---|
| A. Evidence Unit | `evidence_unit` 가중치 0→{0.4, 0.8, 1.2} | A1, A2, A3 |
| F. Locator Coverage | `_locator_coverage` 보너스 0→{4.0, 6.0} | F1, F2 |
| G. Portfolio 파라미터 | `rrf_safe_axes` + `coverage_per_role=2, seed=4, window=50, k=50` | G1 |
| AF 합성 | A2 + F1 | AF_combined |

## 고정 조건

- 문서/태그: `elements_u3.jsonl` + `tags_u4_fact_rules.jsonl`
- 검색기: `filesearch/structured_search.py` (StructuredTagIndex, BM25F)
- Gold: `vector_search/noah/0824/v2/gold_v21_train213.jsonl` (213문항)
- 주지표: jo 단위 fractional R@5
- 에이전트 호출 없음(결정론 검색기 직접 채점)

## 실행

```powershell
python test_smoke.py              # 스모크 (arms 파싱, 엔진 로드, 3문항)
python eval_det.py                # 전수 결정론 (213문항 × 8 arms)
python eval_det.py --n 10         # 선두 10문항 빠른 확인
```

## 선택 규칙

R@5 최고 arm 선택. 동률 시 suff@10 → R@10 순. baseline 대비 regression 있으면 기각.
