# Semantic Tag 최종 조합 — 재현 패키지

`2026-08-12` · donggyu-ralph

## 무엇인가

미팅 3항(*"최종 조합 확인 · 다른 사람이 바로 작업 재현할 수 있도록"*)의 산출물.
**코드·SHA·수치만** 담는다. 원자료(약관 md · QA · gold · 태그)는 별도 경로.

## 파일

| 파일 | 내용 |
|---|---|
| `MANIFEST.json` | 최종 조합 전체 정의 + 전 SHA + 재현 기준값 + 표기 규약 + 필수 유보 |
| `reproduce.sh` | SHA 대조 → 재앵커링 → 결정론 지표 2종 실행 |

## 최종 조합 한눈에

```
적재(2.5)  tags_run1.jsonl  edb581c5…      04B 8슬롯, 활성 5종
           qtags A/B_run1 + v3_extra(127)  질의측
검색(3.3)  CLM(조정수준) + 어휘채널        v2ds32_tiebreak.TieBreak.ordered_tie(qid,'T1a')
           동점 → P-section 문서순         BM25·reranker·임베딩·학습정렬 미사용
예산(S5)   page40 preview160 search20 read8 output64k · submit 순위 10개
우주       elements_rev4.jsonl 36e8c9e6…   P-section 5,924 / 원문 md 40a471d5…
QA         noah_qaset 98662c1b…            재앵커링 337문항
```

## ★ 지표 두 종류를 구분할 것

| 절 | 용도 | 예 |
|---|---|---|
| `결정론_재현_기준값` | **환경 검증용 체크섬.** LLM 0회 — 누가 돌려도 소수점까지 동일 | core118 R@5 `.4322` |
| `에이전트_실측_성능` | **보고용 성능** | core30 R@5 `.6667` · MRR@10 `.5708` |

**두 값은 프로토콜(기계적 top-5 vs 에이전트 제출 10개)도 모집단(118 vs 30)도 달라 비교 불가.**

## 실행

```bash
bash _artifacts/reproduce.sh                      # 결정론 재현 (LLM 0회)
# 에이전트 실측(별도, 비용 발생)
python3 v2ds46_runner.py --execute --n -1 --pool all --reps 1 --workers 8 --resume
```

## 인용 전 필수 확인

1. **C 레벨 수치는 "문서 1건이 이미 골라진 상태" 가정.** 11만건에서는 A/B 깔때기의 문서 선택 성능이 곱해진다
2. `core_retrieval=True`는 193/202가 evidence group 1개인 **쉬운 층** — 모집단 명시 필수
3. noah v3는 v2 대비 분할 재추첨(205문항 이동) → **v3-test 를 확증셋으로 쓸 수 없음.** test 미개봉
4. **에이전트는 태그를 보지 못한다** — 태그는 오프라인 정렬 함수의 입력. 설계서 2.5의 "이정표" 방식은 미구현
5. **OLD/NEW 태그 우열은 예산에 따라 부호가 반전** — 미확정
