# semtag-v2 — Semantic Tag 트랙 재구축 (2026-08-18~)

이전 하네스(v2ds*)는 복구 불가하여 재구축한다. 결정 사항(2026-08-18, 사용자 확정):

| 항목 | 결정 |
|---|---|
| 문서 | 원(ONE) 판매약관 **250212판** md, SHA `40a471d5…` (noah v3 QA·lsh v4 gold 앵커 문서) |
| 우주(분할) | hybrid-enrich `build_elements.py` 규칙(문단/표/산식/heading, 4,000자 분절) 이식 → `build_universe.py` |
| gold | **lsh v4 gold** 기준 (우주 독립 span 형식으로 수령 예정) |
| 지표 | Recall@1/5/10/20, MRR@10 (fractional evidence-group + hits/len(gold) 병기) |

## 우주 산출물 (`out/`, 본문 포함이라 비커밋 — 통계 json만 커밋)

| 이름 | 규칙 | n | 중앙값/평균 자 | SHA |
|---|---|---|---|---|
| u1 | 원 규칙 그대로 | 26,477 | 23 / 102 | `558c61b3…` |
| u2 | + 노이즈 블록 제거(페이지 헤더·번호·구분선·문서코드) + 페이지 경계 문장 병합 + 조·관·편 제목 heading 타입 | 17,202 | — | `6c16d9b8…` |

u1은 250212 md 가 "한 줄 = 한 단락, 페이지마다 헤더/번호/구분선 삽입" 형태라 원 규칙을 그대로 두면
전체의 절반 이상(14,555개)이 30자 미만 노이즈·조각이 된다(`SHINHAN LIFE` 614, `---` 1,744, 페이지번호 2,608 …).
u2 는 같은 분할 규칙에 데이터 위생만 더한 것이며, 두 우주 모두 결정론(2-pass SHA 일치).

재현: `python3 build_universe.py --doc <250212.md> [--clean --name elements_u2.jsonl]`
