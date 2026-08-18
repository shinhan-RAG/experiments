# semtag-v2 실험 방향 설계 (사전 등록 초안) — 2026-08-18

설계서 `RAG_성능고도화_실험설계서_v1.0` 2.5(Semantic Tag) · 2.6(Metadata×Tag 결합) · 3.3(Chunk/Element 도달) 범위.
이전 하네스(v2ds*) 복구 불가 → 재구축. **이 문서에 없는 arm·지표·게이트는 사후 추가하지 않는다.**

## 0. 설계서 준수 사항 (검색기 제약)
- 3.3 검색기 = **벡터 서치(청크 Metadata)** + **파일 서치(Semantic Tag 이정표)** + 질문 성격별 두 도구 선택·조합 전략. **별도 reranker 모델은 설계서에 없음** → arm 으로 두지 않는다. 순위 결정은 (i) 파일 서치의 결정론 정렬 함수, (ii) 에이전트의 제출 순위 두 가지뿐.
- 트랙 A(태그 단독)는 이전과 동일하게 **BM25·임베딩·reranker·학습 정렬 미사용**(file_search 전용). 어휘 채널은 "질문 토큰이 본문에 있으면 조정수준 +1"(CLM) 방식.
- BM25 는 **진단 프록시**(LLM 0회 결정론으로 우주·헤드룸을 재는 도구)로만 쓰고 결과 등록 arm 이 아니다. `RESULT_*_20260818.txt` 의 수치는 전부 진단값.
- 4장(추가 실험: Completeness Check·Page Index)은 2·3장 확정 후 증분 측정 — 본 문서 범위 밖.

## 1. 고정 입력
| 항목 | 값 |
|---|---|
| 문서 | 원(ONE) 판매약관 250212 md, SHA `40a471d5…` |
| 우주 | 색인·태깅 = **u2**(항/표/산식/heading 32,366, `build_universe.py --clean`) · 채점·제출 = **u2jo**(조 7,599, `build_jo_universe.py`) — u2jo 직접 색인을 대조로 병기 |
| gold | **lsh v4 gold**(span 형식 수령 후 u2/u2jo 매핑). 수령 전 임시 = `map_gold_spans.py`(train 338, 앵커 다중매치 120/531 group 표기) |
| 모집단 | noah v3 **train 338** 만. test149 봉인(개봉은 최종 챔피언 1회, 사전 등록 별도). 층화 필수: core/비core, task_type |
| 지표 | Recall@1/5/10/20 (fractional evidence-group), hits/len(gold) 병기, MRR@10, Success@5. 주지표 **R@5** 하나 |
| 검정 | 이진 = 정확 McNemar, 연속 = BCa bootstrap, 가족 단위 Holm. MDE 사전 계산(n=338: 클린 패턴 +1.8pp≈6문항, 혼합 2:1 ≈ +5pp) |
| 널 | query_shuffle(3시드 이상), shuffled_tag(태그 arm), constant_dummy |
| 에이전트 | 모델·예산(page/preview/search/read/submit) 고정, reps ≥ 2, 도구 반환 후보·점수 전량 로그, 세션 오류 ITT 0점 |
| 등록 | `results/elements/<date>/<config_id>.json` (README 규약: repro 4필드·population·channel_params·nulls·limitations) |

## 2. 트랙 A — Semantic Tag 단독 (설계서 2.5 + 3.3 파일 서치)
사전 근거(진단, train 338, 조 단위): 후보는 top-20 안 70%·top-100 안 86% → **순위 문제**. gold 특약 scope 오라클 R@5 .325→.78. 질문의 특약명 완전/부분/미포함 95/103/140. 상용구 중복 gold 19%. 어휘 간극(<0.3) 층 20%는 R@5 .06.

| arm | 변수(1개) | 가설 · 근거 |
|---|---|---|
| A0 | 태그 없음, 어휘 채널만 | 대조 |
| A1 | 태그 재생성(u2): 구조 슬롯(contract/article/section/schema = 규칙) + LLM 슬롯(subject/role/qualifier), CLM 동일가중 | 이전 A4 재현 게이트 |
| A2 | + **소프트 scope 부스트**: 질의 라우터가 특약을 식별한 문항만 contract 일치에 가산(w 사전 고정), 미식별 시 무개입 | 오라클 +.45 상한, 하드 필터의 후보0 회피 |
| A3 | + 슬롯 가중 CLM (contract 정확 > subject > role alias) | 상용구 19% 판별, 동점 해소 |
| A4 | + multi-evidence: (특약×역할) 하위질의 분해 → 그룹별 interleave 제출, 형제 확장(같은 조/같은 특약 다른 역할) | 비core R@5 .27 층 겨냥, fractional 채점 구조 |
| A5 | 에이전트 하네스: 후보 목록에 **태그 헤더 노출**(값 선택 강제 없음, 이정표 v2) | M2 실패 원인(판독 39%) 제거형. 태그를 "고르는" 대신 "읽는" 소비 |
| 대조 | u2jo 직접 색인 vs u2→조 map-back | 입도 확인 |

greedy 승계: 각 스텝 dev(train) ΔR@5 McNemar p<.05 시만 승계, 전 조합 기록. A2·A3 는 LLM 0회 결정론(태그 생성 제외).

## 3. 트랙 B — Metadata (설계서 2.4, lsh 담당)
우리 쪽 요구 인터페이스: (1) v4 gold **span**(line 또는 char 구간), (2) champion 의 **문항별 순위**(uid + line span) — u2jo 로 사상해 트랙 C 입력으로 사용.

## 4. 트랙 C — Metadata × Tag 결합 (설계서 2.6 + 3.3 조합 전략)
같은 우주(u2jo 채점)·같은 gold·같은 지표로 **결합 방식들을 나란히** 재고 채택한다.

| arm | 방식 | 형태 |
|---|---|---|
| C0a / C0b | 태그 단독(A 최종) / 메타 단독(B 최종) | 대조 |
| C1 | **각각 검색 → 순위 융합**: RRF(k=60, 가중 1:1 사전 고정) | 오프라인·결정론 |
| C2 | **순차**: 태그 scope(소프트)로 범위 → 그 안에서 벡터 서치 | 역할 분담(2.6) |
| C3 | **에이전트 복합 실행**: 두 도구(slot_search + vector_search) 동시 제공. C3a 자유 선택 / C3b 규칙 라우팅(특약명 있으면 태그 먼저, 어휘 간극 크면 벡터 먼저) | 3.3 "질문 성격별 선택·조합" |
| C4 | 채움: 에이전트 top-5 + 융합 순위로 채움 | 대조(과거 R@5 하락 사례) |

판정(2.6 그대로): 결합 > max(단독) **이고** 한쪽만 맞춘 문항이 양쪽 모두에 존재 → 채택. 한쪽 쏠림 → 단독. paired McNemar. 중복 필드: 임베딩 문맥엔 좌표(특약·조)만, 태그 채널엔 좌표+역할로 분담(변수로 기록).
사전 오라클: C0a·C0b 문항별 순위로 union@5/@10 과 "A만/B만/둘다/둘다실패" 4분할 → 결합 기대치를 먼저 확정.

## 5. 순서
1. 태그 재생성(u2) — 구조 슬롯(규칙, 무비용) → LLM 슬롯. 2-pass 결정론(구조), LLM 슬롯은 캐시 동결.
2. A0/A1 재현 게이트 → A2·A3 결정론 → A4·A5 에이전트(reps 2).
3. lsh 자료 수령 → gold 재매핑 → 트랙 A 재측정 → C0 오라클 → C1~C4.
4. 챔피언 확정 시 test149 개봉 사전 등록(별도 문서) 후 1회.
