# 리뷰 요청 브리프 — Semantic Tag 파일서치 검색기 실험 (사실 기술, 판단 배제)

## 0. 과제 정의(사용자·설계서 기준)
- 설계서(브레인크루 v1.0, 2026-08-06) 2.5 Semantic Tag 고도화 / 2.6 Metadata×Tag 결합 / 3.3 Chunk·Element 도달. 검색기 = 벡터 서치(청크 Metadata) + 파일 서치(Semantic Tag 이정표) + 질문별 두 도구 선택·조합. reranker 언급 없음. 지표 Recall@1/5/10(20), MRR@10, paired 검정.
- 목표(사용자): (1) slot filesearch 검색기 + semantic tag 스키마 고도화로 element 검색 R@5 **0.7**, (2) 그 위에 metadata(임베딩) 결합·에이전트 두 도구 활용으로 R@5 **0.95**.
- 회의(2026-08-19): QA셋 = lsh 것(noah v3 train348/test149), 채점 = 복수 gold 비율(부분점수), 유사/파생 특약 a/a' 동치. 실험 기준 = PR #49 위 hybrid-enrich_v2. **평가 프로세스·메타 검색·예산·모델 고정, Semantic Tag 검색기만 변인.**
- 최종 보고 지표는 에이전트 실측(ITT, reps≥2). 결정론(LLM 0회) 수치는 선별·진단용.

## 1. 데이터
- 문서: 원(ONE) 판매약관 250212 md(2.76M자, json재조립 파서: 한 줄=한 단락, 페이지마다 헤더/번호/구분선 삽입). SHA 40a471d5.
- QA: train 348(확정 338, core 202/비core 146; task_type single_lookup 210·multi_evidence 121·not_answerable 10·기타), test 149 봉인(사용 금지).
- gold: (a) 임시 = 출처 인용문 앵커링 → char span(338문항, 531 group, 앵커 다중매치 120) (b) lsh QA셋 gold(element id+텍스트) → span 재구성 334문항·432 group; 규칙: gap≤200 병합, 동일텍스트 OR, 동일문구 출현 확장(정확일치·40자↑, 특약 지정 질문은 그 scope만), 나머지 AND. 12 group은 상용구라 OR 멤버 >20.
- 채점: fractional evidence-group Recall@K(Top-K가 덮은 AND group/전체, OR은 1개면 충족), Success@K, sufficient@10(전 group 회수), MRR@10. 채점 단위 = 조(u2jo) map-back.

## 2. 우주(element)
- hybrid-enrich `build_elements.py`(문단/표/산식/heading 블록, 4,000자 분절) 규칙을 250212 md에 이식. 원 규칙 그대로면 26,477개 중 55%가 노이즈(페이지 헤더·번호·구분선·문장 조각) + `$$` 산식 규칙이 문서 절반을 산식으로 오포획 → 수정판 u2 = 32,366(paragraph 17,606 중앙 68자·heading 5,903·table 768·formula 135). 조 단위 u2jo = 7,599(≤4,000자).
- BM25 진단(char-bigram, LLM 0회, train 338): 항 단위 R@5 .128 / 조 단위 .325 / 항 색인→조 map-back .277; 글자 예산 고정 시 항 단위 우세(R@1000자 .241 vs .149). 결정: 색인·태깅 = u2, 채점·제출 = u2jo.
- 팀 내 다른 우주: 이전 semtag 최종 P-section 5,924(조), lsh(id ~7,900, 규칙 미공개), khy-noah 26,477(위 버그 포함) + P-section 5,408.

## 3. 이전 트랙(이전 Claude+사용자, 2026-08-05~13) 사실
- 8/5 (구 분할 dev96/test193, 6,852 element, 어휘 프록시 BM25): 태그 필드 확장·별칭·페이지헤더 제거·표 직렬화·조 재순위·질의측 변환·합성 구어 등 11개 실험 → 어휘 채널에서 전부 미검출/무효. LLM listwise 재순위(K50)만 dev 유의(+16.7pp) 단 위치편향 반증. 결론 기록: "태그의 가치는 어휘 질량이 아니라 주소성". 에라타로 프로토콜 교체(McNemar/BCa/Holm, MDE 사전계산).
- 8/12~13 (noah v3, P-section 5,924, sonnet 에이전트, S5 예산 page40/preview160/search20/read8/submit10, reps 1, $767): 태그 OLD(구 v2 스키마) vs NEW(04B 8슬롯: contract/article/schema=규칙, subject/role/qualifier/table/reference=LLM; 활성 5) × 검색기 AND(슬롯 교집합·문서순) vs CLM(슬롯 합집합∪어휘채널, 맞은 조건 수→문서순). 337 전수 fractional R@5: A1 OLD×AND .355 / A2 NEW×AND .140(후보0 214) / A3 OLD×CLM .489 / A4 NEW×CLM .527(core .698, 비core .271, 가시상한@5 .711). 검색기 축 확정(+.387/+.135), 태그 축 CLM 하 +.038[−.013,+.088] 미확정, AND 하 −.215 유의. 이정표 M2(boost/refine/drop/map, n=29): .638 vs A4 동일문항 .736(판독 정확도 39%). 층 분리: 범용층(subject·schema)만으로 86% 유지, 태그 제거 시 2%.
- v2ds 하네스 소스 12개 파일은 git 미추적으로 유실(SHA만 잔존). 스펙 문서(MANIFEST)와 결과 JSON, 구 AND 검색기 소스(slot_filesearch.py)만 남음.
- 메타 트랙(lsh): bm25+BGE-m3-ko RRF(tag 1.0:meta 1.3, bracket boost, Q_BRIDGE) train348 R@5 .487/R@20 .645, test149 .353/.569 (hits/len(gold)). 하이브리드(khy-noah 0814, 173문항 단일청크 부분집합, success@5): BASE .642 / BASE_CLM .705. PR #49(seyeong): gold 감사(broken 8·오매핑 43 region·OR쌍 17), 정답기준 명세 v1(sufficient@10 주지표), 도구: 태그 별도채널 RRF·soft 계층 하강 채택, AND·concat·hard 하강 기각, train167 suff@10 .695→.820(앙상블, 적응편향 포함).

## 4. 재구축(2026-08-18~19, 현 Claude) 사실
- 검색기 CLM 재구현(스펙 기준; 어휘채널을 "토큰 수 count"로 해석 — binary보다 결정론 +.09, 원 구현 방식 미확인). 규칙 라우터(특약명·역할·대상어·값) + LLM qtags(haiku 1회, 질문+특약사전만 입력, 캐시 동결) + contract 소프트 부스트.
- 태그: NEW 8슬롯 규칙본(subject·role·qualifier는 정규식; 커버리지 contract .998/article .902/role .742/subject .365/qualifier .149/reference .102/table .027), OLD 스키마 이식본.
- 결정론 진단(train, 조 채점): 후보는 top-20 안 70%·top-100 안 86%(순위 문제); 정답 특약 scope 오라클 CLM R@5 .364→.646(R@40 .846); 질문의 특약명 완전/부분/미포함 95/103/140; 상용구 중복 gold 19%; 어휘 간극(<0.3) 층 20% R@5 .06.
- 결정론 선별(임시 gold, 338): CLM 규칙 .364 → +contract w3 .392~.404 → LLM qtags 단독 .448 → +w2/w3 .473; 슬롯 가중(subject/role/schema)·역할별 하위질의 분해는 ±.01(null). lsh gold(OR·비율)에서: CLM+qtags .455 / +w2 .488(suff@10 .569) / AND .323.
- 임베딩 채널(우리 R&D, BGE-m3-ko, u2, lsh gold 334): vec base .535 / vec ctx(특약>조 헤더+원문) .561 / RRF ctx .596 / scoped .568; S@5 4분할 tag만 28·vec만 52·둘다 140·둘다실패 114. lsh RRF와의 C0: union@5 .629, 둘다실패 124(37%) — 분해: 비core 78(multi_evidence 68), 우리 순위 40위 밖 70, 앵커 모호 38, 어휘겹침<0.3 53.
- 에이전트 실측 파일럿(60문항 층화×2reps, 임시 gold): sonnet A1 규칙 .558(core .722/비core .313) / A2a qtags .543; Qwen3-32B-FP8(text 프로토콜) A1 .324 / A2 .337 / C3 태그+벡터 .374 / no_think .231. reps 불일치 sonnet 18%, Qwen 5~15%. 세션 중앙 2~4턴.
- 현재 위치: hybrid-enrich_v2/filesearch (PR #50).

## 5. 현 계획(변인 = Semantic Tag만)
① 태그 스키마: subject·role LLM 추출(문서 문자열 verbatim, 폐쇄 role 코드, 조 단위 haiku) vs 규칙본 ② 라우터: qtags 모델/프롬프트·특약 지정 정밀도 ③ 검색 도구: search 응답에 특약·조 facet 추가(재질의 유도, 하드 필터 아님) ④ diverse 검색(특약×역할 interleave)·10슬롯 제출 규칙 ⑤ 보고 2×2(OLD/NEW 검색기×스키마). 각 1변수, 결정론 선별→에이전트 reps2 실측(McNemar), greedy 승계. 예산·모델·메타 고정.

## 6. 리뷰 요청
위 사실만을 근거로 (a) 방법론·측정 설계의 결함/편향/누락, (b) 이전 트랙 결론 중 재검토해야 할 것, (c) 목표(태그 단독 R@5 .7, 하이브리드 .95) 달성을 위한 고도화 우선순위·구체적 방법과 그 근거(공식 문서·상위 학회 논문 우선), (d) .95의 현실성 판단과 그 조건. 근거 없는 주장은 "추정"으로 표시.
