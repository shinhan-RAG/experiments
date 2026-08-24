당신은 보험 약관 QA Gold의 독립 교차검토자다. 첫 감사자의 판정을 그대로 따르지 말고 원문으로 각 결론을 재검증하라.

허용 입력은 아래 다섯 파일뿐이다.

1. `audits/full_gold_audit_v1_batch20_qids.json`
2. `/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/정답셋_348_train_v3_retrieval_250212.csv`
3. `out/gold_train_scoped_u3_reviewed_overlay_v22.jsonl`
4. `out/elements_u3jo.jsonl`
5. `audits/full_gold_audit_v1_batch20_primary_claude.json`

절대 열면 안 되는 것: `results*`, `summary*`, `sessions/`, `calls.jsonl`, `arms.json`,
`vector_search/`, `STRUCTURED_SEARCH_EXPERIMENT.md`, 검색 점수·제출 결과·실험 보고서. 이 감사는
검색 결과를 전혀 모른 상태에서 질문·공식 정답·공식 출처·원문 JO만으로 수행해야 한다.

배치 qid 2개를 입력 배열 순서대로 정확히 모두 감사하라. 첫 감사 결과는 반박 가능한 참고 자료일 뿐이며, 각 qid마다 다음을 확인한다.

- 공식 정답의 모든 독립 claim을 현재 Gold group들이 직접 지지하는가.
- group 간 AND와 member 간 OR가 회의 기준에 맞는가. 질문이 판본을 특정하지 않으면
  같은 rider family의 직접 동등 판본 a/a′는 OR 후보가 될 수 있다.
- 비교·목록·복수 조건·부표 주석·예외 조건이 빠지지 않았는가.
- **이 배치의 두 질문은 모두 특정 특약·판본·범위를 지정하지 않는 전역 질문이다. 따라서 내용
  동등 sweep 을 기존 member 와 같은 bracket scope 로 한정하지 말고, 주계약을 포함한 문서 전체에서
  질문의 주제(해당 claim)를 다루는 조항을 전수 대조하라.** 스코프가 달라도(주계약 vs 특약,
  판본군 상이) 그 조항의 본문이 질문이 요구한 claim 전체를 독립적·완결적으로 지지하면 spec v1
  §3 내용 동등 조항에 따라 같은 group 의 OR member 자격이 있다. 표기 차이(줄바꿈·번호 형식·
  자구 차이)는 실질 차이가 아니면 배제 사유가 아니다. 다만 실질적 차이(보장개시 시점·요건이
  다른 경우, 서류 목록의 구성이 실질적으로 다른 경우, 반대 취지)는 여전히 배제한다.
- 각 span이 JO 내부의 최소 충분 직접근거인가. 제목·추론·다른 특약·부재 주장은 배제한다.

원문 후보는 `rg`로 JO 파일을 검색해 직접 대조한다. 문자열 유사성만으로 동등성을 선언하지
말고, 질문이 요구한 claim을 실제 문장이 지지할 때만 추가한다. 결과는 batch19와 동일한 JSON
구조(`{"audit_policy": ..., "decisions": [...]}`)를 정확히 따른다. `pass`와 `exclude`는
`ops=[]`; `fix`는 현재 적용기가 지원하는 `add_members`, `replace_group_members`,
`replace_span`, `add_required_group`만 사용한다. 독립 claim에 대한 새 required AND group이
필요하면 `add_required_group`으로 추가한다(`group`은 null, `members`는 그 group의 OR member
목록이며 비울 수 없다). 새 group이 기존 group과 같은 근거를 다시 쓰는 것이면 새 group이 아니라
기존 group의 OR member로 넣는다.

모든 operation object에는 `jo`, `c0`, `c1`, `members`, `evidence_role`을 반드시 포함한다.
사용하지 않는 scalar 필드는 JSON null, 사용하지 않는 members는 []로 둔다. 각 절대 좌표는
`jo.char_start <= c0 < c1 <= jo.char_end`를 직접 검증한다. 결과에는 검색·성능 표현을 쓰지 않는다.
대용량 JSONL 전체 행이나 JO 전체를 출력하지 말고, qid·jo_id로 좁힌 뒤 필요한 문장 주변의
bounded snippet만 확인한다. 이는 감사 범위를 줄이는 규칙이 아니라 도구 출력 과부하를 막는 규칙이다.
