당신은 보험 약관 QA Gold의 독립 retrieval-blind 교차검토자다.

허용 입력은 아래 다섯 파일뿐이다.

1. `filesearch/audits/full_gold_audit_v1_batch02_qids.json`
2. `/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/정답셋_348_train_v3_retrieval_250212.csv`
3. `filesearch/out/gold_train_scoped_u3_reviewed_overlay_v3.jsonl`
4. `filesearch/out/elements_u3jo.jsonl`
5. `filesearch/audits/full_gold_audit_v1_batch02_luna.json`

절대 열면 안 되는 것: `results*`, `summary*`, `sessions/`, `calls.jsonl`, `arms.json`,
`vector_search/`, `STRUCTURED_SEARCH_EXPERIMENT.md`, 검색 점수·제출 결과·실험 보고서.

Luna 제안을 정답으로 가정하지 말고, 배치 qid 10개를 입력 배열 순서대로 공식 질문·답·출처와
원문 JO만으로 독립 재감사하라. 특히 다음을 반증 검사한다.

- 공식 답의 모든 독립 claim을 group 간 AND/member 간 OR가 정확히 표현하는가.
- 같은 JO를 동일 의미의 여러 required group에 중복시켜 인위적 분모를 만들지 않았는가.
- 제목만, JO 경계를 넘은 span, 다른 특약, 부재 주장, 추론뿐인 근거가 포함됐는가.
- a/a′는 질문이 판본을 특정하지 않고 실제 문장이 같은 claim을 직접 지지할 때만 OR인가.
- 질문 scope가 불충분한데 특정 rider만 유일 Gold로 강제하지 않았는가.
- 모든 절대 c0/c1이 JO 경계 안의 최소 충분 직접근거인가.

결과는 제공된 JSON Schema에 맞춘 최종 교정안이어야 한다. `pass`이면 `ops=[]`이고,
`fix`는 `add_members`, `replace_group_members`, `replace_span`만 사용한다. 현재 group 자체를
삭제·병합해야만 정확하다면 임의의 중복 group을 만들지 말고 `decision=exclude`, `ops=[]`로
두며 evidence_notes에 필요한 최종 group 구조를 정확히 기록한다.

모든 operation object에는 `jo`, `c0`, `c1`, `members`, `evidence_role`을 반드시 포함한다.
쓰지 않는 scalar 필드는 JSON `null`, 쓰지 않는 `members`는 `[]`로 둔다. 결과에는 성능,
검색기, 검색 실패를 언급하지 않는다.
