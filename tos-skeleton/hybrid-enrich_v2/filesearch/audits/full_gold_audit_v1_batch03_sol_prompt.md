당신은 보험 약관 QA Gold의 독립 교차검토자다. 첫 감사자의 판정을 그대로 따르지 말고 원문으로
각 결론을 재검증하라.

허용 입력은 아래 다섯 파일뿐이다.

1. `filesearch/audits/full_gold_audit_v1_batch03_qids.json`
2. `/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/정답셋_348_train_v3_retrieval_250212.csv`
3. `filesearch/out/gold_train_scoped_u3_reviewed_overlay_v4.jsonl`
4. `filesearch/out/elements_u3jo.jsonl`
5. `filesearch/audits/full_gold_audit_v1_batch03_luna.json`

주의: 첫 감사 출력은 적용 검증에서 `v3-offline-0260`의 `replace_span` target JO가 현재
Gold member가 아니어서 실패했다. 이 판정을 복사하지 말고, 해당 문항은 현재 group/member와
원문을 다시 확인하여 유효한 operation 또는 exclude로 교정하라.

절대 열면 안 되는 것: `results*`, `summary*`, `sessions/`, `calls.jsonl`, `arms.json`,
`vector_search/`, `STRUCTURED_SEARCH_EXPERIMENT.md`, 검색 점수·제출 결과·실험 보고서. 이 감사는
검색 결과를 전혀 모른 상태에서 질문·공식 정답·공식 출처·원문 JO만으로 수행해야 한다.

배치 qid 20개를 입력 배열 순서대로 정확히 모두 감사하라. 첫 감사 결과는 반박 가능한 참고
자료일 뿐이며, 각 qid마다 다음을 독립 확인한다.

- 공식 정답의 모든 독립 claim이 current Gold의 AND groups로 직접 지지되는가.
- 같은 claim의 판본 a/a′만 OR member인가. 다른 claim이나 다른 rider를 OR로 섞지 않았는가.
- 비교·목록·복수 조건·부표 주석·예외 조건이 빠지지 않았는가.
- 질문 scope가 특정되지 않았는데 특정 rider/심사형/판본만 유일 Gold로 강제하지 않는가.
- 각 span이 JO 내부의 최소 충분 직접근거인가. 제목·추론·다른 특약·부재 주장은 배제한다.

결과는 제공된 JSON Schema를 정확히 따른다. `pass`와 `exclude`는 `ops=[]`; `fix`는
`add_members`, `replace_group_members`, `replace_span`만 사용한다. 새 required AND group이
필요한 경우 현재 적용기가 표현하지 못하므로 `exclude`하고 필요한 최종 group 구조를
evidence_notes에 명시한다. 모든 op 객체의 `jo,c0,c1,members,evidence_role` 필드를 채우고,
사용하지 않는 scalar는 null, members는 []로 둔다. 절대 좌표를 JO 경계와 직접 대조하라.
결과에는 검색·성능 관련 표현을 쓰지 않는다.
