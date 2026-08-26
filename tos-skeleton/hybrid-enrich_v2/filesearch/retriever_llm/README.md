# retriever_llm — LLM 태그 검색기 (실험용)

검색 엔진은 retriever_rules 와 **동일**(SlotSearch/BM25F, 결정론). 차이는 태그 자산 생성에
LLM을 쓴다는 것 하나다 — 적재 시 1회 생성 → `out/tags_u4_llm.jsonl` 동결 → sha 검증.
검색 시점에는 LLM 호출이 없다.

- `build_tags_llm.py`: 태그 생성기(스캐폴드). LLM 담당 축 = subject/role/qualifier,
  contract/locator/reference 는 규칙 태그에서 승계(환각 방지). element sha 캐시로 증분 갱신.
- `qtag_llm.py`: (구) LLM 질의 태깅 — llm 라우터 arm 용 동결 산출물 생성기.

## 전환 방법 (파라미터 하나)
    python3 host_agent_runner.py --retriever llm ...   # tags_u4_llm.jsonl 사용
    python3 host_agent_runner.py --retriever rules ... # 기본(공식): tags_u4_fact_rules.jsonl

전제: `out/tags_u4_llm.jsonl` 이 생성돼 있어야 한다(대량 생성은 비용 승인 후).
근거·전례: 대장 'LLM 배치 원칙' 절 — u2 시절 LLM 태깅 기각, 규칙 정제(RULE2B) 기각으로
subject 축 품질만 LLM 재도전 가치 있음.
