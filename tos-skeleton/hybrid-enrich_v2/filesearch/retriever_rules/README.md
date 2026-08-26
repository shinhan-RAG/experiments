# retriever_rules — 정규식(규칙) 검색기

공식 채택 검색기. 전 파이프라인 LLM 0회·결정론.
- `tag_rules.py` → `build_tags_u4_fact.py`: U4 태그 생성 (subject/role/locator/qualifier/reference)
- `build_tags_u5_reference_graph.py`: 참조 그래프 (reference_follow 입력)
- `clm_search.py`(SlotSearch·라우터·BM25F 엔진) + `structured_search.py` + `patterns.py` + `textmatch.py`

엔진은 retriever_llm 과 공유된다 — 두 검색기의 차이는 **태그 자산**뿐.
루트의 동명 파일들은 하위호환 셔틀(shim)이다.
