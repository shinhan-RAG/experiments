# hybrid-enrich_v2

실험 베이스라인. 두 검색 채널이 각자 폴더를 가진다.

- `filesearch/` — Semantic Tag 파일서치 (태그 생성, slot 검색기, 질의 라우터, 에이전트 도구/러너, 채점, gold 매핑)
- `vector_search/` — 메타데이터 검색 (청크 600/100, 메타 생성, BM25+Dense RRF)

원문·QA는 `../hybrid-enrich/noah_qaset/`를 참조한다. test149는 쓰지 않는다. `out/`은 커밋하지 않는다.
이전 hybrid-enrich 스크립트는 여기 복사하지 않았다(원본 폴더 참조).
