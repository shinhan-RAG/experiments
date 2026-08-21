#!/usr/bin/env python3
print("""=== 0821v3 검색 구성 ===

[1] semantic-tag 파일서치: search
- 색인: filesearch/out/elements_u3.jsonl (32,046 element)
- 조 사상: filesearch/out/elements_u3jo.jsonl (7,612 JO)
- 태그: filesearch/out/tags_u4_fact_rules.jsonl
- 질의 라우팅: 규칙 기반 contract/role/subject/qualifier/schema + 범용 축
- 후보 A: CLM 어휘·슬롯 매칭
- 후보 B: structured_search.py의 구조화 BM25F fact 검색
- 결합: unique-JO quota ensemble (기본 CLM 17 + fact 8)
- C29 조건부 확장: 포함/해당/코드 질문에서만 evidence-unit 3 + membership-support 3 후보
- LLM tagging, embedding, reranker, Gold 입력은 파일서치 내부에서 사용하지 않음

[2] 메타데이터 검색: msearch
- corpus view: V9 chunk metadata
- 희소 채널: BM25(k1=1.5, b=0.75)
- 밀집 채널: dragonkue/BGE-m3-ko cosine similarity
- 결합: RRF(k=60), 채널별 top 50
- 검색 결과 chunk는 char span으로 U3 JO에 사상
- 4 worker가 로컬 embedding server의 단일 BGE 모델을 공유

[3] 에이전트/평가
- Codex CLI model: gpt-5.6-luna, reasoning effort: medium
- 병렬 worker: 4
- 모든 문항에서 search와 msearch를 각각 최소 1회 사용한 뒤 submit
- QA/Gold: gold_train_scoped_u3_reviewed_overlay_v1.jsonl + qids JSON, 총 281문항
- 채점: evidence-group fractional Recall@K, suff@K, RR@10
""")
