# Part 0 — collection→document baseline (신한 약관 9,417)

계약: `PETER_TERMS_COLLECTION_DOCUMENT_BASELINE_HANDOFF_20260804.md`
(SHA-256 `44a87dae25af240a16d37f3c62754693148ed84eb4a5d4147fa521bcf694a0e6`)

측정 대상: 단일 컬렉션(9,417 md) 안에서 질의로 올바른 **문서**를 고르는 능력.
Part 1~4(문서 선택 이후 층)와 별개의 상류 실험이며, Part 1~4 패키지는 수정·실행하지 않는다.

## 구성

- `harness/` — 외부 어댑터. PR #14(`tos-skeleton/`) 파일은 **byte 무수정**으로 import하고
  모듈 경로 상수만 런타임 재바인딩한다. 사용한 PR 파일의 SHA-256을 freeze manifest에 기록.
- 모든 경로는 CLI 인자로 주입 (`--source-root --repo-root --work-dir --results-dir`).
- LLM 사용: **없음** (질문은 결정적 템플릿, gold는 기계 확정 — LLM이 gold를 정할 수 없음).

## 4-arm (2×2 factorial, B11 = concat_ablation)

| arm | label | 표현 |
|---|---|---|
| B00 | filename_baseline | base |
| B10 | index_only | base + Index텍스트 |
| B01 | frontmatter_only | base + frontmatter텍스트 |
| B11 | index_plus_frontmatter | base + Index텍스트 + frontmatter텍스트 (**concat**, two-hop 아님) |

- base = 상품폴더명 + 파일명 (NFC). 전 arm 공통 상수 — 2×2 요인이 순수 가산이 되도록
  사전 등록한 구성 규칙(자세한 근거: `harness/config.py:COMPOSITION_RULE`).
  PR #14 `unified_benchmark.py`의 arm 구성(fm arm에는 base 포함, Index+fm에는 미포함)과
  다른 점을 명시한다.
- Index텍스트 = PR `unified_benchmark.py:24-32` ident 공식 그대로.
- frontmatter텍스트 = PR `eval_content_qa.flatten_fm` 그대로.
- 스코어러 = PR `gate2.BM25` (문자 bigram) 그대로. 4-arm 동일 후보군(9,417 전체)·
  동일 질의·동일 top-k.

## QA 500 (frozen)

- identity 200 / content 200 / mixed 100, 대상 문서 유형 A/B/C = 250/200/50 (D 제외·별도 보고)
- 입력: 원문 md + 파일명·경로의 결정적 신원 사실만. Index/frontmatter 텍스트·기존 순위 입력 금지
  (`source_facts.guard`가 차단, RED 테스트 존재)
- identity gold: 컬렉션 전역 술어 재도출(이름 충돌 시 자동 합집합 — PR 582 문제지의
  동일질문·서로소 gold 결함 재발 방지)
- content/mixed gold: rg 공백무시 정규식으로 전 코퍼스 매칭한 합집합, gold 문서마다
  verbatim evidence span + span_sha256 (PR v2의 단일 anchor 과소집계 보완)
- 문서당 target 1회, 상품당 최대 2회, 정규화 질의 중복 금지
- 500건 전부 기계 재검증 + review ledger (`review_ledger.jsonl`)
- 채점 전 QA·arm 표현·유니버스 manifest SHA-256 동결; 동결 후 변경 시 채점 거부

## 지표·통계

Hit@1/5, MRR@10, nDCG@10(binary), Recall@5/10 (multi-gold), per-query rows,
McNemar exact(전 쌍), B00 대비 paired bootstrap 95% CI·win/tie/loss,
2×2 주효과·상호작용, latency p50/p95, 표현 bytes, peak RSS.

## 실행

```bash
python3 -m harness.runner \
  --source-root <parsed_md> --repo-root <experiments checkout> \
  --work-dir <work> --results-dir <results> \
  --source-zip-sha256 3ac469bc32dce53a4172cd33e6c68bfe63b7dc90d272f0bcaedb4134fb17c08c
```

publication: same-target lock → sibling staging → 검증 → 원자 rename.
동일 identity 재실행 = 검증 후 무변경 reuse. 다른 identity 동일 target = loud 실패.
실패 시에도 사유 포함 failure archive 생성.

## 테스트

```bash
python3 tests/test_part0.py --repo-root <experiments checkout>
```

synthetic RED/GREEN 31건: 파서·지표·McNemar 기지값, 금지 입력 guard, end-to-end 발행,
동일 identity 재사용(byte/mtime 불변), 다른 identity 거부, freeze 변조 거부,
오염 gold 검출.

## 제약·미포함

- 결과 패키지에 원문 본문 미포함(QA evidence span은 계약이 요구하는 필드로 포함).
- 질문 표현이 결정적 템플릿이므로 content 질의는 본문 어구를 포함한다 —
  이 실험은 어휘 기반 문서 선택 능력을 측정하며, 의미 간극(패러프레이즈) 측정은
  별도 승인된 후속(벡터/에이전틱) 범위다.
- two-hop B11은 구현하지 않았다(계약: 승인 없는 추가 금지).
