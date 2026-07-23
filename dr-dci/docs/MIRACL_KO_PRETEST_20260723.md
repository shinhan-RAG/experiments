# MIRACL Korean pretest — 20260723

## Gate decision

**suitable for Korean passage-retrieval algorithm screening**, subject to the recorded data revisions and passage-level contract.
This is not a taxonomy result, Agentic RAG result, insurance-domain validation, or 110K physical-document claim.

## Acquisition and unit

- Topics/qrels revision: `5be20db9509754dadad47689368639fcec739c00`
- Corpus revision: `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`
- Approved revision-lock SHA-256: `84d61180fc79180c4b2851c8e19ea8661b334ab0019f6e6dfbed3c137457c8fa`; acquisition requests only these revisions and never adopts remote HEAD.
- Preparation contract SHA-256: `2c806cc432319f24e9b4fbcd34380bf117bd223bc3c299e64090b6a966a92b21`; source Git commit: `9fd2c84cce23dc673a095a051b17b4b025c2dd7b`. The contract hashes the wrapper, preparation algorithm module, and revision lock.
- MIRACL artifact license: Apache-2.0 (MIRACL repository and dataset cards); underlying Wikipedia terms are recorded separately in the hash manifest.
- Retrieval unit: **passage** (`article_id#passage_index`); article aggregation is not used.
- Raw artifacts are under ignored `data/`; no raw or normalized passage text is committed.

## Measured integrity

- Corpus: 1,486,752 passages from 437,373 articles.
- Queries: train 868; dev 213.
- Judgments: train 12,767; dev 3,057; all judged passages 12,601; positive passage union M+ 2,105.
- Orphan qids/passage IDs: 0 / 0; integrity status `passed`.
- Official reference-count comparison: `matches`; NFC changed title/text/query fields: `{'query': 0, 'text': 5, 'title': 0}`.

## Normalized schema
- Corpus: `corpus_id`, `article_id`, `passage_index`, `title`, `text`.
- Query: `qid`, `query`, `split`; qrel: `qid`, `corpus_id`, `relevance`, `split`.
- IDs and numeric relevance are preserved; passage IDs are never collapsed to article IDs.

## Controlled distractor fixtures

All judged passages (including relevance=0) are fixed across 20K/50K/110K; only unjudged SHA-256-ranked distractor passages increase.
The deterministic rank is `SHA256(miracl-ko-scale-v2\0 + corpus_id), independent of relevance`. Qrels choose mandatory membership only; every serialized scale is independently sorted by this relevance-independent rank.
Thus scale nesting is set inclusion rather than file-prefix inclusion, while every shared passage keeps the same relative order across scales.
Before reporting, the persisted files are reread to verify their SHA-256, all-judged membership, global rank order, set nesting, and shared-passage content invariance.
- 20000: 20,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `1ffc669b3b912473a7d7fdcf50d696a2d39a05bcb9bc3f436b4b6d1bac0d7498`
- 50000: 50,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `66e3f5ccdc7fc5b608c93eceff2345765f253fa0d64d3547f6a90b04e64b0dfc`
- 110000: 110,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `76fc195065f29b96f02e1fc3d7bb8c1e2330c4e701c8c472602fa5ced9626030`

## Leakage and smoke

- No taxonomy artifact was generated. The preparation contract allows taxonomy only from 110K corpus `title`/`text`; qrels and queries are forbidden inputs.
- Train qrels are not used for dev scoring by this preparation layer.
- Lexical plumbing smoke: **blocked** — Pyserini is not installed and no existing Korean lexical retrieval dependency is approved; the repository's small regex BM25 helper is not promoted to a Korean tokenizer

## Next boundary

A later, separately instructed MIRACL experiment may create a taxonomy artifact and approve a passage-specific practical-effect rule. It must not reuse the TREC document-level 0.01 screening rule automatically.
