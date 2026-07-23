# MIRACL Korean pretest — 20260723

## Gate decision

**suitable for Korean passage-retrieval algorithm screening**, subject to the recorded data revisions and passage-level contract.
This is not a taxonomy result, Agentic RAG result, insurance-domain validation, or 110K physical-document claim.

## Acquisition and unit

- Topics/qrels revision: `5be20db9509754dadad47689368639fcec739c00`
- Corpus revision: `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`
- Approved revision-lock SHA-256: `d201546006b1f85864bdc5e62d7f0d886f4fcaaf0cdf60b32beec98b4554d253`; acquisition requests only these revisions and never adopts remote HEAD.
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
The deterministic rank is `SHA256("miracl-ko-scale-v1\0" + corpus_id)`. Qrels choose mandatory membership only; every serialized scale is independently sorted by this relevance-independent rank.
Thus scale nesting is set inclusion rather than file-prefix inclusion, while every shared passage keeps the same relative order across scales.
Before reporting, the persisted files are reread to verify their SHA-256, all-judged membership, global rank order, set nesting, and shared-passage content invariance.
- 20000: 20,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `a01130d3bf5540960a8b6c24cfac78565ef6698f45cd3195327b588a84e92b56`
- 50000: 50,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `9608f260646db6c8a931cb1387a664df63399a7d7c36c9b055b3df7366598cf3`
- 110000: 110,000 passages; judged preservation `True`; positive preservation `True`; corpus SHA-256 `72876a87de9de6ba1e3f7a00e3867673a8b7a61ac21a1aa8ca58a7337aff5107`

## Leakage and smoke

- No taxonomy artifact was generated. The preparation contract allows taxonomy only from 110K corpus `title`/`text`; qrels and queries are forbidden inputs.
- Train qrels are not used for dev scoring by this preparation layer.
- Lexical plumbing smoke: **blocked** — Pyserini is not installed and no existing Korean lexical retrieval dependency is approved; the repository's small regex BM25 helper is not promoted to a Korean tokenizer

## Next boundary

A later, separately instructed MIRACL experiment may create a taxonomy artifact and approve a passage-specific practical-effect rule. It must not reuse the TREC document-level 0.01 screening rule automatically.
