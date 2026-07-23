# MIRACL Korean pretest — 20260723

## Gate decision

**suitable for Korean passage-retrieval algorithm screening**, subject to the recorded data revisions and passage-level contract.
This is not a taxonomy result, Agentic RAG result, insurance-domain validation, or 110K physical-document claim.

## Acquisition and unit

- Topics/qrels revision: `5be20db9509754dadad47689368639fcec739c00`
- Corpus revision: `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`
- MIRACL artifact license: Apache-2.0 (MIRACL repository and dataset cards); underlying Wikipedia terms are recorded separately in the hash manifest.
- Retrieval unit: **passage** (`article_id#passage_index`); article aggregation is not used.
- Raw artifacts are under ignored `data/`; no raw or normalized passage text is committed.

## Measured integrity

- Corpus: 1,486,752 passages from 437,373 articles.
- Queries: train 868; dev 213.
- Judgments: train 12,767; dev 3,057; positive passage union M+ 2,105.
- Orphan qids/passage IDs: 0 / 0; integrity status `passed`.
- Official reference-count comparison: `matches`; NFC changed title/text/query fields: `{'query': 0, 'text': 5, 'title': 0}`.

## Normalized schema
- Corpus: `corpus_id`, `article_id`, `passage_index`, `title`, `text`.
- Query: `qid`, `query`, `split`; qrel: `qid`, `corpus_id`, `relevance`, `split`.
- IDs and numeric relevance are preserved; passage IDs are never collapsed to article IDs.

## Controlled distractor fixtures

Positive passages (`relevance > 0`) are fixed across 20K/50K/110K; only SHA-256-ranked distractor passages increase.
The deterministic rank is `SHA256("miracl-ko-scale-v1\0" + corpus_id)`; the 110K fixture is built first, then ordered 50K/20K prefixes are derived.
- 20000: 20,000 passages; positive preservation `True`; corpus SHA-256 `4ff8dc4803db997aef56414bfe857537e69c3f2dd71397976dbefbb003985a78`
- 50000: 50,000 passages; positive preservation `True`; corpus SHA-256 `ba7423a8de3a11f19c3eeb16d5c345cd418293265e6a427f2a02dbc51438c3b5`
- 110000: 110,000 passages; positive preservation `True`; corpus SHA-256 `fd3088c041916f4bc98f21e54689c2a3eab3fd6ec45daac45fa46591c511217c`

## Leakage and smoke

- No taxonomy artifact was generated. The preparation contract allows taxonomy only from 110K corpus `title`/`text`; qrels and queries are forbidden inputs.
- Train qrels are not used for dev scoring by this preparation layer.
- Lexical plumbing smoke: **blocked** — no existing Korean lexical retrieval dependency is approved for MIRACL; the repository's small regex BM25 helper is not promoted to a Korean tokenizer

## Next boundary

A later, separately instructed MIRACL experiment may create a taxonomy artifact and approve a passage-specific practical-effect rule. It must not reuse the TREC document-level 0.01 screening rule automatically.
