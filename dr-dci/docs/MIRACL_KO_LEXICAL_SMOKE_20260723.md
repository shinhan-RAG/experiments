# MIRACL Korean lexical plumbing smoke — 20260723

## Scope

This is a standalone passage-retrieval plumbing smoke. It is not a taxonomy, Agentic RAG, focused Part 1/2, insurance-domain, or physical-document operations result.

## Pinned backend

- Backend: `anserini==2.1.1` fat JAR distributed in `pyserini==2.1.0`, with `org.apache.lucene.analysis.cjk.CJKAnalyzer` for `ko`.
- Container base: `python:3.12-slim-trixie@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de`; Java package `21.0.11+10-1~deb13u2`.
- Runtime execution mode: `anserini_java_cli_via_pyserini_distribution`; only NumPy is added for paired bootstrap. No embedding, model, or external API client is installed or invoked.

## Completed checks

- Passage IDs returned by Lucene were checked against each scale corpus.
- Dev qids, qrels, raw per-query rows, metrics, latency, index size, and raw-result hashes were recorded.
- 110K−20K comparison is paired by query and is a plumbing diagnostic only.
- 20000: index build `14.248s`; index bytes `4690855`; passage Recall@20 `0.806481`; raw result SHA-256 `c6d2c807eb33eab01c9219a59cffd503f04c65bc18fb23b8ab6eb1d7b287de64`
- 50000: index build `5.857s`; index bytes `9781200`; passage Recall@20 `0.767391`; raw result SHA-256 `69ac1b9e8d48a2d006638c41f8ac135c423030099be5f35773bc6d3ee756fb0b`
- 110000: index build `15.519s`; index bytes `19394763`; passage Recall@20 `0.732751`; raw result SHA-256 `9b9e036b08e48d51e295b0e6ba60da5e3ff5a86a5391538e7e531989e67ff7aa`

## Interpretation boundary

The fixtures fix all judged passages and increase only unjudged distractors. These numbers test adapter and metric wiring, not natural-corpus growth or Shinhan-domain retrieval quality.
