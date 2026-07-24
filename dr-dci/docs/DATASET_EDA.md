# Experiment Dataset EDA

## 1. Purpose

This analysis characterizes the datasets already used by the experiment before
changing retrieval logic. It is intended to explain dataset-dependent results
and to prevent conclusions from a short public benchmark being generalized to
Shinhan's long insurance documents.

The analysis changes no retrieval treatment. It scans the complete local raw
files deterministically and records SHA-256 hashes in the machine-readable
reports.

## 2. Data Sources Recorded in the Code

| Experiment name | Hugging Face source configured in code | Role |
|---|---|---|
| TREC-COVID | [BeIR/trec-covid](https://huggingface.co/datasets/BeIR/trec-covid) | corpus and queries |
| TREC-COVID qrels | [BeIR/trec-covid-qrels](https://huggingface.co/datasets/BeIR/trec-covid-qrels) | test relevance judgments |
| FiQA | [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | corpus and queries |
| FiQA qrels | [BeIR/fiqa-qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels) | test relevance judgments |
| Ko-StrategyQA | [taeminlee/Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA) | corpus, queries, and dev qrels in `download_base.py` |
| Augmented outputs | [braincrew-dev/dr-dci-augmented](https://huggingface.co/datasets/braincrew-dev/dr-dci-augmented) | upload destination configured by Peter's code |

There is a reproducibility gap in the Ko-StrategyQA downloader. The main
`download_datasets.py` first tries `KETI-AIR/ko-strategy-qa`, then
`wics/strategy-qa`; `download_base.py` instead uses
`taeminlee/Ko-StrategyQA`. The first path was unavailable and the second path
was incompatible with the installed Hugging Face `datasets` version during
this run. The existing local raw set contains 9,251 documents and matches the
size documented by the `taeminlee` path, but file content alone cannot prove
its remote revision. Future downloads should pin one source ID and revision in
a provenance manifest.

## 3. Method

The EDA reads the complete BEIR-style `corpus.jsonl`, `queries.jsonl`, and
`qrels.jsonl` files. It measures:

- document and query length distributions;
- language-script presence and paragraph counts;
- positive-gold cardinality and missing gold IDs;
- query-token coverage and Jaccard overlap against gold documents;
- zero-overlap and rare-query-term rates;
- local source file size and SHA-256;
- augmentation shape when taxonomy, tags, prefix, or metadata files exist.

Lexical metrics use only queries with at least one positive qrel. FiQA and
Ko-StrategyQA include many queries outside the loaded test/dev qrels; counting
those unjudged queries as misses would materially distort the result.

Token counts use a deterministic Unicode regular expression. They are useful
for relative shape analysis but are not embedding-model or chat-model token
counts.

## 4. Full-Corpus Results

| Metric | TREC-COVID | FiQA | Ko-StrategyQA |
|---|---:|---:|---:|
| Documents | 171,332 | 57,638 | 9,251 |
| All queries | 50 | 6,648 | 2,833 |
| Queries with positive gold | 50 | 648 | 592 |
| Positive qrels | 24,673 | 1,706 | 1,145 |
| Document token p50 / p95 | 176 / 355 | 95 / 380 | 64 / 150 |
| Documents at most 256 tokens | 73.66% | 88.09% | 98.91% |
| Documents at most 4,000 characters | 99.76% | 99.23% | 99.96% |
| Multi-gold query rate | 100.00% | 66.05% | 64.36% |
| Gold documents/query p50 / max | 478 / 1,266 | 2 / 15 | 2 / 7 |
| Query-token coverage in gold p50 | 100.00% | 66.67% | 25.00% |
| Zero token-overlap query rate | 0.00% | 0.62% | 25.84% |
| Rare query-term ratio p50 | 0.00% | 0.00% | 50.00% |

Exact raw reports are stored in `analysis/eda/*.json`; concise human-readable
reports are stored in `analysis/eda/*.md`.

No local taxonomy, semantic-tag, prefix, or metadata artifact was present in
this checkout, so this run does not claim an EDA result for the generated
augmentations themselves.

## 5. Interpretation for Peter's Experiment

### 5.1 Semantic tags

The premise behind Peter's observation is supported: these datasets are
dominated by short documents. In Ko-StrategyQA, 98.91% of documents are at
most 256 regex tokens. There is therefore little intra-document search space
for element tags to reduce. This is a plausible explanation for a weak tag
effect, not causal proof. The treatment still needs a paired A/B run.

This benchmark shape is unlike a 3,000-4,000-page insurance document. A null
tag result here must not be used to reject structural tags for long parsed
documents.

### 5.2 Dense versus BM25 or hybrid pull

Ko-StrategyQA has substantially lower query/gold lexical alignment than the
other two datasets: 25.84% of judged queries have no regex-token overlap with
their gold documents, and the median rare-term ratio is 50%. This suggests
that dense retrieval remains important there. Conversely, exact entities,
codes, article numbers, and rare phrases in insurance data are plausible cases
for BM25. The correct next test is the already defined one-factor comparison:
dense pull versus dense+BM25 pull with the same RRF fusion, Top-K, agent,
prompts, and evaluation set.

### 5.3 Agentic-search claims

All TREC-COVID queries have many gold documents, with a median of 478. This
relevance structure is not equivalent to multi-hop reasoning and is not
representative of insurance clause retrieval. Multi-gold rate is therefore a
dataset descriptor, not evidence that agentic search solved a multi-hop task.

The DR-DCI paper motivates bounded dynamic workspaces: global retrieval finds
candidates, while direct corpus interaction investigates materialized
documents. Its 100K-10M and 20M results support the interface concept, but they
do not establish performance on this code, these datasets, or Shinhan data.

## 6. Experimental Consequences

1. Keep the next comparison narrow: only the pull backend changes.
2. Report metrics on positive-gold query populations and publish the exact
   denominator.
3. Separate retrieval recall from answer quality and from tool-call cost.
4. Use paired per-query deltas and confidence intervals rather than comparing
   only aggregate means.
5. Treat public-dataset results as hypothesis screening. A long-document and
   insurance-domain gate remains mandatory.
6. Pin Hugging Face source ID, subset, split, revision, and local file hashes
   before the next shared benchmark run.

## 7. Reproduction

```bash
cd dr-dci
PYTHONPATH=. python scripts/run_eda.py
```

The command writes deterministic reports except for the generation timestamp.
Raw files under `data/` remain untracked.

## 8. Research Basis

- [DR-DCI: Scaling Direct Corpus Interaction via Dynamic Workspace Expansion](https://arxiv.org/html/2606.14885v1): primary source for retriever-steered bounded workspaces and direct corpus interaction.
- [Google Research: Agentic RAG](https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/): primary source for iterative planning, retrieval, and evidence sufficiency checks.
- [PyTorch Korea discussion of Google Research Agentic RAG](https://discuss.pytorch.kr/t/google-research-agentic-rag/10599): Korean secondary summary shared with the project; primary claims should be cited to Google Research.
