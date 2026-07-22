# DR-DCI Pull Backend Controlled Experiment

## 1. Purpose

Peter's experiment showed that Hybrid RAG was stronger on some datasets while the DR-DCI pull action used dense retrieval only. This experiment changes one factor only:

- Control: `dense`
- Treatment: `hybrid_rrf` (`dense + BM25 -> RRF`)

The agent prompt, number of turns, pull Top-K, taxonomy, semantic tags, contextual prefix, metadata, dataset, query set, and model endpoints remain fixed.

This is an augmentation experiment on the repository implementation. It is not presented as a reproduction of the official DR-DCI paper.

## 2. Research Basis

- Google Research describes standard single-step RAG as insufficient for multi-source and multi-hop questions and evaluates iterative query planning, routing, retrieval, and sufficient-context checks. The published result is evidence for iterative retrieval, not proof that every agentic pipeline is better on every dataset: <https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/>
- Direct Corpus Interaction argues that a fixed top-K similarity interface can discard evidence before an agent can refine its hypothesis. DCI instead exposes composable local corpus operations: <https://arxiv.org/abs/2605.05242>
- DR-DCI separates corpus-level candidate discovery from workspace-level investigation. Its retriever ablation reports that both BM25 and dense retrieval can back the same dynamic workspace interface, with dataset-dependent trade-offs: <https://arxiv.org/abs/2606.14885>
- RISE uses BM25 to construct a bounded interaction space and keeps local shell-style exploration inside it. Its result supports testing lexical candidate discovery as a boundary mechanism rather than scanning the full corpus: <https://arxiv.org/abs/2606.06880>
- Reciprocal Rank Fusion combines independently ranked result lists without requiring score calibration: Cormack, Clarke, and Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009, <https://doi.org/10.1145/1571941.1572114>

## 3. Baseline Integrity Repairs

The following items are measurement repairs, not treatment variables.

1. The previous judge parser checked whether `correct` was a substring. Therefore `incorrect` could be classified as correct. The parser now accepts exact labels only, and judge errors are reported separately instead of being counted as wrong answers.
2. The embedding cache key previously used only the first five sorted IDs, corpus size, and prefix flag. It now covers the model, every ordered document ID, and the exact embedded text.
3. A cached retriever could retain taxonomy state from a previous ablation. Every arm now replaces or clears that state.
4. `workspace_max_docs` was configured but not passed to the runtime workspace. It is now wired to the agent.
5. Actual turns, candidate count, unique workspace documents, and latency are now recorded.
6. The result file stores the fixed controls, Git commit, query count, seed, and the single changed variable.

The previously reported answer accuracy values must be rerun before reuse. In particular, a repeated value such as `0.78` may partly reflect judge-call success rather than answer correctness under the old parser.

## 4. Hypothesis

Adding BM25 candidates through RRF will improve workspace gold recall for queries containing exact entities, identifiers, or discriminative phrases that dense retrieval misses.

The opposite result is also plausible. Dense retrieval can be stronger for paraphrases, and adding lexical candidates can displace useful semantic candidates when Top-K is fixed. The experiment is therefore two-sided.

## 5. Fixed Controls

| Item | Value |
|---|---|
| Dataset | TREC-COVID |
| Corpus subset | 20,000 documents |
| Query set | Dataset query set, unchanged between arms |
| Pull Top-K | 20 |
| BM25 candidate Top-K | 20 |
| RRF constant | 60 |
| Taxonomy | enabled |
| Semantic tag | approach A |
| Contextual prefix | enabled |
| Metadata | enabled |
| Agent maximum turns | 10 |
| Workspace maximum documents | 100 |
| Random seed | 42 |

These values reproduce Peter's current final stack for the first controlled comparison. They are not claimed as production-optimal values.

## 6. Metrics and Decision Rule

Primary endpoint:

- Paired per-query delta in workspace gold recall (`hybrid_rrf - dense`), with a seeded 95% bootstrap confidence interval.

Secondary diagnostics:

- Answer accuracy and valid judge count
- Judge error count
- Recall per pull
- Recall per 100 retrieved candidates
- Average pull count
- Average and unique workspace documents
- Average, p50, and p95 query latency
- Actual agent turns

Interpretation:

- If the primary confidence interval is entirely above zero, the hybrid backend is a promising candidate for the next dataset gate.
- If the interval crosses zero, this run is inconclusive; do not stack additional techniques to force a positive result.
- If the interval is entirely below zero, keep dense as the control and analyze displaced gold documents.
- Do not use answer accuracy when `judged_n` is lower than the expected query count without reporting judge coverage.
- No production latency threshold is inferred from the development environment.

### 6.1 Measurement Expansion (2026-07-21)

These are measurement additions, not treatment changes. The single variable
remains the pull backend.

- Retrieval-only probe: before the agent run, each backend answers every
  judged query once with the original query text on the same retriever
  instance (shared index and embedding cache). The probe reports paired
  Recall@5, Recall@20, Hit@5, Hit@10, and probe latency with seeded bootstrap
  confidence intervals. Queries without positive gold are excluded from the
  denominator. The probe isolates retrieval quality from agent query
  rewriting; the agent loop remains the endpoint for call efficiency.
- Agent accounting: per-tool call counts (`pull`, `grep`, `find`, `read`,
  `answer`), total tool calls (separate from assistant turns, because one turn
  can issue several tool calls), and LLM prompt/completion token sums when the
  endpoint returns a `usage` block.
- Both arms share one retriever construction path (`build_pull_retriever`), so
  the probe and the agent cannot diverge on anything except the backend.

## 7. Execution

Prepare the existing datasets, augmentations, model endpoints, and API credentials exactly as used by Peter's original run. Then execute:

```bash
cd dr-dci
python run_experiment.py --part 5
```

Results are written under:

```text
results/part5_pull_backend/<timestamp>.json
```

Run the local contract tests before the model-backed experiment:

```bash
PYTHONPATH=dr-dci python -m unittest discover -s dr-dci/tests -v
```

## 7.1 Probe Measurement Run (2026-07-21)

A retrieval-only probe run was executed. This run is **not** a reproduction of
Peter's stack and must not be merged with his numbers:

- Embedding: OpenAI `text-embedding-3-small` (the configured H200
  `gte-Qwen2-1.5B-instruct` endpoint was unreachable from this environment).
- Prefix/taxonomy/tag/metadata artifacts were absent locally, so the corpus
  was embedded without contextual prefixes.
- Both arms shared one retriever construction and one embedding cache
  (`dense` embedded; `hybrid_rrf` reused the identical cache), so the paired
  comparison is internally valid for the backend variable only.
- Command: `python run_experiment.py --part 5 --probe-only
  --embedding-url https://api.openai.com/v1/embeddings
  --embedding-model text-embedding-3-small` (result
  `results/part5_pull_backend/20260721_181041.json`, untracked by policy).

Paired results (TREC-COVID 20K subset, 50 judged queries, seed 42,
delta = hybrid_rrf − dense, bootstrap 95% CI):

| Metric | dense | hybrid_rrf | delta | 95% CI |
|---|---:|---:|---:|---|
| Recall@5 | 0.0127 | 0.0126 | −0.0001 | [−0.0005, +0.0003] |
| Recall@20 | 0.0479 | 0.0446 | **−0.0033** | **[−0.0052, −0.0016]** |
| Hit@5 | 1.0 | 1.0 | 0 | saturated |
| Hit@10 | 1.0 | 1.0 | 0 | saturated |
| Probe latency (s) | 0.579 | 0.676 | +0.097 | [−0.010, +0.259] |

Interpretation under the Section 6 decision rule:

- The Recall@20 interval is entirely below zero: on this dataset and this
  embedding, **dense stays the control**. Displacement analysis over the saved
  ranked lists: 28 of 50 queries got worse versus 6 better; fusing BM25
  displaced 367 gold documents out of dense's Top-20 while promoting 304 gold
  and 103 non-gold — a net gold loss, matching the two-sided hypothesis's
  displacement branch.
- This matches the EDA: TREC-COVID's judged queries have full query-token
  coverage inside gold documents (p50 100%) and no rare-term tail, so BM25
  adds mostly redundant evidence and the fixed Top-20 fusion pays a
  displacement cost.
- Hit@5/Hit@10 are saturated at 1.0 because the median gold set is 478
  documents; these metrics are uninformative on TREC-COVID and must not be
  read as quality evidence. Absolute recall is structurally small for the same
  reason (Top-20 against a 478-document gold set caps Recall@20 near 0.042).
- This result screens the idea on one short-document dataset only. It says
  nothing about Peter's embedding stack, agent-loop behavior, FiQA,
  Ko-StrategyQA, or Shinhan documents.

Next single experiment (one only): run the same probe on FiQA, where the EDA
shows partial lexical alignment (66% query-token coverage, 2-document median
gold) and Peter's Part 4 already observed Hybrid RAG ahead of DR-DCI. That is
the dataset where the hybrid pull hypothesis has its best prior.

## 7.2 FiQA Probe Run (2026-07-21)

Same probe, same deviations as Section 7.1 (OpenAI `text-embedding-3-small`,
no augmentation artifacts), full 57,638-document corpus, all 648 queries with
positive gold (EDA confirms zero missing gold documents). A first attempt
failed mid-embedding on a transient network outage (`[Errno 51]`, partial
billing, no cache written); the embedding client now retries transient
connection failures with exponential backoff (4 attempts, then raises — no
silent degradation), and the rerun completed
(`results/part5_pull_backend/20260721_203516.json`, untracked by policy).

| Metric | dense | hybrid_rrf | delta | 95% CI |
|---|---:|---:|---:|---|
| Recall@5 | 0.4325 | 0.3697 | **−0.0627** | **[−0.0848, −0.0401]** |
| Recall@20 | 0.5910 | 0.5607 | **−0.0303** | **[−0.0443, −0.0165]** |
| Hit@5 | 0.6343 | 0.5802 | **−0.0540** | **[−0.0818, −0.0262]** |
| Hit@10 | 0.6975 | 0.6929 | −0.0046 | [−0.0247, +0.0154] |
| Probe latency (s) | — | — | +0.0839 | [+0.0631, +0.1056] |

Displacement over ranked lists (recall@20 basis): 82 queries worse versus 26
better (540 unchanged); fusion pushed 5,020 non-gold candidates into the
Top-20 while admitting only 33 new gold and displacing 97 dense gold. Hit@5
flipped from hit to miss on 61 queries versus 26 gains.

Interpretation:

- Every interval that excludes zero is negative. Under the Section 6 decision
  rule, **dense remains the control on FiQA as well**, and this dataset had
  the hypothesis's best prior (66% lexical alignment; Peter's Part 4 showed
  Hybrid RAG ahead of DR-DCI there). On FiQA's natural-language finance
  questions, BM25's Top-20 is dominated by lexically overlapping but
  non-relevant posts, and equal-weight RRF lets them displace dense gold.
- This does **not** contradict Peter's Part 4. His "Hybrid RAG" was a
  different full pipeline (single-shot dense+BM25+reranker) compared against
  the DR-DCI agent loop; this probe isolates only the pull-backend fusion.
  The two results together suggest Part 4's Hybrid advantage did not come
  from BM25 candidates per se.
- Open confound: both probe runs used `text-embedding-3-small`, a strong
  general embedder. A weaker embedding (such as the configured gte-Qwen2
  stack) could leave more room for BM25 to help. The conclusion is therefore
  scoped to this embedding.

Two-dataset verdict: the hybrid_rrf pull hypothesis is refuted on both tested
datasets under this embedding. Next single experiment (one only): rerun both
probes on the configured H200 `gte-Qwen2-1.5B-instruct` endpoint when it is
reachable, which removes the embedding deviation and tests whether the
hybrid pull's value depends on embedding strength. (Ranked pull preview
remains the next agent-loop candidate per Section 8; it is not an additional
proposal here.)

## 8. Scope Boundary

The following are intentionally not changed in this experiment:

- Ranked pull previews
- Dynamic Top-K
- Number of mandatory pulls
- Reranker use inside DR-DCI
- Taxonomy boost formula
- Semantic-tag schema or filtering timing
- Metadata filter semantics
- Contextual-prefix length
- Model replacement

The official DR-DCI implementation exposes ranked previews and much larger dynamic workspace expansion budgets, while this repository currently returns counts from a Top-20 pull. Ranked preview is a strong next candidate because the DR-DCI paper reports a controlled gain, but it must be tested after this backend experiment to preserve one-factor attribution.

Semantic tags should be revisited on long, internally structured documents. In the current code they operate after pull, inside the workspace, so they cannot recover a relevant document that the pull backend never materialized. This explains why a short-document benchmark can show little tag effect without proving that tags are generally ineffective.

## 9. Current Status

- Experiment code: complete (probe and accounting added 2026-07-21)
- Offline contract tests: complete (13 passing)
- Model-backed probe runs: TREC-COVID and FiQA executed 2026-07-21 with the deviations in Sections 7.1-7.2; both refute hybrid_rrf under this embedding; agent-loop benchmark still pending Peter's stack
- Production or Shinhan-specific conclusion: not available from this run
