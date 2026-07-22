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
- Model-backed benchmark run: pending model endpoints and experiment data
- Production or Shinhan-specific conclusion: not available from this run
