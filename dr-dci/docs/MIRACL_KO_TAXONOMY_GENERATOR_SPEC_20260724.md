# MIRACL-ko taxonomy generator specification decision — 20260724

## Decision and status

**Status: blocked pending external execution inputs.** This is a generator and
execution-receipt specification, not a generator run. No model/API call,
110K artifact, taxonomy A/B, Agent run, or Part 1/2 execution was performed.

The one provisional execution path is the repository's existing local
`Qwen/Qwen3-8B` OpenAI-compatible vLLM chat-completions path. It is selected
only because `scripts/utils.py` and the historical taxonomy helper already
reference that path; it does **not** constitute a model/revision approval.
Qwen's official repository documents Qwen3-8B deployment through vLLM, while
vLLM documents the OpenAI-compatible serving and structured-output surfaces
needed by a later transport implementation. [Qwen official repository](https://github.com/QwenLM/Qwen3)
[vLLM official documentation](https://docs.vllm.ai/en/latest/)

The configured `Alibaba-NLP/gte-Qwen2-1.5B-instruct` endpoint is not adopted
as an alternative generator: it is configured in this repository as a
retrieval embedding endpoint, and this checkout has neither an approved
clustering implementation nor an immutable embedding/runtime provenance for
taxonomy construction. No new model or clustering library is introduced.

## Gate S0 evidence

| Check | Local evidence | Result |
|---|---|---|
| Existing generation route | `scripts/utils.py` targets local vLLM chat completions with `Qwen/Qwen3-8B`; `scripts/build_taxonomy.py` uses that helper. | Candidate transport exists, but the old helper is not reusable: it passes IDs and has a fallback path. |
| Model and tokenizer immutability | `requirements.txt` has no model revision lock; local Hugging Face cache has no verified Qwen3 snapshot. | Blocked; revision values must not be guessed. |
| Runtime pin | The repository has no vLLM dependency lock or container digest for taxonomy generation. | Blocked. |
| Clustering dependency | No approved `transformers`, `sentence-transformers`, `scikit-learn`, HDBSCAN, or UMAP lock is present for a new clustering path. | Blocked; do not add one at this gate. |
| Existing taxonomy script | `scripts/build_taxonomy.py` is historical TREC helper code and has error fallback. | Explicitly excluded from the MIRACL generator contract. |

## Required approved generator specification

An actual `TaxonomyGenerationPlan` is valid only when its generator object
contains all of the following exact values and their canonical
`generator_spec_sha256`. Placeholder values, a branch name, `latest`, or a
floating model tag are invalid.

| Contract area | Required value before approval |
|---|---|
| Generator | Generator type, algorithm/classification rule, implementation version, clean generator source commit, code-contract and aggregate hashes. |
| Model/tokenizer | Repository, immutable model revision, tokenizer repository and immutable tokenizer revision. |
| Representation | Pooling and normalization, or explicit `not_applicable` only when the selected LLM algorithm does not create embeddings. |
| Category construction | Classification/clustering algorithm, library/version or explicit `not_applicable`, cluster-count/selection rule, stable `label_id` rule, unknown/outlier handling, and display-label rule. |
| Prompt | Complete prompt/template text, UTF-8 SHA-256, structured response schema, temperature and max tokens. The prompt must not interpolate `corpus_id`, query, qrel, relevance, answer, gold, or evidence. |
| Controls | Integer seed, determinism/replay mode, JSON-safe parameters, batch size, corpus-ID-sorted contiguous grouping, ordinal order, timeout, retry cap, resume policy, and request-hash idempotency mode. |
| Runtime | Dependency-lock SHA-256 or immutable container digest, exact library versions, and model/tokenizer values repeated in the execution runtime receipt. |

The mandatory semantic payload is one JSON object per passage:

```json
{"title": "...", "text": "..."}
```

The 110K fixed fixture is the only semantic input. `corpus_id` may occur only
in the mapping envelope for deterministic join, duplicate detection, and
serialization order. Query, qrel, relevance, gold/evidence IDs, answer, and
evaluation outputs are rejected at the public input boundary.

## Plan, run, response, receipt

1. The approved plan includes `run_controls` and hashes the complete generator
   specification. A run whose batch size/grouping/order/retry/resume controls
   differ is rejected before receipt validation.
2. `TaxonomyGeneratorRun` sorts opaque `corpus_id` once, then creates
   contiguous batches. A batch has a non-semantic mapping envelope and a
   title/text-only semantic payload.
3. Each request carries `generation_request_sha256`, derived from the plan,
   envelope, semantic payload, batch schema, and ordinal. It never enters the
   model prompt. A response must return the same request hash and its local
   `request_index` values; response order is irrelevant.
4. A `TaxonomyGenerationReceipt` binds the plan, approval, code contract,
   source commit, run hash, ordered request hashes, each raw response's byte
   size/SHA-256, result assignment SHA-256, batch success/failure/retry counts,
   timestamps, runtime, and determinism/replay declaration.
5. A retry may replace a failed attempt but the receipt contains exactly one
   verified successful response for each batch. It cannot reuse a response
   from another run or plan. `partial`/`failed` receipts are diagnostic only;
   only `complete` can create an artifact manifest.
6. The authorized audit re-hashes every raw response, recomputes the assignment
   list through request envelopes, and compares it with the 110K artifact
   assignment list before the manifest's `generation_receipt_sha256` is
   accepted.

The 20K and 50K artifacts remain pure projections of the 110K assignment
mapping. They do not retrain, rename labels, or change shared assignment
scores.

## Exact external inputs required to unblock a real plan

1. Approved Qwen model and tokenizer immutable revisions, with retrieval date
   and official model-card source.
2. Approved vLLM/runtime deployment identity: dependency lock or immutable
   container digest, library versions, CUDA/driver capture policy, and H200
   execution environment owner.
3. Approved Korean taxonomy algorithm: complete prompt/JSON schema, category
   construction rule, label ID/display-label rule, unknown/outlier rule,
   seed, determinism/replay declaration, and acceptance QA thresholds.
4. Approved execution controls: batch size, timeout, retry/resume policy,
   request-response storage location, retention/access policy for raw
   responses, and maximum allowed retries.
5. An actual execution approval record identifying the approver, timestamp,
   basis, exact plan hash, source commit, code-contract hash, and code hash.
6. A clean generator-source checkout and separately read-only control root;
   the two user-untracked AIHUB documents in this checkout are not a valid
   clean source root and must remain untouched.

Until all six inputs exist, schema and synthetic fixtures are the only valid
outputs. The status is **not approved for generation**.

## Non-results and next step

- Generator/receipt schemas and synthetic validation harness: implemented.
- Actual model resolution, taxonomy generation, artifact, lexical rerun,
  embedding, taxonomy boost, Agent, Part 1, and Part 2: not executed.

The next permitted action is review/approval of the six external inputs, then
creation of a real plan in a separate control root. It is not model execution.
