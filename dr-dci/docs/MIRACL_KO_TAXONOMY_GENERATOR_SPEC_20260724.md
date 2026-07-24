# MIRACL-ko taxonomy generator specification decision — 20260724

## Decision and status

**Status: blocked pending external execution inputs.** This is a generator and
execution-receipt specification, not a generator run. No model/API call,
110K artifact, taxonomy A/B, Agent run, or Part 1/2 execution was performed.

The one provisional execution path is the repository's existing local
`Qwen/Qwen3-8B` OpenAI-compatible vLLM chat-completions path. It is selected
only because `scripts/utils.py` and the historical taxonomy helper already
reference that path; it does **not** constitute a model/revision approval.
Qwen's official vLLM deployment guide documents Qwen3 thinking-mode and
reasoning-parser controls; vLLM documents structured-output and sampling
controls needed by a later transport implementation. [Qwen3 vLLM deployment guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)
[vLLM structured outputs](https://docs.vllm.ai/en/latest/examples/features/structured_outputs/)
[vLLM sampling parameters](https://docs.vllm.ai/en/v0.9.0/api/vllm/sampling_params.html)
[vLLM engine arguments: model, revision, tokenizer, and served name](https://docs.vllm.ai/en/latest/configuration/engine_args/)

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
| Model/tokenizer | For `actual_execution`, model and tokenizer repositories must be remote Hugging Face repository identifiers (not local snapshot paths), and both revisions must be exact lowercase 40-hex Hub commit SHAs. Branches (`main`/`master`), tags (`latest`/release tags), and short SHAs are rejected. The same commits must appear in generator metadata, runtime metadata, and the vLLM `--revision`/`--tokenizer-revision` arguments. `synthetic_test` fixtures may use explicit synthetic revision strings, but cannot authorize preflight or mint a verified receipt. |
| Representation | Pooling and normalization, or explicit `not_applicable` only when the selected LLM algorithm does not create embeddings. |
| Category construction | Classification/clustering algorithm, library/version or explicit `not_applicable`, cluster-count/selection rule, stable `label_id` rule, unknown/outlier handling, and display-label rule. |
| vLLM serving | Exact launch command and arguments plus their canonical SHA-256; exactly one positional model or one `--model`; exactly one each of `--revision`, `--tokenizer`, and `--tokenizer-revision`; and at most one `--served-model-name`. All four identity values must exactly match the declared generator/runtime model and tokenizer fields. Duplicate/conflicting forms fail. If a served name is present, the canonical request `model` is that name; otherwise it is the approved model repository. `model_tokenizer_binding_kind=actual_execution` is required for real preflight/receipts; `synthetic_test` is fixture-only. `--trust-remote-code` is prohibited for the current actual-execution contract: no immutable remote-code revision contract exists yet, and this work does not infer a need for it. The plan also fixes `generation_config_mode` and the complete canonical server generation-config object, source launch value, and SHA-256, or explicit `not_applicable`. `request_controls_only` must launch with exactly `--generation-config vllm` and may not set an unmodelled `--override-generation-config`; `server_generation_config` must use the approved source launch value. |
| Template and reasoning | Chat-template mode, template SHA-256, content format, and the exact `--chat-template` launch value when explicit; Qwen thinking enable/disable state, reasoning-parser policy, and whether reasoning content is retained. The canonical request template must carry `chat_template_kwargs.enable_thinking`; enabled reasoning must have the matching `--reasoning-parser`, while disabled thinking forbids it. Unused values must be explicit `not_applicable`. |
| Prompt and structured output | Complete prompt/template text and UTF-8 SHA-256; JSON structured-output schema, schema name/SHA-256, and content format, or explicit `not_applicable`. The canonical request template must include the same `response_format` JSON schema. The prompt must not interpolate `corpus_id`, query, qrel, relevance, answer, gold, or evidence. |
| Request controls | Integer seed, determinism/replay mode, and every sampling/request control: `temperature`, `max_tokens`, `top_p`, `top_k`, `min_p`, `stop`, `stop_token_ids`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`, `n`, and `logprobs`. Each uses either an exact JSON-safe value or approved explicit `not_applicable`; `temperature`, `max_tokens`, and `seed` must match the top-level generator controls. |
| Batch controls | Batch size, corpus-ID-sorted contiguous grouping, ordinal order, timeout, retry cap, resume policy, and request-hash idempotency mode. `max_retries` is a **per-successful-batch** limit; the receipt total is the sum of each successful batch's retry count and can therefore be larger than the per-batch limit. |
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
   specification, including vLLM launch model/tokenizer identity, served-name
   request routing, serving/template/thinking/structured-output
   and sampling controls. It also contains a canonical OpenAI-compatible
   request-body template and its SHA-256. That template is derived from, and
   must exactly equal, the declared launch/template/thinking/schema/sampling
   values; it uses only a title/text payload placeholder. A run whose batch
   size/grouping/order/retry/resume controls differ is rejected before receipt
   validation.
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
5. A retry may replace a failed attempt but each successful batch's
   `retry_count` must not exceed the plan's per-batch `max_retries`; the
   receipt summary is the exact sum across raw successful response records.
   The receipt contains exactly one verified successful response for each
   batch and cannot reuse a response from another run or plan.
   `partial`/`failed` receipts are diagnostic only; only `complete` can create
   an artifact manifest.
6. The authorized audit re-hashes every raw response, recomputes the assignment
   list through request envelopes, and compares it with the 110K artifact
   assignment list before the validator mints the opaque
   `VerifiedTaxonomyGenerationReceipt` token. The manifest builder accepts
   that token, never a shape-only receipt mapping, and records its
   `generation_receipt_sha256`. The token directly binds the actual code
   contract aggregate/code hash and the complete generator-spec hash; manifest
   construction rechecks those values against artifact provenance.

The 20K and 50K artifacts remain pure projections of the 110K assignment
mapping. They do not retrain, rename labels, or change shared assignment
scores.

## Exact external inputs required to unblock a real plan

1. Approved Qwen model and tokenizer immutable **40-hex commit** revisions,
   with retrieval date and official model-card source. A local snapshot is not
   an alternative until a separate snapshot identity/file-manifest contract is
   approved.
2. Approved vLLM/runtime deployment identity: dependency lock or immutable
   container digest, library versions, CUDA/driver capture policy, and H200
   execution environment owner.
3. Approved Korean taxonomy algorithm: complete prompt/JSON schema, category
   construction rule, label ID/display-label rule, unknown/outlier rule,
   seed, determinism/replay declaration, and acceptance QA thresholds.
4. Approved execution controls: exact vLLM launch model/revision/tokenizer/
   tokenizer-revision/served-name binding, config/template/thinking/
   reasoning-parser/structured-output/sampling values, batch size, timeout,
   per-batch retry/resume policy, request-response storage location, and
   retention/access policy for raw responses. Any future `--trust-remote-code`
   use requires a separate immutable code-revision contract; it is not covered
   by this plan.
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
