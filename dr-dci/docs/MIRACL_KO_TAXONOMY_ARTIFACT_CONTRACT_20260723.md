# MIRACL-ko taxonomy artifact contract — 20260723

## Status and boundary

This document defines a **preparation contract and synthetic/mock harness** for a future MIRACL-ko taxonomy artifact. It does not create the 110K artifact, call a model or API, run Anserini, connect MIRACL to the Part 1/2 runner, or measure taxonomy retrieval quality.

- Retrieval unit: **passage**, never physical document.
- Source fixture: the pinned MIRACL-ko 110K controlled-distractor fixture from the M0 preparation manifest.
- Smaller scales: fixed **filter projections** of that one 110K artifact; they are not separately learned taxonomies.
- Git boundary: real artifact bodies live under ignored `/data/miracl-ko/taxonomy/`; their real hash/provenance manifest is tracked at `config/miracl_ko_taxonomy_artifact_manifest.json`. This repository currently tracks only the validator, manifest schema/contract, and a three-passage synthetic fixture.

Consequently, an absent real artifact is expected today. The audit CLI and future consumer preflight fail loudly in that state.

## Gate 0: current-code comparison

The existing TREC-COVID path is intentionally not reused or changed by this work.

| Concern | Current code evidence | Contract decision |
|---|---|---|
| Artifact loading | `run_experiment.py:91-142` loads `data/taxonomy/{dataset}_{size}k.json`; a requested missing file raises `FileNotFoundError`. | MIRACL gets a standalone manifest/body validator. It is **not registered** in this loader or runner. `load_integrity_only_taxonomy_projection()` is explicitly non-authorizing; a future experiment consumer must use `load_authorized_taxonomy_projection()`. |
| Existing artifact schema | `scripts/build_taxonomy.py:25-41,77-111` requests and writes a flat `doc_id → {L1,L2,L3}` dictionary; it separately generates every scale and silently assigns an `Other/General/unknown` default on parse/API errors (`:87-103`). | That schema and fallback are unsuitable for a reproducible MIRACL treatment. The new artifact is versioned, hash-bound, coverage-complete, and has explicit `unknown` assignments. |
| Query taxonomy path | The agent receives a static taxonomy schema in its system prompt (`src/agent/dci_agent.py:32-45`) and the `pull` tool accepts optional L1/L2 filter values (`:78-100`). The agent/LLM itself supplies a filter and query to `_execute_tool` (`:432-447`); there is no independent query-taxonomy artifact or tagger. | This contract creates no query taxonomy. If added later, it may receive query text only and must be separately approved; qrel/gold input remains forbidden. |
| Boost action point | `PullRetriever.pull()` identifies matching metadata (`src/agent/retriever.py:137-154`) then multiplies only positive dense cosine scores before backend fusion/reranking (`:156-165,245-265`). | A later MIRACL A/B changes only this soft score boost. It must not add filtering, prompt expansion, or workspace taxonomy exploration in the same arm. |
| Agent wiring | `run_experiment.py:247-295` passes a taxonomy map to retriever scoring and conditionally to workspace taxonomy. | No MIRACL mapping is wired into this path in this task. A future adapter must translate a verified mapping into the then-approved consumer schema without changing labels or scores. |
| Focused arm delta | `focused_taxonomy_steps()` starts at `run_experiment.py:145`; the runner loads the shared taxonomy prompt schema at `:262-290`. | The future pair must retain the existing invariant: shared prompt schema, disabled workspace taxonomy, and only `taxonomy`/soft boost differs. `validate_focused_taxonomy_treatment_pair()` freezes that requirement as a pure contract test. |

The historical `scripts/build_taxonomy.py` also passes an ID through an LLM batch payload (`:77-103`). The MIRACL generator-facing API below deliberately returns no passage ID, preventing that identifier from being tokenized, embedded, clustered, or used in label naming.

## Input and leakage contract

`corpus_id` is an opaque mapping/join key, not semantic content. The only generator-facing record is:

```json
{"title": "...", "text": "..."}
```

The transport record may initially contain `corpus_id`, `title`, and `text`; its only allowed non-semantic uses are recorded exactly as:

```json
["mapping_join", "duplicate_detection", "deterministic_output_order"]
```

Thus ID text, prefixes, article numbers, and namespaces cannot be used in embeddings, clustering, label naming, or classification. The deterministic assignment ordering is only a serialization/reproducibility property; it supplies no semantic feature.

The validator rejects query/qrel/qid/relevance/positive-negative/answer/gold/evidence/ranked-result fields and any unsupported transport field. It permits neither gold IDs nor an alternative ID that conveys relevance. Query taxonomy is outside this contract; a future query tagger must accept query text only.

### Generator batch and output rejoin

`build_taxonomy_generator_batch()` is the sole external transport contract. It validates input transport records (`corpus_id`, `title`, `text`), sorts once by opaque `corpus_id`, then writes two hash-bound parallel arrays: `mapping_envelope` contains only `{request_index, corpus_id}` and never enters a semantic generator; `semantic_payload` contains only `{title, text}` in the same deterministic order.

A generator returns exactly one `{request_index, label_id, score, status}` object for every semantic payload row. `request_index` is transport metadata, never prompt content; neither request nor response contains `corpus_id`. Output order may vary. `rejoin_taxonomy_generator_outputs()` indexes responses by verified request index and rejects positional/legacy output, short/extra output, duplicate/missing/out-of-range/bool/non-integer request index, a changed envelope/hash/order, or an ID in semantic payload. Therefore reversed responses produce the same final passage mapping. `build_semantic_generator_inputs()` is retained only as a deprecated deterministic compatibility extractor; it has no output-binding contract and is prohibited for real generation/rejoin.

## Artifact schema and provenance

The schema versions are `dr-dci.miracl-ko-taxonomy-artifact.v1` and `dr-dci.miracl-ko-taxonomy-manifest.v1`.

Each ignored artifact body contains:

```json
{
  "schema_version": "...artifact.v1",
  "taxonomy_artifact_id": "stable-artifact-id",
  "dataset": "MIRACL",
  "language": "ko",
  "source_revisions": {"topics_qrels": "<40-hex>", "corpus": "<40-hex>"},
  "retrieval_unit": "passage",
  "source_scale": 110000,
  "projection_scale": 110000,
  "input_contract": {"mapping_key": "corpus_id", "semantic_generator_input_fields": ["title", "text"], "corpus_id_semantic_use": "prohibited", "...": "..."},
  "provenance": {"revision_lock_sha256": "...", "preparation_contract_sha256": "...", "input_110k_corpus_sha256": "...", "input_subset_manifest_sha256": "...", "generator": {"generator_type": "...", "generator_version": "...", "generator_code_sha256": "...", "model_or_algorithm": "...", "model_or_tokenizer_version": "...", "prompt_template_sha256": null, "seed": 0, "parameters": {}, "determinism_mode": "deterministic|replay_required"}},
  "label_catalog": [{"label_id": "stable-id", "label": "display label"}, {"label_id": "unknown", "label": "Unknown"}],
  "assignments": [{"corpus_id": "opaque-passage-id", "label_id": "stable-id", "score": 0.0, "status": "assigned|unknown"}],
  "projection": {"method": "source_110k_identity_v1", "source_artifact_sha256": null}
}
```

`unknown` is explicit (`label_id=unknown`, `score=0`, `status=unknown`); it is never an unrecorded fallback. Assignment IDs are unique, the label catalog has unique IDs and labels, assignments have finite `[0,1]` scores, and every expected passage has exactly one assignment.

The body deliberately has no wall-clock timestamp so that an actually deterministic generator can be compared with a canonical content SHA-256. The **outer tracked manifest** records an RFC3339 timezone-bearing `generated_at`, exact artifact body byte size/SHA-256 for all scales, source Git commit, source revisions, the input/provenance contract, `generator_contract_sha256`, and each projection's source-artifact content hash. A non-deterministic generator must declare `replay_required`; it cannot claim deterministic regeneration. Generator parameters must be JSON-safe with no NaN/Infinity; boolean values are not valid seeds or assignment scores. Display labels must already be Unicode NFC with normalized single-space whitespace, so the artifact and consumer never silently normalize a category at match time.

## Fixed-scale projection

1. Generate exactly one identity artifact from the **110K fixture title/text** only.
2. For 50K and 20K, filter that assignment list by the already-verified scale `corpus_id` set.
3. Preserve `taxonomy_artifact_id`, label catalog, provenance, every shared `label_id`, `score`, and `status` byte-for-value in the projected rows.

The source body declares `source_110k_identity_v1`. Projections declare `filter_110k_mapping_by_corpus_id_v1` and the source body’s canonical SHA-256. The validator rejects a projection containing a non-110K ID, a missing target-scale ID, a changed assignment, label catalog drift, provenance drift, revision mismatch, or body/hash mismatch.

This is transductive controlled distractor scaling. It must be described as “taxonomy generated on the fixed 110K passage fixture and projected to 20K/50K,” never as a taxonomy trained independently on a smaller corpus.

## Flat-L1 consumer adapter contract

The existing retriever exact-matches document taxonomy dictionaries against Agent L1/L2 filter strings. The sole approved preparation-level bridge is therefore `build_flat_l1_consumer_adapter()`:

- artifact `label` is copied unchanged to document `{"L1": label}` and to the Agent L1 list;
- `label_id` remains an internal artifact/audit key and is never used as an L1 match value;
- the Agent schema is exactly `{"L1": [artifact display labels except Unknown], "L2": {}}`;
- `unknown` assignments are omitted from boost-eligible document mapping;
- assignment `score` stays in artifact QA only and is `not_used_by_existing_soft_boost`.

This is not wired to `run_experiment.py`, the Agent, or `PullRetriever`. The validator rejects a label-ID/display-label mismatch, unknown mapping, score-weight mode, or passage omission/addition. A score-weighted bridge is rejected because the current retriever only has the configured global taxonomy multiplier; confidence weighting would be a second retrieval treatment.

## Generation plan, provenance, and approval boundary

The artifact manifest is not an approval record. Before any artifact exists, an artifact-free `TaxonomyGenerationPlan` binds MIRACL identity/unit, source scale 110K, four fixed input SHA-256 values, source commit, code-contract/aggregate SHA, generator/model/tokenizer/prompt/seed/parameters, title/text-only input contract, batch schema version, output artifact schema version, fixed projection method, and determinism mode. The plan cannot contain an artifact body or artifact hash.

A real generation requires the plan plus three separate tracked objects: (1) a generator code contract with uniquely sorted relative file list, each byte SHA-256, and `generator_code_sha256 = SHA256(canonical JSON of aggregate method + file list)`; its canonical JSON SHA-256 is `generator_contract_sha256`; (2) an approval record that directly names `generation_plan_sha256`, source commit, code-contract/aggregate SHA, approver, RFC3339 time, and basis; and only after execution (3) a result manifest with body hashes and the approved plan hash.

`validate_taxonomy_generation_preflight()` requires only the plan, approval, independently rechecked generator files/code contract, current Git HEAD/clean state, and independently verified input hashes. It does not require an artifact manifest. `validate_taxonomy_artifact_authorization()` then binds a completed integrity-checked artifact to the same plan. No approval record is created in `config/` by this work. The fixture contains a marked `synthetic_test` record which validates fixtures only and cannot authorize execution.

## Preflight, audit, and synthetic RED → GREEN harness

`src/miracl_ko/taxonomy_artifact.py` is preparation-only. It validates coverage, unknown rate, number/distribution of assigned labels, minimum/maximum label sizes, maximum label share, duplicate/conflicting mapping, orphan passage IDs, provenance, exact nested projections, and deterministic regeneration claims. `scripts/audit_miracl_ko_taxonomy_artifact.py` re-reads the real ignored subset manifest, revision lock, artifact body files, and their hashes; it creates neither labels nor a search index.

The following RED counterexamples are fixed in `tests/test_miracl_ko_taxonomy_artifact.py`, then made GREEN by the validator:

- query/qrel/relevance/gold/evidence or unsupported input field reaches the generator boundary;
- corpus revision or pinned input SHA differs from the independently verified fixture;
- missing manifest/body, tampered artifact byte hash, missing seed/provenance, duplicate ID, and orphan/missing mapping;
- changed common-passage label/score, source-outside projection, and non-exact projection;
- deterministic generator content change;
- reverse-order batch/rejoin ambiguity, missing/extra generator output, and semantic-payload ID leakage;
- label-ID/display-label exact-match mismatch, unknown boost eligibility, score-weight drift, and consumer passage coverage drift;
- reversed/duplicate/missing/out-of-range/bool request index and prohibited positional output rejoin;
- artifact-free plan approval, wrong-plan/source/code/input/parameter drift, missing approval, dirty source, and synthetic approval execution attempt;
- self-declared code/source provenance, non-RFC3339 timestamp, bool seed/score, non-finite parameter, and non-canonical label;
- future focused control/treatment differences other than the taxonomy boost toggle.

The tracked three-passage fixture under `tests/fixtures/miracl_ko_taxonomy_artifact/` validates the body SHA and all three projection roles without putting real MIRACL taxonomy content in Git.

## Reproduction commands

```bash
cd /Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci
PYTHONPATH=. pytest -q tests/test_miracl_ko_taxonomy_artifact.py

# Real audit: all three authorization inputs are required by default; this
# command fails loudly until they and the artifact manifest exist.
PYTHONPATH=. python scripts/audit_miracl_ko_taxonomy_artifact.py

# Explicit diagnostic only: never experiment-ready and cannot feed a consumer.
PYTHONPATH=. python scripts/audit_miracl_ko_taxonomy_artifact.py --integrity-only
```

The required temporal order is: (1) build/recheck the artifact-free plan and generator code contract in a clean worktree; (2) obtain the separate actual-execution approval for that exact plan hash; (3) run the one 110K generator through request-index correlation, then derive 50K/20K projections; (4) build the tracked artifact manifest from real body hashes and approved plan hash; (5) run the default authorized audit. It must not use query/qrel/relevance/gold/evidence inputs.

## Non-results and next gate

- Taxonomy contract code and synthetic/mock harness: **implemented**.
- Actual 110K MIRACL taxonomy artifact: **not generated**.
- Taxonomy soft-boost A/B, Anserini rerun, embedding/LLM/Agent execution, focused Part 1/2: **not executed**.
- Passage-retrieval result: no taxonomy quality or insurance-domain claim is made here.

The next external input is an approved, leakage-safe taxonomy generator specification and execution approval. Only after a real artifact and audit are ready may a separately directed consumer adapter and the existing single-variable taxonomy A/B be considered.
