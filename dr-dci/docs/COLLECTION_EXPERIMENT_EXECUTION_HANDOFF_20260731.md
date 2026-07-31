# Collection Experiment Execution Handoff

Date: 2026-07-31

## 1. Integration baseline

All collection experiment work starts from the latest remote
`feature_ralph`. Do not start from `dev` and do not push directly to either
integration branch.

The contract baseline was merged by PR #10:

- base branch: `feature_ralph`
- merge commit: `ae6b2f13bfa1b8203eeeb7a6775d1cddc4175b0e`
- contract:
  `docs/COLLECTION_RETRIEVAL_EVALUATION_CONTRACT_20260731.md`
- validator:
  `scripts/validate_collection_eval_contract.py`
- schemas:
  `config/collection_eval/`

Before creating any task branch:

```bash
git fetch origin --prune
git rev-parse origin/feature_ralph
git status --short
```

If `origin/feature_ralph` has advanced, use the advanced commit. Do not reset,
rebase, or delete another worker's branch or worktree.

## 2. Local EDA and branch order

EDA is performed in a user-owned local directory without creating or modifying
a Git branch. Only Claude implementation sessions modify the repository. Use
one branch and one worktree per Claude implementation stage, and merge each PR
before creating a dependent branch.

| Order | Location or branch | Owner scope | Depends on |
|---:|---|---|---|
| 1 | local EDA directory, no branch | read-only data profiling and reports | contract baseline |
| 2 | `feat/ralph-element-alignment` | adapter/alignment/parser and fixtures | accepted local EDA |
| 3 | `feat/ralph-collection-eval` | three retrieval modes and local/global evaluation | element alignment |
| 4 | `feat/ralph-part1-4-integration` | Peter Part 1-4 wiring only | collection evaluation |

Recommended worktree command:

```bash
git worktree add \
  ../worktrees/experiments-<task> \
  -b <branch> \
  origin/feature_ralph
```

Every PR targets `feature_ralph`. A `feature_ralph` to `dev` PR is created only
after all experiment outputs and integration gates are approved.

After a task PR is merged:

1. fetch and fast-forward the main `feature_ralph` worktree;
2. confirm the task worktree is clean;
3. remove the task worktree;
4. delete only that merged local task branch;
5. do not delete colleague or unmerged branches.

The existing `feat/shinhan-uw-element-retrieval` branch is reference material.
Do not merge it wholesale. EDA may identify generic commits to port through a
new reviewed commit.

## 3. Session A: data EDA prompt

Use the following prompt after the dataset location is available.

> Work in a new user-owned local EDA directory. Read the
> `shinhan-RAG/experiments` contract files as reference, but do not create a
> branch, commit, PR, or worktree and do not modify the repository.
>
> Perform read-only EDA for every supplied collection. Do not modify source
> data, call an LLM, generate QA, generate semantic tags, choose retrieval
> models, run Peter Part 1-4, or implement a parser.
>
> Read and follow:
>
> - `docs/COLLECTION_RETRIEVAL_EVALUATION_CONTRACT_20260731.md`
> - `config/collection_eval/collection_eval_manifest.schema.yaml`
> - `config/collection_eval/document.schema.json`
> - `config/collection_eval/chunk.schema.json`
> - `config/collection_eval/element.schema.json`
> - `config/collection_eval/qa.schema.json`
> - `config/collection_eval/semantic_tag.schema.json`
>
> First inventory files, byte sizes, hashes, encodings, archive structure,
> record counts, and observed schemas. Treat all text as sensitive. Commit no
> raw text or source archive.
>
> For each collection inspect:
>
> - stable collection, document, element, and chunk identifiers;
> - element-to-document links;
> - element text, type, order, hierarchy, and parent/child links;
> - source coordinates and the coordinate basis;
> - whether element text can be aligned to source document text;
> - whether element evidence can be aligned to one or more chunks;
> - missing, empty, malformed, duplicate, or conflicting records;
> - ID collisions within and across collections;
> - exact and normalized content duplicates across collections;
> - text and element length distributions;
> - metadata fields that are safe inputs for semantic-tag generation;
> - fields that contain query, answer, qrel, gold, evidence, labels, or prior
>   rankings and must be excluded from tag generation;
> - whether a common QA generation contract is feasible for every collection.
>
> Produce only aggregate statistics, hashes, schemas, and redacted samples:
>
> - `dataset_profile.json`
> - `collection_inventory.json`
> - `schema_observations.json`
> - `element_position_capability.json`
> - `duplicate_report.json`
> - `id_collision_report.json`
> - `text_length_distribution.json`
> - `redacted_samples.jsonl`
>
> `element_position_capability.json` must select exactly one implementation:
>
> - `adapter`: stable document/element IDs and text are sufficient;
> - `alignment_layer`: boundaries exist but deterministic text/offset mapping
>   is required;
> - `parser`: element boundaries or hierarchy cannot be recovered reliably.
>
> Record evidence for the decision per collection. If collections need
> different implementations, report the maximum required implementation and
> the per-collection decision. Do not infer unavailable coordinates.
>
> Also produce:
>
> - `eda_handoff_manifest.json`
> - `eda_handoff_manifest.json.sha256`
>
> The handoff manifest records every report's relative path, bytes, SHA-256,
> analyzer code path and SHA-256, source inventory SHA-256, creation time,
> redaction status, and final adapter/alignment/parser decision. The sidecar
> verifies the manifest itself.
>
> Keep deterministic EDA code and synthetic tests in the same local EDA
> directory. External data paths must be CLI arguments, not hardcoded local
> paths. Fail loudly on unreadable files, unsupported schemas, duplicate IDs,
> and truncated input. Do not copy raw or report artifacts into Git.

## 4. Local EDA acceptance gate

The local EDA handoff can be passed to Claude only when all statements below
are true:

- every source file has byte size and SHA-256;
- raw source files remain unchanged and untracked;
- every collection has a stable namespace proposal;
- element representation and document linkage are measured, not assumed;
- coordinate availability distinguishes source, element-local, and absent
  positions;
- cross-collection duplicates and ID collisions are reported;
- tag-safe and forbidden fields are enumerated;
- the parser decision is one of the three allowed values with evidence;
- synthetic EDA tests pass without access to private data;
- reports contain no unredacted sensitive text.
- `eda_handoff_manifest.json.sha256` verifies;
- every report and analyzer hash in the handoff manifest verifies.

If these conditions are not met, the next implementation branch must not start.

## 5. Session B: element adapter/alignment/parser prompt

Use this prompt only after the local EDA handoff is accepted and all hashes
verify.

> Create `feat/ralph-element-alignment` from the latest
> `origin/feature_ralph` in a separate worktree. Verify the local
> `eda_handoff_manifest.json` sidecar and every recorded report hash. Read the
> accepted EDA reports and implement only the selected `adapter`,
> `alignment_layer`, or `parser`.
>
> Convert each collection to the document, chunk, and element schemas under
> `config/collection_eval/`. Generate namespaced IDs and deterministic output
> order. Preserve source text and record source hashes.
>
> Do not generate QA or semantic tags. Do not call an embedding, reranker, or
> LLM. Do not change Peter Part 1-4 or `run_experiment.py`.
>
> Positions may remain explicitly `unavailable` when the accepted EDA proves
> they cannot be recovered. Never synthesize offsets. When text alignment is
> possible, record the basis, method, ambiguity count, and verified offsets.
>
> Add RED to GREEN tests for duplicate IDs, missing document references,
> ambiguous alignment, dropped source text, nondeterministic output, invalid
> offsets, and unsupported element types. Open a PR to `feature_ralph`.

## 6. Session C: collection evaluation implementation prompt

Use this prompt only after the alignment PR is accepted and merged.

> Create `feat/ralph-collection-eval` from the latest
> `origin/feature_ralph`. Implement the frozen collection evaluation contract.
>
> Generate QA per collection from anchor elements. Generate semantic tags
> independently from QA. A semantic-tag generator must not receive query,
> answer, qrel, gold, evidence, or prior ranking fields.
>
> Every accepted QA must map evidence to at least one document, element, and
> chunk. Reject unmapped QA before model execution and report rejection
> reasons.
>
> Run the same QA through `chunk`, `semantic_tag`, and `combined` modes. Use
> per-collection and all-collections candidate scopes. The all-collections QA
> set is the union of per-collection QA; do not generate separate global QA.
>
> Freeze model IDs and immutable revisions, candidate budgets, RRF parameters,
> reranking policy, final top-k, metric code revision, and random seeds. Validate
> the manifest with `scripts/validate_collection_eval_contract.py` before any
> model call.
>
> Report required metrics per collection, macro, micro, and paired
> all-collections minus per-collection deltas. Add deterministic unit tests and
> a one-collection synthetic smoke test. Open a PR to `feature_ralph`.

## 7. Session D: Peter Part 1-4 integration prompt

Use this prompt only after the collection evaluation PR is accepted and merged.

> Create `feat/ralph-part1-4-integration` from the latest
> `origin/feature_ralph`. Wire the validated collection evaluation into Peter
> Part 1-4 without redefining the existing parts.
>
> Add `retrieval_mode` and `collection_scope` as controlled dimensions.
> Preserve each part's original independent variable and model configuration.
> Do not select a preferred mode using test results.
>
> Store raw per-query outputs before aggregation. Bind every result to data,
> manifest, model, prompt, and code SHA-256 values. Fail loudly if any expected
> collection, scope, mode, QA, or metric is missing.
>
> Run a synthetic smoke first. Model-backed execution requires a separately
> approved runtime package. Open a PR to `feature_ralph`, not `dev`.

## 8. File ownership boundaries

To reduce conflicts:

| Branch | Primary paths |
|---|---|
| Element alignment | `src/data/collection_*`, conversion scripts and tests |
| Collection eval | `src/eval/collection_*`, QA/tag builders, eval tests |
| Part 1-4 integration | `run_experiment.py`, `config/experiment*.yaml`, integration tests |

If a task needs to modify a path owned by a later task, stop and document the
required interface instead of editing it opportunistically.

## 9. Final integration gate

Do not create a `feature_ralph` to `dev` PR until:

- every task PR above is merged into `feature_ralph`;
- all task branches and worktrees are clean;
- raw data and credentials are absent from Git;
- contract validation reports `status=ok`;
- all expected collection/scope/mode cells exist;
- per-query result counts match the frozen QA manifests;
- model, prompt, data, and code revisions are immutable;
- the complete test suite passes;
- experiment limitations and failed/excluded QA are reported without
  reinterpretation.
