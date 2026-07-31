# Collection Retrieval Evaluation Contract

Date: 2026-07-31

## Scope

This contract prepares the collection-based evaluation requested for Peter's
Part 1-4 experiments. It defines data and validation boundaries only. It does
not select a parser, generate QA or semantic tags, call a model, or execute an
experiment.

The controlled comparison has three retrieval modes:

1. `chunk`
2. `semantic_tag`
3. `combined`

Every mode uses the same QA records, embedding revision, reranker revision,
final top-k, and metric implementation. Evaluation is repeated with two
candidate scopes:

1. `per_collection`: search only the query's source collection.
2. `all_collections`: search the union of all collections.

The all-collections QA set is the union of per-collection QA records. A second
global QA generation pass is forbidden because it would change the queries and
confound the scope comparison.

## Artifact layout

Each collection supplies five immutable JSONL artifacts:

| Artifact | Required relation |
|---|---|
| `documents` | Namespaced document IDs and normalized content SHA-256 |
| `chunks` | Every chunk references an existing document |
| `elements` | Every element references an existing document |
| `semantic_tags` | Exactly one or more tags cover every element |
| `qa` | Every QA maps gold evidence to document, element, and chunk IDs |

All IDs start with `<collection_id>::`. The manifest records relative paths,
record counts, and SHA-256 values. Absolute paths and parent traversal are
rejected.

Raw provider data and generated artifact bodies remain outside Git. Git tracks
schemas, validators, prompts, manifests, aggregate statistics, and hashes.

## Element position and parser decision

Source coordinates are not required to build a valid semantic tag. Every chunk,
element, and evidence record must declare one of these states:

- `verified`: `char_start` and `char_end` identify a span in `source_text` or
  `element_text`.
- `unavailable`: basis and both offsets are explicitly unavailable.

An unavailable source coordinate is not silently treated as zero. QA evidence
must still contain a quote and a verified alignment method. Before evaluation,
each QA must map to at least one existing element and one existing chunk.

EDA decides the minimum implementation:

| Observed data | Implementation |
|---|---|
| Stable element/document IDs and text | Schema adapter |
| Element text/order but no coordinates | Deterministic IDs plus text alignment |
| Ambiguous element-to-source or element-to-chunk mapping | Alignment layer |
| Implicit boundaries, missing hierarchy, or unresolvable evidence | Parser |

No parser choice is made before this EDA.

## QA contract

QA generation starts from an anchor element. A final QA row contains:

- one source collection;
- one or more gold documents;
- one or more gold elements;
- one or more gold chunks;
- evidence quotes;
- verified evidence-to-element and evidence-to-chunk alignment;
- generator ID, immutable revision, prompt hash, and anchor element ID.

QA that cannot map to both an element and a chunk is excluded before model
execution and reported as a generation/alignment failure. It is not retained
with an empty qrel.

## Semantic-tag leakage boundary

Allowed inputs:

- element text;
- element type;
- document title;
- approved document metadata.

Forbidden inputs:

- QA query or answer;
- qrels or relevance judgments;
- gold document, element, or chunk lists;
- evidence quotes supplied by QA;
- prior model scores or rankings.

Every tag row records generator identity, immutable revision, prompt SHA-256,
actual input fields, and explicit `qa_access=false`, `qrel_access=false`, and
`gold_access=false`.

## Candidate budget

The initial frozen budget is:

| Mode | Candidate acquisition | Final output |
|---|---:|---:|
| Chunk | 100 chunk candidates | top 20 |
| Semantic tag | 100 tag candidates | top 20 |
| Combined | 50 chunk + 50 tag, RRF with `k=60` | top 20 |

The combined source pools must sum to the single-source pool. The reranker is
applied under the same policy in all three modes. Changing a model, revision,
prompt, metric implementation, candidate budget, or reranking policy creates a
new experiment ID.

## Metrics and reporting

Required metrics:

- `evidence_ndcg@10`
- `evidence_recall@20`
- `parent_hit@10`
- `evidence_coverage@20`
- `latency_p95_ms`

Results report each collection, a collection macro average, an all-query micro
average, and paired all-collections minus per-collection deltas. A later
analysis stage may add paired bootstrap confidence intervals without changing
retrieval outputs.

## EDA handoff

The data EDA session must produce, without committing raw text:

- `dataset_profile.json`
- `collection_inventory.json`
- `schema_observations.json`
- `element_position_capability.json`
- `duplicate_report.json`
- `id_collision_report.json`
- `text_length_distribution.json`
- `redacted_samples.jsonl`

`element_position_capability.json` records whether document links, order,
coordinates, hierarchy, and chunk/evidence alignment can be recovered. Its
conclusion is exactly one of `adapter`, `alignment_layer`, or `parser`.

## Validation

Before model execution:

```bash
python scripts/validate_collection_eval_contract.py \
  /path/to/collection_eval_manifest.yaml \
  --output /path/to/contract_validation.json
```

The validator checks artifact hashes and counts, namespaced ID uniqueness,
cross-artifact references, full tag coverage, QA gold coverage in both
representations, leakage declarations, candidate-budget equality, model
revision locks, and cross-collection duplicate policy.

The template manifest is
`config/collection_eval/collection_eval_manifest.example.yaml`.
