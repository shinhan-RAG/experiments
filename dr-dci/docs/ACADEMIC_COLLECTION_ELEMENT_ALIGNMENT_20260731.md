# Academic Collection Element Alignment Stage

Date: 2026-07-31
Stage: `element_alignment` (accepted EDA decision: `alignment_layer`)

## Scope

This stage converts the academic-paper source archives (read-only) into the
frozen `config/collection_eval` document/chunk/element schemas plus a separate
element-alignment artifact and a conversion manifest. It implements the
accepted EDA handoff (`eda_handoff_manifest.json`, SHA-256 pinned in
`config/collection_academic/academic_paper.yaml`).

Out of scope, by contract: QA generation, semantic tags, embedding/reranker/
LLM calls, retrieval experiments, Peter Part 1-4 wiring, and any change to
`run_experiment.py`.

## Data facts this stage relies on (accepted EDA)

- Collections: `academic_ha` (HA), `academic_ss` (SS), `academic_st` (ST).
  Training/Validation are splits of each collection, never collections.
- 9,000 label JSON documents, 177,341 section elements, 35,883 image records.
- Every JSON stem pairs with exactly one source `.pdf` and one `.pptx`.
- Archive members are stored with one leading slash; names are normalized by
  stripping exactly that slash and rejecting traversal/nested layouts.
- Source character offsets are unavailable; hierarchy fields are absent;
  visual coordinates have no documented basis. Therefore every
  `source_location` is emitted as the explicit unavailable object.

## Content policy

- Element/document/chunk text comes only from
  `training_data_info.section_info[].original_text`, joined in array order.
- `summary_text`, `summary_cnt`, `original_cnt`, `procede`, image captions,
  image categories, and visual location/page are never used as content.
- `doc_title`/`doc_keyword` are not injected anywhere in this stage.
- `image_info` records (image-only and caption-backed) are excluded and
  counted per cell with an explicit reason until source provenance is proven.
- Repeated element text is retained with distinct IDs; it is a retrieval
  candidate-competition fact, not document leakage.

## Identifiers

- document: `{collection_id}::{json_stem}`
- element: `{collection_id}::{json_stem}::{paragraph_id}`
- chunk: `{collection_id}::{json_stem}::chunk::{index:04d}`
- image (reserved, not emitted): `{collection_id}::{json_stem}::image::{id}`

## Constructed document text and alignment

The document text is the deterministic concatenation of section
`original_text` values in `section_info` array order with the frozen `"\n\n"`
separator (`academic.constructed-document-text.v1`). Element ranges within
that text are exact by construction and recorded in the alignment artifact
under `basis=constructed_document_text`, `method=ordered_concatenation`.
These ranges are NOT PDF/PPTX source offsets and must never be relabeled as
such. Any search-based alignment must go through
`locate_unique_normalized`, which fails loud on ambiguous matches.

Chunking follows the frozen
`config/collection_academic/chunking.academic_element_packed.v1.yaml`
(`academic.element-packed-chunk.v1`): consecutive whole elements are packed
up to `max_chars=2000`; elements are atomic, so an oversized element becomes
one oversized chunk and every element maps to exactly one chunk.

## Outputs (per run, outside Git or under the ignored `data/` path)

```
<output-dir>/
  <collection_id>/documents.jsonl   # shinhan.collection-document.v1
  <collection_id>/chunks.jsonl      # shinhan.collection-chunk.v1
  <collection_id>/elements.jsonl    # shinhan.collection-element.v1
  <collection_id>/alignment.jsonl   # shinhan.collection-element-alignment.v1
  conversion_manifest.json          # shinhan.collection-conversion.v1
```

`document.metadata` carries split/domain/stem/raw IDs plus label and source
member bytes+SHA-256 provenance, clearly separated from `text`. The manifest
records archive hashes, per-cell counts, per-artifact path/records/bytes/
SHA-256, policy IDs, code fingerprints, exclusion counts, and the embedded
verification summary. The manifest intentionally contains no timestamps so a
deterministic rerun is byte-identical.

## CLI

```bash
python scripts/build_academic_collections.py \
  --data-root <read-only-root-containing-3.개방데이터> \
  --output-dir <path-outside-git> \
  [--collections academic_ha ...] [--splits Training ...] \
  [--limit-documents N] [--skip-archive-sha256] [--skip-source-member-hash]
```

Only aggregate counts and hashes are printed; raw text never reaches stdout,
logs, or Git.

## Interface for the next `collection_eval` stage

The full `shinhan.collection-eval-manifest.v1` bundle requires QA and
semantic-tag artifacts, so this stage does not emit one (and must not emit
placeholders). The next stage builds it as follows:

1. For each collection, reference this stage's `documents.jsonl`,
   `chunks.jsonl`, and `elements.jsonl` verbatim; `path`, `records`, and
   `sha256` values can be copied from
   `conversion_manifest.json#collections.<id>.artifacts`.
2. Set `source_revision` per collection to this stage's conversion identity,
   e.g. `conversion_manifest.json` SHA-256 (or the manifest's
   `code_fingerprint.git_commit` plus config hashes) so the corpus revision
   is immutable.
3. Generate QA anchored on `elements.jsonl` records and map evidence to
   chunks through `alignment.jsonl` (`chunk_memberships`), never by ad-hoc
   text search; ambiguous matches must fail loud.
4. Generate semantic tags with only the allowed input fields; the alignment
   artifact and document metadata are provenance, not tag inputs.
5. Validate the finished bundle with
   `scripts/validate_collection_eval_contract.py` before any model call.

## Verification

`src/data/collection_academic.verify_collection_outputs` re-reads every
artifact and fails loud on: duplicate or non-namespaced IDs, references to
absent documents/chunks/elements, offset inversions, numeric offsets on
unavailable locations, missing basis/method on verified ranges, dropped or
duplicated or reordered text, lost separators or tails, elements mapped to
zero or unexpectedly multiple chunks, mutated chunk text, and duplicate
document content. Synthetic RED-to-GREEN tests live in
`tests/test_academic_collection_alignment.py` and run without private data.
