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
  run_metrics.json                  # operational telemetry (see below)
```

`document.metadata` carries split/domain/stem/raw IDs plus label and source
member bytes+SHA-256 provenance, clearly separated from `text`. The manifest
records archive hashes, per-cell counts and chunk-length distributions,
per-artifact path/records/bytes/SHA-256, policy IDs, the run identity, the
embedded verification summary, and the machine-readable acceptance decision.
The manifest intentionally contains no timestamps so a deterministic rerun
into a fresh target is byte-identical; `run_metrics.json` (stage timings,
peak RSS, byte counters) is operational telemetry and is explicitly excluded
from that guarantee.

## Atomic publication and operational idempotency

Deterministic output and operational idempotency are different properties:
determinism says identical inputs produce identical bytes; idempotency says
rerunning against an already published target is safe. The converter
provides both (`src/data/collection_publication.py`):

1. an exclusive advisory lock (`<target>.lock`) is acquired before any
   write; a concurrent same-target process gets a stable
   `LockConflictError` (CLI exit code 3) instead of interleaving writes;
2. all artifacts, the verification result, the acceptance decision, and the
   manifest are written into a sibling `<target>.staging` directory on the
   same filesystem;
3. the staging result is fully verified, then published with one atomic
   `rename`;
4. any failure removes the staging directory; an existing published target
   is never touched, and no partial target is ever exposed;
5. a rerun whose input identity (source/chunking/acceptance configs,
   selection, options, and behavior-affecting code/schema hashes) matches
   the published target re-verifies the target in place and returns
   `publication_status=reused` without rewriting a byte;
6. a target with a different identity, or an incomplete/corrupt target, is
   a loud failure — remove it explicitly if it is really obsolete.

## Acceptance gate

`config/collection_academic/accepted_full_run.v1.yaml` pins the accepted-EDA
per-cell and total denominators in a reviewed contract. Every manifest embeds
`acceptance.eligible` with machine-readable `reasons`; smoke, subset,
hash-skipped, or dirty-worktree runs are always `eligible=false`. A full
accepted conversion additionally requires all 12 archives SHA-256-verified,
source member hashing on, all collection verifications `ok`, exact
denominator matches, and a clean, known git commit.

The `collection_eval` stage must call
`require_accepted_conversion(<manifest path>, acceptance_contract=<reviewed
contract>, expected_manifest_sha256=<pinned>)` before any QA/tag/model work.
The gate never trusts persisted flags: it recomputes the run-identity
self-hash, requires manifest `selection`/`options` to equal the identity,
re-binds the caller's reviewed contract against the hash recorded at
publication, re-hashes every declared artifact, re-runs the streaming
semantic verifier, recomputes `evaluate_acceptance`, and requires the
persisted decision to equal the recomputation exactly. A flipped
`eligible` flag, deleted `reasons`, or tampered selection/totals/runtime/
options therefore fails; callers should also pin the manifest SHA-256 so a
consistently regenerated forgery is rejected too. The same audit runs on
the publication reuse path before any published target is returned as
`reused`. The code identity covers every behavior-affecting local module —
adapter, publication, `src/eval/collection_contract.py`, both CLIs, and the
alignment schema — and an import-coverage test fails if a `src` import of
the adapter is missing from that list.

## Complexity and resource profile

With archive bytes `A`, documents `D`, elements `E`, chunks `C`, and output
text bytes `T`:

- conversion runs in O(A + T) time plus the explicit per-cell stem sort
  O(S log S); peak memory is one document bundle plus per-cell member
  listings and chunk-length lists, never total corpus text;
- verification is a streaming four-way merge join over the deterministically
  ordered artifacts: O(T) time, peak memory bounded by one document's
  records plus an O(D) content-hash map; no disk-backed temporary index is
  needed, so there is no index cleanup path; all invariants (missing or
  duplicate IDs, reference errors, offset errors, dropped/reordered text,
  uncovered chunks/elements, content duplication, count/hash mismatches)
  are preserved;
- `run_metrics.json` records per-stage wall time, peak RSS, and logical
  bytes read/written without any private text;
- `tests/test_academic_collection_publication.py` includes a 1x/2x/4x
  synthetic scale test asserting exact record-count scaling and bounded
  verifier memory.

## Chunk bound and model compatibility (retrieval-unapproved)

`max_chars=2000` bounds multi-element packing only; elements are atomic, so
an oversized section becomes one oversized chunk. The accepted EDA reports
section maxima up to 32,767 chars, so oversized chunks exist and silent
tokenizer truncation would confound the later chunk/semantic-tag/combined
comparison. Therefore:

- every manifest reports per-cell and per-collection chunk-length
  distributions with exact oversized counts/rates and SHA-256-hashed source
  element ID samples;
- the chunk corpus is marked `retrieval_compatibility.status =
  retrieval_unapproved` until the embedding tokenizer/revision and input
  budget are frozen;
- `scripts/validate_chunk_model_compatibility.py` (contract template:
  `config/collection_academic/chunk_model_compat.template.yaml`) later
  proves every chunk fits the frozen input contract — it never calls a
  model. Recording an owner-approved deterministic long-element split
  policy (whose file bytes/SHA-256 are pinned and verified) does NOT make
  an oversized corpus usable: the status becomes `requires_rebuild` and the
  CLI exits non-zero (4) until the split corpus is rebuilt, alignment is
  regenerated, and the validation passes with zero violations;
- exact one-to-one element-chunk mapping does NOT prove retrieval
  compatibility, and moving from atomic oversized chunks to a multi-chunk
  element mapping requires owner approval before QA or retrieval execution.

## Retrieval approval attestation

Retrieval use additionally requires an explicit attestation
(`academic.retrieval-approval-attestation.v1`) built by
`build_retrieval_approval_attestation` on an acceptance-gated corpus. It
binds: the conversion manifest SHA-256 and run identity, every collection's
chunks artifact path/records/SHA-256, the model/tokenizer ID with an
immutable revision, the input template (and its SHA-256), the token budget,
the validator code-identity hashes, and the measured result
(`chunks_total`, `max_tokens_observed`, `token_violations`,
`approved`/`rejected`). Token counts come from a caller-loaded FROZEN
tokenizer callable — local tokenization only, never a model or API call.
`require_retrieval_approved` is the fail-loud gate: it re-verifies every
hash binding, re-runs `require_accepted_conversion` with the pinned
manifest SHA-256, re-tokenizes every rendered chunk with the frozen
tokenizer, and requires zero violations plus exact agreement with the
attested scan. Approval flags are never trusted on their own.

## Normalized-text search offsets

`locate_unique_normalized` returns offsets in the ORIGINAL document text:
the whitespace-collapsed match is mapped back through an index map, so the
returned slice re-normalizes to the searched text. Ambiguous (zero or
multiple) normalized matches fail loud. Normalized-coordinate offsets are
never labeled as original or constructed-document offsets.

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
