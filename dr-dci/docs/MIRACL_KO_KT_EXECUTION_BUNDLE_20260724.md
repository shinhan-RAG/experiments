# MIRACL-ko KT execution bundle

## Scope

This is a Git-independent, user-operated KT execution package for exactly one
approved MIRACL-ko 110K taxonomy generation. The user uploads the generated
tar archive and runs its documented local commands. No Codex process connects
to KT, and the package never runs `git pull`.

The package is an execution transport, not a new taxonomy method. It preserves
the frozen taxonomy generation plan/approval/receipt contracts and does not
authorize taxonomy A/B, Agent execution, focused Part 1/2, or AIHub legal
evaluation.

## Contents and exclusions

`scripts/build_miracl_taxonomy_kt_bundle.py` creates
`miracl-taxonomy-kt-bundle.tar.gz` with executable generator source, validators,
revision/preparation locks, a clean-source manifest, checksums, a non-secret
configuration template, and the operator README.

The archive intentionally excludes MIRACL corpus bodies, model weights,
credentials, API keys, `config.env`, control approval records, and result
files. The operator supplies those as separate local files. Bundle output is
ignored under `kt-bundles/`.

## Execution boundary

Before a model request, locked preflight verifies the exact plan, actual
approval record, copied generator source hash, immutable model/tokenizer
revisions, container digest, MIRACL revision lock, subset manifest, and every
20K/50K/110K corpus byte hash. It also records a redacted environment report.

Generation requires a passed locked preflight and local served-model identity
report. It writes an operation receipt that blocks a duplicate completed run;
only missing batches in a matching partial run may resume within the approved
per-batch retry cap. Partial/failed receipts cannot produce an artifact.

The source artifact must pass standalone audit before 20K/50K filter
projection. Full audit must pass before result collection. Collection preserves
response bytes and their hash manifest for receipt replay, but excludes corpus,
model cache, and local configuration files.

## Current status

Only synthetic bundle tests run locally. No KT connection, model/API call,
taxonomy artifact creation, retrieval A/B, Agent, focused Part 1/2, or legal
primary evaluation has occurred.
