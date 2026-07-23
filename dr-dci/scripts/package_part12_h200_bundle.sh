#!/usr/bin/env bash
# Package the exact Part 1/2 harness for transfer to H200 without git pull.
# This does not include data, credentials, caches, or result files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$ROOT/part12_h200_bundle_${STAMP}.tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FILES=(
  run_experiment.py
  requirements.txt
  config/experiment.yaml
  config/judge_prompt.txt
  config/taxonomy_schemas
  src
  scripts/audit_part12.py
  scripts/run_scale_probe_gte.sh
  scripts/validate_scale_probe_result.py
)

{
  printf 'bundle_schema=dr-dci.part12-h200.v1\n'
  printf 'source_git_commit='
  git -C "$ROOT" rev-parse HEAD
  printf 'source_git_state='
  if test -n "$(git -C "$ROOT" status --short)"; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
  printf 'created_utc='
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf '\nsha256\n'
  (
    cd "$ROOT"
    find "${FILES[@]}" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' \
      -exec shasum -a 256 {} + | sort
  )
  printf '\nrequired_data_not_in_bundle\n'
  printf '%s\n' \
    data/raw/trec-covid/corpus.jsonl \
    data/raw/trec-covid/queries.jsonl \
    data/raw/trec-covid/qrels.jsonl \
    data/subsets/trec-covid/20k.json \
    data/subsets/trec-covid/50k.json \
    data/subsets/trec-covid/110k.json
  printf '\nrequired_before_taxonomy_run\n'
  printf '%s\n' \
    data/taxonomy/trec-covid_20k.json \
    data/taxonomy/trec-covid_50k.json \
    data/taxonomy/trec-covid_110k.json
} > "$TMP/BUNDLE_MANIFEST.txt"

mkdir -p "$(dirname "$OUT")"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$OUT" \
  -C "$ROOT" "${FILES[@]}" -C "$TMP" BUNDLE_MANIFEST.txt
printf 'created %s\n' "$OUT"
