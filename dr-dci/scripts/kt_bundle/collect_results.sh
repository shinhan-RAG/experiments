#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
MODE="${2:-complete}"
OUTPUT="$(output_dir "$CONFIG")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="$OUTPUT/collection"
mkdir -p "$RESULT_DIR"
if [[ "$MODE" == "--preflight-only" ]]; then
  tar -C "$OUTPUT" -czf "$RESULT_DIR/miracl-taxonomy-kt-preflight-$STAMP.tar.gz" environment/preflight.json
  sha256sum "$RESULT_DIR/miracl-taxonomy-kt-preflight-$STAMP.tar.gz" > "$RESULT_DIR/miracl-taxonomy-kt-preflight-$STAMP.tar.gz.sha256"
  echo "$RESULT_DIR/miracl-taxonomy-kt-preflight-$STAMP.tar.gz"
  exit 0
fi
if [[ "$MODE" != "complete" ]]; then
  echo "collect mode must be complete or --preflight-only" >&2
  exit 2
fi
require_passed_preflight "$CONFIG"
python3 - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for relative in ("audit/full_audit.json", "projection/projection_result.json"):
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise SystemExit(f"results collection is blocked until {relative} has status=passed")
for forbidden in ("config.env", ".env"):
    if (root / forbidden).exists():
        raise SystemExit("credential/config file must not be inside the output directory")
PY
tar -C "$OUTPUT" -czf "$RESULT_DIR/miracl-taxonomy-kt-results-$STAMP.tar.gz" \
  environment/preflight.json environment/vllm_launch.json environment/model_identity.json \
  generation/generation_run.json generation/operation_receipt.json generation/generation_receipt.json \
  generation/raw_response_hash_manifest.json generation/raw-responses generation/raw-vllm \
  artifacts/110000.json artifacts/50000.json artifacts/20000.json artifacts/taxonomy_artifact_manifest.json \
  audit/source_audit.json audit/full_audit.json projection/projection_result.json
sha256sum "$RESULT_DIR/miracl-taxonomy-kt-results-$STAMP.tar.gz" > "$RESULT_DIR/miracl-taxonomy-kt-results-$STAMP.tar.gz.sha256"
echo "$RESULT_DIR/miracl-taxonomy-kt-results-$STAMP.tar.gz"
