#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
STAGE="${2:-source}"
if [[ "$STAGE" != "source" && "$STAGE" != "full" ]]; then
  echo "audit stage must be source or full" >&2
  exit 2
fi
exec python3 "$BUNDLE_ROOT/validators/receipt_artifact_validator.py" --config "$CONFIG" --stage "$STAGE"
