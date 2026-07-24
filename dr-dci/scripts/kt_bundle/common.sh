#!/usr/bin/env bash
# Shared, non-interactive helpers for the Git-independent KT bundle.
set -euo pipefail

BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

require_bundle_root() {
  if [[ ! -f "$BUNDLE_ROOT/MANIFEST.json" || ! -f "$BUNDLE_ROOT/SHA256SUMS" ]]; then
    echo "Run this script from an intact miracl-taxonomy-kt-bundle archive." >&2
    exit 2
  fi
}

config_path() {
  local candidate="${1:-config.env}"
  if [[ "$candidate" = /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "$BUNDLE_ROOT/$candidate"
  fi
}

config_value() {
  local config="$1"
  local key="$2"
  awk -F= -v wanted="$key" '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    $1 == wanted { print substr($0, length(wanted) + 2); found = 1; exit }
    END { if (!found) exit 1 }
  ' "$config"
}

output_dir() {
  local config="$1"
  local value
  value="$(config_value "$config" KT_OUTPUT_DIR)"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$BUNDLE_ROOT/$value"
  fi
}

require_passed_preflight() {
  local config="$1"
  local report
  report="$(output_dir "$config")/environment/preflight.json"
  python3 - "$report" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except Exception as error:
    raise SystemExit(f"KT preflight result is unavailable: {error}")
if value.get("status") != "passed":
    raise SystemExit("KT preflight must pass with --require-lock before this stage")
PY
}

require_passed_model_identity() {
  local config="$1"
  local report
  report="$(output_dir "$config")/environment/model_identity.json"
  python3 - "$report" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except Exception as error:
    raise SystemExit(f"KT model identity result is unavailable: {error}")
if value.get("status") != "passed":
    raise SystemExit("KT model identity verification must pass before generation")
PY
}
