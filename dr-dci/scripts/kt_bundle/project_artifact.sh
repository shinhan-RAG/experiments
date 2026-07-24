#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
require_passed_preflight "$CONFIG"
exec python3 "$BUNDLE_ROOT/validators/project_artifact.py" --config "$CONFIG"
