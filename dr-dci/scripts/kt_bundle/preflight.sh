#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
shift || true
exec python3 "$BUNDLE_ROOT/validators/preflight_validator.py" --config "$CONFIG" "$@"
