#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
require_passed_preflight "$CONFIG"
require_passed_model_identity "$CONFIG"
exec python3 "$BUNDLE_ROOT/generator/taxonomy_vllm_generator.py" --bundle-root "$BUNDLE_ROOT" --config "$CONFIG"
