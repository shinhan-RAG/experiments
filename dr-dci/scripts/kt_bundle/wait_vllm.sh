#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
require_passed_preflight "$CONFIG"
HOST="$(config_value "$CONFIG" KT_VLLM_HOST)"
PORT="$(config_value "$CONFIG" KT_VLLM_PORT)"
for attempt in $(seq 1 120); do
  if python3 - "$HOST" "$PORT" <<'PY'
import sys
from urllib.request import urlopen
try:
    with urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "vLLM is healthy"
    exit 0
  fi
  sleep 5
done
echo "vLLM did not become healthy within 10 minutes" >&2
exit 1
