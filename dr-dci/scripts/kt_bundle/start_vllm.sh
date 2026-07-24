#!/usr/bin/env bash
set -euo pipefail
source "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/common.sh"
require_bundle_root
CONFIG="$(config_path "${1:-config.env}")"
require_passed_preflight "$CONFIG"

if docker ps -a --format '{{.Names}}' | grep -Fxq miracl-taxonomy-vllm; then
  echo "The fixed miracl-taxonomy-vllm container name already exists; remove it only after inspecting its logs." >&2
  exit 1
fi

SPEC="$(python3 "$BUNDLE_ROOT/validators/vllm_runtime_contract.py" launch-spec --config "$CONFIG")"
python3 - "$SPEC" "$CONFIG" "$BUNDLE_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

spec = json.loads(sys.argv[1])
config = Path(sys.argv[2])
root = Path(sys.argv[3])
values = {}
for line in config.read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        key, value = line.split("=", 1)
        values[key] = value
cache = values["KT_MODEL_CACHE_DIR"]
command = [
    "docker", "run", "-d", "--name", "miracl-taxonomy-vllm", "--gpus", "all", "--network", "host",
    "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
    "--mount", f"type=bind,src={cache},dst=/root/.cache/huggingface,readonly",
    spec["vllm_image"], spec["launch_command"], *spec["launch_arguments"],
]
container_id = subprocess.check_output(command, text=True).strip()
launch = {
    "generation_plan_sha256": spec["generation_plan_sha256"],
    "generator_source_commit": spec["generator_source_commit"],
    "container_digest": spec["container_digest"],
    "launch_arguments_sha256": spec["launch_arguments_sha256"],
    "container_id": container_id,
}
output = Path(values["KT_OUTPUT_DIR"])
if not output.is_absolute():
    output = root / output
target = output / "environment" / "vllm_launch.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(container_id)
PY
