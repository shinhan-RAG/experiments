#!/usr/bin/env python3
"""Collect a redacted KT environment report and optionally validate execution locks.

This program does not start a model, contact a remote service, read a
credential, or inspect corpus text.  It writes JSON even for a failed
preflight so the user can collect an actionable, non-sensitive report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = BUNDLE_ROOT / "generator"
if str(GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATOR_ROOT))

from src.miracl_ko.kt_bundle import (  # noqa: E402
    atomic_write_json,
    image_matches_entrypoint,
    load_bundle_controls,
    parse_env_file,
    path_from_config,
    require_config,
    validate_execution_lock,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_output(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""
    return completed.returncode == 0, completed.stdout.strip()


def gpu_inventory() -> list[dict[str, str]]:
    okay, output = command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits",
    ])
    if not okay or not output:
        return []
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            result.append({"model": fields[0], "vram_mib": fields[1], "driver": fields[2]})
    return result


def cuda_version() -> str | None:
    """Read the driver-reported CUDA compatibility version without host IDs."""
    okay, output = command_output(["nvidia-smi"])
    if not okay:
        return None
    for line in output.splitlines():
        marker = "CUDA Version:"
        if marker in line:
            return line.split(marker, 1)[1].strip().split()[0]
    return None


def docker_inventory(image: str) -> dict[str, Any]:
    okay, output = command_output(["docker", "image", "inspect", image, "--format", "{{json .}}"])
    if not okay:
        return {"available": False, "image_present": False, "image_id": None, "image_digests": [], "entrypoint": None}
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return {"available": True, "image_present": False, "image_id": None, "image_digests": [], "entrypoint": None}
    digests = value.get("RepoDigests") if isinstance(value, dict) else []
    config = value.get("Config") if isinstance(value, dict) else None
    entrypoint = config.get("Entrypoint") if isinstance(config, dict) else None
    return {
        "available": True,
        "image_present": True,
        "image_id": value.get("Id") if isinstance(value, dict) else None,
        "image_digests": sorted(
            digest.rsplit("@", 1)[-1]
            for digest in (digests or [])
            if isinstance(digest, str) and "@" in digest
        ),
        "entrypoint": entrypoint if isinstance(entrypoint, list) and all(isinstance(item, str) for item in entrypoint) else None,
    }


def image_matches_digest(image: dict[str, Any], digest: str) -> bool:
    """Require the local inspected image, not config text, to carry the lock."""
    return image.get("image_id") == digest or any(
        isinstance(value, str) and (value == digest or value.endswith(digest))
        for value in image.get("image_digests", [])
    )


def vllm_version(image: str) -> str | None:
    okay, output = command_output([
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", image,
        "-c", "import vllm; print(vllm.__version__)",
    ])
    return output if okay and output else None


def cache_has_snapshot(cache_dir: Path, repository: str, revision: str) -> bool:
    safe_repository = repository.replace("/", "--")
    return (cache_dir / f"models--{safe_repository}" / "snapshots" / revision).is_dir()


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.settimeout(0.25)
        return stream.connect_ex((host, port)) == 0


def safe_lock(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        key: lock[key]
        for key in (
            "generation_plan_sha256", "approval_record_sha256", "generator_contract_sha256",
            "generator_code_sha256", "generator_source_commit", "model_revision",
            "tokenizer_revision", "container_digest", "expected_container_entrypoint", "input_provenance",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-lock", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any] = {"schema_version": "dr-dci.miracl-ko-kt-preflight.v1", "generated_at": utc_now()}
    output_dir: Path | None = None
    try:
        config = require_config(parse_env_file(args.config))
        output_dir = path_from_config(BUNDLE_ROOT, config["KT_OUTPUT_DIR"], label="KT_OUTPUT_DIR")
        image = docker_inventory(config["KT_VLLM_IMAGE"])
        python_version = platform.python_version()
        cache_dir = path_from_config(BUNDLE_ROOT, config["KT_MODEL_CACHE_DIR"], label="KT_MODEL_CACHE_DIR")
        lock: dict[str, Any] | None = None
        if args.require_lock:
            controls = load_bundle_controls(path_from_config(BUNDLE_ROOT, config["KT_CONTROL_DIR"], label="KT_CONTROL_DIR"))
            lock = validate_execution_lock(
                bundle_root=BUNDLE_ROOT, config=config, plan=controls[0], approval=controls[1], supplied_contract=controls[2],
            )
        runtime = lock if lock is not None else None
        model_cache = (
            cache_has_snapshot(cache_dir, runtime["model_repository"], runtime["model_revision"])
            if runtime is not None and "model_repository" in runtime else None
        )
        tokenizer_cache = (
            cache_has_snapshot(cache_dir, runtime["tokenizer_repository"], runtime["tokenizer_revision"])
            if runtime is not None and "tokenizer_repository" in runtime else None
        )
        # Repositories are intentionally not emitted; revisions are immutable public IDs.
        gpu = gpu_inventory()
        report.update({
            "status": "passed" if args.require_lock else "environment_only",
            "gpu": {"count": len(gpu), "devices": gpu, "cuda_version": cuda_version()},
            "os": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine()},
            "python_version": python_version,
            "docker": image,
            "vllm_version": vllm_version(config["KT_VLLM_IMAGE"]) if image["image_present"] else None,
            "disk_free_bytes": shutil.disk_usage(output_dir.parent if output_dir else Path.cwd()).free,
            "port_in_use": port_in_use(config["KT_VLLM_HOST"], int(config["KT_VLLM_PORT"])),
            "model_cache_present": model_cache,
            "tokenizer_cache_present": tokenizer_cache,
            "offline_load_possible": bool(image["image_present"] and model_cache and tokenizer_cache) if runtime else None,
            "execution_lock": safe_lock(lock) if lock is not None else None,
        })
        if args.require_lock:
            if not image["image_present"] or not model_cache or not tokenizer_cache:
                raise RuntimeError("required local image or pinned model/tokenizer cache is unavailable")
            if not image_matches_digest(image, lock["container_digest"]):
                raise RuntimeError("local Docker image does not match the approved container digest")
            expected_entrypoint = lock.get("expected_container_entrypoint")
            if expected_entrypoint is not None and not image_matches_entrypoint(image, expected_entrypoint):
                raise RuntimeError("local Docker image entrypoint does not match the approved generation plan")
            if not gpu:
                raise RuntimeError("no GPU is available for the approved local taxonomy generation")
            if report["vllm_version"] is None:
                raise RuntimeError("approved local image does not expose a vLLM version")
            if report["port_in_use"]:
                raise RuntimeError("approved vLLM port is already in use before server launch")
    except Exception as error:
        report["status"] = "failed"
        report["error_class"] = type(error).__name__
        report["error"] = "preflight validation failed"
        report["error_fingerprint_sha256"] = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
    target = (output_dir or (BUNDLE_ROOT / "output")) / "environment" / "preflight.json"
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] in {"passed", "environment_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
