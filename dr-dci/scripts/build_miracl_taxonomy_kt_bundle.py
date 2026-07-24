#!/usr/bin/env python3
"""Build a deterministic, Git-independent KT taxonomy execution archive.

The bundle copies only executable source, validators, and non-secret lock
records. MIRACL bodies, model weights, control approvals, and config.env stay
outside the archive by design.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = Path(__file__).resolve().parent / "kt_bundle"
BUNDLE_NAME = "miracl-taxonomy-kt-bundle"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GENERATOR_FILES = (
    "taxonomy_vllm_generator.py",
    "src/miracl_ko/__init__.py",
    "src/miracl_ko/preparation.py",
    "src/miracl_ko/taxonomy_artifact.py",
    "src/miracl_ko/kt_bundle.py",
)
SHELL_SCRIPTS = (
    "common.sh", "preflight.sh", "verify_model_identity.sh", "start_vllm.sh", "wait_vllm.sh",
    "run_generation.sh", "run_audit.sh", "project_artifact.sh", "collect_results.sh",
)
VALIDATOR_SCRIPTS = (
    "preflight_validator.py", "receipt_artifact_validator.py", "project_artifact.py", "vllm_runtime_contract.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments], check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def require_clean_tracked_source() -> str:
    for arguments in (("diff", "--quiet", "HEAD"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments], check=False, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("KT bundle must be built from a clean tracked source checkout")
    return run_git("rev-parse", "HEAD")


def copy_file(source: Path, destination: Path, *, executable: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if executable:
        destination.chmod(destination.stat().st_mode | 0o111)


def file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in {"MANIFEST.json", "SHA256SUMS"}):
        records.append({
            "relative_path": str(path.relative_to(root)),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def write_deterministic_tar(root: Path, target: Path) -> None:
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(item for item in root.rglob("*") if item.is_file()):
                    info = archive.gettarinfo(str(path), arcname=f"{BUNDLE_NAME}/{path.relative_to(root)}")
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)


def build(output_dir: Path) -> tuple[Path, Path]:
    source_commit = require_clean_tracked_source()
    from src.miracl_ko.preparation import build_preparation_contract
    from src.miracl_ko.taxonomy_artifact import build_generator_code_contract, generator_contract_sha256

    with tempfile.TemporaryDirectory(prefix="miracl-taxonomy-kt-bundle-") as temporary:
        stage = Path(temporary) / BUNDLE_NAME
        stage.mkdir()
        copy_file(TEMPLATE_ROOT / "README.md.template", stage / "README.md")
        copy_file(TEMPLATE_ROOT / "config.example.env", stage / "config.example.env")
        for name in SHELL_SCRIPTS:
            copy_file(TEMPLATE_ROOT / name, stage / "scripts" / name, executable=True)
        for name in VALIDATOR_SCRIPTS:
            copy_file(TEMPLATE_ROOT / name, stage / "validators" / name, executable=True)
        copy_file(TEMPLATE_ROOT / "taxonomy_vllm_generator.py", stage / "generator" / "taxonomy_vllm_generator.py", executable=True)
        for relative_path in GENERATOR_FILES[1:]:
            copy_file(REPO_ROOT / relative_path, stage / "generator" / relative_path)
        (stage / "control").mkdir()
        (stage / "control" / "README.md").write_text(
            "Place only approved generation_plan.json, approval_record.json, and generator_code_contract.json here.\n",
            encoding="utf-8",
        )
        contract = build_generator_code_contract(stage / "generator", GENERATOR_FILES)
        source_dir = stage / "source"
        source_dir.mkdir()
        (source_dir / "generator_code_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        source_manifest = {
            "schema_version": "dr-dci.miracl-ko-kt-clean-source.v1",
            "source_git_commit": source_commit,
            "generator_contract_sha256": generator_contract_sha256(contract),
            "generator_code_sha256": contract["generator_code_sha256"],
            "files": contract["generator_files"],
        }
        (source_dir / "clean_generator_source_manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        copy_file(REPO_ROOT / "config" / "miracl_ko_revision_lock.json", source_dir / "miracl_ko_revision_lock.json")
        preparation_contract = build_preparation_contract(REPO_ROOT)
        (source_dir / "preparation_contract.json").write_text(
            json.dumps(preparation_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        manifest = {
            "schema_version": "dr-dci.miracl-ko-kt-bundle.v1",
            "bundle_name": BUNDLE_NAME,
            "source_git_commit": source_commit,
            "generator_contract_sha256": source_manifest["generator_contract_sha256"],
            "generator_code_sha256": source_manifest["generator_code_sha256"],
            "contains": "executable source, validators, and non-secret lock records only",
            "excludes": ["MIRACL corpus", "model weights", "credentials", "API keys", "approval records", "config.env"],
            "files": file_records(stage),
        }
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        checksums = []
        for path in sorted(item for item in stage.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            checksums.append(f"{sha256_file(path)}  {path.relative_to(stage)}")
        (stage / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / f"{BUNDLE_NAME}.tar.gz"
        write_deterministic_tar(stage, archive)
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
        return archive, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive, checksum = build(args.output_dir.resolve())
    print(json.dumps({"archive": str(archive), "sha256_file": str(checksum), "sha256": sha256_file(archive)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
