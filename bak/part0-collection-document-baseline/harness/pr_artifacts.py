"""Rebuild PR #14 Index/frontmatter artifacts by runtime injection.

PR files are imported byte-unmodified from the pinned checkout; only their
module-level ROOT/OUT path constants are rebound at runtime. The SHA-256 of
every PR file used is recorded.
"""
import importlib.util
import json
import sys
from pathlib import Path

from .config import Config, sha256_file

PR_FILES = (
    "tos-skeleton/frontmatter/doc_frontmatter.py",
    "tos-skeleton/frontmatter/build_collection_index.py",
    "tos-skeleton/frontmatter/collection_fm_v2.py",
    "tos-skeleton/frontmatter/eval_content_qa.py",
    "tos-skeleton/frontmatter/gate2.py",
)


def _load(repo_root: Path, rel: str, name: str):
    path = repo_root / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pr_modules(cfg: Config) -> dict:
    frontdir = cfg.repo_root / "tos-skeleton" / "frontmatter"
    # gate2 / eval_content_qa are imported by name inside PR modules
    sys.path.insert(0, str(frontdir))
    import gate2  # noqa: PR module, byte-unmodified
    import eval_content_qa  # noqa: PR module, byte-unmodified
    import collection_fm_v2  # noqa
    mods = {
        "gate2": gate2,
        "eval_content_qa": eval_content_qa,
        "collection_fm_v2": collection_fm_v2,
        "doc_frontmatter": _load(cfg.repo_root,
                                 "tos-skeleton/frontmatter/doc_frontmatter.py",
                                 "pr_doc_frontmatter"),
        "build_collection_index": _load(cfg.repo_root,
                                        "tos-skeleton/frontmatter/build_collection_index.py",
                                        "pr_build_collection_index"),
    }
    return mods


def pr_file_hashes(cfg: Config) -> dict:
    return {rel: sha256_file(cfg.repo_root / rel) for rel in PR_FILES}


def build_artifacts(cfg: Config, mods: dict) -> dict:
    """Run PR builders with ROOT/OUT rebound. Returns artifact paths + hashes."""
    out = cfg.work_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    dfm = mods["doc_frontmatter"]
    dfm.ROOT = cfg.source_root
    dfm.OUT = out
    dfm.main()

    bci = mods["build_collection_index"]
    bci.ROOT = cfg.source_root
    bci.OUT = out / "collection_index.json"   # PR constant is the file path
    bci.main()

    fm_path = out / "doc_frontmatter.jsonl"
    ci_path = out / "collection_index.json"
    docs = [json.loads(l) for l in open(fm_path, encoding="utf-8")]
    ci = json.loads(ci_path.read_text(encoding="utf-8"))
    n_ci = sum(len(p["documents"]) for p in ci["products"])
    if len(docs) != cfg.expected_docs or n_ci != cfg.expected_docs:
        raise SystemExit(
            f"artifact coverage mismatch: fm={len(docs)} index={n_ci} "
            f"expected={cfg.expected_docs}")
    return {
        "doc_frontmatter_jsonl": str(fm_path),
        "collection_index_json": str(ci_path),
        "doc_frontmatter_sha256": sha256_file(fm_path),
        "collection_index_sha256": sha256_file(ci_path),
        "n_docs": len(docs),
    }


def build_ident_text(ci: dict) -> dict:
    """Exact PR formula (unified_benchmark.py:24-32) for the Index entry text."""
    ident = {}
    for p in ci["products"]:
        for dd in p["documents"]:
            key = p["product_name"] + "/" + dd["file_name"]
            ident[key] = " ".join([
                p["product_name"], dd.get("title", ""), dd["doc_type"],
                ((dd.get("effective_date") or "")[:4] + "년")
                if dd.get("effective_date") else "",
                "최신판 정본 현행" if dd.get("is_representative") else "구판"])
    return ident
