"""Loader for document-agnostic hybrid-enrich configs."""

import os
from pathlib import Path

import yaml

_PATH_EXTS = (".jsonl", ".json", ".yaml", ".yml", ".npy", ".md")

DEFAULT_CONFIG = "configs/doc_1040aa492c.yaml"


def resolve_paths(cfg, base: Path):
    """Recursively resolve string values ending in known path extensions
    to absolute Paths, relative to `base`. Returns a new structure."""
    if isinstance(cfg, dict):
        return {k: resolve_paths(v, base) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [resolve_paths(v, base) for v in cfg]
    if isinstance(cfg, str) and cfg.lower().endswith(_PATH_EXTS):
        p = Path(cfg)
        return p if p.is_absolute() else (base / p).resolve()
    return cfg


def load_config(config_path=None) -> dict:
    """Load a doc config YAML and resolve all path-like values to
    absolute Paths, relative to the hybrid-enrich/ directory.

    Resolution order for config_path: explicit arg > DOC_CONFIG env var
    > default (configs/doc_1040aa492c.yaml).
    """
    base = Path(__file__).resolve().parent.parent  # hybrid-enrich/

    if config_path is None:
        config_path = os.environ.get("DOC_CONFIG", DEFAULT_CONFIG)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (base / config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return resolve_paths(cfg, base)


if __name__ == "__main__":
    import json

    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
