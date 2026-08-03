"""Production loader for the frozen retrieval tokenizer.

The retrieval gates construct the token counter HERE, only from the verified
snapshot root. There is NO mutable-global or caller-provided injection seam:
the previous ``TEST_ONLY_LOADER_OVERRIDE`` is removed because a runtime
global is invisible to file-hash code identity and could approve fabricated
token counts (fifth-review finding C1).

Two tokenizer kinds are supported, both resolved deterministically from the
snapshot's own ``tokenizer_manifest.json`` (which is part of the frozen file
inventory):

- ``huggingface``: loaded via ``AutoTokenizer.from_pretrained`` with
  ``local_files_only=True`` and ``trust_remote_code=False``; the loaded
  class must equal the declared ``tokenizer_class``.
- ``reference_whitespace_v1``: a small dependency-free deterministic
  tokenizer defined entirely by snapshot files, so the whole gate can run
  (and be tested) without network access or the ``transformers`` package.

Regardless of kind, the loader runs the contract's bound self-test
(fixed probe text -> expected ``input_ids``) immediately after construction
and fails loud on mismatch, proving the real tokenizer executed. This
performs local tokenization only; it never calls a model or any API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class TokenizerRuntimeError(ValueError):
    """Raised when the frozen tokenizer cannot be loaded/verified as declared."""


NORMALIZATION_TOKENIZER_BUILTIN = "tokenizer_builtin"
REFERENCE_WHITESPACE_KIND = "reference_whitespace_v1"
HUGGINGFACE_KIND = "huggingface"
TOKENIZER_MANIFEST_NAME = "tokenizer_manifest.json"


class _BatchEncoding(dict):
    """Minimal BatchEncoding-like mapping: len() is the field count."""


def _read_manifest(snapshot_root: Path) -> dict[str, Any]:
    manifest_path = snapshot_root / TOKENIZER_MANIFEST_NAME
    if not manifest_path.is_file():
        raise TokenizerRuntimeError(
            f"tokenizer snapshot lacks {TOKENIZER_MANIFEST_NAME}: {snapshot_root}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerRuntimeError(
            f"unreadable tokenizer manifest: {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise TokenizerRuntimeError("tokenizer manifest must be an object")
    return manifest


def _build_reference_whitespace(
    snapshot_root: Path, manifest: dict[str, Any], *, add_special_tokens: bool
) -> Callable[[str], _BatchEncoding]:
    for key in ("bos_id", "eos_id", "unk_id", "vocab"):
        if key not in manifest:
            raise TokenizerRuntimeError(
                f"reference tokenizer manifest missing {key!r}"
            )
    vocab_path = snapshot_root / str(manifest["vocab"])
    if not vocab_path.is_file():
        raise TokenizerRuntimeError(f"reference vocab missing: {vocab_path}")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise TokenizerRuntimeError("reference vocab must be an object")
    bos, eos, unk = manifest["bos_id"], manifest["eos_id"], manifest["unk_id"]

    def encode(text: str) -> _BatchEncoding:
        ids = [int(vocab.get(token, unk)) for token in text.split()]
        if add_special_tokens:
            ids = [int(bos), *ids, int(eos)]
        return _BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})

    return encode


def _build_huggingface(
    snapshot_root: Path, tokenizer_class: str, *, add_special_tokens: bool
) -> Callable[[str], Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenizerRuntimeError(
            "the transformers package is required to load a huggingface "
            "tokenizer; it is not installed (no model or API call is involved)"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_root), local_files_only=True, trust_remote_code=False
    )
    actual_class = type(tokenizer).__name__
    if actual_class != tokenizer_class:
        raise TokenizerRuntimeError(
            "loaded tokenizer class does not match the frozen contract: "
            f"declared={tokenizer_class!r} actual={actual_class!r}"
        )

    def encode(text: str) -> Any:
        return tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
        )

    return encode


def _run_self_test(
    encode: Callable[[str], Any], self_test: dict[str, Any]
) -> None:
    """Prove the real tokenizer executed by matching the bound probe vector."""
    from src.data.collection_academic import _extract_input_ids  # local import

    probe = self_test["probe_text"]
    expected = self_test["expected_input_ids"]
    actual = _extract_input_ids(encode(probe))
    if actual != list(expected):
        raise TokenizerRuntimeError(
            "tokenizer self-test failed: the tokenizer built from the verified "
            "snapshot did not reproduce the bound probe input_ids "
            f"(expected {len(expected)} ids, got {len(actual)})"
        )


def load_frozen_tokenizer(
    contract: dict[str, Any], snapshot_root: Path
) -> Callable[[str], Any]:
    """Build and self-test the token counter from the VERIFIED snapshot root.

    ``snapshot_root`` must be the resolved path the snapshot inventory was
    verified against; the gate passes that exact path. The returned callable
    has already passed the contract's bound self-test.
    """
    snapshot_root = Path(snapshot_root)
    options = contract["options"]
    if options.get("normalization") != NORMALIZATION_TOKENIZER_BUILTIN:
        raise TokenizerRuntimeError(
            "frozen options.normalization must be "
            f"{NORMALIZATION_TOKENIZER_BUILTIN!r}; external pre-normalization "
            "is not part of the contract"
        )
    add_special_tokens = options["add_special_tokens"]
    manifest = _read_manifest(snapshot_root)
    kind = manifest.get("kind")
    if kind == REFERENCE_WHITESPACE_KIND:
        encode = _build_reference_whitespace(
            snapshot_root, manifest, add_special_tokens=add_special_tokens
        )
    elif kind == HUGGINGFACE_KIND:
        encode = _build_huggingface(
            snapshot_root,
            contract["tokenizer_class"],
            add_special_tokens=add_special_tokens,
        )
    else:
        raise TokenizerRuntimeError(
            f"unsupported tokenizer manifest kind: {kind!r}"
        )
    _run_self_test(encode, contract["self_test"])
    return encode
