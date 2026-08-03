"""Production loader for the frozen retrieval tokenizer.

The retrieval gates construct the token counter HERE, from the verified
snapshot root — an arbitrary caller-provided loader is never accepted in
production. Loading is strictly local (``local_files_only=True``,
``trust_remote_code=False``) from the exact verified snapshot directory,
the loaded class must equal the declared ``tokenizer_class``, and the
frozen options are applied on every call. This performs local tokenization
only; it never calls a model or any API.

``TEST_ONLY_LOADER_OVERRIDE`` exists solely so the synthetic test suite can
substitute a deterministic fake tokenizer without installing
``transformers``. It must stay ``None`` in committed code; this module is
part of the attestation code identity, so any modification (including a
non-None default) changes the recorded loader implementation hash and is
rejected by the retrieval gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class TokenizerRuntimeError(ValueError):
    """Raised when the frozen tokenizer cannot be loaded as declared."""


# Test-only dependency-injection seam; see the module docstring.
TEST_ONLY_LOADER_OVERRIDE: (
    Callable[[dict[str, Any], Path], Callable[[str], Any]] | None
) = None

NORMALIZATION_TOKENIZER_BUILTIN = "tokenizer_builtin"


def load_frozen_tokenizer(
    contract: dict[str, Any], snapshot_root: Path
) -> Callable[[str], Any]:
    """Build the token counter from the VERIFIED snapshot root only.

    ``snapshot_root`` must be the resolved path that the snapshot file
    inventory was verified against; the caller (the retrieval gate) passes
    that exact path.
    """
    if TEST_ONLY_LOADER_OVERRIDE is not None:
        return TEST_ONLY_LOADER_OVERRIDE(contract, snapshot_root)

    options = contract["options"]
    if options.get("normalization") != NORMALIZATION_TOKENIZER_BUILTIN:
        raise TokenizerRuntimeError(
            "frozen options.normalization must be "
            f"{NORMALIZATION_TOKENIZER_BUILTIN!r}; external pre-normalization "
            "is not part of the contract"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenizerRuntimeError(
            "the transformers package is required to load the frozen "
            "production tokenizer; it was not imported because it is not "
            "installed (no model or API call is involved)"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    actual_class = type(tokenizer).__name__
    if actual_class != contract["tokenizer_class"]:
        raise TokenizerRuntimeError(
            "loaded tokenizer class does not match the frozen contract: "
            f"declared={contract['tokenizer_class']!r} actual={actual_class!r}"
        )

    add_special_tokens = options["add_special_tokens"]

    def encode(text: str) -> Any:
        return tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
        )

    return encode
