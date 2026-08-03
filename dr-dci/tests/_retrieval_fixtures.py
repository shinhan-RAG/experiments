"""Synthetic retrieval-approval fixtures: reference tokenizer snapshot, an
Ed25519 fixture authority + trust root, and a signed approval record.

These build REAL objects exercised through the production path (there is no
loader-injection seam): a deterministic reference-whitespace tokenizer
defined by snapshot files, and a genuine Ed25519 signature verified by the
gate. No production key or approver is fabricated; the keypair here is an
explicit, ephemeral, test-only authority.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from src.data import retrieval_approval, tokenizer_runtime
from src.data.collection_academic import canonical_json_sha256, tokenizer_contract_sha256
from src.eval.collection_contract import sha256_file

IMMUTABLE_REV = "0123456789abcdef0123456789abcdef01234567"
BOS, EOS, UNK = 101, 102, 100


def _reference_expected_ids(probe: str, *, add_special_tokens: bool) -> list[int]:
    ids = [UNK for _ in probe.split()]
    if add_special_tokens:
        ids = [BOS, *ids, EOS]
    return ids


def build_reference_snapshot(
    root: Path, *, add_special_tokens: bool = True, probe: str = "aaa bbb ccc"
) -> dict[str, Any]:
    """Write a deterministic reference-whitespace tokenizer snapshot + contract."""
    snapshot = Path(root)
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "vocab.json").write_text(
        json.dumps({}, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (snapshot / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "kind": tokenizer_runtime.REFERENCE_WHITESPACE_KIND,
                "bos_id": BOS,
                "eos_id": EOS,
                "unk_id": UNK,
                "vocab": "vocab.json",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files = [
        {
            "path": name,
            "bytes": (snapshot / name).stat().st_size,
            "sha256": sha256_file(snapshot / name),
        }
        for name in ("tokenizer_manifest.json", "vocab.json")
    ]
    return {
        "tokenizer_id": "fixture/reference-whitespace",
        "revision": IMMUTABLE_REV,
        "tokenizer_class": "ReferenceWhitespaceTokenizer",
        "trust_remote_code": False,
        "local_snapshot_path": str(snapshot),
        "options": {
            "add_special_tokens": add_special_tokens,
            "truncation": False,
            "padding": False,
            "max_length": None,
            "normalization": tokenizer_runtime.NORMALIZATION_TOKENIZER_BUILTIN,
        },
        "files": files,
        "self_test": {
            "probe_text": probe,
            "expected_input_ids": _reference_expected_ids(
                probe, add_special_tokens=add_special_tokens
            ),
        },
    }


def make_fixture_authority(root: Path, *, signer_id: str = "fixture-owner") -> dict[str, Any]:
    """Create an ephemeral Ed25519 fixture authority and pinned trust root."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    Path(root).mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    private_b64 = base64.b64encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode("ascii")
    public_b64 = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    trust_root = {
        "schema_version": retrieval_approval.TRUST_ROOT_SCHEMA_VERSION,
        "authorized_signers": {signer_id: {"ed25519_public_key_b64": public_b64}},
    }
    trust_root_path = Path(root) / "trust_root.json"
    trust_root_path.write_bytes(retrieval_approval.canonical_json(trust_root))
    return {
        "signer_id": signer_id,
        "private_b64": private_b64,
        "public_b64": public_b64,
        "trust_root_path": trust_root_path,
        "trust_root_sha256": sha256_file(trust_root_path),
    }


def make_signed_approval(
    *,
    authority: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    acceptance_contract: dict[str, Any],
    model_id: str,
    revision: str,
    tokenizer_contract: dict[str, Any],
    token_budget: int,
    input_template_sha256: str,
    policy_id: str,
    policy_version: str,
    approval_request_id: str,
    signer_id: str | None = None,
    private_b64: str | None = None,
) -> dict[str, Any]:
    """Build the fully-bound approval subject and sign it with the fixture key."""
    subject = retrieval_approval.build_approval_subject(
        conversion_manifest_sha256=manifest_sha256,
        run_identity_sha256=manifest["run_identity"]["identity_sha256"],
        acceptance_contract_sha256=canonical_json_sha256(acceptance_contract),
        model_id=model_id,
        model_revision=revision,
        tokenizer_contract_sha256=tokenizer_contract_sha256(tokenizer_contract),
        token_budget=token_budget,
        input_template_sha256=input_template_sha256,
        policy_id=policy_id,
        policy_version=policy_version,
        approval_request_id=approval_request_id,
    )
    signature = retrieval_approval.sign_approval_subject(
        subject, private_b64 or authority["private_b64"]
    )
    return {
        "subject": subject,
        "signer_id": signer_id or authority["signer_id"],
        "signature_b64": signature,
    }
