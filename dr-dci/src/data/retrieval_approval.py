"""Retrieval-approval authority binding (fifth-review finding C2).

An approval must be more than a file that exists: it must be structurally
bound to the exact artifact it approves and signed by an authority anchored
OUTSIDE the PR-writable tree.

Binding: the "approval subject" is a canonical object carrying every identity
the approval commits to — conversion manifest SHA, run identity, acceptance
contract SHA, model id + immutable revision, tokenizer identity/contract/
inventory hash, token budget, input-template SHA, policy/version, and the
approval-request id. Its canonical SHA-256 is the ``approval_subject_sha256``.

Authority: an owner-managed trust root (a JSON file OUTSIDE this repository,
whose own SHA-256 the caller pins out of band) lists the authorized Ed25519
public keys. The approval carries a detached Ed25519 signature over the
canonical subject; the gate verifies it with the ``cryptography`` library
against an allowlisted key. Signatures are verified, never hand-rolled.

Fail-closed: if the trust root is absent, its pin mismatches, ``cryptography``
is unavailable, the signer is not allowlisted, the signature is invalid, or
any bound identity differs from the attestation, approval fails and retrieval
stays unapproved. No production key or approver is fabricated here; tests
supply an explicit ephemeral fixture authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


APPROVAL_SUBJECT_SCHEMA_VERSION = "academic.retrieval-approval-subject.v1"
TRUST_ROOT_SCHEMA_VERSION = "academic.retrieval-approval-trust-root.v1"
_SHA256_HEX_LEN = 64

# The subject fields the approval commits to, in a fixed set. Every one must
# equal the corresponding value recomputed from the attestation at the gate.
APPROVAL_SUBJECT_FIELDS = (
    "conversion_manifest_sha256",
    "run_identity_sha256",
    "acceptance_contract_sha256",
    "model_id",
    "model_revision",
    "tokenizer_contract_sha256",
    "token_budget",
    "input_template_sha256",
    "policy_id",
    "policy_version",
    "approval_request_id",
)


class ApprovalError(ValueError):
    """Raised when a retrieval approval cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def approval_subject_sha256(subject: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(subject)).hexdigest()


def _sha256_hex(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and len(value) == _SHA256_HEX_LEN
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be lowercase SHA-256",
    )
    return value


def build_approval_subject(
    *,
    conversion_manifest_sha256: str,
    run_identity_sha256: str,
    acceptance_contract_sha256: str,
    model_id: str,
    model_revision: str,
    tokenizer_contract_sha256: str,
    token_budget: int,
    input_template_sha256: str,
    policy_id: str,
    policy_version: str,
    approval_request_id: str,
) -> dict[str, Any]:
    """Assemble the canonical, fully-bound approval subject."""
    subject = {
        "schema_version": APPROVAL_SUBJECT_SCHEMA_VERSION,
        "conversion_manifest_sha256": _sha256_hex(
            conversion_manifest_sha256, "conversion_manifest_sha256"
        ),
        "run_identity_sha256": _sha256_hex(run_identity_sha256, "run_identity_sha256"),
        "acceptance_contract_sha256": _sha256_hex(
            acceptance_contract_sha256, "acceptance_contract_sha256"
        ),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_contract_sha256": _sha256_hex(
            tokenizer_contract_sha256, "tokenizer_contract_sha256"
        ),
        "token_budget": token_budget,
        "input_template_sha256": _sha256_hex(
            input_template_sha256, "input_template_sha256"
        ),
        "policy_id": policy_id,
        "policy_version": policy_version,
        "approval_request_id": approval_request_id,
    }
    for field in APPROVAL_SUBJECT_FIELDS:
        value = subject[field]
        _require(
            value not in (None, ""),
            f"approval subject field {field} must be set",
        )
    _require(
        isinstance(token_budget, int) and token_budget > 0,
        "approval subject token_budget must be a positive integer",
    )
    return subject


def load_trust_root(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load the owner-managed trust root and verify the caller-pinned SHA-256.

    The trust root lives OUTSIDE the repository; the caller pins its hash out
    of band. A missing file, hash mismatch, or malformed content fails closed.
    """
    path = Path(path)
    _require(path.is_file(), f"approval trust root missing: {path}")
    _require(not path.is_symlink(), f"approval trust root is a symlink: {path}")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    _require(
        actual == _sha256_hex(expected_sha256, "expected_trust_root_sha256"),
        "approval trust root does not match the pinned SHA-256: "
        f"expected={expected_sha256} actual={actual}",
    )
    try:
        trust_root = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApprovalError(f"unreadable approval trust root: {error}") from error
    _require(isinstance(trust_root, dict), "approval trust root must be an object")
    _require(
        trust_root.get("schema_version") == TRUST_ROOT_SCHEMA_VERSION,
        f"approval trust root schema_version must be {TRUST_ROOT_SCHEMA_VERSION}",
    )
    signers = trust_root.get("authorized_signers")
    _require(
        isinstance(signers, dict) and bool(signers),
        "approval trust root must list authorized_signers",
    )
    for signer_id, entry in signers.items():
        _require(isinstance(entry, dict), f"signer {signer_id} entry must be a mapping")
        _require(
            isinstance(entry.get("ed25519_public_key_b64"), str)
            and bool(entry["ed25519_public_key_b64"]),
            f"signer {signer_id} must carry an ed25519_public_key_b64",
        )
    return trust_root


def sign_approval_subject(subject: dict[str, Any], private_key_b64: str) -> str:
    """Ed25519-sign the canonical subject; returns a base64 detached signature.

    Signing helper for owner tooling and fixture authorities. Uses the vetted
    ``cryptography`` library; no cryptographic primitive is hand-rolled.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(private_key_b64)
    )
    signature = private_key.sign(canonical_json(subject))
    return base64.b64encode(signature).decode("ascii")


def verify_approval(
    approval: dict[str, Any],
    *,
    expected_subject: dict[str, Any],
    trust_root: dict[str, Any],
) -> dict[str, Any]:
    """Fail-closed verification of a signed, fully-bound retrieval approval.

    The approval's embedded subject must equal ``expected_subject`` field for
    field (which the gate builds from the attestation), the signer must be
    allowlisted in the trust root, and the Ed25519 signature over the
    canonical subject must verify. Any deviation raises ``ApprovalError``.
    """
    _require(isinstance(approval, dict), "approval must be a mapping")
    subject = approval.get("subject")
    _require(isinstance(subject, dict), "approval.subject must be a mapping")
    _require(
        subject.get("schema_version") == APPROVAL_SUBJECT_SCHEMA_VERSION,
        f"approval.subject schema_version must be {APPROVAL_SUBJECT_SCHEMA_VERSION}",
    )
    _require(
        set(subject) == {"schema_version", *APPROVAL_SUBJECT_FIELDS},
        "approval.subject fields do not match the required binding set",
    )
    # Structural binding: every committed field must equal the attestation-derived value.
    for field in APPROVAL_SUBJECT_FIELDS:
        _require(
            subject.get(field) == expected_subject.get(field),
            f"approval subject field {field} does not match the attestation: "
            f"approval={subject.get(field)!r} expected={expected_subject.get(field)!r}",
        )
    _require(
        approval_subject_sha256(subject)
        == approval_subject_sha256(expected_subject),
        "approval subject digest does not match the attestation",
    )

    signer_id = approval.get("signer_id")
    _require(
        isinstance(signer_id, str) and bool(signer_id),
        "approval.signer_id must be a non-empty string",
    )
    signers = trust_root["authorized_signers"]
    _require(
        signer_id in signers,
        f"approval signer is not authorized by the trust root: {signer_id!r}",
    )
    signature_b64 = approval.get("signature_b64")
    _require(
        isinstance(signature_b64, str) and bool(signature_b64),
        "approval.signature_b64 must be a non-empty string",
    )

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as error:  # fail-closed: no hand-rolled crypto
        raise ApprovalError(
            "the cryptography library is required to verify approval "
            "signatures; approval fails closed and retrieval stays unapproved"
        ) from error

    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(signers[signer_id]["ed25519_public_key_b64"])
        )
        public_key.verify(base64.b64decode(signature_b64), canonical_json(subject))
    except InvalidSignature as error:
        raise ApprovalError(
            f"approval signature is invalid for signer {signer_id!r}"
        ) from error
    except (ValueError, TypeError) as error:
        raise ApprovalError(f"malformed approval signature/key: {error}") from error
    return approval
