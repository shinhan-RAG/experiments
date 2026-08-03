"""Atomic, lock-protected publication of conversion output directories.

Deterministic output (same inputs produce byte-identical artifacts) and
operational idempotency (a rerun against an already published target is safe)
are different properties. This module provides the second one:

1. an exclusive same-target advisory lock is acquired before any write;
2. all artifacts are written into a sibling staging directory on the same
   filesystem;
3. the fully verified staging directory is published with one atomic rename;
4. a failure removes the staging directory and never touches an existing
   published target;
5. a rerun against a published target with the exact same input identity is
   verified in place and reused without rewriting;
6. a target with a different identity, or an incomplete/corrupt target, is a
   loud failure — it is never merged, overwritten, or silently repaired.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


class PublicationError(ValueError):
    """Raised when a target directory cannot be safely published or reused."""


class LockConflictError(PublicationError):
    """Stable error for a concurrent process holding the same target lock."""


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lock_path_for(target: Path) -> Path:
    return target.parent / f"{target.name}.lock"


def staging_path_for(target: Path) -> Path:
    return target.parent / f"{target.name}.staging"


class TargetLock:
    """Exclusive advisory flock on a sibling lock file; non-blocking acquire."""

    def __init__(self, target: Path):
        self.target = Path(target)
        self.path = lock_path_for(self.target)
        self._fd: int | None = None

    def acquire(self) -> "TargetLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise LockConflictError(
                    f"target lock is held by another process: {self.path}"
                ) from error
            raise
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "TargetLock":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


def prepare_staging(target: Path) -> Path:
    """Create a fresh staging sibling; leftover staging is never published state."""
    staging = staging_path_for(target)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def discard_staging(target: Path) -> None:
    shutil.rmtree(staging_path_for(target), ignore_errors=True)


def target_state(target: Path, manifest_name: str) -> str:
    """Classify the target: absent, empty, complete, or partial."""
    if not target.exists():
        return "absent"
    if not target.is_dir():
        raise PublicationError(f"target exists and is not a directory: {target}")
    entries = list(target.iterdir())
    if not entries:
        return "empty"
    if (target / manifest_name).is_file():
        return "complete"
    return "partial"


def publish_staging(staging: Path, target: Path) -> None:
    """Atomically publish the verified staging directory as the target."""
    if target.exists():
        raise PublicationError(
            f"refusing to publish over an existing target: {target}"
        )
    os.rename(staging, target)
