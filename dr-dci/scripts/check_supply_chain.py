#!/usr/bin/env python3
"""Supply-chain integrity check: constraints, hash-pinned lock, interpreter.

Fifth-review C3 / sixth-review owner action 1. Verifies, failing loud on any
deviation:

- ``--constraints``: every entry is an exact ``==`` pin and the installed
  version of each pinned package matches (drift detection).
- ``--lock`` (optional): the hash-pinned transitive lock is structurally
  sound — every requirement is an exact ``==`` pin carrying at least one
  ``--hash=sha256:`` value; no editable, VCS, URL, local-path, nested
  requirements, or option lines other than hashes are allowed; every
  constraints pin appears in the lock at the same version; and the installed
  version of every locked package matches the lock (post-install drift).
- ``--expect-python`` (optional): the running interpreter version equals the
  expected patch version (CI passes 3.12.8 to match the workflow pin and the
  lock resolution target).

Prints only package/version aggregates. Exit codes: 0 ok, 2 violation.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import platform
import re
import sys


PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!-]*)$")
LOCK_PIN_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!-]*)(\s*\\)?$"
)
LOCK_HASH_RE = re.compile(r"^--hash=sha256:[0-9a-f]{64}(\s*\\)?$")
FORBIDDEN_LOCK_RE = re.compile(
    r"^(-e\b|--editable\b|-r\b|--requirement\b|-c\b|--constraint\b)"
    r"|git\+|hg\+|svn\+|bzr\+|^https?://|^file:|^\.{1,2}/"
)


def canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def parse_constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_RE.match(line)
        if not match:
            raise SystemExit(f"constraint is not an exact == pin: {line!r}")
        pins[canonical(match.group(1))] = match.group(2)
    if not pins:
        raise SystemExit("no constraints found")
    return pins


def parse_lock(path: Path) -> dict[str, dict[str, object]]:
    """Parse the hash-pinned lock; any structural relaxation fails loud."""
    entries: dict[str, dict[str, object]] = {}
    current: str | None = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if FORBIDDEN_LOCK_RE.search(line):
            raise SystemExit(
                f"lock line {lineno} is not a plain hash-pinned requirement: {line!r}"
            )
        hash_match = LOCK_HASH_RE.match(line)
        if hash_match:
            if current is None:
                raise SystemExit(f"lock line {lineno}: hash without a requirement")
            entries[current]["hashes"].append(line.split("sha256:", 1)[1].rstrip(" \\"))
            continue
        if line.startswith("--"):
            raise SystemExit(
                f"lock line {lineno}: option lines other than --hash are not "
                f"allowed: {line!r}"
            )
        pin_match = LOCK_PIN_RE.match(line)
        if not pin_match:
            raise SystemExit(
                f"lock line {lineno} is not an exact hash-pinned == pin: {line!r}"
            )
        current = canonical(pin_match.group(1))
        if current in entries:
            raise SystemExit(f"lock line {lineno}: duplicate package {current!r}")
        entries[current] = {"version": pin_match.group(2), "hashes": []}
    if not entries:
        raise SystemExit("no lock entries found")
    unhashed = sorted(name for name, entry in entries.items() if not entry["hashes"])
    if unhashed:
        raise SystemExit(f"lock entries without any --hash=sha256: {unhashed}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--expect-python", default=None)
    args = parser.parse_args()

    violations: list[str] = []

    if args.expect_python is not None:
        actual = platform.python_version()
        if actual != args.expect_python:
            violations.append(
                f"python: running {actual} != expected {args.expect_python}"
            )

    pins = parse_constraints(args.constraints)
    lock = parse_lock(args.lock) if args.lock is not None else None

    if lock is not None:
        for name, expected in sorted(pins.items()):
            locked = lock.get(name)
            if locked is None:
                violations.append(f"{name}: pinned in constraints but not in lock")
            elif locked["version"] != expected:
                violations.append(
                    f"{name}: lock {locked['version']} != constraint {expected}"
                )

    installed_targets: dict[str, str] = dict(pins)
    if lock is not None:
        installed_targets.update(
            {name: str(entry["version"]) for name, entry in lock.items()}
        )
    for name, expected in sorted(installed_targets.items()):
        try:
            actual = pkg_version(name)
        except PackageNotFoundError:
            violations.append(f"{name}: MISSING (expected {expected})")
            continue
        if actual != expected:
            violations.append(f"{name}: installed {actual} != pinned {expected}")

    for name in sorted(installed_targets):
        try:
            print(f"{name}=={pkg_version(name)}")
        except PackageNotFoundError:
            pass
    if violations:
        print("supply-chain violation detected:", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(2)
    scope = "constraints+lock" if lock is not None else "constraints"
    print(f"supply-chain: no drift ({scope})")


if __name__ == "__main__":
    main()
