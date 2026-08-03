#!/usr/bin/env python3
"""Lock-drift check: installed direct dependencies must match constraints.txt.

Fifth-review C3. Compares the currently importable versions of the pinned
direct dependencies against the exact versions in ``constraints.txt`` and
fails loud on any drift. This does not replace a hash-pinned lock (an owner
action), but it detects silent version drift of the direct dependencies in
CI. Prints only package/version aggregates.

Exit codes: 0 no drift, 2 drift or missing package.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import re
import sys


PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!-]*)$")


def parse_constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_RE.match(line)
        if not match:
            raise SystemExit(f"constraint is not an exact == pin: {line!r}")
        pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    if not pins:
        raise SystemExit("no constraints found")
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", type=Path, required=True)
    args = parser.parse_args()
    pins = parse_constraints(args.constraints)
    drift = []
    for name, expected in sorted(pins.items()):
        try:
            actual = pkg_version(name)
        except PackageNotFoundError:
            drift.append(f"{name}: MISSING (expected {expected})")
            continue
        if actual != expected:
            drift.append(f"{name}: installed {actual} != pinned {expected}")
    for name in sorted(pins):
        try:
            print(f"{name}=={pkg_version(name)}")
        except PackageNotFoundError:
            pass
    if drift:
        print("supply-chain drift detected:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(2)
    print("supply-chain: no drift")


if __name__ == "__main__":
    main()
