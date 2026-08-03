"""CI/supply-chain integrity contract (fifth-review C3, sixth-review owner action 1).

These tests run inside the required ``synthetic-tests`` check, so weakening
the workflow body while keeping the job name makes the required check fail:
if an action is unpinned from its commit SHA, the Python patch pin changes or
diverges from the lock target, the hash-pinned lock install is removed or
relaxed back to an unhashed requirements install, the lock loses hashes or
gains a non-pinned dependency, or the drift-check/full-suite steps disappear,
a test here fails. The authoritative protection against unreviewed workflow
edits is owner branch protection (C4: required review + dismiss-stale); this
contract is the in-repo detection layer.
"""

from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "collection-contract-tests.yml"
LOCK = ROOT / "requirements.lock"

# The single supported CI interpreter; must equal the workflow pin and the
# resolution target recorded in the lock header.
EXPECTED_PYTHON = "3.12.8"

SHA_PINNED_USES_RE = re.compile(r"^[^@\s]+@[0-9a-f]{40}(\s+#.*)?$")


class WorkflowIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WORKFLOW.is_file(), f"workflow missing: {WORKFLOW}")
        self.text = WORKFLOW.read_text(encoding="utf-8")
        self.doc = yaml.safe_load(self.text)

    def test_required_job_name_present(self):
        self.assertIn("synthetic-tests", self.doc["jobs"])

    def test_all_actions_are_sha_pinned(self):
        for uses in re.findall(r"uses:\s*(\S+)", self.text):
            self.assertRegex(
                uses,
                r"@[0-9a-f]{40}$",
                f"action is not pinned to a full commit SHA: {uses}",
            )
        # sanity: at least the two actions we expect are present and pinned
        self.assertIn("actions/checkout@", self.text)
        self.assertIn("actions/setup-python@", self.text)

    def test_python_is_pinned_to_the_expected_patch_version(self):
        versions = re.findall(r'python-version:\s*"([^"]+)"', self.text)
        self.assertTrue(versions, "no python-version pin found")
        for version in versions:
            self.assertEqual(
                version,
                EXPECTED_PYTHON,
                f"python-version must equal the lock target {EXPECTED_PYTHON}",
            )

    def test_install_uses_hash_pinned_lock_only(self):
        self.assertIn(
            "pip install --require-hashes -r requirements.lock",
            self.text,
            "install must come from the hash-pinned lock",
        )
        self.assertNotIn(
            "-r requirements.txt",
            self.text,
            "unhashed requirements.txt install is not allowed in CI",
        )
        self.assertNotIn(
            "pip install --upgrade pip",
            self.text,
            "unpinned pip self-upgrade is not allowed in CI",
        )
        self.assertTrue(LOCK.is_file(), "requirements.lock missing")

    def test_lock_drift_and_full_suite_steps_present(self):
        self.assertIn("check_supply_chain.py", self.text)
        self.assertIn("--constraints constraints.txt", self.text)
        self.assertIn("--lock requirements.lock", self.text)
        self.assertIn(f"--expect-python {EXPECTED_PYTHON}", self.text)
        self.assertIn("pytest tests/ -q", self.text)

    def test_constraints_are_exact_pins(self):
        from scripts.check_supply_chain import parse_constraints  # noqa: E402

        pins = parse_constraints(ROOT / "constraints.txt")
        self.assertIn("cryptography", pins)
        for name, version in pins.items():
            self.assertRegex(version, r"^\d", f"{name} pin is not a concrete version")

    def test_lock_is_fully_hash_pinned_and_covers_the_dependency_set(self):
        from scripts.check_supply_chain import (  # noqa: E402
            canonical,
            parse_constraints,
            parse_lock,
        )

        lock = parse_lock(LOCK)
        for name, entry in lock.items():
            self.assertTrue(entry["hashes"], f"{name} has no sha256 hash")
            for digest in entry["hashes"]:
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        pins = parse_constraints(ROOT / "constraints.txt")
        for name, version in pins.items():
            self.assertIn(name, lock, f"constraint {name} missing from lock")
            self.assertEqual(
                lock[name]["version"], version, f"lock/constraint drift for {name}"
            )
        for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            direct = canonical(re.match(r"^([A-Za-z0-9._-]+)", line).group(1))
            self.assertIn(direct, lock, f"direct dependency {direct} missing from lock")

    def test_lock_header_records_the_resolution_target(self):
        header = "\n".join(
            LOCK.read_text(encoding="utf-8").splitlines()[:8]
        )
        self.assertIn("autogenerated by pip-compile", header)
        major_minor = ".".join(EXPECTED_PYTHON.split(".")[:2])
        self.assertIn(
            f"with Python {major_minor}",
            header,
            "lock was not resolved with the expected interpreter line",
        )
        self.assertIn("--generate-hashes", header)

    def test_relaxed_lock_is_rejected(self):
        from scripts.check_supply_chain import parse_lock  # noqa: E402

        cases = {
            "unhashed pin": "requests==2.34.2\n",
            "floating spec": "requests>=2.0 \\\n    --hash=sha256:" + "0" * 64 + "\n",
            "vcs source": "git+https://example.invalid/repo.git\n",
            "editable": "-e ./local\n",
            "nested requirements": "-r other.txt\n",
            "non-hash option": "--index-url https://example.invalid/simple\n",
        }
        with TemporaryDirectory() as tmp:
            for index, (label, content) in enumerate(cases.items()):
                with self.subTest(case=label):
                    bad = Path(tmp) / f"bad_lock_{index}.txt"
                    bad.write_text(content, encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        parse_lock(bad)

    def test_owner_actions_documented(self):
        owner_doc = ROOT / "config" / "collection_academic" / "OWNER_ACTIONS.md"
        self.assertTrue(owner_doc.is_file(), "OWNER_ACTIONS.md missing")
        text = owner_doc.read_text(encoding="utf-8")
        for marker in ("require-hashes", "branch protection", "dismiss_stale_reviews"):
            self.assertIn(marker, text, f"owner action not documented: {marker}")


if __name__ == "__main__":
    unittest.main()
