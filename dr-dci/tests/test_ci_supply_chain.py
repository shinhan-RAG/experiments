"""CI/supply-chain integrity contract (fifth-review C3).

These tests run inside the required ``synthetic-tests`` check, so weakening
the workflow body while keeping the job name makes the required check fail:
if an action is unpinned from its commit SHA, Python loses its patch pin, the
constraints install is removed, or the full-suite step disappears, a test
here fails. The authoritative protection against unreviewed workflow edits is
owner branch protection (C4: required review + dismiss-stale); this contract
is the in-repo detection layer.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "collection-contract-tests.yml"

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

    def test_python_is_patch_pinned(self):
        versions = re.findall(r'python-version:\s*"([^"]+)"', self.text)
        self.assertTrue(versions, "no python-version pin found")
        for version in versions:
            self.assertRegex(
                version, r"^\d+\.\d+\.\d+$", f"python-version is not patch-pinned: {version}"
            )

    def test_install_uses_constraints(self):
        self.assertIn("-c constraints.txt", self.text)
        self.assertTrue((ROOT / "constraints.txt").is_file())

    def test_lock_drift_and_full_suite_steps_present(self):
        self.assertIn("check_supply_chain.py", self.text)
        self.assertIn("pytest tests/ -q", self.text)

    def test_constraints_are_exact_pins(self):
        from scripts.check_supply_chain import parse_constraints  # noqa: E402

        pins = parse_constraints(ROOT / "constraints.txt")
        self.assertIn("cryptography", pins)
        for name, version in pins.items():
            self.assertRegex(version, r"^\d", f"{name} pin is not a concrete version")

    def test_owner_actions_documented(self):
        owner_doc = ROOT / "config" / "collection_academic" / "OWNER_ACTIONS.md"
        self.assertTrue(owner_doc.is_file(), "OWNER_ACTIONS.md missing")
        text = owner_doc.read_text(encoding="utf-8")
        for marker in ("require-hashes", "branch protection", "dismiss_stale_reviews"):
            self.assertIn(marker, text, f"owner action not documented: {marker}")


if __name__ == "__main__":
    unittest.main()
