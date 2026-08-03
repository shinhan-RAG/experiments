# Owner actions for PR #13 (fifth-review C3/C4)

Some trust-boundary items cannot be fully closed by an in-PR code change and
must be applied by the repository owner. They are recorded here rather than
overclaimed as done in the PR.

## C3 — hash-pinned dependency lock (`pip --require-hashes`)

Status: **owner action pending.** In-PR we pinned GitHub Actions to full
commit SHAs, pinned Python to a patch version, added `dr-dci/constraints.txt`
(exact `==` pins for the direct dependencies), a lock-drift check
(`scripts/check_supply_chain.py`) run in CI, and a workflow-integrity test
(`tests/test_ci_supply_chain.py`). A full hash-pinned lock of the transitive
closure was NOT fabricated here because valid artifact hashes are
platform-specific and cannot be produced from a developer machine offline
without overclaiming.

Owner steps (run on the CI platform, linux / CPython 3.12.8):

```bash
pip install pip-tools
pip-compile --generate-hashes --output-file dr-dci/requirements.lock \
  dr-dci/requirements.txt -c dr-dci/constraints.txt
```

Then switch the workflow install step to:

```yaml
run: python -m pip install --require-hashes -r requirements.lock
```

and update `tests/test_ci_supply_chain.py` to require `--require-hashes`.
Commit `requirements.lock` on a reviewed PR.

## C4 — branch protection for `feature_ralph`

Status: **owner action pending.** This implementation stage must NOT change
GitHub settings. The required final protection on `feature_ralph` is:

- `required_status_checks`: `synthetic-tests` (strict)
- `required_pull_request_reviews.required_approving_review_count`: 1
- `required_pull_request_reviews.dismiss_stale_reviews`: **true**
- `required_pull_request_reviews.require_last_push_approval`: **true**
- `required_conversation_resolution`: **true**
- `enforce_admins`: **true**

Do not report PR #13 as merge-ready or approved until these values are
applied and verified with `GET repos/<owner>/<repo>/branches/feature_ralph/protection`.

## Retrieval-approval trust root

Status: **owner action pending (retrieval stays unapproved).** Retrieval
approval requires an owner-managed Ed25519 trust root
(`academic.retrieval-approval-trust-root.v1`) stored OUTSIDE this repository
and pinned by SHA-256 at the gate. No production key or approver is created
in this PR; without a configured trust root the retrieval gate fails closed
and the corpus remains `retrieval_unapproved`.
