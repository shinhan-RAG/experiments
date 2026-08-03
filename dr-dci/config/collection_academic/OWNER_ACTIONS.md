# Owner actions for PR #13 (fifth-review C3/C4)

Some trust-boundary items cannot be fully closed by an in-PR code change and
must be applied by the repository owner. They are recorded here rather than
overclaimed as done in the PR.

## C3 — hash-pinned dependency lock (`pip --require-hashes`)

Status: **done (owner-actions execution, 2026-08-03).**
`dr-dci/requirements.lock` was generated on the CI resolution target
(linux/x86_64, CPython 3.12.8; pip 24.3.1, pip-tools 7.4.1) with:

```bash
pip-compile --generate-hashes --allow-unsafe --strip-extras \
  --output-file requirements.lock requirements.txt -c constraints.txt
```

The workflow installs ONLY via
`python -m pip install --require-hashes -r requirements.lock`, and
`scripts/check_supply_chain.py --constraints constraints.txt --lock
requirements.lock --expect-python 3.12.8` plus
`tests/test_ci_supply_chain.py` fail loud on missing hashes, lock/constraint
drift, relaxed installs, unpinned actions, or a Python-version mismatch.
Regenerate the lock only with the command recorded in its header, on the same
platform, in a reviewed PR; never hand-edit resolved versions or hashes.

## C4 — branch protection for `feature_ralph`

Status: **applied (owner-actions execution, 2026-08-03); keep verifying.**
The required protection on `feature_ralph` is:

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
