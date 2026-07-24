# MIRACL-ko KT actual execution approval request — 20260724

## Status and decision requested

This is **Phase K1: an execution-approval request**, not an execution plan
approval and not a model run. No model/image/cache download, local or remote
inference, taxonomy generation, artifact creation, taxonomy A/B, Agent,
focused Part 1/2, or AIHub legal evaluation was performed.

The requested one-upload design is a monolithic offline capsule: code,
MIRACL 20K/50K/110K fixtures, final control records, a digest-pinned
`docker save` archive, an immutable Hugging Face cache snapshot, and offline
run/result wrappers. The KT host is allowed to supply only the NVIDIA driver,
Docker, NVIDIA Container Toolkit, H200 access, and enough disk.

Two P0 inputs still prevent a real `TaxonomyGenerationPlan` SHA-256 from
being issued. This document records the exact immutable candidates that are
available now and requests the missing inputs; it does **not** invent them.

The machine-readable record is
[`MIRACL_KO_KT_ACTUAL_EXECUTION_APPROVAL_REQUEST_20260724.json`](MIRACL_KO_KT_ACTUAL_EXECUTION_APPROVAL_REQUEST_20260724.json).
The intended outer-capsule inventory is
[`SINGLE_UPLOAD_PACKAGE_MANIFEST_DRAFT.json`](SINGLE_UPLOAD_PACKAGE_MANIFEST_DRAFT.json).

## Fixed candidates verified from official sources

| Area | Candidate to review | Exact value |
|---|---|---|
| Model | Hugging Face repository | `Qwen/Qwen3-8B` |
| Model revision | Immutable Hub commit | `b968826d9c46dd6066d109eabc6255188de91218` |
| Tokenizer | Repository and revision | `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218` |
| Runtime image | Linux/amd64 vLLM OpenAI server | `docker.io/vllm/vllm-openai:v0.9.0@sha256:df2c55e5107afea09ea1a50f9dd96c99ebf97a795334c4d08f691f3d79b2ab12` |
| Registry image bytes | Published amd64 image size | `11,068,018,467` bytes |
| vLLM source version | Release/tag | `0.9.0` |
| Source build identity | CUDA / Python / core Torch packages | CUDA `12.8.1`; Python `3.12`; `torch==2.7.0`; `torchvision==0.22.0` |

The fixed model page resolves the shown full 40-hex commit, identifies the
repository as Apache-2.0, and lists its vLLM serving path. [Qwen3-8B fixed
revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)

The Qwen deployment guide identifies Qwen3 vLLM behavior and the per-request
`enable_thinking` control. It recommends vLLM 0.9.0 or newer for Qwen3. The
v0.9.0 source Dockerfile and CUDA requirements identify the candidate build's
CUDA/Python and Torch dependencies. [Qwen official vLLM guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)
[vLLM v0.9.0 Dockerfile](https://raw.githubusercontent.com/vllm-project/vllm/v0.9.0/docker/Dockerfile)
[vLLM v0.9.0 CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.9.0/requirements/cuda.txt)

The pinned registry digest is the Docker Hub `v0.9.0` Linux/amd64 image
record; the published registry size is a provenance fact, not a disk-capacity
approval. [Docker Hub tag record](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.9.0)

At package build time the locally imported image must still be inspected and
its digest and in-container `vllm`, Python, CUDA, Torch, and installed-library
versions recorded. That observation proves that the imported bytes match this
candidate; no mutable tag or `latest` is acceptable.

## Candidate launch contract

The candidate uses the official image's default OpenAI API-server entrypoint.
The arguments after the image are exactly:

```text
--model Qwen/Qwen3-8B
--revision b968826d9c46dd6066d109eabc6255188de91218
--tokenizer Qwen/Qwen3-8B
--tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218
--host 127.0.0.1
--port 8000
--generation-config vllm
```

The offline container run also requires `--gpus all`, `--network host`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and a read-only bind of the
exact cached snapshot. It does not use `--trust-remote-code`, a local model
path, a remote inference API, or a reasoning parser. The eventual plan must
still fix the request-level thinking policy, full prompt, JSON schema,
sampling controls, batch controls, and request-body hash.

### P0: existing wrapper cannot yet launch this official image

The tracked `b00b6e9` `start_vllm.sh` adds
`plan.server_launch.command` before the plan arguments. The official
`vllm/vllm-openai` image has an API-server default entrypoint and expects the
first supplied container argument to be `--model`; an injected extra command
token is not an executable official-image launch. Therefore the current
60KB harness remains a validation harness, not the approved monolithic
capsule runner.

This is deliberately a **P0 pre-package correction**, not a fallback. The
future correction must be source-pinned, have a new generator code contract,
and preserve the existing preflight/receipt/audit locks. It is outside Phase
K1, so no runner code was changed here.

## Exact plan status

The following input hashes are already fixed:

| Input | SHA-256 / identity |
|---|---|
| Generator source commit | `b00b6e927c810dd870b004da6a6af19ecae9113e` |
| Existing generator code-contract hash | `82d54968c0010e36c44771d289699abb1b33ef709f1e0b339a696d5d90ab3842` |
| Existing generator code aggregate hash | `bfda9828a09074a0131c6192b0c4f629d3d1b1b67b9c2b74c2b39cf869c195b3` |
| Existing code-harness tar | `c4bd88e79174dbd7525193f83848165cf1e815a70e9d08562beabe126f55d7d1` (`60,369` bytes) |
| MIRACL revision lock | `cc27c0a53a6eb002620a2dc03a16d5cd0dcf1663ab04983cf5d1032c4f356783` |
| MIRACL preparation contract | `2c806cc432319f24e9b4fbcd34380bf117bd223bc3c299e64090b6a966a92b21` |
| MIRACL subset manifest | `5523b9d94dea2731915daf2cd88afb305f346339a7bf6fd3530e0db41dfd116b` |
| 110K corpus | `76fc195065f29b96f02e1fc3d7bb8c1e2330c4e701c8c472602fa5ced9626030` |

An actual execution plan SHA is **not available**. That is intentional: the
current taxonomy contract requires the complete taxonomy algorithm, Korean
prompt, structured-output schema, label catalog, artifact ID, unknown rule,
sampling/seed/batch/retry controls, template/thinking policy, and corrected
runner contract. The repository contains only synthetic test examples for
these values; using them for a real run would violate the contract.

The JSON approval-request candidate itself has a canonical content SHA-256,
but it is expressly *not* a `TaxonomyGenerationPlan` SHA and cannot appear in
an approval record.

## Required approvals and inputs — separate decisions

1. **Model/runtime selection:** approve or reject the repository/revisions,
   Linux/amd64 image digest, and source-level runtime identities in the table
   above.
2. **Semantic taxonomy specification:** supply and approve the complete
   frozen algorithm, prompt, JSON schema, label catalog, artifact ID,
   unknown/outlier rule, sampling controls, seed, batch/retry/resume controls,
   thinking/template policy, and acceptance QA thresholds. Inputs remain
   title/text only; query, qrel, relevance, gold, evidence IDs, and answers
   remain prohibited.
3. **Corrected offline runner:** approve a source-pinned correction of the
   P0 container-entrypoint issue and its new code-contract hashes.
4. **Exact plan:** after 1–3, generate the valid canonical
   `generation_plan.json`, review its exact SHA-256, then issue a real
   `approval_record.json` with approver, UTC timestamp, and basis.
5. **Package import:** approve acquisition of the digest-pinned Docker image,
   the exact Hub snapshot cache, `docker save`, and construction of one
   offline upload capsule.
6. **One KT/H200 execution:** approve one 110K generation only.
7. **Post-generation evidence:** approve standalone source audit, then only
   on success the 20K/50K filter projection and full audit/result collection.

The template in the JSON file has blank `approved_by`, `approved_at`, `basis`,
and plan/code values. It is not an approval record and cannot authorize any
preflight or generation.

Not included in any of the above decisions: taxonomy A/B, Agent execution,
focused Part 1/2, AIHub legal evaluation, external inference API use, or
post-hoc tuning.

## Single-upload capsule size and disk contract

Known fixture payloads total `100,106,970` bytes: 20K/50K/110K corpora
(`100,100,008` bytes), subset manifest (`6,079`), and revision lock (`883`).
The existing code harness is `60,369` bytes; it is not the final code
component.

No model cache, Docker archive, or final outer tar has been downloaded or
constructed. Their actual byte sizes and checksums are therefore unknown and
must not be estimated into an approval.

```text
outer_uncompressed_bytes =
  code + MIRACL_fixture + control_records + docker_save_archive
  + HF_snapshot_cache + run_wrappers + manifests

outer_upload_tar_bytes = measured final compressed tar bytes

peak_free_disk_required_while_retaining_upload_tar =
  outer_upload_tar_bytes + outer_uncompressed_bytes
  + docker_graph_after_load + docker_load_temporary
  + generation_outputs + result_tar + safety_reserve
```

Before release, the package builder must record the actual `docker save` byte
size/SHA-256, exact HF snapshot file manifest/byte total/SHA-256, final outer
tar byte size/SHA-256, Docker graph bytes after `docker load`, and available
KT disk. The registry's `11,068,018,467` image bytes are only a remote
reference point, not a substitute for any of those measurements.

## Outcome interpretation

- **Complete generation + standalone audit + projection + full audit:** an
  audited MIRACL 110K taxonomy artifact and its 20K/50K filter projections
  are available.
- **Partial or failed generation:** only receipt, raw-response hash, and
  diagnostic evidence are available. No successful artifact is claimed.
- Neither outcome is a taxonomy performance A/B result, Agent result, Part
  1/2 result, or AIHub legal result.

## Approval reply text

Because a valid plan SHA does not yet exist, an actual execution approval
would be false. The following is the only truthful approval available now:

```text
I approve only Phase K1 candidate review of Qwen/Qwen3-8B at
b968826d9c46dd6066d109eabc6255188de91218 and
docker.io/vllm/vllm-openai:v0.9.0@sha256:df2c55e5107afea09ea1a50f9dd96c99ebf97a795334c4d08f691f3d79b2ab12.
This does not approve a generation plan, cache/image download, capsule build,
KT execution, artifact, projection, A/B, Agent, Part 1/2, or AIHub evaluation.
Provide the frozen semantic taxonomy specification and approve the corrected
runner before requesting the exact plan SHA.
```

After the missing specification and runner correction are independently
approved, the final approval text must include the **then-generated** plan
SHA-256, final code-contract SHA-256, exact outer-capsule SHA-256, actual
approver, UTC timestamp, and basis. No placeholder from this document may be
substituted for those values.
