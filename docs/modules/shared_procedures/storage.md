# storage.md

**Owner:** `shared_infrastructure`. **Implementation status:** implemented with actual local CPU tests; mandatory live integrations remain separately gated.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/shared_procedures/storage.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/shared_procedures/storage.mmd`](../../../spec/mermaid/workflows/shared_procedures/storage.mmd) |
| This specification | `docs/modules/shared_procedures/storage.md` |
| Unit test slot | `tests/unit/shared_procedures/test_storage.py` |

**Scientific sources:** E.1.1,E.1.3; E.2.3,E.2.5. [p. 49](../../../spec/experiment.pdf#page=49) · [p. 50](../../../spec/experiment.pdf#page=50) · [p. 51](../../../spec/experiment.pdf#page=51) · [p. 55](../../../spec/experiment.pdf#page=55) · [p. 56](../../../spec/experiment.pdf#page=56).

## Responsibility and controlling details

Persist immutable sources and raw outputs before downstream consumption. Use append-safe/locked writes and atomic final publication when supported; reopen/verify written artifacts before marking complete. Enforce separate writer namespaces across jobs/queues and prevent accidental reopening of a completed job under a changed configuration. Cache invalidation follows dependency identities and must include affected baseline derivatives while preserving valid raw records.

Use verified durable provider storage or equivalent external storage, not an assumed durable path name. The PDF distinguishes container, Pod-volume and independent network-volume lifetimes and requires testing actual survival/access in the intended deployment. Do not assume one volume can be simultaneously attached to every worker. Copy irreplaceable manifests/checkpoints to a separate durable location; record capacity, retention, access and storage charges. Credentials stay out of artifacts/Git.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../contracts.md), [frozen prompt instructions](../../prompts.md), [decision ledger](../../decisions.md) and [validation requirements](../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

No separate child procedure is declared in this diagram. This does not exempt shared recovery, storage, provenance or validation contracts.

## Completion and failure boundary

Implement and test `src/shared_procedures/storage.py` against its specified behavior in `tests/unit/shared_procedures/test_storage.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/shared_procedures/storage.mmd"
---
flowchart TD
%% Source-derived support responsibility, separated into this module by the requested repository organization.
%% Scientific source: E.1.1,E.1.3; E.2.3,E.2.5; PDF pages 49, 50, 51, 55, 56.
%% Repository module: src/shared_procedures/storage.py
%% Module documentation: docs/modules/shared_procedures/storage.md
%% Unit test: tests/unit/shared_procedures/test_storage.py
%% Owner: shared_infrastructure
    MODULE["src/shared_procedures/storage.py"]
    CONTRACT["Publish verified artifacts atomically where supported, enforce exclusive ownership and durable storage, and invalidate stale dependencies."]
    DOCUMENT["docs/modules/shared_procedures/storage.md"]
    TEST["tests/unit/shared_procedures/test_storage.py"]
    MODULE --> CONTRACT
    MODULE -.-> DOCUMENT
    MODULE -.-> TEST
```


## Implemented public interfaces and evidence

`atomic_write_json`, `atomic_write_bytes`, `read_json`, `append_jsonl`, `file_lock`, `publish_artifact` and `verify_artifact` implement fsync, atomic publication, one-writer locking, immutable identities and integrity checks. `register_artifact_hook`, `unregister_artifact_hook`, scoped `artifact_hook` and `notify_artifact` support verified replication; directory notifications verify the file inventory and send payloads before the completion manifest. Hook failures propagate. Local flush proves local publication; provider lifetime survival still requires actual evidence.

Local verification uses `PYTHONPATH=src python -m pytest tests/unit/shared_procedures tests/integration/test_recovery_handoff.py --import-mode=importlib`. Tests exercise positive behavior and actual negative cases, including schema/model/hash failures, no-auth spending blocks, persistent retry exhaustion, fixture-cache exclusion and real CPU optimizer restoration. CUDA/API/lifecycle evidence is reported separately; passing local tests does not authorize or certify scientific launch.
