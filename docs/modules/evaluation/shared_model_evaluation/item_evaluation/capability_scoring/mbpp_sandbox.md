# mbpp_sandbox.md

**Owner:** `evaluation`. **Implementation status:** implemented; unit/integration verification and remaining live gates are recorded below.

| Artifact | Exact repository path |
|---|---|
| Implementation | `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py` |
| Workflow/support diagram | [`spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.mmd`](../../../../../../spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.mmd) |
| This specification | `docs/modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.md` |
| Unit test slot | `tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_mbpp_sandbox.py` |

**Scientific sources:** D.1.2; E.2.7; H.4. [p. 41](../../../../../../spec/experiment.pdf#page=41) · [p. 57](../../../../../../spec/experiment.pdf#page=57) · [p. 72](../../../../../../spec/experiment.pdf#page=72).

## Implemented public interface and validation

`DockerMBPPSandbox(image, *, runner=subprocess.run, scratch_root=None)` requires a locally available immutable image digest. `execute(code, tests, *, entrypoints=None)` runs a trusted supervisor and a permanently unprivileged candidate worker in a disposable container with no network, read-only inputs, one CPU, one GiB, five CPU seconds, and ten candidate wall seconds. Hidden original tests arrive through supervisor stdin and remain outside candidate memory/files; candidate stdout cannot publish correctness. The supervisor requires namespace PID 1 before terminating all other namespace processes, including descendants that create a new session; measured candidate wall time is recorded. Safe primitive/standard-container, mutation, iterator, and regex-match proxies preserve the original tested semantics. Unsupported transports are infrastructure failures. `preflight()` exercises isolation, completion forgery, supervisor mutation, supported values, escaping descendants, and resource limits. `run_mbpp_preflight(config, output_dir)` retains exhaustive 427-source diagnostics and gates on the exact original selected 203 references plus real isolation probes. `verified_reference_callback(report_path, *, panel, image=...)` verifies hashes and selected bindings before data freezing. Hash-verified per-reference journals retain completed positive and negative checks across interruption or workspace disk exhaustion. Prior failed attempts remain evidence; they do not select replacement items.

Owned unit and cross-owner fixture checks are recorded in the [evaluation validation receipt](../../../../../evidence/evaluation_validation.json). Actual restricted execution evidence is recorded in the [MBPP sandbox check](../../../../../evidence/mbpp_sandbox_check.json). Fixture checks are not scientific outcomes. Native CUDA generation, live pinned judge compatibility, baseline repeatability, statistical validation, and durable cloud storage remain launch gates until their actual reports pass.

The current real Docker certification passed all 427 original sanitized reference suites, including the immutable selected 203, with no unsupported reference cases. The [complete reference report](../../../../../evidence/mbpp_reference_tests.json) retains every task outcome and its code/test/source hashes. The [isolation report](../../../../../evidence/mbpp_isolation_checks.json) records actual protection against completion forgery and supervisor mutation, supported Python values, and resource probes. The descendant-process wall probe stopped at 10.010 seconds; the CPU probe stopped at 5.007 seconds. Candidate code, original tests, selected IDs, and resource caps were unchanged. The [earlier failed executor attempt](../../../../../evidence/mbpp_sandbox_prior_attempt.json) remains available for review.

## Responsibility and controlling details

Implement the source-required separate disposable executor, not code execution in the trainer, judge service or privileged host. Enforce no outbound network, least privilege, read-only allowed inputs and no access to host secrets/training artifacts. Limits are one CPU, one GiB memory, five CPU seconds and ten wall seconds. Use only the original hidden scoring tests for functional correctness; do not expose tests or their outputs to the generating model for repairs.

Verify reference tests and interface extraction before evaluation. Do not modify generated solutions, inject missing algorithms or fix interfaces after generation. Differentiate syntax/failing assertions/task-defined timeouts from failure of the executor itself. Record resource and infrastructure statuses with test/scorer/raw-output identity. A normal subprocess timeout on a privileged trainer host is explicitly insufficient isolation. Sandbox correctness must be tested in the real deployment, not inferred from a blank file's existence.

## Interfaces and dependent procedures

Use the common [artifact contracts](../../../../../contracts.md), [frozen prompt instructions](../../../../../prompts.md), [decision ledger](../../../../../decisions.md) and [validation requirements](../../../../../validation.md). Typed signatures/serialization are delegated implementation details; their scientific meaning is not. The named owner defines this module with its consumers before dependent integration.

No separate child procedure is declared in this diagram. This does not exempt shared recovery, storage, provenance or validation contracts.

## Completion and failure boundary

Implement and test `src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py` against its specified behavior in `tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_mbpp_sandbox.py`. Validate every emitted artifact and preserve its source/configuration identity. Do not claim completeness from a placeholder or passing import. Propagate required child failures and maintain ordinary model outcomes separately from infrastructure faults. Mandatory real preflights and permission gates remain in force. Unspecified allowed conventions must be documented/frozen before outcomes; new scientific choices are not silently authorized.

## Exact path-synchronized diagram

The following diagram is byte-for-byte the active `.mmd` content at generation time. Its procedural content is from the supplied workflow or the cited PDF support requirement; file labels and role assignment are the requested repository organization.

```mermaid
---
title: "spec/mermaid/workflows/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.mmd"
---
flowchart TD
%% Source-derived support responsibility, separated into this module by the requested repository organization.
%% Scientific source: D.1.2; E.2.7; H.4; PDF pages 41, 57, 72.
%% Repository module: src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py
%% Module documentation: docs/modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.md
%% Unit test: tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_mbpp_sandbox.py
%% Owner: evaluation
    MODULE["src/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.py"]
    CONTRACT["Isolate untrusted generated code; withhold hidden tests from prompts and secrets from execution; enforce one CPU, one GiB, five CPU seconds and ten wall seconds."]
    DOCUMENT["docs/modules/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/mbpp_sandbox.md"]
    TEST["tests/unit/evaluation/shared_model_evaluation/item_evaluation/capability_scoring/test_mbpp_sandbox.py"]
    MODULE --> CONTRACT
    MODULE -.-> DOCUMENT
    MODULE -.-> TEST
```
