# runbook.md

## Executable stage interfaces

Install using [environment instructions](operations/environment.md), then use the single `illegibility` CLI. Discovery commands do not provision paid infrastructure.

```bash
illegibility hardware
illegibility validate
illegibility plan --target rl_14b
illegibility status
illegibility provision --include-weights
illegibility preflight
illegibility preflight --execute-live
illegibility run --target panels
illegibility run --target rl_data
illegibility run --target sft_data
illegibility run --target rot13_7b_level_01
illegibility run --target kazitext_14b_level_10
illegibility run --target rl_7b
illegibility run --target evaluate_rl_7b_step_010
illegibility run --target analysis
illegibility run --target sync
```

Inspect `plan` for exact IDs before choosing a target. Dependencies run automatically and completed outputs are reused only after hash/configuration/source/dependency verification. A changed or corrupt upstream artifact invalidates dependent completion. Trained stages require a passing canonical live preflight report. Unavailable authorization or scientific evidence produces BLOCKED rather than invented completion. `smoke` runs explicit fixture integrations; it does not authorize a scientific launch. Recovery uses the same target after restoring compatible persistent state.

See [RunPod lifecycle](deployment/runpod.md), [Drive authorization](operations/google_drive.md) and [artifact compatibility](operations/artifacts.md). Local validation evidence is in [validation_report.md](validation_report.md); empirical results are intentionally absent.

## Before real data or paid work

Confirm authorization for source data and accounts, exact construction identities and permitted spending. Resolve/lock compatible dependencies and profile/tokenizer/dictionary provenance. Do not put credentials into source, YAML or checkpoints. Set actual durable data/artifact roots and test the intended lifecycle before relying on them. Check quotas, provider model/schema behavior and actual rates. The PDF's historical price examples and $300 envelope are planning inputs, not live provider quotations or unlimited funding approval. The conflicting warning values remain explicit in `decisions.md`.

## Prepare frozen specifications and test implementation

Shared contracts, state/recovery and producer/consumer interfaces are implemented and exercised against explicit fixtures; run the required live gates before launch. Construct benchmark manifests and public interfaces using frozen source sampling, and use those IDs for detectable-overlap exclusions. Build authorized LMSYS candidates and once-per-model responses under fixed settings. Validate every candidate through the same exact transforms later used for final datasets; freeze 10,000/100 common IDs only after all checks. Produce all forty verified SFT train/validation artifacts and the shared ordered 800-question RL manifest.

Complete the actual native evaluator, question-aware audit, MBPP sandbox and analysis routines before paid scientific training is treated as ready. Source-required baseline repeatability and statistical simulation validation are pre-outcome gates. Their failures cannot be handled by silently changing retention thresholds or sampling. Actual masks, gradients, grouping, checkpoints, providers, storage and authorization require real preflight evidence.

## Schedule training only after required gates

The source allows three GPU queues: ROT13 SFT, KaziText SFT and GRPO. Each SFT queue has twenty independent fits; each uses the shared two-epoch fit engine starting from the corresponding untouched base. The RL queue runs 7B then 14B sequentially, each with its own optimizer state and same ordered 800 questions. These queues may run concurrently but bill independently. The PDF recommends an 80-GB A100/H100-class target but does not establish that all 14B RL configurations fit. The standard Transformers generation path with `use_vllm=False` remains specified.

Every run has a separate verified output namespace. Worklist/config locks prevent duplicate writers or reopening completed conditions with changed identities. Save full state at source cadences and final adapters/scientific snapshots; verify actual outputs rather than infer them from intended step counts. End each GPU's billing when that queue ends rather than wait for all branches to finish.

## Evaluate and analyze without changing the question

Use the same 1233 main capability IDs and nested 200 legibility IDs for all 62 states. Base evaluations are shared across SFT/RL and include the same independent audit. Main native settings remain identical across domains/states. Cache raw generations before scoring. Grade capability deterministically and generated code only inside the isolated executor. Schedule two unchanged scores and the separate tri-state audit for every assessable selected trace, preserving independent caches and all missingness.

Produce per-domain capability and raw legibility reports, retention/common sets, full and conditional option-A opacity, all 216 candidate windows and validated simultaneous bounds where supported. Keep main baseline unknown evidence from becoming favorable zero opacity. Preserve padding diagnostics, actual denominators and same-transfer-pair conditions. Separate point-estimate candidates, confirmatory qualification and indeterminate evidence. Do not add another run or panel merely because none qualifies.

## Planned workload ledger

| Work | Source count |
|---|---:|
| Accepted paired prompt IDs | 10100, split 10000 train + 100 validation |
| Final SFT condition datasets/fits | 40 |
| Independent RL trajectories | 2 |
| Planned RL rollouts | 6400, before recovery attempts |
| Main evaluated states | 62 |
| Main capability generations | 76446 |
| Additional baseline diagnostic generations | 4932 |
| Capability generations including diagnostics | 81378 |
| Evaluation scoring/audit requests | at most 37200 before retries |
| RL reward requests | at most 12800 before retries |

These are planned counts from [H.6.4](../spec/experiment.pdf#page=76) and C.3.5, not completed work or a price forecast. Actual judging can be lower because only eligible/correct/assessable cases are called as specified. Candidate-source generation, transformations, training token counts and storage are separate resource accounting.

## Recovery and safe stopping

On an ordinary bad model answer or rejected candidate, record the outcome and do not resample for a better one. On an eligible execution fault use the shared bounded recovery contract, preserving counters, exact payloads and valid upstream outputs. At most three retries per operation and 100 per worker include SDK retries. Quota/billing/frozen-input/unclassified failures do not get endless retries.

On a known intentional pause, finish the current logical update if safe, save full state, flush caches/manifests, verify persistence and release the GPU. If a needed RL reward is unavailable, pause before the update; never update an incomplete group. After abrupt termination/OOM, restore only the last verified matching completed checkpoint, not a newly imagined save from a failed process. Document any replay and hardware change; exact bit-for-bit trajectories are not guaranteed across different hardware/kernels. A later policy never substitutes for a missing earlier snapshot.

Keep original immutable manifests and irreplaceable checkpoints backed up separately. Report incomplete jobs/records and spending authorization requests honestly. The finished program may legitimately report inability to launch or indeterminate evidence; it must not achieve a neat report by silently changing experimental treatment.
