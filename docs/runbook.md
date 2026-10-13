# runbook.md

## Current SFT status: 14B ROT13 level 10 complete; $240 authorized for remaining work

The default `efficient_one_epoch_v1` recipe processes **all 10,000 certified training examples once per selected fit**: 625 optimizer updates, a complete 625-update cosine schedule and 19 warmup updates. The first fresh fit, 14B ROT13 level 10, completed all 625 updates on 12 October 2026; KaziText level 10 has not completed under this recipe. Both 7B/14B, ROT13/KaziText and all ten levels remain available in the forty-condition Drive-verified inventory. The completed adapter is retained, leaving 39 available conditions uncompleted. On 13 October the user authorized $240 for SFT; no additional GPU launch is recorded here. See [the training redesign](sft_short_pilot.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. Other conditions remain unmeasured. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). GPQA Diamond (198 questions) is appended to the existing capability suite, matching the José et al. dataset variant; the 198-question panel is prepared, while comparison-state freeze and evaluation remain pending. It is a supplemental stress test and does not change the original four-domain criteria.

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

See [RunPod lifecycle](deployment/runpod.md), [Drive authorization](operations/google_drive.md) and [artifact compatibility](operations/artifacts.md). Validation evidence is in [validation_report.md](validation_report.md); trained experimental results remain pending.

## Certified entrypoint and historical receipt scope

The historical NF4 [launch receipt](evidence/sft_launch_readiness.json) records all thirteen mandatory gates passed for frozen consumer `773cfd3`, following [final-field scoring repair](final_field_scoring.md) and reuse of compatible completed controls. The [existing H100 validation](evidence/sft_launch_pod_validation.json) accepted all forty certified conditions and the ten selected ROT13 7B fits. Historical failed baseline and statistical reports remain preserved; they are superseded by the separately versioned passing evidence. No trained adapter is claimed at this readiness handoff.

Use the committed `scripts/run_certified_sft.py` with `--config`, `--certificate`, `--bundle`, `--worklist`, `--preflight` and `--deadline` naming the reconciled handoff files. Select `--family rot13 --model 7b`; add `--level` only to select one independent fit. `--validate-only` verifies the saved launch bindings without GPU work or Drive requests. For the efficient recipe, external SFT checks bind certified data, durability, actual spending and compatible statistical evidence; native model/mask/full-state checks execute as ordinary startup before updates and are not previously passed GPU approval. The production launch verifies Drive storage before model initialization, preserves full checkpoints at the existing cadence, and pauses at a complete optimizer update on deadline or SIGTERM. Compatible controls need not be rerun merely because documentation changed.

## Before real data or paid work

Confirm authorization for source data and accounts, exact construction identities and permitted spending. Resolve/lock compatible dependencies and profile/tokenizer/dictionary provenance. Do not put credentials into source, YAML or checkpoints. Set actual durable data/artifact roots and test the intended lifecycle before relying on them. Check quotas, provider model/schema behavior and actual rates. The PDF's historical price examples and $300 envelope are planning inputs, not live provider quotations or unlimited funding approval. The conflicting warning values remain explicit in `decisions.md`.

## Prepare frozen specifications and test implementation

Shared contracts, state/recovery and producer/consumer interfaces are implemented and exercised against explicit fixtures; run the required live gates before launch. Construct benchmark manifests and public interfaces using frozen source sampling, and use those IDs for detectable-overlap exclusions. Build authorized LMSYS candidates and fresh once-per-model vLLM responses through separate model workers under fixed settings and whole-batch Drive barriers. Validate every candidate through the same exact transforms later used for final datasets; freeze 10,000/100 common IDs only after all checks. Produce all forty verified SFT train/validation artifacts. The shared ordered 800-question RL manifest has already been selected and Drive-verified.

The actual evaluator, question-aware audit, MBPP sandbox and analysis routines retain their required launch evidence. Source-required baseline repeatability and statistical simulation validation are pre-outcome gates; both now have separately retained passing evidence. Their failures cannot be handled by silently changing retention thresholds or sampling. Actual masks, gradients, grouping, checkpoints, providers, storage and authorization require real compatible preflight evidence.

## Legacy full-grid scheduling (superseded for the active pilot)

The source allows three GPU queues: ROT13 SFT, KaziText SFT and GRPO. Each SFT queue has twenty independent fits; each uses the shared two-epoch fit engine starting from the corresponding untouched base. The RL queue runs 7B then 14B sequentially, each with its own optimizer state and same ordered 800 questions. On the currently authorized single H100, run one GPU worker at a time; separate hardware may support concurrent queues. The PDF recommends an 80-GB A100/H100-class target but does not establish that all 14B RL configurations fit. The standard Transformers generation path with `use_vllm=False` applies to GRPO. Fresh SFT source generation used the approved separate vLLM workers.

Every run has a separate verified output namespace. Worklist/config locks prevent duplicate writers or reopening completed conditions with changed identities. Save full state at source cadences and final adapters/scientific snapshots; verify actual outputs rather than infer them from intended step counts. End each GPU's billing when that queue ends rather than wait for all branches to finish.

## Legacy full-grid evaluation and analysis

Use the same 1233 main capability IDs and nested 200 legibility IDs for all 62 states. Base evaluations are shared across SFT/RL and include the same independent audit. Apply the frozen vLLM measurement protocol and repaired final-field scorer to untouched bases and trained adapters; the generic native evaluation default must not replace it. Per-request generation settings remain identical across domains/states. Cache raw generations before scoring. Grade capability deterministically and generated code only inside the isolated executor. Schedule two unchanged scores and the separate tri-state audit for every assessable selected trace, preserving independent caches and all missingness. Baseline legibility judging remains pending and reuses its saved seed-42 responses.

Produce per-domain capability and raw legibility reports, retention/common sets, full and conditional option-A opacity, all 216 candidate windows and validated simultaneous bounds where supported. Keep main baseline unknown evidence from becoming favorable zero opacity. Preserve padding diagnostics, actual denominators and same-transfer-pair conditions. Separate point-estimate candidates, confirmatory qualification and indeterminate evidence. Do not add another run or panel merely because none qualifies.

## Historical full-grid workload ledger

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

## Archived launch procedure: 11 October 2026

At this historical launch, fresh vLLM corpus generation, all forty transformed conditions and their Drive copies were complete. The repaired baseline gate and all thirteen reconciled SFT launch gates passed; the existing H100 accepted the certified ROT13 7B queue. Training outcomes, full legibility judging and RL trajectories were still pending at that time. The current completion and prospective GPQA supplement are described above and in the [updated procedure](procedure_update.md).
