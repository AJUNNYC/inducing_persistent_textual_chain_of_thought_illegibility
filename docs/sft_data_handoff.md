# SFT producer-to-consumer data handoff

The prepared corpus keeps its original generation configuration, source commit,
model traces and transformed targets. A repaired training implementation has its
own configuration and native dependency lock. These identities are explicitly
separated, rather than rewriting records or pretending current code is the old
producer.

`scripts/certify_sft_handoff.py` checks all 122 local artifact sizes and SHA-256
hashes against the completed independent Google Drive readback. It validates the
40-condition production worklist, including ordered shared IDs, model-specific
clean traces, exact final answers and 10,000/100 splits. Scientific configuration
projections must agree; only implementation revision and dependency lock hash
may differ beyond existing operational exclusions.

The certificate binds both configurations, original Drive proof, consumer source
files, native lock and exact ordered record hashes. `build_worklist` accepts the
certificate explicitly. Pass the same certificate to `run_family_queue` or
`run_sft_fit`; a fit with producer identity fails without its matching certificate.
The certificate identity enters fit/checkpoint identity. All existing record,
model, target and preflight validations remain in force.

Run the committed script with `--producer-config`, `--consumer-config`, `--audit`,
`--bundle` and `--output`. Consumer implementation revision must name the actual
source commit and dependency lock hash must match `requirements.lock`. Store the
certificate and worklist in Drive. Compatibility certification does not satisfy
baseline repeatability or authorize a scientific fit by itself.

The certificate, worklist, consumer configuration and launch evidence for the paused fit are in the [verified Drive launch folder](https://drive.google.com/drive/folders/1x7_brQz7cCutiTZ42adyQ4C5ilZQsrGV). Its consumer source remains frozen at `773cfd3cc6a776262e9ca65ca10667aaad456b69`. The [certificate](evidence/sft_data_handoff_773cfd3_certificate.json) retains the original corpus and training settings. The newer [optimization implementation](sft_optimizations.md) changes training source bytes and requires a fresh consumer certificate plus affected target-GPU evidence; it is not covered by this original certificate. The prior `a50de2c` handoff remains historical in its original folder.


## Certified single-GPU launch

`scripts/run_certified_sft.py` consumes the consumer configuration, certificate,
production `transformed/manifest.json`, certified worklist, canonical preflight
report and UTC deadline. `--family rot13 --model 7b` runs the ten independent
encoding levels sequentially; `--level 1` selects only the first fit. Each fit
uses a fresh base or an exact full checkpoint. The fixed optimizer, loss, counts
and two epochs remain unchanged. Checkpoints and final adapter payloads pass
verified Drive barriers. SIGTERM and the deadline pause at a whole optimizer
update. `--validate-only` checks every launch binding and all thirteen gates
without loading a model or starting training.

`scripts/reconcile_sft_launch.py` reuses compatible passed control evidence,
retaining its original source/configuration hashes and adding verified current
source/recipe bindings. The newly certified consumer changes implementation
identity only. It still requires the repaired full baseline gate to pass.
Scoring, prompt and GPU controls are not declared passed through fixture tests.

The SFT consumer configuration is separate from the vLLM measurement protocol.
Subsequent adapter evaluations must explicitly use the saved frozen vLLM
contract; the generic experiment evaluator must not silently default to native
inference. Capability-only repeatability does not complete legibility judging.


## Validated launch on 11 October 2026

The repaired baseline gate passed for all four domains of both models using the exact 7,398 preserved outputs. The [launch reconciliation receipt](evidence/sft_launch_readiness.json) records all thirteen mandatory training gates passed for consumer `773cfd3`, and the [actual H100 validation](evidence/sft_launch_pod_validation.json) returned READY with forty certified conditions and ten selected 7B ROT13 fits. This snapshot preceded the first fit, now paused at step 214. Compatible GPU and statistical controls retained their original evidence and explicit source/recipe bindings. These receipts do not authorize new execution profiles or replacement hardware.

From the pinned Git archive `/workspace/sft_launch_773cfd3` on the existing H100, the tested launch validation is:

```bash
SOURCE_CODE_COMMIT=773cfd3cc6a776262e9ca65ca10667aaad456b69 \
ARTIFACT_ROOT=/workspace/artifacts/sft_launch_773cfd3 \
/workspace/illegibility_gpu_env/bin/python scripts/run_certified_sft.py \
  --config /workspace/bootstrap/sft_launch_configuration_773cfd3.json \
  --certificate /workspace/artifacts/sft_data_handoff_773cfd3/certificate.json \
  --bundle /workspace/data/vllm_sft_preparation/transformed/manifest.json \
  --worklist /workspace/artifacts/sft_data_handoff_773cfd3/worklist.json \
  --preflight /workspace/artifacts/sft_launch_773cfd3/preflight/report.json \
  --deadline 2026-10-11T20:20:00Z --family rot13 --model 7b --validate-only
```

For training, remove `--validate-only` and provide the existing private `GOOGLE_OAUTH_TOKEN_FILE` and central `GOOGLE_DRIVE_FOLDER_ID` bindings. The deadline must still be in the future; the worker preserves full checkpoints to Drive before pausing. The frozen measurement contract is `/workspace/artifacts/sft_launch_773cfd3/measurement_protocol.json`; its configuration binding differs explicitly from the original baseline binding, while measurement settings, panels and code hashes agree. Legibility judging remains pending.


## Original H100 first-fit launch

The [launch receipt](evidence/sft_training_launch.json) records 7B ROT13 level 1 training on the existing H100, after native model initialization and successful actual completion-loss/adapter-gradient verification. The original ten independent levels remain queued. The launch-only `HF_HUB_OFFLINE` flag was removed after it blocked a tokenizer metadata request; model revisions and cached weights are unchanged. Verified Drive checkpoints occur every 100 optimizer updates, and the authorized 20:20 UTC deadline pauses after a whole update and preserved full checkpoint. The readiness snapshot above predates this launch.


## Safe pause and resumption

Source `05f42b0` now has [accepted no-recomputation evidence](evidence/sft_optimization_pro6000_selected_profile.json) and [READY affected-runtime controls](evidence/sft_optimization_pro6000_readiness.json) for 7B ROT13 level 1 on RTX PRO 6000 Blackwell 96GB. CUDA input caching failed the speed gate and remains disabled. The launcher accepts `--execution-benchmark` and `--execution-profile` only with passed source/GPU/condition-matched evidence. Migrating the original step-214 fit additionally requires `--migration-checkpoint`, `--migration-receipt` and `--recovery-benchmark`, selecting that one condition. It verifies parent identity and full-state bytes, preserves immutable scientific fields, inherits adapter/optimizer/scheduler/sampler/RNG state and writes separate consumer lineage while retaining original checkpoint bytes. The two paired 96-example sequences had bitwise state agreement, with 1.2695x and 1.2806x median throughput ratios. The full matrix and statistical validation were not repeated; compatible upstream evidence remains reusable. The [continuation worker](evidence/sft_optimization_pro6000_training_resume.json) launched from inherited step 214 with verified Drive child state; parent bytes remain unchanged.

The original H100 7B ROT13 fit was [paused at step 214](evidence/sft_training_pause_214.json), with the whole optimizer boundary and full recovery state preserved in Drive. For an unchanged-source resume, restore the registered `full_state.pt` and `manifest.json` into their original checkpoint directory using their exact provenance; restore the certified corpus, config, certificate and worklist from the existing Drive indices. Execute pinned source `773cfd3` with the same consumer identity and a new authorized future deadline. The fit automatically restores its latest matching verified full checkpoint, including optimizer, scheduler, RNG and sampler position. The previous 20:20 UTC deadline is historical and must not be extended implicitly. Stopping the training process does not stop provider billing; the PI account operator stops the Pod through the console.

Replacement hardware must additionally pass the affected runtime checks and record an explicit migration. Changing source, attention kernels or execution profile is not an exact continuation under the old certificate. The historical source `fd23db9` benchmark used fresh adapters without reading the paused checkpoint. The source `05f42b0` recovery screen instead loads its immutable full state into isolated comparisons, evaluates the next ordered updates, and never rewrites or advances the official checkpoint.

The [actual source fd23db9 Blackwell benchmark](evidence/sft_optimization_pro6000_benchmark.json) completed fresh baseline, deferred-check and SDPA profiles on RTX PRO 6000 Blackwell Server Edition 96GB. It neither read nor modified the official step 214 state. Deferred checks retained bitwise numerical agreement but missed the 5% speed requirement; SDPA failed the numerical screen and was slightly slower. Neither profile is approved for production. The [verified Drive index](evidence/sft_optimization_pro6000_drive_index.json) preserves both the measured attempt and the earlier cache-lock failure. Benchmark input/source bindings and longest-example gradient probes do not replace the affected production migration gates or authorize the old checkpoint under a new certificate.
