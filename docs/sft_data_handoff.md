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

The completed certificate and worklist are in the [verified Drive handoff folder](https://drive.google.com/drive/folders/15wW4PXRzf_i4yNaN3jyOeMMFddXrLpRv). Source/configuration identities are frozen at `a50de2c`; subsequent evidence-only commits do not redefine that consumer.


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
