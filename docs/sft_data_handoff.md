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
