# Google Drive authorization and preservation

Use the existing Google Cloud project `chain-of-thought-illegibility`, the existing desktop OAuth client, and the enabled Drive API. Authenticate as `antoniojun.nyc@gmail.com`. No new cloud project or broader scope is required. The only requested scope is `https://www.googleapis.com/auth/drive.file`.

Save the authorized desktop client JSON and token JSON below a protected directory outside source and artifact roots. The files must be readable by the runtime user and excluded from Git. The program writes token files with0600 permissions and refreshes expired access tokens with the refresh token. API keys and tokens must never be placed in `config.yaml`, Pod JSON, logs or artifact manifests.

```bash
export GOOGLE_OAUTH_CLIENT_SECRET_FILE=/workspace/.secrets/google_client.json
export GOOGLE_OAUTH_TOKEN_FILE=/workspace/.secrets/google_token.json
.venv/bin/illegibility drive-authorize --port 8765
.venv/bin/illegibility drive-init
```

For consent on your own computer,download the existing desktop client JSON from Google Cloud Console → `chain-of-thought-illegibility` → APIs & Services → Credentials → the existing desktop client. Keep it in a protected local directory. Clone this repository and run:

```bash
python -m pip install google-auth-oauthlib==1.2.3 google-api-python-client==2.187.0 requests==2.32.5
python scripts/authorize_google_drive.py --client /PROTECTED/google_client.json --token /PROTECTED/google_token.json
```

This lightweight helper opens your browser,requests only drive.file,verifies actual granted scopes and the designated account,and resolves the canonical folder. It prints only paths/folder IDs. Transfer the protected token file through an authorized encrypted channel or secret-file facility to the runtime; do not paste its contents into chat. Desktop loopback consent does not require SSH forwarding through the RunPod proxy.

For a remote host, forward localhost8765 through the authorized SSH connection before opening the emitted consent URL. Complete consent with the designated account. The program verifies the account through Drive `about`, resolves the exact root title **Inducing Persistent Textual Chain-of-Thought Illegibility**, rejects ambiguous duplicate roots and records the stable folder ID in `artifacts/state/drive_folder.json`. Preserve that ID through `GOOGLE_DRIVE_FOLDER_ID` or `operations.drive_folder_id`. With `drive.file`, an unrelated existing folder may not be visible until the application has access; do not broaden scope or create ambiguous duplicates.

`drive-init` performs an actual upload/download/checksum roundtrip. Successful local transport-double tests do not establish OAuth consent, folder creation or remote persistence. A roundtrip is transport evidence; retained compatible artifacts and actual restoration are required for the durability gate.

Artifacts are placed below `<stage>/<branch>/<model>/<run_id>/`. Stages include clean collection, transforms, baseline/evaluation, SFT, RL, analysis and reports. Each remote file has an application artifact ID, canonical project ID, scientific configuration hash and SHA256. The indexed local registry records MD5, size, source commit, model/dataset revisions, seeds, status and remote IDs. Required raw outputs and checkpoints trigger immediate incremental replication; a failed upload preserves pending state and stops further affected work.

```bash
.venv/bin/illegibility sync
.venv/bin/illegibility retrieve ARTIFACT_ID /workspace/restored/adapter.safetensors \
  --expected-provenance /workspace/compatible_expected_provenance.json
```

Retrieval requires configuration hash, source commit, model and dataset revisions, model identity and seeds. Matching filenames alone are insufficient. Final sync reconciles the full remote inventory and exports `artifact_registry.json`; raw hooks use indexed SQLite transactions and process only pending/failed uploads. Preserve `artifact_registry.sqlite3` and its WAL/SHM through a SQLite backup or checkpointed filesystem snapshot. Never copy a live SQLite database file alone.

Google resumable uploads use8MiB chunks and zero SDK retries. Shared persistent recovery permits three retries per operation and100 per worker. In-process interruptions retain the resumable request; after process loss, an existing pending remote file ID is repaired without creating a duplicate, which can restart the file transfer from its beginning. Downloads publish atomically only after actual size, MD5 and SHA256 checks. A verified remote copy permits recovery when a local checkpoint file has been retired.

When standard HTTP/HTTPS proxy variables are configured, `drive_service` uses the Requests transport with the same OAuth credentials and exact scope verification. It verifies TLS, permits only the named Google HTTPS hosts, rejects redirects, preserves Drive's resumable-upload308 responses, and bounds both request and response bodies to9MiB. Media transfers retain8MiB chunks and the shared retry policy; the transport adds no hidden retries. Direct environments retain the ordinary Google SDK transport.

## Google Drive as the primary durable store

The user directed this deployment on11October2026 to rely entirely on Google Drive. Data, artifacts, model caches and local ledgers on the RunPod are scratch. Completed required artifacts must be verified on Drive before the worker proceeds; an interrupted unverified operation remains incomplete. This mode does not certify that Pod storage survives stopping or deletion.

Set `runtime.durability_evidence.mode` to `google_drive_primary`; this evidence is operational, excluded from the scientific configuration hash and included in the audited snapshot. Supply `live: true`, a `registered_probe` containing the complete VERIFIED/COMPLETED artifact-registry record, and `independent_restore` containing the actual independently downloaded probe report. The latter records `status: INDEPENDENT_DRIVE_RETRIEVAL_VERIFIED`, observation time, `source_commit`, `configuration_hash`, exact `probe_identity`, payload `content_hash`, artifact and Drive file IDs, complete-file SHA256, MD5 and size, and `pod_filesystem_read_for_retrieval: false`. A fixture, a generic success flag, or a payload hash alone cannot establish this gate.

The retained probe contains `data: {payload: durable_probe_v1}` and identity `{configuration_hash: <frozen hash>, purpose: durability_probe_v1}`. Its registry provenance must match the frozen source commit, configuration, aggregate model/dataset revisions, model `all` and seed42. A new source/configuration freeze requires a newly bound probe and independent restore; preserve earlier immutable artifacts under their original namespace.

Every launch authenticates the intended account and canonical project again, checks the retained remote artifact's metadata/publication identity and ancestry, then downloads it into a new temporary directory. `DriveStore.restore_registered_artifact(record, destination, expected_provenance=...)` verifies complete-file SHA256, MD5 and size against the separately preserved registry record before publishing the restored path. The preflight also verifies the JSON identity and payload hash. It never uses the original local probe or a cached destination. Unavailable Drive, mismatched provenance or corrupt bytes block paid work. No RunPod volume or stop-survival flag is required in this explicit mode. The default `runpod_lifecycle` mode retains its existing provider-survival and independent-backup requirements.

If scratch is lost, recreate the exact approved source/runtime and supply protected OAuth credentials from the authorized credential store, or repeat consent. Preserve the canonical Drive folder ID and independently verified recovery-inventory identity outside the Pod. Restore the compatible inventory from Drive first, then restore each required payload into an absent destination with complete expected provenance. Rebuild local registry/ledger state only from verified completed records; do not mark missing or pending transfers completed, reset spending/retry counters, or regenerate completed responses. Source model weights may be reacquired at their original immutable revisions. Run the scoped live preflights again before resuming. A complete training recovery additionally requires every checkpoint payload and its compatible optimizer, scheduler, RNG and progress manifests; adapter-only exports cannot resume training.

For a collection started with Drive-primary recovery enabled, each immutable artifact has a small recovery descriptor published before its payload under `recovery/<source commit>/<scientific configuration>/<stage>`. After restoring the same source checkout, frozen configuration and protected OAuth access, recover collection files without the Pod's old registry:

```bash
.venv/bin/illegibility --config /PATH/same_frozen_config.yaml recover-collection
```

The command validates descriptor identities, safe paths and current remote metadata before downloading evaluation/candidate/raw-response/pair/accepted-pool artifacts into the configured scratch tree. A resumed collection action reconstructs mutable progress from the original response cache; a missing or pending payload blocks regeneration. Matching coherent native retry snapshots preserve original operation IDs, starts and retry counts. Completed uploads also record their actual worker retry floor and operation retry count; recovery merges these records with newly charged restore-download retries. Missing snapshots or retry metadata block native resume rather than reset the three-per-operation or100-per-worker allowance.

The validated native recovery record at `artifacts/state/collection_recovery.json` permits scoped native collection under the same immutable inputs and current externally anchored GPU spending authorization. Native collection invokes no provider APIs. This command imports no provider spending or training state; later training/provider recovery still requires compatible coherent ledgers and checkpoint state.
