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

`drive-init` performs an actual upload/download/checksum roundtrip. Successful local transport-double tests do not establish OAuth consent, folder creation or remote persistence. None of those live operations has been performed in the supplied CPU environment because the protected client/token files are unavailable.

Artifacts are placed below `<stage>/<branch>/<model>/<run_id>/`. Stages include clean collection, transforms, baseline/evaluation, SFT, RL, analysis and reports. Each remote file has an application artifact ID, canonical project ID, scientific configuration hash and SHA256. The indexed local registry records MD5, size, source commit, model/dataset revisions, seeds, status and remote IDs. Required raw outputs and checkpoints trigger immediate incremental replication; a failed upload preserves pending state and stops further affected work.

```bash
.venv/bin/illegibility sync
.venv/bin/illegibility retrieve ARTIFACT_ID /workspace/restored/adapter.safetensors \
  --expected-provenance /workspace/compatible_expected_provenance.json
```

Retrieval requires configuration hash, source commit, model and dataset revisions, model identity and seeds. Matching filenames alone are insufficient. Final sync reconciles the full remote inventory and exports `artifact_registry.json`; raw hooks use indexed SQLite transactions and process only pending/failed uploads. Preserve `artifact_registry.sqlite3` and its WAL/SHM through a SQLite backup or checkpointed filesystem snapshot. Never copy a live SQLite database file alone.

Google resumable uploads use8MiB chunks and zero SDK retries. Shared persistent recovery permits three retries per operation and100 per worker. In-process interruptions retain the resumable request; after process loss, an existing pending remote file ID is repaired without creating a duplicate, which can restart the file transfer from its beginning. Downloads publish atomically only after actual size, MD5 and SHA256 checks. A verified remote copy permits recovery when a local checkpoint file has been retired. Volume teardown durability still requires a real lifecycle check on the intended infrastructure.
