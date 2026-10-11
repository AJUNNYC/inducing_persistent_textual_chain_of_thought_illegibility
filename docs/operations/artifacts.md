# Artifact identity, reuse and recovery

The Git repository holds source, documentation, schemas, locks and sanitized verification evidence. Runtime benchmark rows, clean/accepted/transformed traces, native generations, judge requests/results, training weights, optimizer state and research results live below configured persistent data/artifact/cache roots. Credentials are never artifacts.

Canonical outputs include `data/evaluation/evaluation_package.json`, the nested panel and prospective plan; `data/rl/manifest.json`; candidate/clean/accepted/transformed data; `artifacts/preflight/report.json`; `artifacts/training/sft/<family>/<model>/level_<nn>/`; `artifacts/training/rl/<model>/`; native generation/judge caches; and `artifacts/reports/`. A selected RL orchestrator may add an outer selected-model namespace; state descriptors resolve the returned verified snapshot path rather than guessing it. Verify the stage's result manifest for actual paths.

Every artifact record contains experiment, branch, model, run, stage, producer,40-character source commit,64-character scientific configuration hash, model/dataset immutable revisions, seeds, relative path, size, SHA256, MD5, completion/validation/sync/retention status and remote file/folder IDs. It is registered only after actual bytes exist. SQLite transactions with `synchronous=FULL` and a pending-state index preserve incremental updates across interruptions. A full JSON inventory is exported during final reconciliation; it is not rewritten for every raw record.

`StageRunner` hashes completed output files and dependency completion content. Re-running a target reuses verified compatible outputs. Corrupt output triggers repair/reexecution; identical repaired bytes permit dependent reuse. Changes to scientific configuration, source or upstream content invalidate downstream completion. Unknown or missing outputs never become completed from a directory name.

SFT/GRPO recovery states include adapter, optimizer, scheduler, RNG and progress/sampler state. Matching adapter-only exports support evaluation, not continuation of training. Full checkpoint publication is atomic and hash-verified before its completion manifest becomes visible. Directory notifications publish every verified file with manifests last. Latest-two recovery retention follows successful required replication; all final SFT adapters and all ten RL scientific snapshots per model remain required.

Ordinary malformed/wrong model answers are recorded outcomes. Infrastructure retries share persistent three-per-operation/100-per-worker counters and never silently discard completed upstream records. An RL update waits for all32valid reward records. A failed step rolls back to matching full state; no failed update advances completion. OOM halts affected execution for a resource-compatible restart rather than retrying the same impossible allocation. Checkpoint compatibility guards source/configuration/data/model/seeds and the actual serialized state.

[Drive operation](google_drive.md) covers automatic raw/checkpoint replication and compatibility-aware retrieval. Before deleting a local required copy, verify the permanent remote copy and preserve the registry. Before retiring a volume, perform a real remount/retrieval check. Neither a local fsync nor a transport-double test establishes cloud lifetime durability.
# Batched clean collection

Set `CLEAN_GENERATION_BATCH_SIZE=32` and `DRIVE_UPLOAD_WORKERS=8` when invoking
the committed collection command. Native Transformers decoding groups whole
prompts with left padding and independent per-example RNGs under the frozen
temperature, top-p and token budget. Responses retain their individual EOS,
token IDs and original candidate order. Each complete GPU batch is written to
the raw cache, then published through eight independent Drive HTTP transports.
Every raw file retains its checksum verification and recovery descriptor before
the next GPU batch starts. Stop requests drain and publish the current batch.
Folder lookups are cached within a publisher; upload retries still share the
durable worker ledger and its original limits.

