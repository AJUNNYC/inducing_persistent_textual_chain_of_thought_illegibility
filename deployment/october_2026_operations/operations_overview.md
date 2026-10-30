# October 2026 training operations

These are the operational scripts used to launch, extend, recover and monitor the October 14B SFT campaign. They complement the scientific training code already in `scripts/run_certified_sft.py`, `scripts/run_unencoded_sft_control.py` and `scripts/continue_sft_level.py`.

## Historical configuration

ROT13 pod `x7j03ag24vz0gk` ran the 0% control for one epoch followed by a matched second-epoch continuation, then continued existing levels 8 and 9. KaziText pod `tla6hi5owr4prx` ran levels 5, 10, 1, 3, 7, 8 and 9, retaining both epoch endpoints.

First epochs use the existing BF16 rank-8 LoRA recipe, 10,000 training examples, effective batch 16, 625 updates and peak learning rate 2e-4. Second epochs preserve optimizer state and use 625 additional updates with peak learning rate 5e-5. These scripts do not change scientific training settings.

## How to read these scripts

This directory is a deployment record, not an unattended fresh-machine installer. Paths, pod IDs, source revisions, deadlines and prerequisite files are specific to this campaign. Some scripts generate additional remote scripts. Read them before executing; do not blindly replay completed launches or repairs. Original source revision 9de8813 is retained for first-epoch recovery identity; continuation uses 89609d1, or 71fe856 for the zero-percent control.

- `ordered_training_queue.py` and `start_second_kazi_queue.py` establish the original queues. Later queue builders supersede their initial ordering.
- `queue_kazi_backup.py` adds levels 1, 3 and 7 after the initial queue succeeds.
- `resume_kazi_after_cleanup.py` skips completed first epochs and adds export/cleanup between jobs.
- `install_kazi_89.py` adds levels 8 and 9 using that same recovered queue.
- `fix_rot_adapter.py` restores missing supporting files only when hashes exactly match the parent manifest, verifies both parents, then resumes 8 and 9.
- `repair_kazi_sync_launch.py` retries a failed Drive upload through the existing registry before resuming from the latest verified checkpoint.
- `volume_checkpoint_exporter.py` creates unique global-volume archives, checks readback hashes and writes verification receipts.
- `kazi_verified_cleanup_source.py` removes only redundant local intermediate checkpoints after archive verification. Final checkpoints, adapters and global-volume archives remain intact.
- Monitoring and audit scripts report status and verified backup records. The historical health monitor can flag stale first-epoch status files after successful completion; final results and the active queue take precedence over those warnings.

## Storage and credentials

Training state, locks and SQLite registries live on local `/opt/runpod_work/<pod-id>`. Immutable archives are exported to `/workspace/pod_exports/<pod-id>` on the global volume. Google Drive uploads use existing restricted credentials through private runtime paths. Credential values, tokens, model weights, training datasets and generated traces are not included here. Paths to credentials are references only.

No RL or new evaluation jobs are launched by this archive.
