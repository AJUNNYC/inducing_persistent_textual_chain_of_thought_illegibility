# 14B control and remaining SFT conditions

Authorized order is the zero-percent ROT13 control, ROT13 level 8 and 9 continuations, then KaziText levels 5 and 10. RL remains paused. KaziText levels 3 and 7 are not authorized for automatic launch.

The explicit ROT13 level 0 control restores the certified accepted pool's original reasoning, retaining prompt bytes, final answers, common ordered 10,000/100 split and seed. It is a separate condition with strength zero, not a relabelled level 1 or level 10. `scripts/run_unencoded_sft_control.py` verifies the pool and transformed-source correspondence before writing a checksummed working dataset. It uses the existing efficient one-epoch engine, rank 8/alpha 16 LoRA, learning rate 2e-4, 625 optimizer updates, effective batch 16 and 4096-token eligibility. Native mask, capacity and full-state recovery guards remain active. The control is initially one epoch.

`continue_sft_level.py --family kazitext` extends the established continuation to KaziText while retaining ROT13 defaults and identities. A second epoch is a separate continuation from the completed own-family one-epoch full state, at peak 5e-5 with preserved AdamW moments and seed-42 next-epoch ordering. The original one-epoch endpoint is retained. This staged schedule matches the existing low-rate ROT13 continuations; it is not a single two-epoch cosine schedule.

Outputs and recovery checkpoints are written to the mounted volume and synchronously published through the existing Drive artifact service. Each condition has a separate namespace. A sign-in, installed runtime, or launched process is not evidence of optimizer progress. Completed runs require their real result and publication receipts.

## Zero-percent control continuation

The requested control now retains epoch 1 and adds a matched second epoch before ROT13 levels 8 and 9. It uses the same continuation recipe as the encoded conditions, including peak learning rate 5e-5, 625 additional updates, and preserved AdamW state. Epoch 2 has a separate output directory and the existing Drive and volume backup procedures.
