# Colab CLI automation handoff

## Current status: all ten 14B ROT13 fits complete; GPQA long-profile evaluation next

All ten **14B ROT13 SFT levels are complete**, with final adapters, recovery states and results verified in Google Drive. The nine fresh levels 1–9 each reached 625 updates under scientific source `9de8813031da737453f491723a10b305679dc42d`; retained level 10 keeps source `2ea4fb9a7737cff806226366e73743eec5935fad`. The priority queue was observed complete on 14 October 2026, with zero pending or failed publications and its GPU idle. Each independent fit used all 10,000 certified training examples once, a 625-update cosine horizon and 19 warmup updates. KaziText level 10 remains paused at 444/625 with verified full recovery state and no automatic restart. All forty prepared condition datasets and historical 7B results remain available; their availability does not imply completed fits. At that 14 October handoff, the next selected work was GPQA Diamond for the untouched 14B base plus all ten ROT13 levels, using the separate `long_32000` evaluation profile. Training, data, recipe and original four-domain evaluation budgets are unchanged. See [the campaign guide](gpqa_14b_rot13_campaign.md) and [the training record](sft_completed_rot13_14b_level10.md).

The declared efficient baseline uses an unquantized frozen BF16 base and BF16 head, SDPA with efficient-kernel preference, direct full-gradient Cut Cross-Entropy (`cce_exact`) with FP32 loss and hidden-gradient accumulation, and plain AdamW with FP32 optimizer states. CCE is pinned to upstream commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; `filter_eps=None`, `filter_e_grad=False` and `filter_c_grad=False` disable gradient filtering. No Qwen patch or full token-by-vocabulary logits matrix is used. Policy `published_bf16_lora_full_gradient_cce_v2` requests microbatch 2/accumulation 8. A non-updating duplicated-longest-example capacity guard tries 2/off, 1/off, 2/on, then 1/on, where on/off denotes activation checkpointing; only CUDA OOM permits fallback. The selected actual mode is recorded and cannot change during exact recovery.

Keep completion-only masks and per-example weighting, all seven LoRA targets (rank 8/alpha 16/dropout 0.05), learning rate `2e-4`, requested microbatch 2/accumulation 8/effective batch 16 (capacity fallback 1/16/16), seed 42, all 10,000/100 records, full 4096-token eligibility without truncation and verified Drive saves. Checkpoints occur at updates 100, 200, 300, 400, 500, 600 and final 625. Each update retains the same ordered 16 examples with per-example weight 1/16; execution grouping, padding, dropout draws and floating-point rounding may differ. No measured batching speedup or bitwise-equivalent outcome is claimed. This prospectively declared scientific baseline changes exposure, precision/head numerics, optimizer and execution; it does not prove identical final results. The original NF4 `legacy_two_epoch_v1` and intermediate NF4 `one_epoch_pilot_v1` remain explicit alternatives, with incompatible checkpoints and separate identities.

The user requested implementation without another optimization test or timing campaign. Normal native model/mask/gradient/capacity/full-state guards passed for the completed 14B ROT13 level-10 fit on RTX PRO 6000 Blackwell 96GB. Actual execution was microbatch 1 / accumulation 16 with activation checkpointing off. The recorded fit interval was 5,793.143751 seconds (96m33s); worker launch through completion was 6,322.565654 seconds (105m23s). At $2.99/hour these imply $4.81 and $5.25, respectively, calculated from runtime rather than invoices. That standalone L10 timing remains a dated measurement, not a bound for other levels or hardware. Archived PRO step-987 and A100 update-28/30 states remain in Drive, never warm starts for this recipe. One epoch removes 50% of nominal example passes versus the same-condition two-epoch recipe; no one-hour finish or measured hardware speedup is promised. Diagnostics use the 100 certified validation IDs and base companions; final test panels and original confirmatory thresholds remain reserved. No additional training after the completed 625-update cosine horizon is an exact continuation of the old 1,250-update schedule.

See the [completion, provenance and budget record](sft_completed_rot13_14b_level10.md). The historical untouched 14B GPQA Diamond baseline remains **49/198 (24.75%)**, with 133 truncated and three malformed outputs under its 3,072-token cap; all 198 count. Its Drive-verified records remain separate from the new `long_32000` comparison. The complete long-profile base and original L1-6/L10 are retained; original one-pass L7 is running, with L8-9 remaining. The separate high-rate L10 pilot is complete and native/Drive-verified: strict 52/198 versus original L10 98/198 and base 99/198; historical first-128 human/wordfreq audits retain subset scope. Low-rate v2 training is safely paused at 64 added/689 cumulative updates with verified full recovery and no automatic resume. Only GPQA receives the larger response/context allowance; GSM8K, MMLU, ARC-Easy and MBPP retain their existing 3,072-token measurement contract. The [GPQA contract](modules/evaluation/supplemental_gpqa.md) and [campaign guide](gpqa_14b_rot13_campaign.md) describe strict profile/cache isolation and verified Drive publication. KaziText remains suspended and unscheduled.

## Historical access and deployment evidence


## Purpose and verified tooling

Continue this project from a local Codex CLI session, using the existing consumer
Colab account. The user wants automated runtime management, training execution,
Google Drive checkpointing, and progress reporting after initial authentication.
An Enterprise subscription is not required for this route.

Google's official [Colab CLI](https://github.com/googlecolab/google-colab-cli)
supports runtime provisioning, Python/notebook execution, file transfers, and
SSH over WebSocket. Version **0.7.4** was installed in an isolated cloud venv at
`/workspace/colab_cli_env`; its version and SSH help completed successfully.
After an age-encrypted transfer authorized by the user, the cloud CLI login was
installed privately with mode 0600. Authenticated session listing succeeded.
The user then authorized an A100 runtime, allocated as `sft_a100`. After
`*.prod.colab.dev` was added to the network allowlist, runtime file access and
remote Python execution both succeeded. The [connection receipt](evidence/colab_a100_connection.json)
records an NVIDIA A100-SXM4-40GB (39.49 GiB, SM80), Python 3.13.15,
PyTorch 2.11.0+cu130, and CUDA 13.0 in the stock runtime. At that connection receipt, training had not
started: the stock software environment and a memory-compatible A100 execution
profile still need qualification against the pinned training recipe. The
accepted PRO 6000 no-checkpointing profile must not be assumed to fit 40GB.
Never put the credential cache or session state (which includes runtime tokens)
in Git or public artifacts.
The upstream README currently supports Linux and macOS; use a Linux
environment such as WSL on Windows, subject to verifying that environment.

## One-time local authentication

With Python 3.12 or newer and `uv` available on the local machine:

```bash
uv tool install google-colab-cli==0.7.4
colab --auth=oauth2 sessions
```

`sessions` is read-only. On first use, the CLI prints Google's consent URL and
asks for the authorization code displayed by Google. Enter that code directly
in the local terminal. Never put authorization codes, refresh tokens, browser
cookies, or passwords in Git, this handoff, or a chat message. The bundled
OAuth client supports a remote consent flow; do not substitute the project's
Drive-only desktop OAuth client, which uses a different registered redirect.

The CLI caches credentials locally. This local login does not authenticate a
different cloud machine. Run the continuing Codex agent on the authenticated
machine unless a separate secure credential handoff is explicitly arranged.
The existing Drive-only OAuth grant in the cloud environment does not establish
Colab authorization. `colab auth` is a VM-side GCP command, not CLI login.

Inspect `colab --help` and the installed `colab skill` before operating sessions;
the installed version's global auth default is `oauth2`. Some upstream skill
text still says `adc`; use the explicit flag above to avoid that discrepancy.
The CLI's `drivemount` command requires human interaction. For unattended
checkpointing, use this repository's existing verified Drive API storage
workflow with independently authorized credentials rather than assuming a
Drive filesystem mount is unattended.

## Context for the continuing Codex agent

The user's latest instruction supersedes the earlier resume plan: restart both
7B family queues from untouched bases on the two A100s, retaining the old
checkpoint as an archive. See [the two-A100 restart](colab_sft_restart.md).

- Repository: `AJUNNYC/inducing_persistent_textual_chain_of_thought_illegibility`.
- Last publication before this handoff: `0d713be02a84f406c0fc820f6d69faf54afbb6dd`.
- Data generation and the certified 7B SFT handoff have already been completed;
  inspect [SFT data handoff](sft_data_handoff.md) and its evidence rather than
  restarting collection or repeating completed scientific suites.
- Archived two-A100 baseline source was `05f47f48f2cc2bbaaff326bb83b3594e5ae973c5`; it does not authorize the current one-epoch pilot.
  Two A100-SXM4-40GB workers run fresh 7B baseline family queues:
  `sft_a100` ROT13 levels 1–10 and `sft_a100_kazi` KaziText levels 1–10.
  Native 7B controls passed on each device; all thirteen launch gates passed.
- The prior PRO 6000 step-987 checkpoint remains archived in Drive and must not
  be used as the parent of these new queues. Its original receipt is historical.
- Both new queues completed one optimizer update and Drive-verified full state,
  then resumed that exact state solely to remove the operator time cutoff.
  The user imposes no wall-clock deadline. The runner's legacy mandatory deadline
  uses datetime.max (`9999-12-31T23:59:59.999999+00:00`) as a sentinel.
  Explicit stop requests and Colab/provider termination remain possible.
- Google Drive is the sole durable checkpoint store. Verify uploaded bytes
  before treating a checkpoint as recoverable or releasing a runtime.
- Both A100s use the eager checkpointed baseline; Blackwell-specific optimized
  kernels are not activated. Qualification covers 7B only, not 14B or optimized
  A100 profiles. No measured A100 throughput forecast is claimed yet.
- See [the proposed worker allocation](sft_multi_gpu_execution_plan.md) and
  [measured optimizations](sft_optimizations.md). The four-worker allocation is
  a proposal, not proof that four sessions can run or that Colab has sufficient
  accelerator quota or balance. Check authenticated account availability before
  spending compute units.
- The user wants concise status updates and action, with changes pushed to
  GitHub before scientific execution. Use GitHub source for scientific commands;
  private connection/authentication glue is permitted. Keep documentation and
  the integrated protocol PDF accurate when the implemented procedure changes.
- Never stop another worker or assign the same condition twice. Preserve the
  current level-1 lineage and immutable per-condition outputs. Provision only
  user-authorized workers, and release owned runtimes after verified recovery
  state is saved.

## Local Codex continuation

Clone or update the repository, start Codex CLI in its root, and give it:

> Read docs/colab_cli_handoff.md and the applicable AGENTS.md instructions.
> Continue Colab automation using the official google-colab-cli and my locally
> authorized account. First verify authenticated session access without creating
> a runtime. Preserve the paused checkpoint and scientific recipe. Prepare and
> qualify the A100-compatible execution profile before launching training, reuse
> completed verification, and save all recoverable state to Google Drive.

This document transfers the project state and constraints; it does not import
the original chat transcript or its private authentication/session state.

## Browser alternative

If editing the Colab web UI itself is necessary, Microsoft's
[Playwright MCP](https://github.com/microsoft/playwright-mcp) supports
`--extension` with its browser extension for an existing Chrome/Edge session.
The MCP server and the extension must run on the user's machine. This can use
an already signed-in tab; the cloud browser does not inherit the desktop login.
Direct Colab CLI runtime control is the primary automation route here.
