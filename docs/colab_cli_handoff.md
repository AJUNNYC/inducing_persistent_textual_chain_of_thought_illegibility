# Colab CLI automation handoff

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
PyTorch 2.11.0+cu130, and CUDA 13.0 in the stock runtime. Training has not
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
- The accepted production source is
  `a6e3295bfeca2c213360ae6d772dbfec6a08017e`, specifically for 7B ROT13 level 1
  on RTX PRO 6000 Blackwell 96GB. The later memory-copy optimization was
  rejected and must not be activated.
- Training is **PAUSED at optimizer step 987/1250**. The complete state is
  verified in Google Drive. There are **263 updates plus final validation**
  remaining. See [the pause receipt](evidence/sft_pro6000_session_pause_987.json).
- Full-checkpoint Drive folder:
  `142UamKwaFNprt3DbsXP1F0YNR3iizBU5`; full-state file:
  `1aOXFsl8kpXbDGbkMps9UKpS36gILXzvr`, SHA-256
  `c0e84a69de64b6c2cd113b601f7ccee2fa25ba84ac1a0d2cbcbe9b29e15a3231`.
- Preserve the adapter, optimizer, scheduler, RNG, sampler, exposure order, and
  scientific condition identity when resuming. The recorded deadline is
  `2026-10-12T02:24:01.558671+00:00`; if it has elapsed, obtain a newly authorized
  deadline rather than silently extending it.
- Google Drive is the sole durable checkpoint store. Verify uploaded bytes
  before treating a checkpoint as recoverable or releasing a runtime.
- An A100 runtime is allocated and command access is verified, but no
  additional training workers or A100 training profiles have been deployed or measured. The fast Triton helpers were qualified for SM120 with Triton 3.5.0;
  do not assume those helpers work on an A100. Qualify a compatible execution
  profile while preserving the scientific recipe, starting with changes least
  likely to alter results. Reuse completed checks and run only affected checks.
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
