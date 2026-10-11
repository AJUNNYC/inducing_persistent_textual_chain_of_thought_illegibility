# Docker and RunPod lifecycle

Build on Linux x86_64 with Docker. The CUDA12.8.1/Ubuntu24.04 base is pinned by an actual inspected platform manifest digest. The locked PyTorch/NVIDIA packages supply CUDA runtime and cuDNN libraries. The scientific image installs the hashed GPU lock, Aspell and English dictionaries, Python3.12 and source. It excludes runtime data, secrets, caches and Git internals.

```bash
export SOURCE_CODE_COMMIT="$(git rev-parse HEAD)"
docker build --build-arg SOURCE_CODE_COMMIT="$SOURCE_CODE_COMMIT" \
  -f dockerfile -t YOUR_AUTHORIZED_REGISTRY/illegibility:"$SOURCE_CODE_COMMIT" .
docker run --rm YOUR_AUTHORIZED_REGISTRY/illegibility:"$SOURCE_CODE_COMMIT" hardware
docker push YOUR_AUTHORIZED_REGISTRY/illegibility:"$SOURCE_CODE_COMMIT"
```

Use the registry's resulting immutable image digest in an approved copy of `deploy/runpod/pod.json`. The template requests one80GB A100/H100-class GPU,64 GB RAM,8 vCPU,250 GB workspace target and30GB container disk. These are planning targets; the template is an optional API deployment route, not the layout used by the completed generation runs. vLLM generation throughput and memory controls were observed; GRPO capacity and full-study runtime remain unverified. No claim of measured GPU capacity or complete-experiment runtime is made. The observed development host has CPU-only PyTorch and no NVIDIA GPU.

For the active generation procedure, RunPod holds disposable working copies and model caches. Google Drive is the permanent store for required experimental data and artifacts; protected credentials remain outside source and artifact roots. Use separate CUDA devices/Pods for the ROT13, KaziText and RL queues when authorized; each SFT queue has20independent fits. Set `CUDA_VISIBLE_DEVICES` consistently:the first actual device identity supplies the process lock. Stages sharing that device serialize. RL runs each selected model's100updates and retains all ten scientific snapshots. Stop billing for an exhausted queue rather than wait for others.

MBPP requires the hardened Docker executor with the pinned Python image. A RunPod Pod does not automatically supply a usable nested Docker daemon. Before launch, verify an authorized Docker-capable sandbox host. An external CPU Linux VM may supply a Docker context over SSH, with host-key verification and the Docker CLI installed in the controller. Use `DOCKER_HOST=ssh://AUTHORIZED_USER@AUTHORIZED_SANDBOX_HOST`; the candidate containers receive no host socket, credentials or network. Run the real sandbox probes and all original reference tests through that context. Missing executor capability blocks deployment readiness. Do not disable isolation to make a Pod pass.

```bash
.venv/bin/illegibility runpod-deploy \
  --payload /workspace/approved_pod.json --authorization /workspace/approved_deployment.json
.venv/bin/illegibility runpod-status --pod-id ACTUAL_POD_ID
python scripts/manage_runpod.py watch --pod-id ACTUAL_POD_ID \
  --authorization /workspace/approved_lifecycle.json --state /workspace/controller/pod_state.json
```

Deployment authorization must have `approved:true`, a nonempty authorization ID, the exact `approved_payload`, positive `max_total_usd` and `max_runtime_seconds`. Pod creation is never blindly retried after an ambiguous timeout; inspect the account for the possible already-created billable Pod. Never put credentials into tracked Pod templates.

The external lifecycle controller additionally needs the selected `pod_id`,`authorized_hourly_usd`,`price_evidence` and actual `billing_started_at_unix`. It persists its start across restarts, checks the provider's observed rate and stops before the next interval can exceed its runtime/cost bound. Missing or higher price evidence also stops the Pod. Run this controller on a durable host outside the GPU Pod. If the controller host disappears, its Python process cannot guarantee termination; retain an independent provider-console billing check.

Use an authorized SSH or Jupyter connection. For API-authorized deployments, configure protected credentials and disposable working roots, then run:

```bash
python scripts/freeze_runtime_config.py --output /workspace/runtime_config.yaml --resolve-native-assets
illegibility --config /workspace/runtime_config.yaml provision --include-weights
illegibility drive-init
illegibility preflight --execute-live
illegibility run --target rl_7b
illegibility run --target rot13_7b_level_01
illegibility run --target sync
```

Confirm exact target IDs with `illegibility plan`; a full run contains62main evaluated states. Paid execution checks source/dependency identities, funding, Drive replication, baseline repeatability, statistical validation and all live requirements. Failed mandatory scientific preflights stop launch. On pause, finish a safe logical boundary, save complete optimizer/scheduler/RNG/progress state, replicate required artifacts and stop billing:

```bash
python scripts/manage_runpod.py stop --pod-id ACTUAL_POD_ID
python scripts/manage_runpod.py resume --pod-id ACTUAL_POD_ID --authorization /workspace/approved_resume.json
```

Resume authorizes that selected billable Pod and restores only verified compatible complete checkpoints; adapter-only exports are inference artifacts. Final `sync` verifies remote integrity before discarding scratch data. The active procedure relies on verified Google Drive artifacts; validate actual complete-state restoration before claiming interrupted training can resume. Both H100 Pods completed fresh vLLM generation and exact-deficit refills. All final datasets are Drive-verified, including a subsequent full byte readback without Pod connections. Training and evaluation remain unrun and subject to their remaining gates. The current PI account cannot use RunPod API keys: operate the existing Pods through authenticated Jupyter and stop billing through the provider console. The API lifecycle commands above describe an optional route for accounts with that permission.


The user initially authorized the existing 7B H100 up to $90 and 30 hours, then separately authorized generation on the supplied 14B Pod. Additional Pod creation still requires authorization. Account for its actual billing start and prior cost before allocating the remaining budget. Native clean generation does not require judge calls. Later phases must reserve provider costs within the same total rather than assigning the full ceiling independently to GPU time and API calls. Missing actual price/billing evidence blocks a claim that the controller enforces the combined authorization.

For a graceful clean-generation pause, create `data/clean_traces/STOP_REQUESTED` on persistent storage. The generator saves and replicates completed outputs at its next boundary, records exact resume progress and returns `PAUSED`; downstream transformations/training remain unscheduled. After verified final synchronization and provider stop confirmation, a later continuation removes the request and reuses the same immutable cached responses. A separate host must enforce the Pod budget even if the worker exits.

## Current procedure: 11 October 2026

The [current procedure](../procedure_update.md) records the user-approved fresh vLLM SFT generation on separate Pods, 128-request chunks with a full 3072-token allowance per request, Drive as permanent storage, archived old responses, and the verified shared RL manifest. Final pair acceptance, exact-deficit refills, all 40 transformed conditions and Drive copies are complete. Training and actual evaluations remain pending. The generation backend change does not alter GRPO rollout or native evaluation settings or waive failed statistical gates.
