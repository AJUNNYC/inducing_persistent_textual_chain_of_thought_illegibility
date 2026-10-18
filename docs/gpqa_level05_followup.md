# Automatic GPQA Diamond after the level-5 continuation

The dispatcher `scripts/watch_gpqa_continuation.py` waits for the level-5 training
watcher's `COMPLETE_DRIVE_VERIFIED` state at 625 additional updates and a ready,
separate inference environment. It then reruns the published training campaign's
idempotent completion gate, including full-state and live Drive verification,
before selecting its exact final adapter. It never launches another training level.

Evaluation retains source `5e6be1b0848e0e88725323591baddef37d69d8ae`, vLLM 0.11.0,
the original 198-question GPQA Diamond panel, choice permutations, prompts, seeds,
scorer and `long_32000` generation contract. The 32,000-token output allowance and
40,960-token context are unchanged. All 198 original baseline raw records are
restored using their sealed Drive inventory and full SHA-256 hashes, validated
against the frozen request identities, and republished. No new baseline inference
is performed. Only the completed two-epoch 14B ROT13 level-5 adapter generates
new answers. The original one-epoch results remain intact.

Raw answers and scores use the existing synchronous Drive artifact hooks. After
evaluation, the native completion barrier checks all question/score coverage,
protocol, report hashes and the verified Drive inventory. The dispatcher also
uploads the native completion receipt and the handoff configuration, operation
script and frozen source archive. Hardware observations are recorded per attempt.
The final dispatcher state is `COMPLETE_DRIVE_VERIFIED`, not merely process exit.
Failed child phases have a bounded three-recovery allowance; scientific settings
are never relaxed on failure. A stopped or failed dispatcher needs inspection.

## Moving to another GPU

Create `PAUSE_REQUESTED` in the evaluation output root and wait for dispatcher
state `PAUSED`. This stops its owned subprocess group and prevents automatic
restart. Completed and Drive-verified responses are reusable; an in-flight batch
may need regeneration. Export and verify the live artifact registry before stopping
the old pod. Restore the exact adapter, input protocol, baseline/raw/score files,
registry and handoff on the replacement pod using their checksum records.

Keep the same absolute model-cache, adapter and output paths, frozen source,
inference dependency versions, prompts, seeds and token budgets. Provision a GPU
with sufficient memory for the existing context contract; do not silently reduce
the output allowance to fit. After the old worker is confirmed stopped and restored
files are verified, remove `PAUSE_REQUESTED` on the new pod and run the same dispatcher
with its saved configuration. Cached answers are validated and reused, so unanswered
questions are the remaining inference work. Hardware changes can introduce numerical
differences; they are recorded, not represented as bitwise-identical execution.

The supervisor does not stop RunPod billing. Never terminate a pod merely because
all GPU inference appears finished; wait for the verified upload barrier.
