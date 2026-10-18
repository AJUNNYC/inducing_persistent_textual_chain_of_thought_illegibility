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

## Additional levels on one GPU

`scripts/run_sft_gpqa_queue.py --configs LEVEL3_CONFIG LEVEL7_CONFIG --output QUEUE_OUTPUT`
serializes each selected level's published continuation and the same frozen GPQA
dispatcher. The October 18 extension selects only 14B ROT13 levels 3 and 7, each
from its own original one-epoch parent, with one additional epoch, peak LR 5e-5,
restored optimizer state and gradient checkpointing, matching the level-5 pilot.
Order is level 3 training, level 3 evaluation, level 7 training, level 7 evaluation.
Each receives the original 198-question Diamond panel and 32,000 output tokens;
the existing baseline is reused, and all prior results remain in separate paths.
The queue advances only after native training endpoints and evaluation completion
receipts are Drive-verified. It stops on failure, rather than skipping a level.
`PAUSE_REQUESTED` in the queue directory or SIGTERM requests a checkpoint-safe
training pause, or stops the evaluation supervisor. A restart runs the same
idempotent paths; never run another GPU job concurrently on the same device.
No new training recipe, score parser or inference settings are introduced.
