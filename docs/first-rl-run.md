# Run an RL experiment with oellm-rlvr

This is the operator entry point. Start with **one small math run**, verify that the whole loop works, then
decide whether to spend more GPU time. A YAML file is a run proposal, not an already-running job. `validate`,
`topology`, and `render-slurm` do not allocate GPUs; `sbatch` does.

## What you provide and what happens

You provide (1) a local Hugging Face-format model checkpoint and tokenizer, (2) a prepared Parquet task pool
with prompts and ground-truth answers, and (3) a YAML config that names the LUMI container, pinned training
backend, GPU layout, learning settings, and output paths. The [math dataset contract](data-and-verifiers.md)
describes the task fields and reward check. Model and data must already be on shared LUMI storage: compute
nodes run offline. Use different semantic problems for training and evaluation, including across translations.

The job reserves GPUs through Slurm, starts learner and vLLM rollout workers, samples several answers per
prompt, verifies each answer, computes a grouped training signal, updates the weights, and syncs the updated
weights to the rollout workers. It writes logs, rollout records, exported model checkpoints, and (when enabled)
restartable learner state. `oellm-rlvr` is the configuration, launch, verification, and monitoring layer;
the pinned TMAX/Open-Instruct backend performs the math learner update. SkyRL + Harbor is a separate agentic
path; verl is optional for on-policy distillation. You do **not** need those paths for a first math run.

## The shortest safe path on LUMI

1. **Choose one checkpoint and task pool.** Use an existing, tested configuration as a template, not as an
   untouched production command. For the current OpenEuroLLM 9B math experiment, the bounded one-update
   example is `configs/lumi-grpo-math-oellm9b-promotion-loadfix-1step.yaml`; the proposed 32-update continuation
   is `configs/lumi-grpo-math-oellm9b-promotion-phase-a-32step.yaml`. The latter is **not yet qualified as a
   completed 32-update run**.
2. **Create a new YAML under your run directory.** Change `name`, `model.local_path`, `model.revision`, every
   `datasets[].path`, `output.directory`, `output.rollout_directory`, `output.experiment_name`, and
   `training.checkpoint_state_directory` to immutable inputs and *unused* output locations. Verify the model
   and data exist, and pin `backend.commit`, `platform.container`, and the control-repo commit. Do not reuse
   an earlier attempt's output/state directory or its generated Slurm script. For a large 9B checkpoint,
   keep `model.stage_to_local: true` and a finite learner-startup timeout.
3. **Validate without spending GPUs** from the checkout and environment used on LUMI:

   ```bash
   oellm-rlvr validate --config /path/to/my-run.yaml
   oellm-rlvr topology --config /path/to/my-run.yaml
   oellm-rlvr render-slurm --config /path/to/my-run.yaml --output /path/to/my-run.sbatch
   ```

   Read the reported learner/rollout GPU split and inspect the rendered script, container, account, walltime,
   paths, and pinned backend. Save the YAML, script, and their checksums alongside the run. The
   [LUMI runbook](lumi.md) covers environment bootstrap and preflight details. Do not `uv sync` the CUDA
   backend environment over LUMI's ROCm image.
4. **Submit deliberately** from a LUMI login node: `sbatch /path/to/my-run.sbatch`. Record the job ID and
   inspect it with `squeue -j JOB_ID`, `sacct -j JOB_ID`, and the Slurm output file. A submitted or
   `COMPLETED 0:0` job alone does not prove learning.
5. **Check the evidence.** Require initialized learner and rollout workers, completed trajectories, mixed
   reward outcomes within at least some prompt groups, finite nonzero gradients, a post-update weight sync,
   and a readable checkpoint/restart state. Inspect rollout files with
   `oellm-rlvr inspect-rollouts --rollouts /path/to/rollouts_000000.jsonl`. The exact rollout filename may
   differ; use the path logged by the backend. Zero reward variance, widespread truncation, verifier errors,
   or stale rollout weights mean the run needs diagnosis before scaling. The [LUMI runbook](lumi.md#6-monitoring-and-promotion)
   lists the health metrics and recovery rules.
6. **Evaluate before continuing.** Compare the saved model with its starting checkpoint on the *same frozen,
   unseen* evaluation prompts, using a separately recorded dataset and evaluator revision. Look at overall
   accuracy, each language, response form, length/truncation, and regressions—not only online training reward.
   The [current math phase plan](oellm9b-math-promotion-2026-09-28.md) specifies the frozen evaluation and
   the conditions for a harder second phase. Do not start phase B merely because phase A finishes.

The sample config contains project-specific absolute LUMI paths. New users need their own account, staged
container/backend/checkpoint/data, and writable project-scratch paths. A local `pytest` or a rendered script
does **not** demonstrate that those cluster dependencies work.

## Completed runs versus plans

The [experiment history](experiment-history.md) separates completed, failed, and proposed work and links to
the dated evidence. [Campaigns](../campaigns/) and plans describe intended sequences; they are not evidence
that every stage has run. The next proposed multilingual math experiment is described in the
[phase plan](oellm9b-math-promotion-2026-09-28.md).

## Terms used in this repo

| Term | Plain meaning |
|---|---|
| Rollout | A model response generated during the training job, together with the information needed to verify and learn from it. |
| RLVR | Reinforcement learning from verifiable rewards: a deterministic checker, such as exact math answers or hidden code tests, supplies the reward. |
| Canary / reasoning canary | A deliberately small reasoning-training run that tests whether the full loop works and whether an early checkpoint improves on held-out questions. It is not a production run. |
| Promotion gate | A predeclared go/no-go check before using a checkpoint as the starting point for the next stage. It includes frozen evaluation and regression checks, not just training reward. “Promoted” means selected for the next experiment, not publicly released. |
| Bounded interface bridge | An optional, explicitly limited SFT step to teach a model the desired response format (for example, balanced `&lt;think&gt;` tags and in-language reasoning) when its current responses provide too little useful RL signal. It is not an RL run or an assumption that all translated traces are good training data. |
| Mixed-reward group | Several answers to the same prompt include both successes and failures. Group-relative RL can learn from this contrast; all-equal rewards often yield little or no gradient. |
| Verifier | Code that checks an answer or executes hidden tests and returns a reward. An infrastructure/verifier failure must be counted separately from a valid wrong answer. |
| Checkpoint / restart state | The exported model is for inference or the next phase; restart state additionally includes optimizer and training state needed to resume an interrupted phase. |
| GCD | One MI250X GPU compute die. LUMI accounting and our GPU-hour estimates count GCDs, not dual-die MI250X packages. |

For implementation details see [architecture](architecture.md); for math/code task fields and sandbox boundaries
see [data and verifiers](data-and-verifiers.md). Start with this guide when the goal is simply to train a model.
