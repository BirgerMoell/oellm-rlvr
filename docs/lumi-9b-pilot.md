# OELLM 9B RLVR pilot on LUMI

This pilot answers one narrow question: can the current LUMI stack sustain a real 9B online RL loop with
eight rollout engines, useful grouped reward variation, nonzero optimizer updates, live weight transfer,
and restartable state? Ten updates are enough to expose system failures and measure throughput. They are not
enough to claim a model-quality improvement.

## Fixed inputs

- Model: `openeurollm/oellm-9b-256k-sft`, staged at the path in the config. Replace both model fields when
  qualifying a newer SFT/DPO checkpoint.
- Backend: `OpenEuroLLM/tmax-reproduction` at
  `3f80d37042402b8363f39c9535723b0d4cb8de54`.
- Math source: `birgermoell/oellm-math-rlvr` revision
  `0ffc9d6dc82717c25733b3172f4dbd63e48bab68`.
- Pilot subset: 256 distinct English semantic groups, difficulty 1–3, seed `20260904`. The staged Parquet has
  SHA-256 `ec33cc0fcbd6be73d87ed8f0582683cc6f2e4e56297ffa7db92f3712a5ec3aff`; it contains 55/92/109 rows at
  difficulties 1/2/3, 146 integer and 110 rational verifiers, and 22 subdomains.
- Compute: two LUMI-G nodes, 8 learner GCDs plus 8 TP=1 rollout GCDs. The two-hour limit is a safety margin;
  billing is for actual runtime. The maximum allocation is 4 node-hours or 32 GCD-hours.

Recreate the subset on a LUMI login node with the LAIF container and control venv active:

```bash
oellm-rlvr sample-math \
  --source /scratch/project_465002530/users/bmoell/oellm-rlvr/data/sources/oellm-math-rlvr-0ffc9d6c.parquet \
  --output /scratch/project_465002530/users/bmoell/oellm-rlvr/data/math-pilot-en-d1-3-seed20260904-n256.parquet \
  --count 256 --language en --min-difficulty 1 --max-difficulty 3 --seed 20260904
```

## Run sequence

First require the bounded hierarchical 9B qualification to finish: eight vLLM engines must initialize, two
learner updates must complete, and both the initial and post-update hierarchical weight transfers must be
visible in the log. Do not spend the pilot allocation if that job fails.

Then use a clean, revision-pinned control checkout on LUMI:

```bash
CONFIG=configs/lumi-math-oellm9b-256k-sft-pilot-2node.yaml
oellm-rlvr validate --config "$CONFIG"
oellm-rlvr topology --config "$CONFIG"
oellm-rlvr doctor --config "$CONFIG"
oellm-rlvr render-slurm --config "$CONFIG" --output oellm9b-pilot.sbatch
sbatch oellm9b-pilot.sbatch
```

The run performs 10 updates of 64 trajectories each (8 prompts × 8 samples), or 640 accepted trajectories.
Active sampling discards all-equal prompt groups and records the filtered groups. It writes a complete model
at step 10 and DeepSpeed restart state at steps 5 and 10. Re-submitting the unchanged config resumes the
state directory; choose new output paths to start over intentionally.

The prerequisite qualification, LUMI job `21734954`, completed two updates in 14m52s including cold startup
and final model save; its post-initialization learner steps took 38.73s and 8.57s. The pilot's two-hour limit
is deliberately conservative because broader prompts, active resampling, and two restart checkpoints can be
slower. Treat the first pilot as a measurement run and keep the walltime unchanged until its timing breakdown
is archived.

## Go/no-go decision

Promote this stack to a longer run only if all of the following are true:

1. The job exits `COMPLETED 0:0`; all eight engines initialize on one rollout node and the learner occupies
   the other node.
2. Initial weight sync and every scheduled post-update sync finish. Rollout model versions advance and the
   maximum observed policy lag is at most four.
3. All 10 optimizer updates finish with finite gradients, and at least 8 have a strictly positive gradient
   norm. There are both reward-0 and reward-1 samples within accepted prompt groups.
4. Across saved rollout shards, zero-standard-deviation groups are at most 80%, truncated completions at most
   15%, and verifier/infrastructure errors at most 2%. Report task failures separately from infrastructure
   errors.
5. The step-10 model and step-10 restart state are readable. Submit one additional 64-trajectory update from
   that state and require it to begin at step 11, proving recovery rather than only checkpoint writing.
6. Record total runtime plus generation, verification, learner, checkpoint, and weight-sync timings. If 10
   updates exceed two hours or GPU utilization is dominated by serialization/sync, profile before scaling.

For every generated JSONL rollout shard, run:

```bash
oellm-rlvr inspect-rollouts --rollouts /path/to/rollouts_000000.jsonl
```

Archive the config, rendered Slurm script, stdout/stderr, W&B offline directory, dataset hash, backend/control
commits, checkpoint manifest, and the inspection summaries together. A successful pilot qualifies math RLVR
infrastructure. Code RLVR remains a separate promotion gate because sandbox startup, multi-turn tool parsing,
and test execution have different latency and failure modes.

## After the math pilot

Run a smaller code qualification before mixing tasks: 64–128 easy Python tasks, eight samples per prompt,
three to five updates, the same 8+8 GPU topology, and the prepared or Slurm Apptainer backend already proven
by the code smoke. Require hidden tests to remain absent from prompts/traces, environment errors below 2%,
at least one mixed-reward group, nonzero gradients, and a successful restart. Only then design a joint
math/code curriculum; keep per-domain rewards and sampling quotas separate so cheap math rollouts cannot hide
code-sandbox failures.
