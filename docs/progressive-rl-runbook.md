# Progressive OpenEuroLLM 9B RL runbook

Date: 2026-09-08

## What is working now

It is fair to say that the `oellm-rlvr` **RL system works on LUMI**, provided the claim is kept narrower than
"every target domain is production-ready." The evidence is:

| Capability | Status | Evidence | Remaining gate |
|---|---|---|---|
| Reasoning RL on OELLM 9B | Qualified | Job `21750021`: 32/32 non-zero-gradient updates, 2,048 rollouts, synchronized weights, resumable exports, and a positive paired GSM8K diagnostic | Repeat on the incoming frozen checkpoint and use clean multi-domain selection evals |
| Math RL on OELLM 9B | Qualified | Jobs `21540106` and `21734954`: mixed groups, non-zero gradients, complete saves, and hierarchical learner-to-eight-engine weight sync | Run a broader, re-profiled curriculum and held-out multi-subdomain evaluation |
| Executable-code RL | Backend qualified; OELLM 9B not qualified | Qwen3.5-2B job `21536745` produced mixed rewards, `grad_norm=0.21`, hidden-test execution, and weight sync | Profile and update the actual incoming 9B checkpoint, then pass held-out code and retention gates |
| Agentic rollouts | Qualified | Qwen3.5-9B job `21811157`: 8/8 learner-admissible Harbor trajectories, 27 commands, no parser errors, exact token IDs/log-probs | Build mixed-reward tasks and connect them to one SkyRL optimizer step |
| Agentic RL training | Not yet qualified | The learner-off data plane and Harbor/Singularity contracts pass | Qualify Qwen3.5's SkyRL learner boundary, then complete one synchronous OELLM 9B update and weight sync |

The current system has therefore passed real rollout, verifier, advantage, gradient, optimizer, checkpoint, and
weight-synchronization boundaries. The unresolved work is concentrated in actual-checkpoint code qualification
and the SkyRL + Harbor learner connection, not in basic reasoning or math RL.

## The checkpoint sequence

Use one explicit selection boundary after every training domain:

```text
frozen parent
  -> reasoning candidate -> reasoning selected (or parent)
  -> math candidate      -> math selected (or reasoning selected)
  -> code candidate      -> code selected (or math selected)
  -> agentic candidate   -> final selected (or code selected)
```

"Selected" is important. A later stage must start from the candidate only when paired held-out evaluation and
all retention gates pass. Otherwise it starts from the previous selected checkpoint. This prevents a weak code or
agentic mini-stage from erasing a useful reasoning/math checkpoint.

The machine-readable version is
[`campaigns/lumi-oellm9b-progressive-rl.yaml`](../campaigns/lumi-oellm9b-progressive-rl.yaml). Validate and render
it with:

```bash
oellm-rlvr validate-campaign \
  --campaign campaigns/lumi-oellm9b-progressive-rl.yaml
oellm-rlvr render-campaign \
  --campaign campaigns/lumi-oellm9b-progressive-rl.yaml \
  --output progressive-campaign.md
```

## Difficulty is measured, not assumed

For every new checkpoint and every domain, generate eight samples per prompt before training. Assign bins from
the **current checkpoint's** verifier pass rate:

- easy: 5-7 successes out of 8;
- medium: 2-4 successes out of 8;
- hard: 1 success out of 8;
- saturated: 8/8, retained for evaluation or replay checks but excluded from group-relative updates;
- currently impossible: 0/8, routed to data repair, an SFT bridge, or a later checkpoint rather than repeatedly
  resampled during RL.

This is why the Qwen3.5-9B all-one agentic group proved the pipeline but was not a trainable GRPO group. A source
label such as difficulty 3 is only metadata; the observed within-prompt reward distribution decides admission.
Re-profile at each curriculum-window boundary because learning moves tasks between bins.

Materialize the bins from a standardized profile with:

```bash
oellm-rlvr build-curriculum-pools \
  --profile <profile-dir>/task-profile.parquet \
  --output <profile-dir>/curriculum-pools.json
```

The output keeps infrastructure failures separate from model failures and lists task IDs by domain and bin. It
does not silently admit saturated or currently impossible prompts into RL.

## Qwen's role

Use the official `Qwen/Qwen3.5-9B` checkpoint before expensive OELLM stages to find broken prompts, verifier
mistakes, leaked tests, invalid tool schemas, insufficient turn budgets, and ROCm/runtime regressions. It is a
control, not a teacher or a curriculum oracle:

- Qwen must never determine the OELLM difficulty bin;
- Qwen scores must never select an OELLM checkpoint;
- OELLM receives its own eight-sample profile after the task contract passes;
- Qwen should make one SkyRL + Harbor optimizer step first because its agent behavior is already positive. That
  isolates learner integration from OELLM's current terminal-policy weakness.

## Progressive mixtures

Each domain has warmup, core, and challenge windows. Warmup is easy-heavy. Core emphasizes medium tasks.
Challenge adds hard tasks while preserving easy examples and replay from earlier verified domains. The exact
fractions are validated in the campaign YAML and must sum to one both across windows and inside every mixture.

The guiding shape is:

| Stage | Warmup | Core | Challenge | Retention source |
|---|---|---|---|---|
| Reasoning | 65% easy / 30% medium / 5% hard | 30 / 50 / 20 | 20 / 40 / 40 | Easier reasoning remains in every window |
| Math | 50% easy math plus 25% reasoning replay | 40% medium math plus 20% reasoning replay | 25% hard math plus 25% reasoning replay | Selected reasoning tasks |
| Code | 55% easy code plus 25% prior replay | 40% medium code plus 20% prior replay | 25% hard code plus 25% prior replay | Selected reasoning and math tasks |
| Agentic | Structured actions and short functions dominate | Functions, state, terminal, and small repo repair | Longer terminal/repo/recovery tasks | Verified code and math tasks |

Never normalize raw reward across incomparable environments. Form prompt groups within one task/verifier domain,
compute group-relative advantages there, and enforce explicit per-domain quotas at the batch scheduler.

## Binding the incoming checkpoint

Run configurations no longer need hand-edited checkpoint and output paths. Materialize each validated template
into a campaign-specific configuration:

```bash
oellm-rlvr materialize-config \
  --template configs/lumi-reasoning-gsm8k-oellm9b-32step.yaml \
  --run-name oellm9b-reasoning-01 \
  --model-id <repository@revision> \
  --model-path <frozen-local-checkpoint> \
  --dataset <profiled-reasoning-pool> \
  --output-root <campaign-root> \
  --output <campaign-root>/configs/reasoning.yaml
```

The command validates the resulting topology and isolates model outputs, rollout artifacts, and restart state
under the new run name. Repeat it with the math and code templates, passing the preceding **selected** checkpoint.

## Execution order and stop rules

1. Freeze and hash the incoming SFT/DPO checkpoint.
2. Run Qwen learner-off controls over all proposed task contracts.
3. Profile the frozen OELLM parent and materialize its easy/medium/hard pools.
4. Train and select reasoning.
5. Re-profile, train, and select math while replaying reasoning.
6. Re-profile, train, and select executable code while replaying reasoning/math.
7. Profile agentic tasks with Qwen and the selected OELLM code checkpoint.
8. Complete one Qwen SkyRL + Harbor optimizer step and post-update rollout.
9. Complete one synchronous OELLM SkyRL + Harbor update; only then scale the agentic curriculum.
10. Run the all-domain paired scorecard and replay all deterministic rewards before promotion.

Stop a stage rather than spending through it when any of these occur:

- fewer than 60% of admitted prompt groups have mixed rewards;
- more than 5% of updates have zero, non-finite, or missing gradients;
- verifier or environment errors reach 2%;
- deterministic rewards do not replay;
- policy lag exceeds four;
- the selected prior-domain score regresses beyond its frozen tolerance;
- hidden verifier material appears in model-visible context.

The full dry campaign is budgeted at 468 expected GCD-hours with a 1,176 GCD-hour sum of stage ceilings, inside
a 1,200 GCD-hour campaign ceiling. These are maximum planning boundaries, not a commitment to spend through a
failed gate. The initial checkpoint freeze, Qwen controls, OELLM profiling, and reasoning repeat are the first
launch tranche; later allocations depend on the preceding selection reports.
