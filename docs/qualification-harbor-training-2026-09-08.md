# Qwen3.5-9B SkyRL + Harbor agentic-training qualification

Date: 2026-09-08

## Outcome

The LUMI agentic backend is qualified end to end. Job `21816880` converted real multi-turn Harbor repository
repair trajectories into a SkyRL GRPO batch, completed two full-weight FSDP optimizer steps, synchronized the
updated policy into vLLM, sampled again from the newer policy version, and wrote both distributed restart state
and Hugging Face exports. The machine report passed every gate in
`oellm-harbor-training-qualification-v1`.

This run used the official Qwen3.5-9B checkpoint as a system control. It proves the learner boundary independently
of the current OpenEuroLLM SFT checkpoint's agent-interface weakness. It does not claim that the incoming
OpenEuroLLM checkpoint has passed agentic RL; that checkpoint must first produce its own mixed-reward group and
then pass the same command unchanged apart from `MODEL` and `TASK_GLOB`.

## Evidence

| Measurement | Result |
|---|---:|
| LUMI job | `21816880` (`COMPLETED`, exit `0:0`) |
| Wall time | 22 minutes 57 seconds |
| Allocation | 1 node, 8 MI250 GCDs, 480 GiB host memory |
| Allocated compute | 3.06 GCD-hours |
| Learner | SkyRL v0.3.0, full-weight FSDP on 4 GCDs |
| Sampler | vLLM on 1 GCD, direct lossless Harbor data path |
| Task | `repo-repair-bool`, 8 samples per prompt, 2 turns, temperature 1.0 |
| Harbor trials | 16/16 learner-admissible |
| Optimizer steps | 2 |
| Gradient norms | `[6.535426139831543, 0.0]` |
| Sampler policy versions | `[1, 2]` |
| Weight syncs | 3 (one initial load and two post-update syncs) |
| Exports | `global_step_1/policy/model.safetensors`, `global_step_2/policy/model.safetensors` |

The first batch reward vector, ordered by trial start time, was:

```text
[1, 0, 1, 1, 1, 0, 1, 1]
```

Its mean reward was 0.75 and it produced the non-zero gradient. After the first optimizer step and sync, the
second sampler batch used the next policy version and received:

```text
[1, 1, 1, 1, 1, 1, 1, 1]
```

SkyRL correctly produced zero group-relative advantage and `grad_norm=0.0` for this saturated batch. The change
from 6/8 to 8/8 is encouraging but the sample is too small and non-independent to claim a model-quality gain.
Its operational consequence is clear: the scheduler must remove or re-profile groups that become saturated.

The full artifact is stored at:

```text
/scratch/project_465002530/users/bmoell/oellm-rlvr/harbor-train/21816880
```

The key machine-readable files are `training-qualification.json`, `rollout-qualification.json`,
`campaign-index.jsonl`, `logs/training.log`, the frozen manifests, `checkpoints/`, and `exports/`.

## Why earlier runs had zero gradients

The controls separate three cases:

1. Qwen3.5-9B job `21811157` solved the original four-turn repair 8/8. The system worked, but a constant all-one
   GRPO group has zero relative advantage.
2. OpenEuroLLM job `21816280` attempted four explicit function-calling tasks eight times each. It produced 32
   Harbor results, but all finite rewards were zero, one trajectory was not learner-admissible, and 49 parser
   errors were recorded. That is checkpoint behavior, not evidence of a broken optimizer path.
3. Qwen3.5-9B profile job `21816376` tightened the repair to two turns. All 8 trajectories were learner-admissible,
   with 21 Bash commands, zero parser errors, and reward vector `[1,0,1,0,1,1,0,1]`. This was the mixed group used
   for the training qualification.

The zero gradients were therefore caused by task/model mismatch: all-zero groups were too difficult and all-one
groups were saturated. The fixed workflow profiles the current policy first and trains only mixed groups.

## Reproduction

Run the learner-off profile first:

```bash
sbatch --export='ALL,MODEL=/scratch/project_465002530/users/bmoell/models/Qwen3.5-9B,TASK_GLOB=repo-repair-bool,N_SAMPLES_PER_PROMPT=8,TEMPERATURE=1.0,MAX_TURNS=2,MAX_TOKENS_PER_TURN=512' \
  scripts/lumi_harbor_agentic_rollout.sbatch
```

After its strict report shows both reward 0 and reward 1, run:

```bash
sbatch --export='ALL,MODEL=/scratch/project_465002530/users/bmoell/models/Qwen3.5-9B,TASK_GLOB=repo-repair-bool,N_SAMPLES_PER_PROMPT=8,TEMPERATURE=1.0,MAX_TURNS=2,MAX_TOKENS_PER_TURN=512,TRAIN_STEPS=2,POLICY_GPUS=4,NUM_ENGINES=1,TOTAL_GPUS=8' \
  scripts/lumi_harbor_agentic_train.sbatch
```

The job exits non-zero unless every raw rollout is learner-admissible and the training report observes mixed
rewards, completed optimizer steps, finite metrics, at least one non-zero gradient, post-update weight sync,
a newer sampler policy version, a clean training shutdown, and a restartable HF export.

## Next OpenEuroLLM gate

Do not reuse Qwen's difficulty label blindly. For the incoming SFT/DPO checkpoint:

1. run eight samples per candidate task;
2. route 0/8 tasks to interface SFT or an easier curriculum, exclude 8/8 tasks from GRPO, and admit 1-7/8 tasks;
3. run this same two-step canary on one admitted group;
4. require a non-zero first gradient, post-update sampler version, replayable reward, restart state, and held-out
   agentic evaluation before scaling;
5. re-profile after every short window because this qualification demonstrated that a single update can move a
   small task group into saturation.
