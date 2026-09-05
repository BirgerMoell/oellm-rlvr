# OELLM 9B GSM8K reasoning-RL qualification

Date: 2026-09-05

## Result

The 32-update reasoning-RL reference experiment completed successfully on LUMI. It proves that the current
`oellm-rlvr` control plane can perform online RL on the real OELLM 9B SFT checkpoint, synchronize the learner
to eight vLLM rollout engines, preserve useful grouped reward variation, export resumable checkpoints, and
produce a statistically positive paired change on the locked primary diagnostic.

The selected step-32 checkpoint improved exact GSM8K answer accuracy from **40.80% to 44.46%** on 1,255
paired prompts: **+3.67 percentage points**, 95% paired-bootstrap interval **[+0.96, +6.37]**, exact McNemar
**p=0.00998**. This is an optimization and infrastructure result, not a clean generalization claim: the parent
SFT mixture contains a 49,498-row GSM8K-derived component and does not document a final whole-mixture
decontamination pass.

## Frozen inputs

| Item | Frozen value |
|---|---|
| Logical parent | `openeurollm/oellm-9b-256k-sft` |
| Staged parent origin | `birgermoell/oellm-9b-256k-sft` |
| Parent revision | `08359ad61333263c067edaf290067fea5b103d34` |
| Parent manifest SHA-256 | `717c07f19caecbeb518f2385a79ecdfc748ef824b5c47a8fbcb1b0b31610c4fe` |
| GSM8K revision | `740312add88f781978c0658806c59bc2815b9866` |
| Training rows | 7,473 official `train` rows |
| Prompt calibration | 64 seeded official `test` rows |
| Primary comparison | remaining 1,255 official `test` rows |
| Control-plane commit | `c597cdd776dc82dc0d26d50c8e5f4b4f889fad95` |
| Training backend commit | `3f80d37042402b8363f39c9535723b0d4cb8de54` |

The RL parquet contains prompts and ground truth only. It does not contain assistant traces or GSM8K reference
rationales. Calibration and primary rows never entered training, and the preparation manifest reports zero
train/test and calibration/primary overlap.

## Locked behavior and training protocol

The prompt requests one balanced `<think>...</think>` block with a concise, checkable derivation, followed by
one boxed numeric answer. The direct reward is numeric correctness; tags and style are measured as guardrails,
not substituted for solving the problem.

The run used DPPO with learning rate `1e-6`, eight prompts and eight samples per prompt, active sampling,
32 updates, 2,048 accepted trajectories, a 1,024-token response limit, eight learner GCDs, and eight TP=1 vLLM
rollout GCDs. Restart state was written every eight updates and full Hugging Face exports at updates 16 and 32.

## LUMI jobs and compute

| Job | Purpose | Allocation and runtime | GCD-hours |
|---|---|---:|---:|
| `21742994` | corrected 1,024-token calibration | 1 GCD, 2m50s | 0.047 |
| `21743067` | corrected 2,048-token calibration | 1 GCD, 3m06s | 0.052 |
| `21749008` | parent primary evaluation | 1 GCD, 12m43s | 0.212 |
| `21750021` | 32-update online RL | 16 GCDs, 34m53s | 9.302 |
| `21755329` | step-16 primary evaluation | 1 GCD, 12m33s | 0.209 |
| `21755293` | step-32 primary evaluation | 1 GCD, 12m37s | 0.210 |
| | **Locked protocol total** | | **10.03** |

Including five rejected prompt calibrations and two mismatched jobs canceled after 26 seconds, the complete
exploratory campaign used approximately **10.4 GCD-hours**. The training job's reservation ceiling was 96
GCD-hours; billing followed the 9.30 GCD-hours actually used.

## Training health

| Metric | Result |
|---|---:|
| Completed updates | 32 / 32 |
| Updates with non-zero gradient | 32 / 32 |
| Mean gradient norm | 0.174 |
| Accepted trajectories | 2,048 |
| Prompt groups with mixed rewards | 256 / 256 |
| Zero-standard-deviation groups | 0 |
| Mean reward | 0.3223 |
| Non-zero advantage fraction | 100% |
| Truncated rollouts | 899 / 2,048 = 43.90% |
| Timeout or tool-format errors | 0 |
| Maximum policy lag | 3 (gate: 4) |

The first 16 updates averaged reward 0.313 and 48.5% truncation. The last 16 averaged reward 0.334 and 39.3%
truncation. The final individual batch was noisy (reward 0.22, truncation 52%), which is why promotion relies on
the full paired evaluation rather than the final training metric.

## Paired evaluation

| Metric | Parent | Step 16 | Step 32 | Parent → step 32 |
|---|---:|---:|---:|---:|
| Exact answer accuracy | 40.80% | 43.35% | **44.46%** | **+3.67 pp** |
| Box present | 72.59% | 78.41% | **82.55%** | +9.96 pp |
| Correct and boxed | 39.76% | 42.39% | **42.39%** | +2.63 pp |
| Strict reasoning-channel form | 70.12% | 76.25% | **78.80%** | +8.69 pp |
| High repetition | 38.01% | 30.28% | **25.10%** | -12.91 pp |
| Length stop | 26.93% | 20.88% | **16.57%** | -10.36 pp |
| Mean generated tokens | 587.0 | 524.0 | **463.9** | -123.1 |
| Unbalanced think tags | 26.37% | 20.72% | **16.33%** | -10.04 pp |
| Any think-tag use | 100% | 100% | 100% | unchanged |

Parent to step 32 had 176 wrong-to-right and 130 right-to-wrong transitions, with 382 both right and 567 both
wrong. Step 16 to step 32 gained 1.12 points, but that incremental difference was not significant (95% interval
`[-1.35, +3.75]`, McNemar `p=0.425`). Step 32 is selected because it has the strongest significant parent
delta and the best completion, repetition, and reasoning-channel metrics.

## Blinded trace audit

A 24-pair audit was sampled evenly from wrong-to-right, right-to-wrong, both-right, and both-wrong strata and
read before opening its assignment key. A single Codex reviewer preferred the step-32 trace in 13 pairs, the
parent in 9, with 2 ties. This stratified count is diagnostic and is not a population win rate or an independent
human evaluation.

Typical gains were correct arithmetic setup, reaching a final answer before the token limit, and removing
repeated self-check loops. Important residual failures remain: inverted ratios, double-counting units, ignoring
one clause of a word problem, and occasional loops. Some outputs also put an explanatory sentence after the
reasoning block before the box; the present structural metric accepts this even though the prompt requests only
one final line. A production protocol should tighten that checker without making format a large reward term.

## Artifacts on LUMI

- Parent predictions:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/evals/gsm8k-parent-primary-c597cdd/predictions.jsonl`
- Step-16 predictions:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/evals/gsm8k-candidate-primary-step16-c597cdd/predictions.jsonl`
- Step-32 predictions:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/evals/gsm8k-candidate-primary-step32-c597cdd/predictions.jsonl`
- Paired comparison:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/evals/gsm8k-step32-comparison-c597cdd.json`
- Blinded audit and key:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/evals/gsm8k-step32-reasoning-audit-c597cdd.md`
- Rollouts:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/rollouts/reasoning-gsm8k-oellm9b-32step/lumi_reasoning_gsm8k_oellm9b_32step__42__1788615063_rollouts_000000.jsonl`
- Selected full export:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/outputs/reasoning-gsm8k-oellm9b-32step/lumi_reasoning_gsm8k_oellm9b_32step__42__1788615063_checkpoints/step_32`
- Restart state:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/outputs/reasoning-gsm8k-oellm9b-32step-state`

Prediction SHA-256 values are `366db7b8468da670f4af225727541dd6766f55f2c5bbe317e2963a34487d864a`
(parent), `979cba714c04c82d3f13b9ed72e4abac1ca4a4a8ebf8dab470ae5b120b40c956` (step 16), and
`cbf53b41c7ef02cd554225f302af44f36ddf7d0520d8be89838a5157e5990e50` (step 32).

## Decision

The reasoning path passes its infrastructure, reward-signal, and paired model-signal gates. Keep it as the
single-turn reference backend. Do not promote step 32 as a general model: first run clean math/code,
multilingual, instruction-following, safety, and long-context retention evaluations. Use the result to set
throughput and stability expectations for the production RL plan.
