# OpenEuroLLM 9B instruct math-RL canary — 2026-09-28

## Decision

The two-node LUMI math-RL path is ready for a longer, deliberately evaluated run from
`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`. A 16-update DAPO canary completed all
optimizer steps with useful grouped reward variance, synchronized learner weights to eight rollout engines,
wrote complete step-8 and step-16 exports, and resumed from full state for update 17 on a cold two-node job.

Step 16 is the selected canary checkpoint. On a frozen 1,024-prompt in-domain holdout disjoint by row ID,
semantic group, and prompt hash, greedy exact accuracy moved from **42.97% to 44.63%**. The paired delta was
**+1.66 percentage points**, with a 5,000-sample paired-bootstrap 95% interval of **[+0.20, +3.13] points** and
an exact two-sided McNemar value of `p=0.0396` (39 wrong-to-right and 22 right-to-wrong transitions).

This is a positive canary result, not a production capability claim. Step 16 was selected after examining
step 8 and the holdout comes from the same procedural generator collection as training. Confirm the signal on
additional frozen math families and retention evaluations before a long run or public checkpoint claim.

## Frozen inputs

- parent model: `Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`;
- parent revision: `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`;
- backend: `OpenEuroLLM/tmax-reproduction` at
  `3f80d37042402b8363f39c9535723b0d4cb8de54`;
- math source: `birgermoell/oellm-math-rlvr` at
  `0ffc9d6dc82717c25733b3172f4dbd63e48bab68`;
- train pool: 256 English difficulty-1-to-3 prompts, SHA-256
  `ec33cc0fcbd6be73d87ed8f0582683cc6f2e4e56297ffa7db92f3712a5ec3aff`;
- held-out evaluation: 1,024 English difficulty-1-to-3 prompts, SHA-256
  `c433561d0c3029930120194b182e13d67c5bdac8fc0f573d46db87c4e27ef7b8`;
- train/evaluation overlap: zero IDs, zero semantic groups, and zero prompt hashes;
- topology: two LUMI-G nodes, eight learner GCDs plus eight TP=1 rollout GCDs;
- optimizer: DAPO, constant learning rate `5e-7`, `beta=0`, ZeRO-3, BF16;
- update shape: eight prompts by eight samples, active sampling, 1,024 accepted trajectories over 16 updates;
- response budget: 2,048 tokens during training; greedy evaluation used 1,024 tokens.

The canary configuration and rendered Slurm hashes were
`c55f686197dbbee6961a3f2f734b488a9379f566e21c54fb7116c001864f0974` and
`b1dc7f341eee45654b4ab1643f4704eaa346830daf91f10fb1f2ec22078dca67`. The recovery configuration and
rendered Slurm hashes were `14057e5a87cec484a9c06039021de30696c61057a10fa700d13fb6d2a9d3c6f2` and
`f07200b1ffcfff9ba3e10bef13eb1fde61abe1cab6832f6348d00cbd2f767aaa`.

## Admission profiles

The first candidate curriculum was the cleaned `open-r1/DAPO-Math-17k-Processed` split. LUMI job `22397832`
evaluated 256 prompts with eight stochastic samples each. It had only `4.15%` sample accuracy and 51/256
mixed groups (`19.92%`), failing the predeclared `10–70%` reward and `>=20%` mixed-group gates. Errors were
zero and truncation was `0.88%`, so this was a checkpoint/curriculum mismatch rather than an infrastructure
failure. The corresponding 1,024-prompt greedy holdout job `22397833` measured `3.81%` accuracy. No training
job was launched on that rejected curriculum.

Job `22398367` profiled the selected OpenEuroLLM procedural pool with the same 256-by-8 protocol:

- sample accuracy: `33.74%`;
- mixed groups: 155/256 (`60.55%`);
- pass@8: `67.97%`;
- length-stop rate: `1.17%`;
- think-tag use: `100%`;
- valid reasoning-channel form: `74.32%`;
- mean response length: 209 tokens;
- job result: `COMPLETED 0:0` in 10m27s.

This passed every admission gate and closely matched the actual online training reward regime.

## Training and rollout evidence

LUMI job `22398668` completed `0:0` in 14m30s. All 16 updates had finite non-zero gradients:

| Update | Reward | Gradient norm |
|---:|---:|---:|
| 1 | 0.55 | 0.46 |
| 2 | 0.42 | 0.29 |
| 3 | 0.41 | 0.27 |
| 4 | 0.55 | 0.31 |
| 5 | 0.45 | 0.36 |
| 6 | 0.48 | 0.35 |
| 7 | 0.41 | 0.33 |
| 8 | 0.55 | 0.30 |
| 9 | 0.31 | 0.30 |
| 10 | 0.56 | 0.33 |
| 11 | 0.70 | 0.34 |
| 12 | 0.44 | 0.32 |
| 13 | 0.38 | 0.47 |
| 14 | 0.48 | 0.19 |
| 15 | 0.55 | 0.35 |
| 16 | 0.62 | 0.38 |

Inspection of the 1,024 accepted trajectories found:

- mean reward `0.49121`;
- 128/128 mixed prompt groups;
- zero-standard-deviation fraction `0`;
- non-zero-advantage fraction `1.0`;
- no truncations, timeouts, or tool-format errors;
- maximum observed policy lag of three updates, below the gate of four;
- stale asynchronous results were dropped rather than trained on;
- hierarchical weight synchronization completed at every update.

Complete 18,203,942,400-byte `model.safetensors` exports with `.checkpoint_complete` markers exist at steps 8
and 16 and in the final export.

## Frozen held-out evaluation

All checkpoints used identical native chat templating, greedy decoding, 1,024 response tokens, and the same
1,024 ordered prompts and seed.

| Checkpoint | Job | Exact accuracy | Delta vs parent | Think tags | Reasoning-channel form | Length stops |
|---|---:|---:|---:|---:|---:|---:|
| parent | `22399250` | 42.97% | — | 100% | 75.10% | 4.10% |
| step 8 | `22399251` | 42.77% | -0.20 points | 100% | 75.68% | 4.00% |
| step 16 | `22399891` | 44.63% | +1.66 points | 100% | 75.59% | 3.81% |

Step 8 was statistically neutral: its paired 95% interval was `[-1.37, +0.98]` points and McNemar
`p=0.8746`. Step 16's prediction SHA-256 is
`ad7175f94d5254d7e903d4ea6636294c43847061c6b1dbc84a93169efd27f8ed`; the parent prediction SHA-256 is
`79a8855d7371b7399efa70a35f20d753ef54126833d6dae57856a727c8c9923c`.

## Cold restart

LUMI job `22400164` found the canary's `global_step17` DeepSpeed tag and restored the shared learner state.
It continued at user-facing update 17, not update 1:

- reward `0.53125`;
- gradient norm `0.52`;
- 8/8 mixed prompt groups and non-zero advantages;
- no truncations, timeouts, tool-format errors, or stale results;
- rollout model step min/max exactly 17;
- complete step-17 and final model exports;
- result `COMPLETED 0:0` in 9m48s.

The state directory now retains `global_step9`, `global_step17`, and `global_step19` and occupies about 306
GiB. Backend-internal DeepSpeed tags include setup steps; the logged user-facing training step is the
authoritative update number.

## Resource accounting

The two-node 16-update training job used **3.87 GCD-hours** and the cold restart used **2.61 GCD-hours**.
The admission/baseline jobs and three one-GCD held-out evaluations used **1.31 GCD-hours**. A one-second bad-path
submission and a duplicate evaluator cancelled during startup used **0.03 GCD-hours**. Total compute for the
complete admission, training, evaluation, and recovery sequence was **7.82 MI250 GCD-hours**. A pending
`standard-g` submission was cancelled before allocation and used zero compute.

## Next promotion step

Before scaling beyond a canary:

1. confirm step 16 on frozen non-procedural math families and multilingual retention suites;
2. profile and mix harder examples so the current policy remains in the 10–70% reward band;
3. include 10–20% easy replay and re-profile at each curriculum boundary;
4. run a 64–128 update checkpoint-selection study with predeclared primary metrics and no holdout reuse;
5. keep the DAPO-Math hard pool out until SFT or an easier bridge lifts its current pass rate.
