# OpenEuroLLM 9B instruct math-DAPO qualification — 2026-09-28

## Decision

The two-node LUMI path is qualified for a larger single-turn math RLVR canary from
`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`. It completed grouped rollout,
deterministic verification, five finite learner updates, live hierarchical learner-to-vLLM weight transfer,
full Hugging Face exports, distributed ZeRO-3 state, and a cold restart on different nodes.

This is an infrastructure and reward-signal qualification, not evidence that five updates improve held-out
math quality. The next run must use a frozen parent/candidate evaluation and a larger, difficulty-stratified
training split.

## What was fixed

The pinned TMAX revision does not implement a literal `loss_fn=grpo`; it accepts `dapo`, `cispo`, `dppo`, and
`tvpo`. DAPO is the selected grouped relative-policy objective for this qualification. Configuration validation
now rejects literal `training.loss: grpo` for the TMAX backend with an actionable error, while retaining that
name for the colocated verl adapter. Existing OELLM 9B math profiles were corrected to DAPO.

The runtime also carries the MCP 2 compatibility bridge required by the pinned backend. The resulting control
plane revisions are:

- base dry run: `0b1ecaa`;
- cold-restart probe: `0582e72`;
- TMAX: `3f80d37042402b8363f39c9535723b0d4cb8de54`.

The full local test suite passed before submission. Both run profiles passed `oellm-rlvr doctor`, including the
container, backend checkout, and exact backend revision checks.

## Frozen inputs

- policy revision: `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`;
- local policy cache: `/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/model`;
- learner dataset: 256 deterministic English difficulty-1-to-3 samples from
  `birgermoell/oellm-math-rlvr`;
- dataset SHA-256: `ec33cc0fcbd6be73d87ed8f0582683cc6f2e4e56297ffa7db92f3712a5ec3aff`;
- topology: two LUMI-G nodes, eight learner GCDs and eight TP=1 rollout GCDs;
- update shape: eight unique prompts by eight samples, active sampling, 2,048 response tokens;
- optimizer: DAPO, constant learning rate `5e-7`, `beta=0`, ZeRO-3, BF16;
- weight transfer: hierarchical native NCCL.

The four-update configuration SHA-256 was
`e6675e677c0c88a01af6ae4175d96c549443aab0808cb0b42dd3f6a890e90703`; its rendered sbatch SHA-256 was
`4088ab0400e78138265c6fb3d5c26a9231f3aca559f5f418485d1627b5481d0c`. The restart configuration and
sbatch hashes were `57e8b83d6b8a5f51c400a94d8e820de44c9210402717b8ae1be94c4e35b510cc` and
`f183d112cf44b2a536fc7af2e79d0d48ae5886be3813d4500db074d97ccb5a3f`.

## Results

| LUMI job | Purpose | Result | Runtime | Allocated compute |
|---:|---|---|---:|---:|
| `22395497` | four-update production dry run | `COMPLETED 0:0` | 12m00s | 3.20 GCD-hours |
| `22395748` | cold restart and update 5 | `COMPLETED 0:0` | 9m16s | 2.47 GCD-hours |

Total qualification compute was 5.67 MI250 GCD-hours. Slurm helper steps used to terminate Ray daemons can be
reported as failed or cancelled even when the parent batch exits cleanly; the parent batch result and complete
artifact markers are authoritative.

The first job produced the following per-update signal:

| Update | Verifiable reward | Gradient norm |
|---:|---:|---:|
| 1 | 0.53 | 0.47 |
| 2 | 0.45 | 0.31 |
| 3 | 0.53 | 0.27 |
| 4 | 0.47 | 0.30 |

Across its accepted rollout shard, all 32 prompt groups were mixed reward groups, mean reward was `0.4961`,
zero-standard-deviation fraction was `0`, non-zero-advantage fraction was `1.0`, and there were no timeouts,
truncations, or tool-format errors.

The second job discovered the previous `global_step5` DeepSpeed tag, restored RNG, dataloader, and data-prep
actor state, and began at training update 5 rather than update 1. Update 5 had reward `0.53125`, gradient norm
`0.47`, eight of eight mixed prompt groups, zero truncations, and no timeouts. Its saved client state reports:

- `training_step=5`;
- `episode=496` (including active-sampling candidates);
- `num_total_tokens=51932`;
- RNG, dataloader, and data-prep actor state present.

The backend's final DeepSpeed tag is `global_step7` because ZeRO-3 initialization performs an internal dummy
step; the user-facing client state remains update 5.

## Artifacts and operational notes

The base run wrote complete 18,203,942,400-byte `model.safetensors` exports at updates 2 and 4 and a complete
final export. The restart run wrote the same complete artifacts at update 5. Every export contains
`.checkpoint_complete`.

The restart directory retains three full ZeRO-3 states (`global_step3`, `global_step5`, and `global_step7`) and
currently occupies about 306 GiB. A production schedule should therefore checkpoint restart state less often
than model metrics and keep the existing three-checkpoint retention limit.

On restart, loading the separately saved reference-policy file emitted ZeRO-3 shape warnings and fell back to
the immutable base checkpoint. That fallback is behaviorally correct here because the frozen reference is the
same base model and `beta=0`. Before enabling a non-zero KL coefficient or changing the reference policy,
qualify a sharding-safe reference-policy restore and make fallback identity explicit.

## Promotion gate

Proceed to the 16-update optimization canary only after freezing its train/evaluation split and parent metrics.
Require at least 14 finite non-zero-gradient updates, verifier/system errors at or below 2%, zero-std groups at
or below 80%, truncation at or below 35%, policy lag at or below four, another successful restart, and no
regression on the frozen parent/candidate evaluation. Do not infer capability improvement from this dry run.
