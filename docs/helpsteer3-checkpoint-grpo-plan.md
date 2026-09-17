# RLVR plan for the OpenEuroLLM 9B instruct-SFT checkpoint

## Frozen starting point

- Policy: `Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft` at
  `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`.
- Initial RL data: the English configuration of `open-r1/DAPO-Math-17k-Processed` at
  `31dd309567e3da778038cc87d868b6097a3ccf68` (14,116 published rows), materialized at
  `/scratch/project_465002530/training/collection/post-training/2026q3/open-r1-dapo-math-17k-processed/release/31dd3095`.
- Provenance: processed from `BytedTsinghua-SIA/DAPO-Math-17k`; retain both source records. The processed Hub
  card has no machine-readable license tag, so production promotion retains a license-review gate.
- Backend: `OpenEuroLLM/tmax-reproduction` at
  `3f80d37042402b8363f39c9535723b0d4cb8de54` through the LUMI-qualified `oellm-rlvr` adapter.
- Topology: two LUMI-G nodes: eight learner GCDs and eight TP=1 vLLM rollout GCDs.

The prepared learner parquet contains only the problem prompt and verifier ground truth. Published solution or
source-prompt fields do not enter model context. Dataset QA collapses same-answer duplicates and drops every
normalized prompt group with conflicting answers. In the pinned shard this removes 126 redundant rows and five
two-row answer-conflict groups, leaving 13,980 verified prompt groups. A deterministic seed then reserves 256
rows for decoding/reward calibration and 1,024 rows for frozen evaluation, leaving 12,700 training prompts; all
three splits are disjoint. The frozen train parquet SHA-256 is
`cce9fb9a7a458bd7e8ca08abb9618fac77cfc8e4360bdb355bbf1ddc12f05852` as generated in the pinned LUMI
container.

## Job ladder

| Gate | Shape | Purpose | Allocation ceiling | Promotion requirement |
|---|---|---|---:|---|
| Infrastructure smoke | 2 updates, 128 trajectories, 1,024 response tokens, DAPO loss | Prove model load, grouped generation, verifier, optimizer, hierarchical weight transfer, checkpoint and restart state | 12 GCD-hours | 2/2 finite updates, post-update policy version, readable export and restart state |
| Reward-signal probe | 256 frozen prompts x 8 samples, no learner update | Measure current-policy pass rate, mixed groups, truncation and verifier failures before spending training compute | 16 GCD-hours | mean reward 0.10-0.70, at least 20% mixed groups, verifier errors <=2%, truncation <=30% |
| Optimization canary | 16 updates, 1,024 accepted trajectories, 2,048 response tokens | Establish nonzero gradients and short-horizon reward movement with active sampling | 32 GCD-hours | >=14 finite/nonzero-gradient updates, zero-std <=80%, policy lag <=4, restart succeeds |
| Pilot | 64 updates, 4,096 accepted trajectories, 2,048 response tokens | Compare parent and step 32/64 on the frozen 1,024-row set and external clean math suites | 64 GCD-hours | paired improvement without regression in instruction following, safety, multilingual or long-context checks |
| Production candidate | 256 updates, 16,384 accepted trajectories; difficulty-stratified sampling | Train a promotable math-reasoning candidate; do not treat one pass over trace rows as an objective | 128 GCD-hours | preregistered eval gain, stable KL/entropy/reward, reward replay, clean checkpoint recovery |

Ceilings are reservations, not expected consumption. Based on the qualified 32-update OELLM 9B run, expected
training use is roughly 10-20, 25-45, and 60-100 GCD-hours for the 16-, 64-, and 256-update stages, respectively;
longer DAPO responses can move these estimates upward.

## Reward and evaluation contract

Use correctness as the optimization reward. Keep answer-format compliance, repetition, language, response
length, and truncation as separately reported guardrails; do not let a large format bonus replace solving the
problem. Sample eight completions for every prompt and compute advantages only within that prompt group. All-
equal groups provide no GRPO signal and must be measured explicitly before active sampling is enabled.

The frozen DAPO evaluation split measures in-distribution progress, not clean generalization. Before promotion,
deduplicate the training pool against the chosen external math suites and run paired parent/candidate evaluation.
Keep GSM8K, MATH, AIME and other reported benchmarks outside the training pool used for that evaluation.

## Expansion after the English math gate

1. Add verified multilingual math prompts with per-language sampling quotas and native-language audits.
2. Add code RLVR only after license/provenance review and a hardened execution sandbox; never mix raw math and
   code reward scales without domain-specific normalization and quotas.
3. Add agentic tasks only after repeated-prompt rollouts show within-group reward variance and token-accurate
   action/observation traces.
4. Treat preference data such as HelpSteer3 as DPO or reward-model data, not as GRPO verifier ground truth.
