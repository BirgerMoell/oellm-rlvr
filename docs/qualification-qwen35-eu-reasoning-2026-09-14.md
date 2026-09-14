# Qwen3.5-9B EU multilingual reasoning qualification — 2026-09-14

## Decision

Do not run multilingual reasoning RL directly from `EU-SFT-v2full`. The model has strong exact
math capability and a working Qwen `<think>` interface, but its reasoning language normally switches
to English. Exact-answer-only RL would reinforce that behavior on non-English prompts.

The admitted sequence is therefore:

1. a small multilingual reasoning SFT interface bridge;
2. the same frozen 36-language parent/candidate profile;
3. only if the conjunctive reward has adequate variance, a 16-update language-gated RL run.

The starting checkpoint's lineage and SFT recipe are recorded in
[`checkpoint-qwen35-9b-eu-sft-v2full.md`](checkpoint-qwen35-9b-eu-sft-v2full.md).

## Parent profiles

All profiles used eight stochastic samples per prompt at temperature 0.8 and the checkpoint's native
chat template. That template supplies the opening `<think>` in the generation prompt; the evaluator
reconstructs it before measuring tag balance.

| Profile | LUMI job | Prompts / samples | Response cap | Accuracy | Pass@8 | Zero-std prompt groups | Length stops |
|---|---:|---:|---:|---:|---:|---:|---:|
| Basic symbolic | `22029976` | 36 / 288 | 768 | 58.33% | 86.11% | 33.33% | 42.36% |
| Basic symbolic | `22030182` | 36 / 288 | 1,536 | 90.28% | 100% | 69.44% | 9.38% |
| Eight-family challenge | `22030601` | 72 / 576 | 1,536 | 31.77% | 61.11% | 40.28% | 68.58% |
| Selected four-family signal | `22031165` | 72 / 576 | 2,048 | 79.86% | 100% | 45.83% | 19.44% |

The 768-token basic pool manufactured variance through truncation. At a usable response cap the same
pool saturated. The eight-family challenge pool went too far in the other direction, so family-level
profiling selected affine modular chains, weighted checksums, second-order recurrences, and linear
system checksums. The resulting 2,048-token profile passed the 25% truncation gate.

The first attempt at the selected profile, job `22031076`, failed during model initialization after
72 seconds because the generic evaluation script did not enable the already-qualified Qwen3.5 text
compatibility hook. Commit `0fc265f` makes that hook an explicit script default; the successful
resubmission was job `22031165`.

## Language and reward audit

The offline Lingua audit scores reasoning prose separately from mathematical notation and the boxed
answer. It supports 34 current targets; Galician and Maltese remain explicit manual/alternate-detector
gaps. BCMS is treated as one coarse equivalence group. This metric is diagnostic and does not replace
native review.

On the selected parent profile:

- target-language match: **5.52%** over 543 scored samples;
- correct + valid think/box form + target language: **2.39%**;
- prompt groups with mixed conjunctive reward: **5/68**;
- zero-standard-deviation conjunctive reward groups: **92.65%**.

The configured maximum is 40%, so the parent fails RL admission even though its exact-answer profile
looks strong. This is a behavioral/data problem, not a rollout or optimizer failure.

Commit `76f3bf7` adds a TMAX runtime verifier named `multilingual_math`. Its label carries an ordinary
math answer and target ISO-639-1 language. Reward is strictly conjunctive:

`correct answer AND target language AND valid think/box form`

Language or formatting never awards a positive score to an incorrect answer. The materialized
34-language training pool has 2,176 rows and SHA-256
`677f7b79569635cd028e991474d6a7af154f85149785441e8f25d0ff68332fe3`.

## SFT interface bridge

The bridge contains 3,924 conversations:

- 3,156 accepted translated in-language reasoning traces across 34 non-English target languages;
- 384 English `<think>` reasoning replay rows;
- 384 general-instruction replay rows.

Its Parquet SHA-256 is
`07410754cd114680967e4d23ae556d4d7d11ceba49ff419cace50597db3e92fb`.
The translated source is CC-BY-4.0 and contains only 99 underlying source problems, so this is a
control/interface bridge, not a production reasoning corpus. Georgian is absent and requires newly
verified traces. Mixed replay-source terms must be reviewed before publishing derived weights.

The bridge config uses full-parameter BF16 SFT, 4,096-token packing, 128 optimizer steps, global batch
32, cosine learning rate peaking at `2e-6`, 5% warmup, and gradient checkpointing. LUMI job
`22031504` requests four nodes with four MI250 GCDs per node for at most two hours: 16 active GCDs
and a hard ceiling of 32 allocated GCD-hours. Unlike the original SFT job, it does not reserve four
unused GCDs on every node.

## Admission gates after the bridge

Run the identical 72-prompt, 576-sample profile and require:

- verifier/system errors at or below 2%;
- length stops at or below 25%;
- conjunctive-reward zero-std prompt groups at or below 40%;
- target-language match high enough to produce signal across language families, with every mismatch
  slice reported and Galician/Maltese reviewed separately;
- no unexplained degradation in exact accuracy relative to the frozen parent profile.

If these pass, submit
`configs/lumi-qwen35-9b-multilingual-reasoning-signal.yaml`: two nodes, eight learner and eight
rollout GCDs, 2,048 episodes, 16 optimizer updates, DPPO at `5e-7`, active sampling, hierarchical
weight transfer, and periodic restartable state. That run is a bounded control qualification, not a
production OpenEuroLLM checkpoint.
