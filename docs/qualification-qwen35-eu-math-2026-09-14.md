# Qwen3.5-9B EU-SFT multilingual math qualification — 2026-09-14

## Result

LUMI job `22027705` completed `0:0` in 12m45s on two nodes and 16 MI250X GCDs. It loaded the
EU-SFT-trained Qwen3.5-9B checkpoint, initialized eight ZeRO-3 learner ranks and eight TP=1 vLLM rollout
engines, generated verified non-English math rollouts, completed two DPPO optimizer steps with nonzero
gradients, synchronized the policy, and saved a complete 17,907,663,008-byte model.

This qualifies the Qwen3.5 text-only model path and multilingual math reward-to-update path. It does **not**
yet qualify the target reasoning format or a production language schedule: only 31/128 retained responses had
one balanced `<think>...</think>` envelope, and the weighted smoke happened to sample no English prompt groups.

## Pinned inputs

- Hub identity: `birgermoell/Qwen3.5-9B-EU-SFT`
- immutable local BF16 source:
  `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-sft-bf16`
- text-only inference repair:
  `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-sft-bf16-textfix`
- backend: TMAX commit `3f80d37042402b8363f39c9535723b0d4cb8de54`
- runtime: PyTorch `2.10.0+rocm7.0`, HIP `7.0.51831`, Ray `2.54.0`, vLLM `0.22.1`
- run config: `configs/lumi-qwen35-9b-eu-math-smoke.yaml`
- EU pool: 2,944 rows, 128 per non-English EU official language, SHA-256
  `15c49ec9a9994af43f0dcaafaff8c2f3f465b51ebf46a5f4db66afcee2097ef1`
- English replay pool: 2,048 rows, SHA-256
  `84022474b1a7a2b2f8345b2bd93f00fdf20aee07edab738730bdfdee09d2e1b6`
- semantic overlap between the two pools: zero

CPU job `22026321` produced the text-only checkpoint in 2m29s. It rewrote only the key namespace, preserved
all 427 BF16 tensors, and passed exact missing/unexpected-key parity against `Qwen3_5ForCausalLM`. The original
checkpoint was not modified.

## Successful job evidence

The 8+8 topology used all 16 allocated GCDs. All eight rollout engines resolved `Qwen3_5ForCausalLM`, loaded
16.8 GiB each, constructed the aligned hybrid attention/GDN cache, and accepted the text-only three-channel
M-RoPE positions. The initial hierarchical trainer-to-relay-to-leaf weight transfer completed for 8/8 engines.

| Metric | Step 1 | Step 2 |
|---|---:|---:|
| retained trajectories | 64 | 64 |
| math correct rate | 0.52 | 0.48 |
| gradient norm | 0.52 | 0.43 |
| advantage range | -0.88 to 0.88 | -0.88 to 0.88 |
| truncated completions | 12 | 14 |
| stale results dropped | 0 | 0 |

Active sampling generated 152 trajectories in total. It removed three constant-reward prompt groups from the
first window—one solved group and two all-zero groups—and trained on 128 retained trajectories. The retained
artifact has:

- 16/16 mixed-reward prompt groups and zero zero-standard-deviation groups;
- mean reward `0.5` and nonzero advantages on 128/128 trajectories;
- 64 correct and 64 incorrect verifier outcomes;
- 26/128 length truncations, no timeouts, and no tool-format errors;
- SHA-256 `7d4407032665d5eb0bb97e2b37458c2362ea5f7b9613acf647e94dec70c99a7f`.

The 16 prompt groups covered 12 non-English languages: Bulgarian, Danish, Finnish, Hungarian, Lithuanian,
Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, and Swedish. This is a valid multilingual smoke, not
an all-language evaluation. The random `0.8/0.2` dataset weights produced 16 non-English groups and no English
group; production campaigns need window-level language quotas rather than relying on mixture probabilities.

## Behaviour audit

The chat template opened `<think>` before generation. Of the 128 retained completions:

- 31 closed exactly one balanced think block;
- 23 placed a boxed final answer after the closing tag;
- 26 reached the 1,024-token response limit;
- 97 did not close the think block, commonly emitting the boxed answer inside it;
- sampled traces sometimes switched from the prompt language to English.

The math verifier correctly supplied learning signal, but the format and language-consistency results fail the
production behaviour gates. Before the reasoning stage, teach the interface with a small filtered multilingual
SFT bridge, add correctness-conditioned format reward, increase or adapt the response budget only after length
profiling, and enforce per-language quotas. Never make think-tag validity an independent positive reward for an
incorrect answer.

## Compatibility fixes qualified by this run

The pinned vLLM build predates its complete Qwen3.5 text-only support. The control plane now:

1. registers the shipped native `Qwen3_5ForCausalLM` instead of falling back to the vision wrapper;
2. preserves the post-import loader interface used by vLLM's inspectors;
3. repairs the exported `model.language_model.*` namespace into an immutable text-only checkpoint;
4. restores the hybrid GDN state-shape/copy interface required for cache construction; and
5. backports upstream text-only M-RoPE behavior by broadcasting ordinary token positions over three channels.

The first four fixes were individually validated by bounded failed jobs. Job `22026999` then reached the first
real rollout token and isolated the missing M-RoPE interface; `22027705` proves the backport through generation
and two optimizer steps.

## Compute accounting

Successful job `22027705` used `16 × 765 / 3600 = 3.400` allocated GCD-hours. The six bounded GPU attempts
that isolated the preceding compatibility faults used another 11.671 GCD-hours, for 15.071 GCD-hours across
this qualification sequence. The CPU-only checkpoint repair used no GCD-hours.

This is allocation-based accounting, not a claim about LUMI billing weights. No job from this qualification is
still running; three older, unrelated user-held jobs were left untouched.
