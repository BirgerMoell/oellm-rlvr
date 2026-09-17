# Reasoning RL plan for the OpenEuroLLM 9B instruct-SFT checkpoint

## Objective

The mandatory outcome is reliable English reasoning from
`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft` at revision
`85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`. Reasoning in the canonical OpenEuroLLM languages is a
second-stage objective. A checkpoint is not promoted merely because training reward rises: it must improve on
locked reasoning evaluations while preserving multilingual instruction following, safety, and long-context
behavior.

The policy prompt requests exactly one non-empty `<think>...</think>` block and a boxed final answer. The
optimization reward remains verifier correctness. Think-tag balance, repetition, reasoning length, language,
and truncation are guardrails; format alone never earns a positive task reward.

## Dataset roles

| Dataset | Frozen location on LUMI | Role |
|---|---|---|
| `open-r1/DAPO-Math-17k-Processed@31dd3095` | `/scratch/project_465002530/training/collection/post-training/2026q3/open-r1-dapo-math-17k-processed/release/31dd3095` | Initial English math RLVR pool: 12,700 train prompts after duplicate/conflict removal and frozen calibration/evaluation splits. |
| `birgermoell/oellm-math-rlvr@0ffc9d6c` | `/scratch/project_465002530/training/collection/post-training/2026q3/birgermoell-oellm-math-rlvr/release/0ffc9d6c` | Apache-2.0 procedural math expansion, including aligned prompts in all 24 official EU languages. Profile by family, language, and current-policy pass rate before use. |
| OpenEuroLLM multilingual reasoning signal | `/scratch/project_465002530/training/collection/post-training/2026q3/oellm-multilingual-reasoning-signal/release` | Language-gated RLVR across 34 supported macro-languages. Reward is conjunctive: correct answer, valid reasoning/answer form, and target-language reasoning. |
| `openeurollm/reasoning-traces-multilingual@031163d3` | `/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/raw/datasets/openeurollm--reasoning-traces-multilingual` | Optional reasoning-interface SFT bridge, not an RLVR prompt source. Consume each accepted 16K-eligible row at most once and mix with stronger English reasoning plus general replay. |
| `birgermoell/oellm-code-rlvr@e1cae771` | `/scratch/project_465002530/training/collection/post-training/2026q3/birgermoell-oellm-code-rlvr/release/e1cae771` | Later English code RLVR stage: 100,000 procedural Python tasks with hidden tests. Run only through the qualified offline sandbox and keep tests/reference solutions out of model context. |

The user-supplied `birgermoell/reasoning-traces-multilingual` pilot is useful evidence, but the newer official
OpenEuroLLM v0.2 snapshot is the preferred bridge input: it contains 3,425 accepted rows, of which 3,351 fit
the 16K bridge limit, across 37 non-English labels. It represents only 99 semantic source problems, so it is a
coverage floor and must never be oversampled as if it were a broad reasoning corpus.

## Gated job ladder

1. **Corrected infrastructure smoke.** Run two DAPO updates from the frozen instruct-SFT parent on 128
   trajectories. Require two finite, non-zero-gradient updates, at least one mixed-reward prompt group, no
   verifier/system errors, a post-update policy version, and readable restart/export state.
2. **Frozen parent profile.** Sample eight responses for each of 256 calibration prompts without a learner
   update. Measure exact reward, mixed groups, zero-standard-deviation groups, think-tag balance, repetition,
   truncation, and response length. Admit direct RL when mean reward is 0.10–0.70, at least 20% of prompt groups
   have mixed rewards, errors are at most 2%, and truncation is at most 30%.
3. **English reasoning canary.** Run the checked-in 32-update DAPO configuration with 2,048 accepted
   trajectories and 2,048 response tokens. Save a full export at updates 16 and 32 and restart state every eight
   updates. Compare the parent and both checkpoints on the frozen DAPO evaluation split and clean external
   math suites; select by paired evaluation, not by the final training batch.
4. **Conditional SFT bridge.** If the parent profile has no useful successes, fewer than 20% mixed groups, or
   unreliable non-empty reasoning blocks, stop direct RL. Build a small full-parameter bridge from high-quality
   English reasoning traces, every eligible multilingual pilot row once, and general-instruction replay. Repeat
   the identical parent profile before resuming RL. Do not launch a large reasoning-SFT epoch by default.
5. **Multilingual reasoning canary.** Starting from the selected English checkpoint, profile the frozen
   multilingual signal pool. Run 16 updates only if the conjunctive reward has adequate within-prompt variance.
   Report every language separately; use alternate/manual review for detector gaps. Keep an English reasoning
   replay quota so target-language optimization does not erase the mandatory capability.
6. **Code RLVR.** Profile the full `oellm-code-rlvr` train split by generator family and difficulty with eight
   samples per prompt. Pack only the learnable frontier into private sandbox tasks, then run a 16-update code
   canary. Treat sandbox launch failures as infrastructure errors rather than reward zero. Code RLVR follows,
   rather than substitutes for, the English reasoning gate.

## Reuse the reasoning-v1 recipe, not its checkpoint

`birgermoell/oellm-9b-256k-reasoning-v1@e74926a1` is not a candidate parent for this campaign. It continues an
older SFT lineage and did not pass its retention gates. Its useful contribution is the reproducible training
recipe and the failure evidence: a large synthetic-reasoning continuation can teach visible traces, but 2.086
billion packed tokens with only 14.89% exact prior-SFT replay still regressed English GSM8K, IFEval, and MMLU
computer science and produced language switching and reasoning loops.

If the direct-RL profile triggers the conditional SFT bridge, reuse these mechanics from that recipe:

- full-parameter, assistant-only SFT on structured conversations, with prompt/user tokens masked from loss;
- render before packing, keep only complete 64--16,384-token conversations, and never truncate an answer;
- language-scoped normalized-prompt deduplication, with specialized verified sources claiming duplicates before
  broad synthetic pools;
- a pinned, globally shuffled, token-budgeted mixture with immutable per-source manifests and a recorded seed;
- verified math, code, and STEM reasoning plus multilingual traces and general-instruction replay, rather than
  training only on translated traces;
- BF16, FSDP, gradient checkpointing, FlashAttention 2, AdamW with zero weight decay, and a short cosine schedule
  with 3% warmup as the implementation starting point.

Do not copy the original scale or mixture weights. Begin with a 32-update recipe-validation run, evaluate its
checkpoint, and cap the first real bridge at 128 updates (about 134 million packed tokens at global sequence
batch 64 and length 16,384). Increase the general-instruction replay share substantially above 14.89%, cap each
multilingual pilot row at one exposure, and checkpoint every 32 updates. Promote only the earliest checkpoint
that establishes balanced non-empty reasoning while passing the same reasoning, instruction-following,
multilingual, repetition, and long-context gates used for RL. Then repeat the frozen parent profile and return
to verifier-driven DAPO; SFT teaches the reasoning interface, while RLVR supplies the correctness signal.

## Checked-in LUMI jobs

- `configs/lumi-grpo-dapo-oellm9b-instruct-sft-smoke.yaml`: two-update, 128-trajectory smoke on two LUMI-G
  nodes. The smoke uses DAPO loss at `5e-7`, eight samples per prompt, one learner node, and one rollout node.
- `configs/lumi-grpo-dapo-oellm9b-instruct-sft-32step.yaml`: 32-update English reasoning canary with active
  sampling, 2,048-token responses, resumable state, and full exports at steps 16 and 32.

Both checked-in jobs use the pinned TMAX/Open-Instruct backend at
`3f80d37042402b8363f39c9535723b0d4cb8de54`, the LUMI ROCm 7 container, eight learner GCDs, and eight TP=1
vLLM rollout GCDs. The 32-update canary has a 64 GCD-hour reservation ceiling; the measured 1,024-token
reference run used 9.30 GCD-hours, so 2,048-token responses should be re-estimated from the smoke rather than
budgeted from the four-hour ceiling.

## Promotion gates

- all optimizer steps finite, at least 95% with non-zero gradients;
- verifier/system error rate at most 2% and policy lag at most four;
- direct-reward zero-standard-deviation groups at most 80% for English and at most 40% for the multilingual
  conjunctive stage;
- paired improvement on at least one clean reasoning suite, with no contradictory regression across two
  independent suites;
- balanced think tags improve or remain stable, high repetition and truncation do not rise by more than five
  percentage points, and median reasoning length does not grow by more than 25% at equal accuracy;
- average multilingual/general retention drops by no more than one absolute point, no priority language drops
  by more than three points without an explicit decision, and long-context retrieval drops by no more than 5%
  relative;
- every promoted artifact records parent, tokenizer, repository, backend, container, dataset/verifier
  revisions, Slurm job IDs, rollout hashes, reward replay, and checkpoint finiteness results.

Do not combine math and code rewards in the first run. Their rollout costs, failure modes, and reward scales are
different, and a single raw mixture would make regressions hard to attribute.
