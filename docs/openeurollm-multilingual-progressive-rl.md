# OpenEuroLLM multilingual progressive RL programme

Status: execution proposal, 2026-09-14

## Decision

Train one selected checkpoint through a gated sequence:

```text
frozen SFT/DPO parent
  -> reasoning candidate -> selected reasoning checkpoint (or parent)
  -> math candidate      -> selected math checkpoint (or previous)
  -> code candidate      -> selected code checkpoint (or previous)
  -> agentic candidate   -> selected release checkpoint (or previous)
```

Every transition is conditional. A later stage starts from a candidate only when it improves its target
capability and passes all multilingual, direct-mode, safety, long-context, and prior-capability retention gates.
Qwen3.5-9B is the data and infrastructure control; it is not the production parent, teacher, or checkpoint
selector. Production starts from the frozen OpenEuroLLM SFT/DPO checkpoint.

Use the canonical OpenEuroLLM language file as the source of truth. It currently defines 36 macro-languages
and 43 language/script or written-standard variants: the 24 EU official languages, three co-official languages,
seven languages of candidate member states, Icelandic, and Norwegian. Dataset builders must ingest that file
rather than copy a language list into each campaign.

## Target behaviour

The model has two explicit response modes:

- **Reasoning mode:** exactly one non-empty `<think>...</think>` block followed by one user-visible answer.
  The reasoning stays in the user's language unless mathematical notation, code, quoted material, or a tool
  protocol requires otherwise. It should not restart, repeat, or continue to the token cap.
- **Direct mode:** no required think block for simple chat, translation, extraction, classification, or a user
  request for a short answer. Direct-mode examples remain in every stage as retention replay.
- **Tool mode:** the model may reason between observations, but emits schema-valid actions, uses results,
  recovers from a failed action, and stops when the task is complete. Tool and code identifiers remain stable;
  the user-facing request, explanations, and final answer follow the target language.

Correctness dominates the reward. A valid think envelope is a gate or a small correctness-conditioned bonus,
never a way for an incorrect response to earn a positive training reward. The verifier extracts the answer
after `</think>` and does not grade whether the hidden reasoning matches a preferred trace.

## Data readiness audit

The machine-readable inventory is in
[`campaigns/openeurollm-multilingual-rl-sources.yaml`](../campaigns/openeurollm-multilingual-rl-sources.yaml).

| Stage | Sources already available | Coverage and role | Gap before production |
|---|---|---|---|
| Think-interface bridge | `openeurollm/Dolci-Think-SFT-translated`; `openeurollm/reasoning-traces-multilingual`; decontaminated English Dolci Think | About 1.0M translated rows in 12 languages; the smaller pilot has 3,425 accepted translations over 99 source problems and 37 non-English labels. Use only a filtered, verified subset to teach tags and in-language traces. The pilot and English Dolci snapshots are already staged on LUMI. | Map every row to the canonical 36/43 contract; native review, independent answer checking, PII/safety screening, and missing-language generation. Do not call the 99-problem pilot a production corpus. |
| General reasoning RLVR | `open-thought/reasoning-gym`; Apple's Multilingual Reasoning Gym; NVIDIA `Nemotron-RL-ReasoningGym-v1` | More than 100 procedural, algorithmically scored task families; the multilingual fork provides parallel generation for 10+ languages. | Add project-owned templates for all 36 languages/43 variants, validate every scorer, reserve generator/template/seed clusters for evaluation, and exclude any embedded benchmark-derived tasks. |
| Math RLVR | `birgermoell/oellm-math-rlvr` at `0ffc9d6c`; optional NVIDIA math RL sources after license/decontamination review | 1M deterministic prompts, including 10k semantic problems rendered in all 24 EU official languages; exact/rational answers and explicit semantic groups. The pinned parquet is staged on LUMI. | Extend the generators to the other 12 OpenEuroLLM macro-languages and required script variants. Add symbolic, geometry, probability, and proof-verification families rather than only translating more arithmetic. |
| Code RLVR | `birgermoell/oellm-code-rlvr` at `e1cae771`; NVIDIA competitive-coding data as a separately licensed optional pool | 100k Apache-2.0 procedural Python tasks with 10–13 hidden tests, but currently English-only. Hidden tests are language-neutral and must stay model-invisible. | Stage the full source on LUMI, translate only natural-language problem surfaces into 36 languages, re-run examples and hidden tests, add non-Python and debugging families, and hold out entire generator families/seeds. |
| Agentic RL | Qualified SkyRL + Harbor stack; NVIDIA function-calling, conversational tool-use, calendar, workplace, terminal, and SWE task sources | The LUMI stack has completed real Qwen3.5-9B Harbor rollouts and SkyRL optimizer steps. Public sources provide thousands of function/SWE tasks, but are predominantly English and require their own environment adapters. | Localize user/tool surfaces; prebuild network-free SIFs; pin repositories and tool state; require oracle/failing-agent/model contracts; create native European public-service and workplace tasks; keep release benchmarks entirely out of training. |
| Evaluation only | OpenEuroLLM `ArenaHard-EU`, `eval_dashboard`, project multilingual holdouts, MGSM/Global-MMLU/IFEval-style suites where licensing permits | Selection and regression measurement. | Freeze revisions and contamination signatures before training. Never backfill their prompts into an RL pool. Add disjoint procedural holdouts for every train generator and every target language. |

This audit means math is ready for an EU24 pilot now. Full OpenEuroLLM-language reasoning, code, and agentic
training are buildable from existing generators and environments, but are not yet data-ready without the listed
localization and verification work.

The current infrastructure control is documented in
[`qualification-qwen35-eu-math-2026-09-14.md`](qualification-qwen35-eu-math-2026-09-14.md). LUMI job
`22027705` completed two real Qwen3.5-9B EU-SFT optimizer steps with nonzero gradients over 128 retained math
trajectories and 16/16 mixed-reward groups. It also exposed the next data problem: only 31/128 completions
closed one balanced think block, 26/128 were length-truncated, and random mixture weights covered only 12
languages. Therefore the think-interface bridge, response-length profiling, and deterministic window-level
language quotas are prerequisites, not optional refinements.

## Common data contract

Every admitted prompt group carries:

- canonical `language`, `script`, and written-standard codes;
- immutable source repository/revision, license, generator version, template version, and content hash;
- semantic cluster, generator family, difficulty parameters, and train/calibration/evaluation disposition;
- verifier or environment image revision plus a deterministic reward-replay command;
- parent pass rate from eight sampled completions, response length, truncation, error class, and language ID;
- a flag distinguishing translated, native-authored, language-neutral, and code/tool-protocol content.

Parallel translations of one latent task remain in one split and normally in one prompt group. They must never
cross train/evaluation boundaries. Reserve complete templates, generator families, repositories, and seed ranges
for held-out evaluation so the score cannot be won by memorizing a generator surface.

Difficulty is measured separately for the checkpoint entering each stage:

- easy: 5–7 successes out of 8;
- medium: 2–4 out of 8;
- hard: 1 out of 8;
- saturated: 8 out of 8, excluded from group-relative RL but useful for retention;
- currently impossible: 0 out of 8, routed to repair, a small SFT bridge, or a later stage.

Re-profile at every curriculum-window boundary. Source difficulty labels are not a substitute for observed pass
rates.

## Language scheduler

Use prompt-group quotas rather than raw-row sampling. For the production stages:

- 40% English or language-neutral prompts preserve general capability and rollout efficiency;
- 30% is distributed uniformly over the 35 non-English macro-languages, guaranteeing a minimum exposure floor;
- 30% is weighted by project priorities and measured weakness, capped so high-resource languages cannot crowd
  out low-resource ones;
- every 64-update window covers all 36 macro-languages; every 256-update window covers all 43 internal variants;
- report rewards, pass rates, answer-language consistency, length, and errors by language and language family.

Run 20%, 50%, and 80% non-English canaries from an identical parent before fixing these production weights. The
current Qwen EU-math smoke is the first 80% treatment. Select the smallest non-English share that gives broad
gains without a statistically meaningful English regression; the 60% production value above is a hypothesis,
not a result.

## Stage 0 — freeze, controls, and scorecard

1. Hash the incoming OpenEuroLLM SFT/DPO checkpoint, tokenizer, chat template, generation config, SIF, backend,
   datasets, and verifier sources.
2. Run the same frozen multilingual scorecard on the pre-SFT base, SFT, DPO if present, and existing specialist
   controls. Store generations so rewards can be replayed.
3. Run learner-off Qwen3.5-9B controls over every task contract. Then profile the actual OpenEuroLLM parent;
   Qwen pass rates never define OpenEuroLLM difficulty.
4. Verify direct mode, one balanced think block, answer extraction, 256K architecture loading, and at least one
   successful and one failed tool trajectory.

Exit: immutable manifests, clean held-outs, and an admitted pool with at least 60% mixed-reward prompt groups.

## Stage 1 — multilingual reasoning with think tags

If the parent does not reliably emit the interface, first run a small SFT bridge: 20k–80k highly filtered
examples, 50% English and 50% spread over all target languages, with one correct concise trace and one final
answer. This teaches the protocol, not the capability.

RL curriculum, 256–512 updates, eight prompts by eight samples per update:

| Window | Reasoning mix | Difficulty mix |
|---|---|---|
| warmup, first 25% | arithmetic, logic, state tracking, calendars, strings, simple graphs | 65% easy / 30% medium / 5% hard |
| core, middle 50% | algebra, planning, algorithms, probability, compositional logic | 30% / 50% / 20% |
| challenge, final 25% | longer compositions and unseen templates, plus 25% easy replay | 20% / 40% / 40% |

Use a deterministic answer scorer. Reward can be `correct * (0.95 + 0.05 * format_valid)` with an independently
logged invalid-format flag; an incorrect formatted answer remains zero. Keep 15% direct-mode retention prompts.

Exit: held-out reasoning improves in aggregate and in at least 30/36 languages; no language family regresses
beyond its frozen tolerance; at least 98% of reasoning-mode correct answers have balanced tags; direct-mode tag
intrusion stays below 2%.

## Stage 2 — multilingual mathematics

Start from the selected reasoning checkpoint. Train 384–768 updates with:

- 60% math RLVR: numeric/exact, symbolic algebra, geometry, probability/combinatorics, and later proof checks;
- 20% general-reasoning replay;
- 10% direct multilingual instruction replay;
- 10% English/language-neutral hard math and code-based verification.

Move from 50% easy math in warmup to 40% medium in core and 25% hard in challenge, while keeping at least 25%
easy/prior replay. Symbolic equivalence must be bounded and deterministic; ambiguous proof grading stays out until
an independently replayable checker is qualified.

Exit: gains on disjoint generators and external multilingual math suites; no reasoning/direct-mode regression;
less than 5% answer-language mismatch or truncation; reward replay is exact.

## Stage 3 — multilingual executable code

Keep this stage completion-oriented rather than agentic: one problem, one code artifact, one hidden-test result.
Train 384–768 updates with 55% executable code, 15% math replay, 15% reasoning replay, 10% multilingual direct
mode, and 5% structured output. Translate prose, not identifiers, APIs, formats, or hidden tests.

Warmup uses short functions and basic stdin/stdout tasks; core adds algorithms, parsing, state, tests, and repair;
challenge adds longer solutions, debugging, cross-file context, and 25% easy replay. Every generation runs in an
isolated Apptainer sandbox with no network, secrets, or writable host paths. Infrastructure failures are retried
and masked, not converted into policy reward zero.

Exit: the actual OpenEuroLLM checkpoint completes optimizer updates with mixed hidden-test rewards; held-out
pass@1 rises across language families; sandbox reliability exceeds 98%; math and reasoning retention gates pass.

## Stage 4 — multilingual agentic RL with SkyRL + Harbor

Start from the selected code checkpoint. Run three separately promotable phases:

1. **Function and structured actions, 128–256 updates:** one or two calls, schema correctness, deterministic
   state changes, and recovery from one injected tool error.
2. **Multi-turn tools, 256–512 updates:** calendars, search over local corpora, SQL, workplace tasks, and public
   services; two to eight turns with localized user/tool surfaces.
3. **Terminal and repository work, 256–768 updates:** navigation, test repair, debugging, dependency reasoning,
   and small issue resolution in pinned, network-free Harbor environments.

Each batch contains 25% retention replay: 10% executable code, 5% math, 5% general reasoning, and 5% multilingual
direct/safety tasks. Compute advantages inside a single environment/verifier domain before combining quotas.
Reward final environment success. Invalid calls, unnecessary turns, and truncation may receive small bounded
penalties; never reward calling a tool or producing a long plan by itself.

Exit per phase: verifier replay is deterministic; environment errors stay below 2%; post-update rollouts use the
new sampler version; restart succeeds; held-out task success improves without prior-stage regression. Scale to
long repo tasks only after the actual OpenEuroLLM model passes one synchronous SkyRL optimizer step and a
post-update Harbor rollout.

## Stage 5 — recovery, evaluation, and release selection

Run a short 128–256 update recovery mixture only if the final scorecard shows drift: 30% verified reasoning,
20% code, 15% multilingual instruction, 15% agentic retention, 10% safety/abstention, and 10% long-context/direct
mode. A small preference or on-policy-distillation treatment is preferable to unconstrained RL once correctness
has saturated.

Select among all saved checkpoints, not automatically the last. Publish the winning model only with source and
license manifests, config and code revisions, GCD-hours, rollout/reward distributions by language, deterministic
reward-replay results, paired evaluation confidence intervals, and known limitations.

## Universal stop/go gates

Stop the active stage when any condition persists after one bounded retry or data repair:

- fewer than 60% of admitted groups have mixed rewards;
- more than 5% of updates have zero, missing, or non-finite gradients;
- verifier/environment errors reach 2%, or rewards fail deterministic replay;
- policy lag exceeds four, hidden tests leak, or model-visible context includes verifier secrets;
- truncation exceeds 10% overall or 20% in any language without an understood tokenizer-length cause;
- any prior selected capability falls by more than three relative points or its pre-registered confidence bound;
- language-consistency failures exceed 5% in a macro-language or a script variant receives no coverage.

## Compute envelope

The observed 32-update 9B reasoning pilot cost 9.30 LUMI GCD-hours, but multilingual long traces, executable code,
and Harbor environments will be slower. Use these planning ranges, including stage evaluation:

| Work | Expected GCD-hours |
|---|---:|
| Data/interface canaries and frozen baseline scorecard | 80–180 |
| Reasoning RL | 120–300 |
| Math RL | 180–450 |
| Executable-code RL | 300–800 |
| Agentic phases | 900–2,500 |
| Recovery and final evaluation | 150–400 |
| **Full programme** | **1,730–4,630** |

These are gated envelopes, not a commitment to spend through failure. The first production rehearsal should stay
below 200 GCD-hours: 16-update reasoning, math, and code canaries plus an 8-update agentic phase-A canary, each
starting from the selected predecessor and each covering every macro-language at least once across its profiling
and training window.

## Two-week implementation order

1. Add a canonical-language loader and 36/43 coverage validator to `oellm-rlvr`.
2. Freeze training/evaluation seed ranges and semantic clusters for the existing math and code generators.
3. Import and pin Reasoning Gym; choose 20–30 task families without benchmark-derived content; add translations
   and scorer replay tests for all target variants.
4. Stage full code data on LUMI and build a translation pipeline that re-executes examples and hidden tests.
5. Filter the reasoning SFT bridge for tag balance, correctness, language consistency, license, PII, and length.
6. Build 16 Harbor function/tool tasks in four difficulty bands, then localize and oracle-test them across the
   target language contract before adding repo tasks.
7. Run Qwen data controls, then the incoming OpenEuroLLM parent profiles. Materialize pools from that parent only.
8. Execute the sub-200-GCD-hour rehearsal; publish a selection report and a revised production budget from
   measured tokens/second, active-sampling yield, environment utilization, and evaluation variance.

## Primary references

- [OpenEuroLLM canonical target languages](https://github.com/OpenEuroLLM/training-data-catalogue/blob/main/languages)
- [OpenEuroLLM Hugging Face datasets](https://huggingface.co/openeurollm/datasets)
- [`birgermoell/oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr)
- [`birgermoell/oellm-code-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr)
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
- [Multilingual Reasoning Gym](https://github.com/apple-aiml-research/ml-multilingual-reasoning-gym)
- [NVIDIA Nemotron reinforcement-learning datasets](https://huggingface.co/collections/nvidia/nemotron-reinforcement-learning)
- [NVIDIA Nemotron Post-Training v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)
- [SkyRL](https://github.com/novasky-ai/skyrl)
- [Harbor](https://github.com/harbor-framework/harbor)
