# OpenEuroLLM 9B reinforcement-learning plan

Status: proposed execution plan, 2026-09-05

## Executive decision

The goal should be the **best openly reproducible general and agentic model in the dense 7–10B class**, with a
distinct advantage in European languages. A 9B model cannot win by memorizing the most facts or imitating a
single giant English reasoner. It can win on the combination of verified reasoning, useful code generation,
reliable tool use, concise behavior, multilingual parity, long-context retention, and unusually transparent
training evidence.

Use a multi-stage curriculum rather than one giant mixed run:

1. freeze and profile the incoming SFT/DPO checkpoint;
2. strengthen single-turn math, code, science, and constrained instruction following with RLVR;
3. train multi-turn function calling, terminal use, and software engineering with SkyRL + Harbor;
4. run a short balanced recovery/alignment stage with replay from capabilities that must not regress;
5. select the release checkpoint on a frozen, contamination-controlled scorecard rather than training reward.

Keep this repository as the LUMI-first control plane. Use its already-qualified TMAX backend for the first
single-turn stages. Add SkyRL as an agentic backend and Harbor as the task, sandbox, and verifier layer. Do not
replace the working LUMI path before an alternative completes the same real-checkpoint gradient, weight-sync,
restart, and paired-evaluation gates.

## Product target

The release should support two explicit response modes:

- **Reasoning mode:** exactly one balanced `<think>...</think>` block followed by the user-facing answer. The
  reasoning may be long when the problem requires it, but should not repeat, restart, or run to the token cap.
- **Direct mode:** no mandatory visible reasoning block for ordinary chat, extraction, translation, or simple
  instructions. The final answer should be concise and follow the requested format.

For tool tasks, the model should plan, emit schema-valid calls, use observations, recover from a failed call,
and stop after achieving the task. Reward task success, not the mere presence of think tags or tool calls.

The release claim should be specific: leading open 9B capability across verified reasoning and coding, strong
agentic success, and the best measured coverage across the OpenEuroLLM language set, with retained 256K
architecture and long-context retrieval. “Best” is a scorecard and a Pareto claim, not one averaged leaderboard
number.

## What the completed pilot establishes

The real `openeurollm/oellm-9b-256k-sft` pilot completed 32/32 online updates on two LUMI-G nodes with non-zero
gradients throughout, 256/256 mixed reward groups, policy lag at most three, and a readable step-32 export.
It used 9.30 GCD-hours for training. On the locked 1,255-row GSM8K diagnostic it moved exact accuracy from
40.80% to 44.46%, reduced length stops from 26.93% to 16.57%, and reduced high repetition from 38.01% to
25.10%. See [the qualification record](qualification-gsm8k-reasoning-2026-09-05.md).

This is enough evidence to scale the mechanics, but not the data. GSM8K is potentially contaminated by the
parent SFT mixture, and a 32-update run does not test cross-domain retention.

## Stack architecture

```text
                         oellm-rlvr control plane
             manifests / Slurm / gates / artifacts / promotion
                                  |
              +-------------------+-------------------+
              |                                       |
       single-turn backend                       agentic backend
       TMAX/Open-Instruct                      SkyRL trainer/rollouts
       (qualified on LUMI)                              |
              |                                 Harbor generator
       math/code/IF verifiers                         |
              +----------------------+----------------+
                                     |
                  Singularity/Apptainer task sandboxes
                                     |
                         hidden deterministic verifiers

                 independent oellm-eval + agentic evals
                   decide checkpoint promotion/rejection
```

### Components and ownership

| Component | Choice | Reason |
|---|---|---|
| Cluster control and provenance | `oellm-rlvr` | It already handles LUMI Slurm/Ray, offline assets, topology, checkpointing, hashes, and health gates. |
| Single-turn learner/rollouts | pinned TMAX/Open-Instruct backend | It has completed a real 9B online run on MI250X. |
| Agentic learner/rollouts | SkyRL, after a LUMI qualification | SkyRL separates trainer and generator, has an official Harbor integration, and now documents AMD FSDP + vLLM execution. |
| Agent environments | Harbor | Shared task format for trace generation, evaluation, sandbox lifecycle, and verifiers; current Harbor includes a Singularity/Apptainer environment. |
| Fallback backend | verl FSDP + vLLM/SGLang | verl documents an AMD path, but its published ROCm instructions target MI300 and still require a LUMI-specific qualification. |
| Reference recipes/data | NVIDIA Nemotron v3, OpenR1, OpenThoughts | Mine their curricula, schemas, datasets, and reward implementations; do not adopt CUDA-specific launch stacks wholesale. |
| Independent evaluation | `OpenEuroLLM/oellm-eval`, JudgeArena, Harbor benchmark harnesses | Evaluation must not be owned by the same loop that optimizes reward. |

NeMo RL is a valuable recipe reference, not the first LUMI trainer. NVIDIA's published Nemotron 3 Super guide
uses a staged RLVR → SWE → RLHF sequence and CUDA/B200-oriented containers. Porting that full stack to MI250X
would add risk without replacing capabilities already present here.

## Fourteen-day readiness plan

Do these in parallel before the production checkpoint arrives.

### Days 1–3: freeze contracts and evaluation

- Define a versioned prompt record containing `task_id`, `messages`, `domain`, `language`, verifier identity,
  verifier revision, difficulty/pass-rate stratum, source revision, license, and contamination status.
- Make all training records answer-free from the policy's perspective. Hidden tests, reference answers, and
  judge rubrics remain sandbox/verifier-only.
- Freeze a release scorecard and private holdouts. Hash every prompt and use exact, normalized, MinHash, and
  semantic-neighbor checks against every SFT/DPO/RL source.
- Run parent baselines through `oellm-eval`, the OpenEuroLLM function-calling harness, and Harbor. Do not tune
  prompts on the primary holdouts.

### Days 3–6: build the trainable data catalogue

- Normalize candidate datasets into one manifest and preserve per-row license/provenance.
- Profile each prompt with eight sampled outputs from the actual parent. Store pass rate, output length,
  verifier failures, and reward variance.
- Admit mostly prompts with pass rate 0.10–0.90. Keep small all-wrong and all-right reservoirs for measuring
  frontier movement, but do not let zero-signal groups dominate optimization.
- Split by source problem and semantic cluster, not by row, before translations or prompt variants are made.

### Days 4–9: multi-domain single-turn support

- Add a task registry and reward router for math, code, science/MCQ, structured output, and deterministic
  instruction-following constraints.
- Compute advantages and quotas per domain. Cheap math generations must not numerically swamp slower code or
  minority-language batches.
- Add scheduled mixtures, pass-rate bins, curriculum state in restart checkpoints, and per-domain dashboards.
- Add soft overlength handling and mask infrastructure failures from policy reward. Retain binary outcome
  reward as the default; partial rewards require individually auditable subtests.

### Days 5–11: SkyRL + Harbor on LUMI

- Build a pinned Apptainer image from SkyRL's AMD FSDP/vLLM recipe. Confirm gfx90a/MI250X compatibility rather
  than assuming an MI300 example is portable.
- Use Harbor's Singularity backend with prebuilt SIF images and server dependencies baked into the image; LUMI
  compute nodes cannot install packages from the network during trial startup.
- Convert 16 small code/tool tasks into Harbor format and run oracle, deliberately failing agent, and model
  checks before RL.
- Pass one 9B agentic optimizer step, post-update rollout, weight sync, restart, and deterministic verifier
  replay on two nodes. Set a hard ceiling below 100 GCD-hours.

### Days 10–14: rehearsal

- Stage the exact incoming checkpoint format with a compatible dummy or current 9B export.
- Run a 64-update mixed single-turn rehearsal and a 16-update agentic rehearsal.
- Prove immutable exports, resumability after Slurm preemption, policy-version attribution, reward replay,
  independent evaluation, and artifact collection.
- Make a day-14 backend decision. If SkyRL + Harbor fails any real-checkpoint gate, keep Harbor tasks but use
  the qualified TMAX learner/rollout loop for the first production work; keep verl as the bounded fallback
  experiment, not a third simultaneous production stack.

## Starting-checkpoint gate

The current `oellm-9b-256k-sft` is a pipeline reference. Production RL should start from the incoming frozen
SFT/DPO candidate only after all of the following are recorded:

- immutable Hub/local revision, tokenizer, chat template, generation config, and SHA-256 manifest;
- BF16 load/generation on the exact LUMI container;
- balanced think tags in reasoning mode and clean direct-mode answers;
- baseline scorecard in English and every target European language;
- sampled pass-rate histograms for every proposed RL domain;
- no catastrophic regression relative to the best SFT checkpoint caused by the DPO stage;
- a 64-update real-checkpoint canary with mixed groups, at least 95% non-zero gradients, policy lag at most
  four, verifier/system errors below 2%, and a readable restart plus full export.

If the new checkpoint cannot produce any successes on the intended hard tasks, do not start RL there. First
do a small, high-quality reasoning/tool SFT bridge. Suitable inputs include the project's translated Dolci
Think subset and verified OpenThoughts/Nemotron traces, filtered for one valid think block, correct final
answer, language consistency, reasonable length, and clean licenses. The bridge should teach the interface,
not consume every available synthetic trace.

## Data programme

### Candidate sources

| Capability | Preferred candidates | Treatment before use |
|---|---|---|
| Math | `birgermoell/oellm-math-rlvr`, NVIDIA Nemotron-RL-Math-v2, DAPO-Math-17k, project-authored problems | Deduplicate against GSM8K/MATH/AIME/PolyMath; symbolic/exact verifier; parent pass-rate profile. |
| Competitive code | `birgermoell/oellm-code-rlvr`, NVIDIA competitive-programming data, OpenR1 Codeforces/verifiable Python | Run hidden tests in Apptainer; remove weak or leaked tests; cluster by underlying problem. |
| Science/knowledge | NVIDIA Nemotron-RL-Science-v1 and curated MCQ/open QA | Prefer exact, tool, or calibrated equivalence verification; separate CC BY-SA material in manifests. |
| Structured/instruction | NVIDIA structured-output, free-form-format, calendar, inverse-IFEval, and multi-turn instruction data | Deterministic checkers; adversarial prompt variants; never reward content-free format compliance alone. |
| Multilingual reasoning | `openeurollm/reasoning-traces-multilingual`, translated Dolci Think, newly translated answer-only RL prompts | Native-speaker audit; re-solve a stratified sample; maintain language-consistency checks. |
| Tool/function calling | NVIDIA agentic conversational/function-calling data and OpenEuroLLM function-calling corpora | Execute calls against deterministic local tools; schema and state-transition verification. |
| Terminal/SWE | Harbor/Terminal-Bench tasks, NVIDIA SWE pivot, SWE-Gym/R2E-Gym subsets | Prebuild SIFs, pin repos, validate oracle, hide tests, block network, enforce time/resource limits. |
| General preference/safety | existing OpenEuroLLM DPO data, carefully licensed GenRM/preferences, safety/abstention tasks | Keep separate reward scale; human spot-check; use mainly for the final recovery/alignment stage. |

Do not concatenate these raw datasets. Build an OpenEuroLLM-owned training release with immutable source
revisions, per-row license, normalized task schema, semantic clusters, pass rates from the production parent,
verifier versions, and explicit train/calibration/evaluation disposition.

### Language allocation

RL must not become English-only after multilingual SFT. In the first single-turn stage, reserve **20% of prompt
groups for non-English tasks**. Allocate half uniformly across language families/low-resource languages and
half by expected user volume. Use aligned translations only for comparisons; include independently authored
native tasks so the model cannot succeed by learning translation artifacts.

Start with programmatically verifiable math, code, structured output, and tool tasks, where the same answer or
environment can support many languages. Expand from the nine-language translated Dolci Think release to all
OpenEuroLLM languages only after native review and solver re-verification. Keep at least 50% English or
language-neutral replay in reasoning batches until ablations show that translated reasoning alone is stable.

## Production curriculum

The ratios below are prompt-group quotas, not raw dataset row shares. Each stage resamples by difficulty and
language; it does not attempt one epoch over every source row.

### Stage R0: parent profiling and optimizer selection

- Run three 64-update canaries from the identical parent with learning rates `3e-7`, `6e-7`, and `1e-6`.
- Use 40% math, 30% code, 15% deterministic instruction/structured output, 10% science, 5% simple function
  calling. Within every domain, 20% of groups are non-English.
- Keep group size eight initially. Compare DPPO-TV with the current threshold against one GRPO/Dr-GRPO-style
  control only if the backend implements the loss exactly and logs the estimator normalization.
- Select on held-out deltas, entropy/divergence, truncation, and repetition—not training reward alone.

Expected budget: **55–90 GCD-hours** total, including evaluation.

### Stage R1: single-turn verified reasoning and code

- Run 768 updates from the selected R0 configuration: 30% math, 30% competitive code, 15% science/knowledge,
  15% instruction/structured output, and 10% simple function calling; enforce the 20% multilingual quota
  across applicable domains.
- Curriculum: first 25% of updates draw mainly parent pass rate 0.35–0.80, middle 50% draw 0.15–0.70, final
  25% draw 0.05–0.55 while retaining 25% easier replay.
- Save every 64 updates. Run a fixed fast gate every 64 and the full scorecard at updates 256, 512, and 768.
- Stop a domain if its held-out gain saturates twice while its response length or cross-domain regression grows.

At the observed 1,024-token GSM8K throughput, 768 updates project to about **223 GCD-hours**. Code and longer
answers will raise this; reserve a **400 GCD-hour hard ceiling**.

### Stage R2: agentic function calling and terminal use

- Start from the best R1 checkpoint, not automatically the final R1 step.
- Phase A, 128 updates: single- and two-tool calls, deterministic state changes, repair after one tool error.
- Phase B, 256 updates: multi-turn workplace/data tasks, 2–8 turns, stateful tool use, structured final answer.
- Phase C, 256–512 updates: terminal coding and small repository repairs in Harbor SIF environments.
- Each agentic batch contains 20% retention replay: 8% math/code, 5% instruction, 4% multilingual, 3% safety
  and abstention. Compute domain-relative advantages before combining losses.
- Reward final environment success. Add only small, bounded penalties for invalid calls, unnecessary turns, and
  truncation. Never add a positive reward merely for calling a tool or writing a long plan.

Run initially on 2 LUMI-G nodes (8 learner + 8 rollout GCDs). If rollout utilization is the bottleneck and
weight synchronization is qualified beyond one rollout node, move to 3–4 nodes with 8 learner and 16–24
rollout GCDs. Expected budget is **900–2,000 GCD-hours**, with explicit stop points after each phase.

### Stage R3: software-engineering specialization

- Use 128–256 carefully selected SWE tasks for a first run, then scale the task count only after environment
  reliability exceeds 98% and task reward is independently replayable.
- Train on issue resolution, test repair, dependency/debugging, and repository navigation; do not optimize on
  release benchmark instances.
- Keep repository snapshots, SIF digests, test revisions, agent scaffold, and every observation in the trace.
- Check both success rate and attempts/tokens/time to solution. A 9B model should aim to be an efficient agent,
  not imitate the longest trajectories of a 100B+ teacher.

Expected budget: **600–1,500 GCD-hours**, depending on rollout horizon and sandbox concurrency.

### Stage R4: balanced recovery and final alignment

- Return to a 256–512 update mixture containing 35% verified reasoning/code, 20% multilingual instruction,
  15% agentic retention, 15% general chat/preferences, 10% safety/abstention, and 5% long-context tasks.
- Use a frozen reference checkpoint and monitor divergence. Prefer a small replay/on-policy-distillation or
  preference stage over continuing RL until reward saturates.
- Add a mild length objective only after correctness is stable. NVIDIA's staged recipe similarly ends with an
  RLHF length-penalty stage; this should be treated as a separate behavior objective with its own regression
  tests.

Expected budget: **150–400 GCD-hours**.

## Optimizer and rollout rules

- Use 8 samples per prompt for routine stages; use 16 only for difficult or noisy verifier domains where the
  added variance estimate changes sampling decisions.
- Active-sample zero-variance groups, but cap resampling. Record all rejected groups so apparent throughput
  cannot hide a curriculum that the model has exhausted.
- Maintain a 10–20% easy replay floor. Nemotron's published blends explicitly order higher-pass-rate examples
  before harder examples; DAPO's dynamic sampling likewise focuses updates on informative groups.
- Keep token-level and sequence-level statistics: reward, advantage, entropy, clipped fraction, divergence,
  response length, truncation reason, reward by length, and policy lag.
- Mask infrastructure failures from policy optimization and retry them separately. A timeout caused by the
  model is reward zero; a missing image or crashed verifier is not evidence about the policy.
- Use soft overlength shaping near the cap and mask irrecoverably truncated completions. Do not create a cliff
  where a nearly correct long proof and nonsense both receive the same learning signal for unrelated reasons.
- Require rollout/trainer log-prob agreement checks when the engines differ. NVIDIA documents that an older
  vLLM mismatch required masking affected sequences in its Nemotron recipe.
- Run FP32 reward/advantage aggregation, finite-gradient checks, and exact checkpoint/restart tests.
- Do not optimize a learned judge jointly with the policy. Version it, calibrate it against humans, and keep a
  substantial programmatically verified component in every run.

## Evaluation and promotion scorecard

### Capability suites

| Axis | Primary evaluation |
|---|---|
| Math/reasoning | clean project holdout, MATH-500, current AIME-style sets, and `oellm-eval` PolyMath EU tiers; report pass@1 and pass@k separately |
| Code | LiveCodeBench temporal splits, EvalPlus/MBPP+, BigCodeBench, and private unit-test tasks |
| Agentic terminal/SWE | Harbor-held-out terminal tasks and SWE-bench-style repository tasks with no train-repo overlap |
| Function calling | OpenEuroLLM function-calling suite, BFCL-style checks, and stateful multi-turn tool scenarios |
| European languages | `oellm-multilingual`: Belebele, FLORES, Global-MMLU EU, MGSM EU, PolyMath EU, generic multilingual, and INCLUDE; add ArenaHard-EU/JudgeArena |
| Instruction following | IFEval/IFBench-style deterministic constraints, structured outputs, MultiChallenge, and long multi-turn retention |
| Safety/calibration | jailbreak/refusal balance, prompt injection in tool tasks, truthfulness, and answer/abstain calibration |
| Long context | fixed NIAH/RULER-style binding plus real 32K/128K/256K retrieval and synthesis tasks in multiple languages |
| Efficiency | generated tokens, tool calls, turns, wall time, and success per rollout-GCD-hour |

### Promotion gates

A checkpoint advances only when:

- its target-domain paired delta is positive and the confidence interval is useful, or two independent clean
  suites agree on the gain;
- the average general/multilingual score drops by no more than one absolute point and no priority language
  drops by more than three points without a documented tradeoff decision;
- safety and prompt-injection success do not regress;
- 32K–256K retrieval does not drop by more than 5% relative, and the model still respects the native tokenizer
  and context metadata;
- repetition, unbalanced think tags, and truncation do not increase by more than five points;
- agentic environment errors remain below 2% and verified reward can be replayed from stored artifacts;
- at equal or better accuracy, median reasoning tokens and agent turns do not grow by more than 25%.

Never select the final checkpoint on GSM8K, a public SWE benchmark used in training, training reward, or an
LLM-judge score alone.

## Observability and release evidence

Every update and export must be attributable to:

- parent checkpoint and tokenizer hashes;
- repository, backend, container, SIF, dataset, and verifier revisions;
- exact prompt groups, sampled completions, rewards, advantages, logprobs, policy versions, and stop reasons;
- per-domain and per-language sampling counts;
- Slurm allocation, elapsed time, node/GCD-hours, failures, and retries;
- checkpoint completeness, optimizer/RNG/curriculum restart state, and post-restart equivalence;
- paired before/after evaluation predictions and blinded human-audit packs.

Publish model cards with the actual—not nominal—mixture, compute, selection rule, failed runs, contamination
limits, licenses, and evaluation hashes. Keep private tests private while publishing their construction and
aggregate results.

## Budget and calendar

| Work | Calendar after checkpoint freeze | Planning budget |
|---|---:|---:|
| Baseline, profiling, and R0 canaries | days 1–3 | 100–200 GCD-hours |
| R1 single-turn RLVR | days 4–8 | 250–400 GCD-hours |
| R2 agentic tool/terminal curriculum | weeks 2–3 | 900–2,000 GCD-hours |
| R3 SWE specialization | weeks 3–4 | 600–1,500 GCD-hours |
| R4 recovery/alignment | week 5 | 150–400 GCD-hours |
| full evals, repeats, and failed-job reserve | throughout | 300–700 GCD-hours |
| **Recommended programme** | **5–6 weeks** | **2,300–5,200 GCD-hours** |

These are reservation ranges, not a request to spend to the ceiling. The completed 32-update training run
cost 9.30 GCD-hours, implying 0.291 GCD-hours per update at that short single-turn protocol. Agentic costs are
more uncertain and must be re-estimated from the 16- and 128-update gates. Report physical MI250X devices and
LUMI-visible GCDs separately; the table uses GCD-hours because that is what the current job topology exposes.

## Go/no-go milestones

1. **Day 7 before checkpoint:** SkyRL AMD container and Harbor Singularity oracle trials work on LUMI.
2. **Day 14 before checkpoint:** 9B post-update agentic rollout and restart pass below 100 GCD-hours.
3. **Three days after freeze:** parent scorecard and trainable pass-rate strata are complete.
4. **R0 gate:** at least two learning rates improve clean target metrics without retention failure.
5. **R1 gate:** verified reasoning/code gain survives independent multilingual/general evaluation.
6. **R2 gate:** tool and terminal task success improves, environment failures remain below 2%, and reward replay
   matches online reward exactly.
7. **Release gate:** two independent evaluations and a blinded multilingual/agentic human audit prefer the
   candidate; all artifacts needed to reproduce the selected stages are frozen.

## Immediate implementation order

1. archive the completed GSM8K qualification and run the full retention scorecard on parent and step 32;
2. add multi-domain reward routing, per-domain advantage normalization, scheduled quotas, and curriculum state;
3. build the SkyRL AMD Apptainer image and exercise Harbor's Singularity environment on LUMI;
4. convert a 16-task code/tool canary and pass oracle/failure/model/reward-replay tests;
5. prepare revision-pinned, decontaminated R0 data with parent pass-rate profiles;
6. freeze the production scorecard and promotion thresholds before the new checkpoint is visible;
7. run the incoming checkpoint through the starting gate, then execute R0 rather than immediately launching a
   long run.

## Primary references

- [NVIDIA Nemotron post-training v3 collection](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)
- [Nemotron 3 Super RL blend and staged curriculum](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends)
- [Nemotron 3 Ultra RL blend](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends)
- [NVIDIA NeMo RL Nemotron 3 Super guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/docs/guides/nemotron-3-super.md)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL) and [its AMD example](https://docs.skyrl.ai/docs/examples/amd)
- [SkyRL + Harbor integration](https://novasky-ai.notion.site/skyrl-harbor)
- [Harbor](https://github.com/harbor-framework/harbor) and its [Singularity/Apptainer backend](https://github.com/harbor-framework/harbor/tree/main/src/harbor/environments/singularity)
- [verl AMD installation](https://github.com/verl-project/verl/blob/main/docs/start/install.rst)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [OpenR1](https://github.com/huggingface/open-r1)
- [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [OpenEuroLLM multilingual reasoning pilot](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual)
- [OpenEuroLLM evaluation control plane](https://github.com/OpenEuroLLM/oellm-eval)
