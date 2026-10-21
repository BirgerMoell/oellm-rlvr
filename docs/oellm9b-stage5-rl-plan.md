# OpenEuroLLM 9B — Stage 5 RL plan (data, curriculum, rewards, tiers)

> Canonical copy of the post-training RL plan; the SFT/DPO stages it follows are documented in
> [qwen35-posttrain/docs](https://github.com/BirgerMoell/qwen35-posttrain) (`OELLM_POST_TRAINING_ROADMAP.md`).
> It uses this repository's control plane unchanged: measured-difficulty bins (`build-curriculum-pools`),
> admission gates, fresh-optimizer windows from the selected export, and paired held-out selection.


Status 2026-10-21. Starts from the Stage-4 (DPO) selected checkpoint. Executed with
[`oellm-rlvr`](https://github.com/BirgerMoell/oellm-rlvr) (TMAX/Open-Instruct DAPO backend for single-turn domains;
SkyRL + Harbor for multi-turn agentic tasks). This plan adds the data, curriculum and reward design; the control
plane, gates and checkpoint-selection rules of `oellm-rlvr` are kept as they are.

## 0. What we already know

- The pipeline works on OELLM 9B: rollouts, verifiers, nonzero gradients, weight sync, restart
  ([math canary](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-canary-step16-experimental): DAPO, β=0,
  LR 5e-7, 8 prompts × 8 samples, 16 updates; +1.66 points on an in-distribution procedural holdout).
- Gains so far are small and narrow (multilingual math 36.3 % → 38.9 %, p≈0.06; DAPO-Math near floor), and the
  nominal 80/20 mix realized as ~98/2. Lessons: verify the *realized* mix, profile difficulty on the *current*
  checkpoint, select only on frozen held-out sets, and expect most of the value from harder, more diverse pools
  than from the procedural easy tier.
- Our model will now have a real thinking mode (Stage 2) and tool use (Stage 3), so RL can train long reasoning,
  which the earlier canaries (on an instruct-only parent) could not.

## 1. Chat-template and output contract (all domains)

- Template v2.1, our tokenizer (`<think>`=18, `</think>`=17). Never the threeway repo's template.
- **Thinking-mode domains** (math, code, hard IF, tool reasoning): generation prompt ends with `<think>\n`; the reward
  parser reads only text after the first `</think>`; a missing `</think>`, an empty answer, or a second `<think>` ⇒
  reward 0 (also a loop guard).
- **Default-mode domains** (strict format, short IF, tool calls without reasoning): no think tokens allowed; any
  `<think>`/`</think>` in the output ⇒ reward 0. This keeps the non-thinking mode trained, not just preserved.
- Max response length grows by window (4K → 8K → 16K) with DAPO soft overlong punishment in the last 20 %.

## 2. Algorithm

DAPO-style GRPO, as already qualified in `oellm-rlvr`:
group size 8 (16 for the hard window), temperature 1.0, clip-higher (ε_low 0.2 / ε_high 0.28), token-level policy-gradient
loss, **dynamic sampling** (drop all-0 / all-1 groups; refill), β = 0 (no KL; monitor entropy and length instead),
constant LR 5e-7 (1e-6 only if gradients are tiny and stable), max policy lag 4, full state saved every 16 updates.
Each curriculum window starts a fresh optimizer from the previous window's **selected** export.

## 3. Data — prompt pools with verifiable answers

All pools keep `id`, `source`, `source_license`, `language`, `domain`, `difficulty_prior`, `verifier_kind`,
`ground_truth` outside the model-visible `messages`. Licenses below are from the HF cards (checked 2026-10-21).

### Math
| Pool | Size | License | Notes |
|---|---|---|---|
| [birgermoell/oellm-math-rlvr](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) | 1M (760K en + 10K × 24 EU langs) | Apache-2.0 | exact integer/rational answers, difficulty 1–5, semantic groups aligned across languages — **the multilingual backbone** |
| [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) | 103K | MIT | decontaminated, difficulty 1–9 (95K at 5–9) — main hard tier |
| [POLARIS-Project/Polaris-Dataset-53K](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) | 53K | Apache-2.0 | difficulty-filtered for RL |
| [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | 17K | Apache-2.0 | integer answers; also our external diagnostic family (hold a split out) |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) (verified) | ~94K problems | Apache-2.0 | `correctness_math_verify` rows only |
| [SynthLabsAI/Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) | ~250K | Apache-2.0 (gated) | filtered for RL-verifiable answers |
| [nvidia/Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath) | 1M | CC-BY-4.0 + Qwen-license note | answers from `\boxed{}` in R1-Distill solutions; keep rows whose answer parses and is single-valued; diverse synthetic problems |
**Multilingual math beyond oellm-math-rlvr**: translate a ~20K stratified slice of DeepMath/POLARIS/PrismMath *problems*
(never answers) into the 24 EU languages + Nordic/Balkan targets with the Dolci translator (Gemma), keeping the
English semantic-group ID so all variants stay in one split; verify the answer is unchanged by re-solving a sample.

### Code
| Pool | License | Notes |
|---|---|---|
| [birgermoell/oellm-code-rlvr](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr) | Apache-2.0 | 100K stdin/stdout tasks, 10–13 mutation-checked hidden tests |
| [agentica-org/DeepCoder-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | MIT | competitive-programming problems with tests; exclude anything overlapping LiveCodeBench windows used for eval |
| PrimeIntellect/verifiable-coding-problems | card has no license → verify before use | |
| ~~KodCode~~ | CC-BY-NC-4.0 | excluded |

### Instruction following (precise constraints)
| Pool | License | Notes |
|---|---|---|
| [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) | ODC-BY | IFEval/IFBench-style programmatic constraints (up to 5) |
| [allenai/Dolci-RLZero-IF-7B](https://huggingface.co/datasets/allenai/Dolci-RLZero-IF-7B) | ODC-BY | IF RL prompts used for Olmo 3 |
| [allenai/Dolci-Instruct-RL](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL) | card has no license field → verify | 170K prompts: math, code, IF, chat |
| **Multilingual IF (new)** | ours | translate the *instruction* part of language-neutral constraints (word/paragraph counts, JSON, bullets, case, end phrase) into target languages; drop constraints that do not transfer (English-specific casing/keywords) |
| **Strict-format set (new)** | ours | from our EUROPA failures: "number only", literal keyword (`INSUFFICIENT`), letter-only MCQ, exact JSON — default mode, exact-match verifier |

### Tool use
| Pool | License | Notes |
|---|---|---|
| [birgermoell/oellm-eu-tooluse-v1](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1) `grpo` | Apache-2.0 | **46K prompts with gold calls** — the GRPO split, used as-is (re-rendered with v2.1) |
| [nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) | CC-BY-4.0 | conversational tool-use RL items |
| no-call items | ours | from openeurollm-tool-use-v1 categories `no_call_direct` / `ask_for_info` / `cannot_answer` (held-out from SFT rows) |
Multi-turn agentic RL (Harbor tasks) only after single-turn tool RL passes its gate.

### Hygiene (all pools)
Dedup by normalized problem text and semantic group; decontaminate (13-gram + exact) against every eval below,
our SFT/DPO prompts *are allowed* (RL on seen prompts is fine) but eval prompts are not; reject rows whose ground truth
the verifier cannot parse, multi-answer or proof questions, and anything needing images.

## 3b. RL in three environment tiers (simple → agentic)

RL runs as a sequence of tiers ordered by **environment complexity**, each with its own difficulty windows (§4) and
promotion gate; a tier starts only from the previous tier's selected checkpoint. Licenses are from the HF cards
(checked 2026-10-21); "verify" = card has no license field.

### Tier A — single-turn, programmatic verifiers (TMAX/DAPO; cheap, start here)
Everything in §3 (math, code stdin/stdout, IF, strict format, single-call tool use) plus:
| Pool | License | Adds |
|---|---|---|
| [MiniMaxAI/SynLogic](https://huggingface.co/datasets/MiniMaxAI/SynLogic) | MIT | logic puzzles with verifiers |
| [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1) | CC-BY-4.0 | Reasoning-Gym tasks (algebra, graphs, games, cognition) |
| Reasoning Gym library (generate ourselves) | Apache-2.0 (code) | unlimited procedural tasks with difficulty knobs — and we can render prompts in our 24+ languages, like oellm-math-rlvr |
| [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified) | Apache-2.0 | general STEM / science QA with verified answers (beyond math) |
| [nvidia/Nemotron-CrossThink](https://huggingface.co/datasets/nvidia/Nemotron-CrossThink) | CC-BY-4.0 | multi-domain (STEM, humanities, law, social science) QA with templated, verifiable answers |
| [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | ODC-BY | IF RL prompts |
| [nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) | CC-BY-4.0 | function-calling RL |
| [nvidia/Nemotron-RL-coding-competitive_coding](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding) | CC-BY-SA-4.0 | competitive coding with tests (share-alike applies to a republished pool) |
| nvidia/Nemotron-RL-knowledge-mcqa | verify | knowledge MCQ (letter-only answers also train our strict-format behaviour) |
Excluded: Enigmata-Data and KodCode (CC-BY-NC).

### Tier B — multi-turn with local, simulated environments (SkyRL; moderate infra, no containers per task)
| Environment | Data / license | Reward |
|---|---|---|
| Search agent over a **local Wikipedia index** (Search-R1 / ASearcher setup: BM25 or e5 + FAISS server, ~50–60 GB) | [inclusionAI/ASearcher-train-data](https://huggingface.co/datasets/inclusionAI/ASearcher-train-data) (Apache-2.0); NQ/HotpotQA mixes (verify) | exact match / F1 on the final answer, format guard |
| Conversational tool use with simulated users and mock tools | [Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1), [Nemotron-RL-agent-workplace_assistant](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant) (CC-BY-4.0); τ²-bench gym interface for *evaluation* only | task-state checks, as in τ²-bench |
| Multi-step tool loops on executable mock tools | our openeurollm-tool-use-v1 multi-step rows (held out from SFT) | final-answer / end-state match |
| Agent safety | [Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) (CC-BY-4.0) | does not follow injected instructions in tool outputs |
| Text-to-SQL (optional) | NovaSky-AI/SkyRL-SQL-653-data (verify); BIRD/Spider train splits (check terms) | execution match on a local SQLite DB |
Multilingual: translate user turns of the search and tool tasks (answers stay checkable), as in Stage 3.

### Tier C — sandboxed agents (SkyRL + Harbor; already backend-qualified in `oellm-rlvr`)
| Environment | Data / license | Size |
|---|---|---|
| Terminal tasks | [allenai/open-instruct-endless-terminals](https://huggingface.co/datasets/allenai/open-instruct-endless-terminals) (Apache-2.0), [camel-ai/SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env) (Apache-2.0), [Lite-Coder/LiteCoder-Terminal-RL-preview](https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-RL-preview) (MIT), [Nemotron-RL-Agentic-Terminal-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1) (CC-BY-4.0) | ~1K–4.5K envs each |
| Software engineering | [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (MIT, 2.4K), [SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) (MIT), [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) (Apache-2.0, 4.6K), [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) (CC-BY-4.0; keep only rows whose `license_name` is permissive), [Nemotron-RL-Agentic-SWE-Pivot-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1) (CC-BY-4.0) | 10K+ tasks |
Eval only: Terminal-Bench 2.x ([harborframework](https://huggingface.co/datasets/harborframework/terminal-bench-2.0)), SWE-bench Verified, SWE-rebench leaderboard —
decontaminate training repos/tasks against them. Tier C needs an SFT warm-start on agent trajectories first
(e.g. the Stage-3 multi-step data plus terminal/SWE SFT traces) — RL from a model that cannot yet complete any task
yields all-zero groups; the 0/8 rule (§4) will show this before compute is spent.

Reference blend: [nvidia/Nemotron-RL-Ultra-Training-Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends)
(CC-BY-4.0) documents how NVIDIA mixed these domains — useful to compare against our proportions.

## 4. Slicing into gradually harder problems

Source difficulty labels (oellm 1–5, DeepMath 1–9, code 1–5) are **priors for candidate selection only**; admission
is by measured pass rate on the current checkpoint (8 samples, thinking mode, window's max length):

| Bin | pass@8 count | Use |
|---|---|---|
| saturated | 8/8 | excluded (eval/replay check only) |
| easy | 5–7/8 | warm-up, replay |
| medium | 2–4/8 | main signal |
| hard | 1/8 | frontier signal |
| impossible | 0/8 | parked; re-profiled at every window boundary (some become hard) |

Windows (each 32–64 updates, fresh optimizer from the selected export, re-profiled at the boundary):

| Window | Max response | Mix (by prompts) | Domains |
|---|---|---|---|
| W1 warm-up | 4K | 30 % easy, 60 % medium, 10 % hard | math (oellm EU24 + DeepMath 3–5), strict format, IF ≤2 constraints |
| W2 core | 8K | 10 % easy, 55 % medium, 35 % hard | + DeepMath 5–7, POLARIS, PrismMath, oellm-code d1–3, IF ≤5, tool calls |
| W3 frontier | 16K | 5 % replay, 45 % medium, 50 % hard (incl. re-profiled former 0/8) | + DeepMath 7–9, DeepCoder, code d4–5, multilingual hard math |

Domain proportions inside a window (starting point, adjusted to keep ≥40 % mixed groups per domain):
math 40 % (≥40 % of it non-English), IF + strict format 20 %, code 20 %, tool calls 15 %, default-mode replay 5 %.
The realized mix is logged per batch and must stay within ±5 points of target or the window is stopped.

## 5. Rewards

| Domain | Reward | Notes |
|---|---|---|
| Math | 1 if the final answer (last `\boxed{}` after `</think>`) is symbolically equal to ground truth, else 0 | existing thread-safe `MathVerifier`; reject multiple different boxed answers |
| Code | 1 if all hidden tests pass in the sandbox, else 0 | partial credit only later and only with mutation-tested suites |
| IF | 1 if every constraint passes, else 0 (multi-constraint) | IFEval/IFBench checker functions; language-aware word counting |
| Strict format | 1 on exact normalized match | default mode |
| Tool calls | ToolRL-style in [0, 1]: valid `<tool_call>` JSON (0.2) + correct tool set (0.3) + argument match (0.5); no-call items: 1 iff no call and the right behaviour class | parsed with the same logic as vLLM's `hermes` parser |
| **Language consistency** (multiplier, all non-English prompts) | × 0 if the final answer is not in the prompt language (fastText/lingua ID, code/math spans stripped) | M-Thinker-style LC reward; in thinking mode, trace language is logged, not rewarded, in v1 |
| Structure guards | 0 for missing/duplicated think delimiters, empty answer, repetition loops (n-gram repeat ratio), overlong (soft) | |

No learned reward model or LLM judge in v1 (reward-hacking risk). Every verifier is versioned, unit-tested on
adversarial outputs (multiple boxed answers, answers only inside the trace, prompt echo, empty code), and the
top-reward rollouts of each window are manually audited.

## 6. Evaluation and promotion gates

Frozen, decontaminated, never used for training or for choosing data:
- Math: MATH-500, AIME 2024/2025, GSM8K, MGSM, [PolyMath](https://huggingface.co/datasets/Qwen/PolyMath) (multilingual), the
  oellm-math-rlvr 512-row multilingual holdout, a DAPO-Math held-out split.
- Code: LiveCodeBench (post-cutoff window), HumanEval+/MBPP+, oellm-code-rlvr test split.
- IF: IFEval, IFBench test; multilingual IF holdout.
- Tools: BFCL v4 (incl. irrelevance), When2Call test.
- Retention: EUROPA Eval and Alignment (thinking off and on), think-tag rate in default mode (= 0), closed-trace
  rate in thinking mode, identity check, median response length, language-consistency rate per language.

A window's checkpoint is promoted only if its target-domain holdouts improve (paired test) **and** no retention metric
regresses beyond noise; otherwise the next window starts from the previous selected checkpoint. Training reward is
never a selection criterion.

## 7. Compute and platform

`oellm-rlvr` is qualified on LUMI (MI250X); a CUDA profile exists but has not run on Leonardo. Plan: run W1 on LUMI
with the qualified stack; qualify Leonardo with a 16-update canary (~10 GPU-h) in parallel. Rough budget at 32–64
updates per window, 64–128 prompts × 8–16 samples per update and growing response lengths: W1 ~60–100, W2 ~150–250,
W3 ~250–400 A100-equivalent GPU-h, plus ~10 % for profiling and evals.

## 8. Order of work (Tier A first; B and C only after A's gate)

1. Pool builders (one per source → common schema) + decontamination + verifier-parse checks; publish an aggregated
   **`birgermoell/openeurollm-rlvr-pool-v1`** (prompts + ground truth + priors, per-row license; NC sources excluded).
2. Verifier work: language-consistency multiplier, IF checker port (multilingual word counting), tool-call reward,
   structure guards — each with adversarial unit tests.
3. Multilingual slices: math problem translation (+ answer re-check), IF instruction translation, strict-format set.
4. Profile W1 candidates on the Stage-4 checkpoint (8 samples), apply bins, check admission gates
   (10–70 % accuracy, ≥40 % mixed groups, ≤15 % length stops, ≤2 % errors).
5. W1 → evaluate → select → re-profile → W2 → W3, per the gates above.
