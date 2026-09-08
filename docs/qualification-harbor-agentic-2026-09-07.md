# OELLM 9B Harbor agentic-rollout qualification

Date: 2026-09-07

## Outcome

The LUMI runtime path is qualified for real, learner-admissible Harbor rollouts from the frozen OELLM 9B SFT
checkpoint. Job `21792529` completed one real Terminus-2 episode inside an isolated Singularity sandbox and
passed the strict ATIF gate: exact prompt/completion token IDs, aligned per-token log-probabilities, linked shell
actions and observations, a finite Harbor verifier reward, no exception, and no private verifier leakage.

The parent model did not solve the repository repair. That is a model-interface result, not an infrastructure
failure. It either guessed an edit before observing the source or repeatedly copied an inspection command. An
explicit terminal-edit prompt also failed to produce JSON within the context window. Do not begin agentic RL
from this parent: first run a small agent-interface SFT bridge, or repeat the gate on the incoming SFT/DPO
checkpoint and require held-out positive rewards.

## Frozen components

| Component | Value |
|---|---|
| Logical parent | `openeurollm/oellm-9b-256k-sft` |
| Staged parent origin | `birgermoell/oellm-9b-256k-sft` |
| Parent revision | `08359ad61333263c067edaf290067fea5b103d34` |
| Architecture | `Qwen3ForCausalLM`, 36 full-attention layers, BF16, four shards |
| SkyRL | `NovaSky-AI/SkyRL@f5bc3b78dfddfb352870d5d7430cd226e5785838` (`v0.3.0`) |
| Harbor | `harbor-framework/harbor@4407eb5227a2ff4f0d3f16b2eb48849382fdf276` (`v0.22.0`) |
| LUMI image | `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif` (hash in each run manifest) |
| Qualified control-plane commit | `98a299d7413cabe0a7135ca22e0b05a824126037` |

The deliberately shortened component labels above are not a substitute for the machine manifests. Every run
stores the full source commits, model-config hash, SIF hash, environment, and compatibility report under its
artifact directory.

## Failure isolation and fixes

| Job | Change or question | Result | GCD-hours |
|---|---|---|---:|
| `21790604` | Router data path | Real shell actions, but router returned null vLLM token-ID extension fields | 0.767 |
| `21791260` | Direct one-engine vLLM path | Exact IDs/log-probs recovered; Qwen3.5-2B decoded at about 0.6 token/s and timed out | 0.924 |
| `21792287` | 9B full-attention model, compiled mode, INFO logging | About 62–64 token/s; 192-token responses all truncated before valid JSON | 0.349 |
| `21792529` | 1,024-token turns | **Infrastructure pass**; four real commands, 1,221 completion tokens, reward 0 | 0.252 |
| `21792842` | Forced inspect-first prompt | Zero parser errors, but the policy reread the same file four times; reward 0 | 0.251 |
| `21793673` | State-dependent inspect/edit prompt | Six repeated reads despite visible source; reward 0 | 0.261 |
| `21793845` | Explicit easy terminal edit | Both Harbor attempts filled the response/context budget without a JSON action | 0.362 |
|  | **Total** |  | **3.166** |

The direct vLLM endpoint is currently required for a one-engine session because the intermediate SkyRL router
preserves the extension keys but nulls `prompt_token_ids` and completion `token_ids`. Direct mode refuses a
multi-engine configuration so this cannot silently become a load-balancing/session bug.

The 2B canary was a poor MI250 performance proxy: its Qwen3.5 hybrid uses GDN/linear-attention kernels that fell
back to Triton on gfx90a. The real 9B checkpoint is a conventional full-attention model. With eager mode disabled
it loaded weights in 11.8–13.6 seconds, compiled/warmed in about 32–33 seconds, captured GPU graphs in one second,
and sustained roughly 51–65 generated tokens/s in the agent trials.

## Agent-interface bridge

`build-agentic-sft-bridge` now emits a deterministic, dependency-light bridge in LlamaFactory's OpenAI-message
format. It mirrors the Harbor v0.22.0 Terminus-2 JSON protocol and produces successful multi-turn oracle episodes
for the 16 project tasks. The generated validator checks role alternation, exact JSON, required fields, command
newlines, unique task IDs, and final completion confirmation. The manifest hashes the data, LlamaFactory mapping,
and protocol template.

The seed set is intentionally tiny and leaks its own oracle solutions, so every row is marked SFT-only. Its sole
purpose is to verify that a short bridge can teach the checkpoint to emit valid JSON and condition the next action
on terminal output. It is not an RL or evaluation dataset.

## Next gates

1. Overfit the 16-row bridge with a short LoRA or low-learning-rate full SFT run and merge/export a temporary
   checkpoint. This is a format experiment, not a model candidate.
2. Evaluate on held-out variants that share the protocol but not code, constants, or verifier clusters. Require at
   least 95% parseable actions, zero infrastructure errors, and positive reward on both an explicit edit and a
   repository repair.
3. Expand to at least 256 project-authored bridge episodes with train/calibration/evaluation cluster separation,
   then run a small SFT bridge only if the incoming SFT/DPO checkpoint still fails the same held-out gate.
4. Profile eight samples per held-out prompt at training temperature. Start GRPO only when at least one prompt
   group has both reward 0 and reward 1; all-zero or all-one groups produce no useful group-relative gradient.
5. Connect accepted Harbor ATIF trajectories to one synchronous SkyRL learner update, verify post-update weight
   sync, and only then scale task count or enable asynchronous rollouts.

## Qwen3.5-9B control (2026-09-08)

Job `21810547` repeated the strict `repo-repair-clamp` canary with the official post-trained
`Qwen/Qwen3.5-9B` checkpoint and no other task, sandbox, verifier, or rollout changes. It completed successfully
on two MI250 GCDs in 13 minutes 3 seconds. Most of the wall time was one-time compilation of Qwen3.5's hybrid
Gated Delta Network kernels on gfx90a; the actual four-turn agent interaction took about 39 seconds.

The model inspected `app.py`, made the correct minimal edit, reread the file, and confirmed completion. Harbor
awarded reward `1.0`. The strict qualification accepted the trajectory for RL with three Bash commands, linked
terminal observations, zero parser errors, four prompt/completion token-ID segments, and log-probability arrays
aligned to all 1,283 completion tokens.

This isolates the previous zero-reward result: the LUMI/SkyRL/Harbor data path can produce a successful,
learner-admissible agentic trajectory. The frozen OpenEuroLLM SFT parent's failure was policy behavior, not an
integration failure. A single deterministic success is not yet a training qualification; the next control is an
eight-sample group at training temperature to establish within-prompt reward variance.

The first repetition attempt, job `21810949`, exposed a bug in SkyRL v0.3.0's debugging-only
`main_harbor_generate` entrypoint: it hard-codes `repetition_id=0` and ignores `n_samples_per_prompt`. The strict
expected-trial gate rejected the run after one otherwise valid reward-1 trajectory. The repository now uses a
project-owned learner-off entrypoint that preserves SkyRL's setup but explicitly expands each prompt into unique
repetition IDs. The real learner entrypoint is unaffected; this repair makes its pre-training qualification
faithfully reproduce the grouped sampling shape.
