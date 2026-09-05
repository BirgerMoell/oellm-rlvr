# LUMI full-stack RL dry run

Status: implementation plan and executable campaign contract, 2026-09-05

## Decision

Run a six-day miniature of the complete post-training programme on the current
`openeurollm/oellm-9b-256k-sft` checkpoint before the next SFT/DPO checkpoint arrives. The dry run is not
intended to produce a release model. It is intended to prove every boundary that a production run will depend on:

1. immutable checkpoint and task identities;
2. parent evaluation and parent pass-rate profiling;
3. online rollouts, deterministic rewards, advantages, finite gradients, and optimizer updates;
4. full learner-to-sampler weight synchronization;
5. math, code, structured output, function calling, stateful tools, and repository repair;
6. checkpoint, forced interruption, restart, and policy-version continuity;
7. independent candidate evaluation and exact offline reward replay.

The machine-readable source of truth is
[`campaigns/lumi-9b-end-to-end-dry-run.yaml`](../campaigns/lumi-9b-end-to-end-dry-run.yaml). Validate it before
changing or reserving compute:

```bash
oellm-rlvr validate-campaign \
  --campaign campaigns/lumi-9b-end-to-end-dry-run.yaml
oellm-rlvr render-campaign \
  --campaign campaigns/lumi-9b-end-to-end-dry-run.yaml \
  --output /tmp/oellm-rl-dry-run.md
```

The manifest rejects duplicate stages, forward or missing dependencies, uncommanded stages marked ready, invalid
repository pins, and required stage ceilings above the campaign budget. It deliberately does not launch commands:
the control plane should submit a stage only after its dependency artifacts and gates have been checked.

## Framework selection

The shortlist was refreshed on 2026-09-05. Repository tags and commits are frozen in the campaign manifest rather
than following `main` during a run.

| System | What the upstream project establishes | LUMI evidence | Decision |
|---|---|---|---|
| Existing TMAX/Open-Instruct backend | Online grouped-policy training, vLLM actors, code sandboxes, and weight transfer | A real OELLM 9B run already completed 32 updates on 16 MI250X GCDs | Keep for single-turn reasoning and code |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | FSDP/Megatron training, vLLM generation, custom generators, HTTP inference, async correction, and an upstream Harbor integration | Its [AMD example](https://docs.skyrl.ai/docs/examples/amd) uses FSDP plus ROCm vLLM, but is a Docker/Tinker LoRA example and is not a gfx90a full-weight qualification | Primary agentic candidate; qualify `skyrl-v0.3.0` |
| [Harbor](https://github.com/harbor-framework/harbor) | Agent/task harness, deterministic graders, exportable trajectories, and RL integrations | Its current tree contains a native [Singularity/Apptainer environment](https://github.com/harbor-framework/harbor/tree/main/src/harbor/environments/singularity); the [cookbook](https://github.com/harbor-framework/harbor-cookbook) includes SkyRL training | Primary task, sandbox, verifier, and ATIF layer; qualify `v0.22.0` |
| [verl](https://github.com/verl-project/verl) | Mature multi-backend RL, colocated and fully async modes, FSDP/FSDP2/Megatron, vLLM/SGLang | The [official ROCm guide](https://verl.readthedocs.io/en/latest/amd_tutorial/amd_quick_start.html) covers MI300/MI350 (`gfx942`/`gfx950`), not LUMI MI250X (`gfx90a`), and includes downstream patches | One time-boxed fallback; qualify `v0.9.0` only if SkyRL misses its day-5 gate |
| [slime](https://github.com/THUDM/slime) | High-performance Megatron + SGLang path with custom multi-turn generation and fully async examples | Its [AMD guide](https://github.com/THUDM/slime/blob/main/docs/en/platform_support/amd_tutorial.md) explicitly limits the current ROCm image to MI300/MI325 and requires Megatron checkpoint conversion | Watch `v0.3.2`; not a faster gfx90a route than SkyRL/verl |
| [rLLM](https://github.com/rllm-org/rllm) | Clean general-agent tracing API and many agentic recipes | Distributed training is delegated to verl, so it does not remove the unresolved LUMI backend work | Mine its agent API; do not add a second task abstraction beside Harbor |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Mature Ray + DeepSpeed + vLLM single/multi-turn trainer | No official MI250X qualification was found | Keep as an algorithm/implementation reference, not a third port |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) + [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | The strongest public end-to-end recipe catalogue: multi-environment RLVR, SWE RL, RLHF, and many auditable environments | Published containers and performance recipes are CUDA-oriented | Mine recipes, data, and verifier designs; do not port the trainer first |
| [AReaL](https://github.com/areal-project/AReaL) | Strong asynchronous microservice architecture and agentic examples | No documented ROCm/MI250X qualification found | Watch, do not put on the two-week critical path |
| [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) + [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) | Good asynchronous trainer and task/harness abstractions | Upstream currently states that at least one NVIDIA GPU is required | Mine environment interfaces; do not port now |
| [TorchForge](https://github.com/meta-pytorch/torchforge) | PyTorch-native agentic RL abstractions and a ROCm installer | Its README says development is paused and consolidating elsewhere | Reject as a new dependency |
| [OpenR1](https://github.com/huggingface/open-r1) | Accessible TRL recipes, math/code data filtering, and verifier examples | Useful for references and small NVIDIA experiments, not our multi-turn LUMI control plane | Reference only |

The important choice is architectural: `oellm-rlvr` remains the project control plane, while training backends and
environment libraries remain replaceable. Do not create three competing task formats. New project-owned agentic
tasks use Harbor format, emit ATIF, and are adapted once into the control-plane trajectory record.

## Target stack

```text
                       oellm-rlvr campaign/control plane
          pins / Slurm / dependency gates / artifacts / promotion
                                  |
             +--------------------+--------------------+
             |                                         |
     qualified single-turn path                candidate agentic path
       TMAX/Open-Instruct                        SkyRL FSDP + vLLM
             |                                         |
     math + executable code                     Harbor generator
             |                                         |
             +-------------------+---------------------+
                                 |
                  Harbor Singularity task sandboxes
                                 |
               hidden deterministic tests and state checks
                                 |
                   independent oellm-eval + Harbor eval
```

For the dry run, use synchronous generation for the first 9B agentic update. Only enable asynchronous in-flight
updates for the 16-update mixed stage, after token IDs, loss masks, sampler weights, and policy versions agree.
This isolates correctness before throughput. Use eight samples per prompt, centered group-relative advantages,
and binary outcome rewards by default. If the async stage uses stale trajectories, enable SkyRL's explicit
off-policy correction and retain a policy-lag ceiling of four.

Do not reward `<think>` tags, long plans, or tool calls by themselves. Reasoning prompts should still require one
balanced `<think>...</think>` block followed by a final answer, while direct and tool prompts should follow their
native chat-template mode. Track malformed tags as a behavior metric and gate, not as a substitute for task
success.

## Data and environment plan

### Reuse now

| Capability | Initial source | Dry-run use |
|---|---|---|
| Math | [`birgermoell/oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr) | Select 128 semantically distinct prompts with parent pass rate 0.10–0.90; keep the earlier GSM8K run only as an infrastructure reference because of contamination risk |
| Competitive code | [`birgermoell/oellm-code-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-code-rlvr) | Select 64 easy/medium tasks with strong hidden tests and six-turn sandboxes |
| Function calling | [`birgermoell/oellm-eu-tooluse-v1`](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1) | Use the verifiable `grpo` split for name, arguments, relevance, and format checks; its card reports 46,366 English RL rows |
| Structured output and IF | [Nemotron RL collection](https://huggingface.co/collections/nvidia/nemotron-reinforcement-learning) and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | Start with JSON/schema and deterministic IFEval-style checks; reimplement the small verifier in the project task pack rather than importing all of NeMo Gym |
| Multilingual retention | [`openeurollm/reasoning-traces-multilingual`](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual) and [`Dolci-Think-SFT-translated`](https://huggingface.co/datasets/openeurollm/Dolci-Think-SFT-translated) | Use for format/reasoning retention and SFT bridge experiments, not as unreviewed RL truth; the reasoning-trace card explicitly calls for native review and solver verification |
| Evaluation | [`OpenEuroLLM/oellm-eval`](https://github.com/OpenEuroLLM/oellm-eval) | Freeze multilingual, PolyMath, general, direct-mode, safety, and long-context slices before training |

### Create before the dry run

Create an OpenEuroLLM-owned `oellm-rl-dryrun-v0` catalogue rather than concatenating upstream rows. It should
contain:

- 128 math prompts, 64 competitive-code prompts, 64 structured-output prompts, and 128 exact function-call
  prompts;
- 20% non-English prompt groups across at least Swedish, German, French, Polish, Finnish, and Romanian;
- a 16-task Harbor pack: four exact function-call tasks, four stateful two-tool tasks, four terminal-edit tasks,
  and four micro-repository repairs;
- at least 24 training, four calibration, and four private evaluation variants once the 16-task smoke passes;
- source revision, per-row license, semantic-cluster ID, language, verifier revision, SIF digest, parent pass rate,
  and train/calibration/evaluation disposition on every record.

New tool tasks should be generated from deterministic state machines and parameterized templates. This is better
than translating only the natural-language surface of English traces: the same tool schema and final state can be
verified identically in every language. New SWE dry-run tasks should use tiny project-authored repositories stored
as immutable git bundles or snapshots. Do not train on Terminal-Bench or SWE-bench release instances; use those
only as independent evaluation references.

Every task must pass four checks before it is admitted:

1. the oracle succeeds;
2. a deliberately wrong agent fails;
3. repeated runs produce the same reward;
4. hidden tests and expected state are absent from all policy-visible messages, files, observations, and traces.

## Six-day execution schedule

The dry run is a miniature curriculum, not a shortened production run. The budgets are deliberately generous for
bring-up and debugging. LUMI accounting below uses visible MI250X GCD-hours; one two-node hour is 16 GCD-hours.

| Order | Stage | What is exercised | Expected | Hard ceiling |
|---:|---|---|---:|---:|
| 1 | checkpoint freeze | immutable input and replacement contract | 0 | 0 |
| 2 | cluster preflight | BF16 load, gfx90a, Ray, RCCL, vLLM, offline Apptainer | 1 | 4 |
| 3 | parent fast eval | frozen row-level baseline and score aggregation | 12 | 32 |
| 4 | pass-rate profile | 8 rollouts/prompt, reward variance, difficulty bins, verifier health | 20 | 48 |
| 5 | 16-update reasoning canary | rollout → verify → advantage → gradient → sync → export | 6 | 24 |
| 6 | 4-update code canary | multi-turn bash, hidden tests, reward replay | 20 | 48 |
| 7 | SkyRL AMD smoke | FSDP, ROCm vLLM, optimizer and sampler reload on one node | 4 | 16 |
| 8 | Harbor task contract | SIF startup, oracle/failure checks, ATIF export | 0 GPU | 0 GPU |
| 9 | one 9B agentic update | complete SkyRL + Harbor full-weight boundary | 28 | 80 |
| 10 | 16-update mixed agentic stage | quotas, domain-relative advantages, async correction, retention replay | 96 | 192 |
| 11 | 4-update SWE stage | 8–16 turns, repo state, deferred tests, long trajectories | 64 | 160 |
| 12 | forced restart + 16 recovery updates | optimizer/RNG/queue/curriculum recovery and capability replay | 8 | 32 |
| 13 | final eval and replay | paired evaluation, all-reward replay, go/no-go report | 31 | 84 |
|  | **Required total** |  | **290** | **720** |
| C1 | verl fallback, only if triggered | same one-step agentic contract | 40 | 96 |

Reserve 750 GCD-hours for the required campaign. The expected use is 290 GCD-hours; the difference is failure and
queue-debug headroom, not a target to spend. The 96 GCD-hour verl fallback is separately authorized only by a
SkyRL no-go decision. More compute will not compensate for a failed token, weight-sync, or reward-replay gate.

### Stage transitions

- Do not train until the parent profile contains useful reward variance. If a domain is all wrong, insert a small
  SFT bridge or easier project-authored tasks. If it is all correct, move to a harder bin.
- Do not enable agentic async training until a synchronous 9B update completes below 80 GCD-hours and the
  post-update sampler is proven to serve the new learner weights.
- Do not start the mixed stage until all 16 Harbor tasks pass oracle/failure checks and at least 98 of 100
  environment starts succeed.
- Intentionally stop the mixed job immediately after a saved state. Resume it and compare the next sampled batch,
  policy version, and update with an uninterrupted control.
- Select the dry-run result on pipeline correctness. A small benchmark gain is welcome but is not the purpose;
  large regressions are still a failure.

## What must be implemented first

The campaign validator currently marks two training stages ready and eleven required stages as `build_required`.
The preflight and evaluation components exist, but they need standalone campaign wrappers before they count as
runnable stages. The critical path is:

1. add checkpoint fingerprinting, freeze the fast evaluation scorecard, and add a multi-domain task
   catalogue/profiler;
2. build the pinned SkyRL AMD Apptainer image and the Harbor SIF task pack in parallel;
3. add the SkyRL backend adapter and Harbor-to-ATIF/control-plane trajectory bridge;
4. pass the synchronous 9B agentic one-step gate;
5. add scheduled environment quotas, domain-relative advantages, capped active sampling, and complete restart
   state;
6. add campaign-wide reward replay and paired evaluation reports;
7. run the six-day dry run, freeze the resulting working revisions, then replace only the parent checkpoint
   manifest when the new SFT/DPO model arrives.

The two TMAX dry-run profiles are already committed:

- [`configs/lumi-dryrun-reasoning-oellm9b-16step.yaml`](../configs/lumi-dryrun-reasoning-oellm9b-16step.yaml)
- [`configs/lumi-dryrun-code-oellm9b-4step.yaml`](../configs/lumi-dryrun-code-oellm9b-4step.yaml)

The code profile expects `data/code-dryrun-oellm9b-s6/`, produced by the profiling/data stage. This dependency is
intentional; do not run the profile against an arbitrary unprofiled sample merely to obtain a green job.

## Backend switch rule

SkyRL is a go only if, by the end of its fifth implementation day, it passes the one-node AMD smoke and the
two-node 9B full-weight step with correct post-update sampling, no CUDA-only imports, token/loss-mask agreement,
and a restartable export below 80 GCD-hours. If it fails, spend at most 96 GCD-hours qualifying the identical
contract with verl. If both agentic candidates fail, continue production single-turn RLVR on the qualified TMAX
path while agentic training remains a separate engineering track. Do not hold the entire checkpoint schedule
hostage to a second simultaneous backend port.

## Success criteria

The dry run is successful only when:

- at least 95% of optimizer updates have finite, non-zero gradients;
- every rollout stores its learner policy version and policy lag never exceeds four;
- environment/verifier infrastructure failures are below 2% and never become negative policy examples;
- every online deterministic reward replays from frozen task, test, and environment artifacts;
- hidden tests never become model-visible;
- full model, optimizer, RNG, sampler, queue, and curriculum state resumes after forced interruption;
- direct-mode, reasoning-form, multilingual, safety, and long-context retention stay inside the frozen gates;
- the final report names an exact backend/container/repository set that can accept the upcoming checkpoint by
  replacing one checkpoint manifest.

## Source-derived lessons used here

- NVIDIA's [Nemotron 3 Super recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md)
  uses multi-environment RLVR, SWE RL, and final RLHF as distinct phases. The dry run mirrors those boundaries at
  small scale instead of pretending one mixed job proves everything.
- The public [Nemotron RL data collection](https://huggingface.co/collections/nvidia/nemotron-reinforcement-learning)
  spans math, code, agentic tasks, instruction following, knowledge, and safety. Use it as a catalogue of task and
  verifier designs; preserve each row's license rather than treating the collection as one homogeneous corpus.
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) exposes deterministic environments for code, instruction
  following, structured outputs, workplace tools, calendar state, and SWE. These define useful behavior contracts
  even though the full NVIDIA execution stack is not the first LUMI target.
- SkyRL's [agent integration guide](https://docs.skyrl.ai/docs/tutorials/agent-integration) makes the generator the
  boundary for existing harnesses such as Harbor. That is the correct adapter seam for this repository.
- Harbor's [ATIF specification](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)
  carries multi-step actions, observations, rewards, and optional RL fields. Preserve ATIF as the raw agent trace
  and store the smaller `oellm-rlvr` record as the campaign index.
- [verl's ROCm guide](https://verl.readthedocs.io/en/latest/amd_tutorial/amd_quick_start.html) is credible fallback
  evidence but explicitly targets newer AMD architectures and notes downstream inference patches, which is why a
  gfx90a qualification remains mandatory.
