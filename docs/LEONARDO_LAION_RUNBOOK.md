# Runbook: LAION OT-validated RL environments on Leonardo

For humans and for AI agents picking up this work. It says what exists, what state it is in, how to run each step, what has gone
wrong before and how it was fixed, and what to do next. Written 16 November 2026 from a working session. For the narrative and the
measured results see [leonardo-laion-rl-writeup.md](leonardo-laion-rl-writeup.md). **If something here disagrees with the live
cluster, trust the cluster and fix this file.**

## 0. Read this first (rules for agents)

- **The account and the cluster are shared.** Other sessions run their own experiments on the same Slurm account. Never cancel, edit
  or reuse a job, directory, port, token or file you did not create. Check `squeue -u $USER` before assuming a job is yours.
- **Do not publish or push without the user's explicit instruction.** Commits are local until told otherwise. This repo and the
  benchmark repo are public. Never put tokens, passwords, personal e-mail addresses, usernames or cluster paths into commits,
  issues, pull requests or outside requests.
- **Do not run code you did not read**, and do not install new third-party packages on the cluster without asking.
- **Leonardo compute nodes have no internet.** Anything that downloads must run on a login node or through the build proxy.
- **Do not poll.** Use a background monitor or `squeue` checks spaced out; the harness blocks chained `sleep`.
- **Report honestly.** If a run failed or a number is uncertain, say so. Several early diagnoses here were wrong (see section 7).
- Terminal-Bench 2.1 and TBLite are **evaluation only**. Never train on them.

## 1. Goal and status

Goal: run RLVR on [`laion/ot-validated-rl-datasets`](https://huggingface.co/collections/laion/ot-validated-rl-datasets) on Leonardo,
using Harbor agentic tasks that run in apptainer-style sandboxes.

| Step | Status (16 Nov 2026) |
|---|---|
| Download `laion/rts-1k-rootless-apptainer` (984 tasks) | done |
| Build x86_64 sandbox images (801 unique) | done, 696 built; about 105 fail (task-specific npm `@jsr` 404s) |
| Validate tasks: oracle must pass, nop must fail | done, **741 of 984 usable** |
| Roll out a model against the tasks and score it | working |
| Find tasks with mixed pass/fail outcomes | first pass: 12 tasks, a lower bound; rerun pending |
| Port the SkyRL training step | **not started** |
| The other 11 datasets in the collection | not started |

No training has happened. The reason to hold off is data, not code: with Prelude-9B-SFT-v2 only about 8 % of scored trials pass.

## 2. Facts about Leonardo that shaped everything

- Scheduler: Slurm. GPU partition `boost_usr_prod` (4 × A100 64 GB per node, 32 cores). Debug QoS `boost_qos_dbg` gives nodes quickly
  but only 30 minutes. `lrd_all_serial` is a login-type partition with internet. No `dcgp_usr_prod` budget on our account.
- Container runtime is **SingularityPRO 4.3.1** (`singularity`, no `apptainer`). No podman, buildah, enroot or proot. `ptrace_scope=2`.
- **Login nodes**: internet, but user namespaces disabled. **Compute nodes**: user namespaces enabled, **no internet**.
- `/etc/subuid` is empty, so `singularity --fakeroot` fails everywhere.
- System `python3` is 3.6. Use the Harbor venv's Python 3.12.
- Node-local `/tmp` (the job's `TMPDIR`) is **10 GB** by default; Slurm rejects larger `tmpfs` requests (tried 40 to 200 GB).
- The SSH certificate lasts about 12 hours. Renew from your own machine (needs your login and second factor, an agent cannot do it):
  `step ssh login '<CINECA e-mail>' --provisioner cineca-hpc`, then check `ssh leonardo hostname`.
- `sbatch` copies the script at submit time. Editing a script does not change jobs already queued.
- A `sbatch` job writes logs to `logs/%j...` relative to the submit directory. Submit from a directory that has a `logs/` folder.

Configure site settings once: copy [../leonardo.env.example](../leonardo.env.example) to `leonardo.env` (git-ignored), fill in
`OELLM_ACCOUNT`, `OELLM_WORK`, `OELLM_SCRATCH`, then `source leonardo.env` before `sbatch`. Everything below uses `$OELLM_SCRATCH`.

## 3. Layout

Repo files (this repository):

| Path | Purpose |
|---|---|
| `scripts/laion_rootless/singularity` | Wrapper that makes `singularity build` work without root (section 5.1) |
| `scripts/laion_rootless/fakeown.c` | `LD_PRELOAD` shim: `chown`/`setuid`-style calls return success |
| `scripts/laion_rootless/local_relay.py` | Loopback relay that adds proxy credentials for Java (which ignores URL credentials) |
| `scripts/laion_proxy.py`, `leonardo_laion_proxy.sbatch` | Token-protected HTTP/CONNECT proxy on an internet node (4 h) |
| `scripts/leonardo_laion_sif_build_rootless.sbatch` | Builds sandbox images for one dataset or one shard |
| `scripts/leonardo_laion_validate.sbatch` | Oracle and nop validation, writes `leonardo-compatible.jsonl` |
| `scripts/summarize_laion_validation.py` | Summarises validation results (keyed on the task directory name) |
| `scripts/leonardo_laion_agent_rollout.sbatch` | vLLM + Terminus-2 + sandboxes + verifier, writes `rollout-summary.json` |
| `scripts/analyze_rollout.py` | Classifies rollout outcomes (passed, used all turns, stopped early, each error type) |
| `scripts/leonardo_halu_bench.sbatch` | Hallucination benchmark job (section 5.6) |
| `src/oellm_rlvr/web_gateway.py`, `scripts/leonardo_web_gateway.sbatch` | Web-search gateway sidecar (section 6). Tested locally, never run on Leonardo |
| `docs/leonardo-laion-rl-writeup.md` | Narrative status and measured results |

On Leonardo (under `$OELLM_SCRATCH/oellm-rl/`): `venvs/harbor-laion` (Python 3.12 with Harbor), `src/harbor-laion` (the Harbor fork
`marianna13/harbor` at commit `82efa984`), `data/laion-ot/<dataset>/tasks`, `sif-cache-x86_64/` (built images), `proxy/`
(`proxy.host`, `proxy.token`, `fakeown.so`), `laion-validation/` (validation outputs and
`rts-1k-leonardo-compatible-all.jsonl`), `laion-rollouts/<dataset>-<jobid>/` (rollout outputs), `models/` (downloaded checkpoints).
Copy or clone the repo's `scripts/` to the cluster before submitting (login nodes have internet, so `git clone` works).

## 4. Daily commands

```bash
ssh leonardo 'squeue -u $USER -o "%i %j %T %M %r"'              # what is running or queued
ssh leonardo 'sacct -u $USER -S now-1day -X -o JobID,JobName%20,State,Elapsed'
ls $OELLM_SCRATCH/oellm-rl/sif-cache-x86_64/*.sif | wc -l        # images built (the cache also holds .build.json files)
```

Count `*.sif` only. Counting every file in the cache doubles the number (each image has a `.sif.build.json`). This mistake was made once.

## 5. Procedures

### 5.1 Build images (once per dataset)

Rootless build recipe, in short: `unshare -rm singularity build` (root inside a single-id namespace), the `fakeown.so` shim so file
ownership changes succeed, apt told to run as root, internet through a token-protected proxy plus a per-job loopback relay.

1. Start the proxy on an internet node (it exits after 14,000 s, so plan for that):
   `sbatch -A $OELLM_ACCOUNT scripts/leonardo_laion_proxy.sbatch`. It writes `$PROXY_DIR/proxy.host` and `proxy.token`.
2. Prepare a context-hash list (one hash per line) and split it into shards. **There is no script for this yet**; the lists used so far
   were produced ad hoc. Writing one is open work (section 9).
3. Submit the shards (about 100 images per hour overall with four shards of 3 to 4 workers; the first 696 took about 5 hours):

```bash
DATASET=rts-1k-rootless-apptainer CONTEXT_HASH_FILE=$PWD/shard-00 SIF_BUILD_WORKERS=3 \
  sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL scripts/leonardo_laion_sif_build_rootless.sbatch
```

4. Build jobs read the proxy address once at start. **If the proxy expires first, restart a proxy, then resubmit the shards.**
   Finished images are skipped, so no work is lost.
5. Image failures about 13 % are expected (npm `@jsr/...` 404s and similar). `glibc` images only; alpine/musl images do not work with the shim.

### 5.2 Validate tasks

```bash
DATASET=rts-1k-rootless-apptainer SUBSET=list TASK_LIST=$PWD/val-chunk-00 ATTEMPTS=3 \
  sbatch -A $OELLM_ACCOUNT --qos=boost_qos_dbg --time=00:30:00 --export=ALL scripts/leonardo_laion_validate.sbatch
```

`SUBSET` is `overfit`, `diagnostic`, `all` or `list` (with `TASK_LIST`, one task id per line). About 250 tasks per 30-minute debug job
is the limit. The output `leonardo-compatible.jsonl` lists tasks where the oracle scored 1 and the nop agent 0 on every attempt.
Merge the chunks into one list for the rollout job.

### 5.3 Roll out a model

```bash
MODEL=$OELLM_SCRATCH/oellm-rl/models/<model> DP_SIZE=4 N_TASKS=100 ATTEMPTS=4 CONCURRENT=48 \
  WORKERS_PER_NODE=28 MAX_TURNS=100 SEED=2 \
  sbatch -A $OELLM_ACCOUNT --time=04:00:00 --export=ALL scripts/leonardo_laion_agent_rollout.sbatch
```

What it does: starts vLLM (`DP_SIZE` replicas, one per GPU), starts the Harbor bridge server and workers, waits for vLLM (loading 17 GB
on four GPUs takes about 7 to 10 minutes), then runs Harbor's **Terminus-2** agent on `N_TASKS` shuffled tasks from the validated list,
and writes `rollout-summary.json` (per-task rewards, exceptions). Run `scripts/analyze_rollout.py <run>/jobs/rollout --max-turns <N>` for a
breakdown.

Useful variables: `TASK_LIST` (default is the validated list), `MAX_TURNS`, `TEMPERATURE` (0.6), `MAX_MODEL_LEN` (32768),
`MAX_TOKENS_PER_TURN` (4096), `AGENT_TIMEOUT_MULT` (3.0), `EXCLUDE_BASE` (see 7), `STAGING_ROOT` (default `/dev/shm`),
`BRIDGE_OVERLAY_SIZE_MB` (512), `VLLM_WAIT_STEPS` (5 s steps, default 420).

The job also writes `service-logs/staging-usage.log` (sandbox staging size by component, every minute) and `vllm.log`, `worker.log`,
`server.log`.

### 5.4 Read results

Rewards are in `jobs/rollout/<task>__<id>/verifier/reward.txt` and `result.json`. A **mixed task** (some attempts pass, some fail) is a
training candidate. A task with 0 of 4 passes gives no RL signal, and a task that passes 4 of 4 is too easy. Trials that errored
(`exception_info` set) have no reward; count them separately, never as zero.

### 5.5 Models used so far

- `marianna13/prelude-9b-sft-v2`: Qwen3 architecture, ChatML template with `<think>` and Hermes tool calls. Works in the harness.
- `birgermoell/oellm-9b-256k-stage5-e8-search-rl-step160`: fails in the harness (vLLM rejects think tags in message content). Not a
  terminal-agent model.

### 5.6 Hallucination benchmark

Repo: `BirgerMoell/oellm-hallucination` (500 questions in 10 languages; judge `gemma-4-31B-it` on 2 GPUs).

```bash
git clone https://github.com/BirgerMoell/oellm-hallucination $SCRATCH/halu-bench-<tag>
BENCH=$SCRATCH/halu-bench-<tag> SCRATCH=$SCRATCH MODELS="<label>=<model dir>" \
  sbatch -A $OELLM_ACCOUNT --time=02:00:00 --export=ALL scripts/leonardo_halu_bench.sbatch
```

It uses the project container and a Python with vLLM 0.24.0 (needed for Gemma 4), and the judge model already on disk at
`/leonardo_work/OELLM_prod2026/models/google/gemma-4-31B-it`. That setup matches the benchmark repo's baseline, so numbers compare.
About 25 minutes to generate and 25 minutes to judge.

## 6. Web-search gateway (not in use yet)

`src/oellm_rlvr/web_gateway.py` is a sidecar so agents can search and fetch the live web through a restricted interface: only
`web_search` and `web_fetch`, URLs only from search results, fetched pages or an operator seed, HTTPS on port 443, no IP literals,
allow and deny lists, redirects re-validated hop by hop, connections pinned to validated IPs, size and time caps, HTML reduced to text
with hidden text and chat-template markup neutralised, per-session budgets, an audit log, and a record/replay cache. 38 unit tests pass
(`python -m pytest tests/test_web_gateway.py`). Not done: deploying it on a node with internet (`leonardo_web_gateway.sbatch` needs a venv with
`fastapi`, `uvicorn`, `pyyaml`), registering the tools in a rollout, choosing a backend (Wikipedia keyless, Brave with a key, or
SearXNG), and deciding whether to relay through a small VPS. Compute nodes can reach a service on a serial node over the cluster
network, as another session demonstrated with its own relay.

## 7. Known failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `singularity build --fakeroot` fails | empty `/etc/subuid` | use the rootless wrapper (5.1) |
| Build: `$APPTAINER_ROOTFS` empty, `mkdir /app` fails | SingularityPRO only defines `SINGULARITY_ROOTFS` | wrapper rewrites the def file |
| Build: Maven "Network is unreachable" | Java ignores proxy credentials in URLs | loopback relay + `JAVA_TOOL_OPTIONS` (done). Clojure tooling still ignores it, so 1 known image fails |
| Build: `TOOMANYREQUESTS` from Docker Hub | one shared proxy IP | base images pulled from `mirror.gcr.io` (done) |
| Build: `invalid character 'e'/'}' after top-level value` | corrupted shared singularity cache (written during rate limiting and by parallel jobs) | one cache dir per job (done); delete a poisoned cache |
| Build jobs die when the proxy ends | proxy limit is 4 h and build jobs read the address once | restart proxy, resubmit shards |
| Summary shows task "failed" though it passed | Harbor reports the `task.toml` name, not the directory name | summarizer keys on the task directory (done) |
| vLLM: `libxcb.so.1` missing during startup | spawned workers import OpenCV | `VLLM_WORKER_MULTIPROC_METHOD=fork` (done) |
| Every trial fails at once: `hosted_vllm models require model_info` | Terminus-2 needs token limits and costs | `--agent-kwarg model_info=...` (done) |
| Trial error: `tmux: unknown option -- e` | `tmux` too old in `ubuntu:20.04`, `ubuntu:18.04`, `debian:bullseye`, `debian:buster` images | those bases are excluded by default (`EXCLUDE_BASE`) |
| Job exits at 10 min: vLLM "did not become ready" | 4 replicas load 17 GB from Lustre at once | wait up to 35 min (done) |
| Most trials use every turn and fail | turn limit (30) too short, tasks allow 600 to 1,800 s | use 100 turns and `AGENT_TIMEOUT_MULT=3` |
| `AgentTimeoutError` after about 60 to 90 turns | task agent timeout (600 s × multiplier) | raise the multiplier |
| vLLM `BadRequestError: Think tags in content` | model writes `<think>` in content, the Qwen3 reasoning parser leaves it, Terminus-2 sends it back | open: serve without `--reasoning-parser` or strip think tags |
| `[Errno 28] No space left on device` at sandbox start, late in a run | per-sandbox staging fills the 10 GB node-local disk. **Exact component not identified** | stopgap: staging in `/dev/shm` (RAM, counts against `--mem`). Check `staging-usage.log` |
| `only root user can use sandbox as overlay in setuid mode` | a sandbox (directory) overlay started in setuid mode | 25 of 400 trials; not investigated |
| `Permission denied (publickey…)` | SSH certificate expired | renew (section 2) |
| Debug job fails right at launch: `OELLM_SCRATCH: source leonardo.env first` | environment not passed | `source leonardo.env`, `--export=ALL` |

Diagnostic mistakes to avoid: an error counter grouped by message text is meaningless if the text contains job IDs (normalise
first); "COMPLETING" is a normal state, not an ending; a job that is `PENDING` for hours on a busy node is waiting for priority, not
broken. Prefer the debug QoS for short jobs.

## 8. Measured results to date (Prelude-9B-SFT-v2)

- 3-task smoke test: oracle 1.0, nop 0.0 on all three.
- Validation: 79 of 100 (overfit set), 741 of 984 (full set).
- Agent rollouts: 0 of 15 (20 turns); 2 of about 110 scored (30 turns; 87 % used every turn); **17 of 203 scored (100 turns)** with 12
  mixed tasks. 186 of 400 trials had no score because of sandbox errors (154 disk, 25 overlay, 7 tmux).
- Hallucination benchmark: 97.7 % fabrication on made-up subjects (default mode), 93.6 % (thinking), 2.3 % and 6.4 % honest "I don't
  know". Worse than every baseline in the benchmark repo.

## 9. What to do next (in order)

1. Check the profile rerun (the same 100 tasks, `/dev/shm` staging, usage log). Read `staging-usage.log` to see which component grows,
   and fix it: limit or drop the per-instance `/tmp` bind, shrink or skip the pre-copy of writable prefixes (`BRIDGE_WRITABLE_PREFIXES`),
   or clean up leaked staging. Rerun if needed.
2. Build the training pool from the mixed tasks. If there are too few (under a few dozen), widen: more tasks and attempts, easier
   datasets such as `laion/nemotron-terminal-easy-clean1k` or `laion/tasktrove-curriculum-easy`, or a stronger starting checkpoint.
3. Port the SkyRL training step with the apptainer environment: bridge and workers inside the training job, CUDA and A100 memory
   limits, the reward and verifier plumbing. The LUMI version is `scripts/lumi_harbor_agentic_train.sbatch`.
4. Write a script to generate and split the image context-hash lists (step 5.1.2).
5. Decide on the gateway (section 6). Fix the E8 think-tag error if that model matters.

## 10. Glossary

- **SIF**: a Singularity/Apptainer image file; one per unique task build context.
- **Harbor**: the task and sandbox framework. A **task pack** is a directory of tasks (`task.toml`, `environment/Dockerfile`, `tests/`).
- **Bridge**: Harbor's apptainer environment service. A server plus workers start sandboxes and execute commands in them.
- **Terminus-2**: Harbor's terminal agent; it drives a `tmux` session in the sandbox and talks to the model through litellm.
- **Oracle / nop agent**: the reference solution (must score 1) and the do-nothing agent (must score 0). Both must hold for a task to be
  usable for RL.
- **Mixed task**: some attempts pass and some fail. The only kind that gives a policy-gradient signal.
