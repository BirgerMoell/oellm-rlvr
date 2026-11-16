# RL on the LAION OT-validated datasets on Leonardo: what we have done so far

Status as of 16 November 2026. Written from this working session; every number below was measured on Leonardo unless it says
otherwise. This file contains cluster paths and job IDs, so check it before publishing it.

## 1. Goal and where we are

**Goal:** run RLVR (RL with verifiable rewards) on the LAION collection
[`laion/ot-validated-rl-datasets`](https://huggingface.co/collections/laion/ot-validated-rl-datasets) on Leonardo (CINECA), using agentic tasks that run in Harbor sandboxes. Terminal-bench-2-1 and tblite-dev are evaluation-only and must
never be trained on.

**Where we are:** the environments now run on Leonardo and a model can be rolled out against them and scored. We have measured the
first pass rates and found a small set of tasks with mixed outcomes. **No training step has run yet.** The training hookup
(SkyRL + Harbor) is not ported, and the first rollout results are too thin to justify it until the rerun described in section 6
finishes.

| Stage | State |
|---|---|
| Download `rts-1k` (984 tasks) | done |
| Build x86_64 sandbox images | done: 696 of 801 unique images built |
| Validate tasks (oracle passes, nop fails) | done: 741 of 984 tasks usable |
| Roll out a model and score it | working; first pass rates measured |
| Find tasks with mixed outcomes | 12 found, a lower bound (46 % of trials were lost to errors) |
| Port the SkyRL training step | not started |
| Other 11 datasets | not started |

## 2. How the sandboxes work on Leonardo

Leonardo has no `apptainer` (SingularityPRO 4.3.1), login nodes have internet but disable user namespaces, and compute nodes have user
namespaces but no internet. The Harbor fork (`marianna13/harbor`, apptainer environment) builds images with `--fakeroot`, which fails
here (empty `/etc/subuid`). The working recipe:

- **Rootless builds on compute nodes:** `unshare -rm singularity build` with a small `LD_PRELOAD` shim (`fakeown.so`, from
  `scripts/laion_rootless/fakeown.c`) that makes `chown`/`setuid`-style calls succeed, plus apt set to run as root. glibc images only.
- **Internet for the builds:** a token-protected forward proxy on an internet-enabled serial node (`scripts/laion_proxy.py`,
  `leonardo_laion_proxy.sbatch`, 4 h limit) and a same-user loopback relay on each build node (`local_relay.py`) because Java
  ignores credentials in proxy URLs.
- **Problems solved on the way:** Docker Hub's anonymous pull limit (base images now come from `mirror.gcr.io`), a corrupted shared
  singularity cache (now one cache per job), and `$APPTAINER_ROOTFS` being empty under SingularityPRO (rewritten in the wrapper).
- **Runtime:** a Harbor "bridge" server plus per-node workers run the sandboxes; one bridge per job on the compute node.

Result: **696 of 801 unique images built** (four shards). About 105 fail, mostly task-specific npm `@jsr/...` 404s inside the
task images.

## 3. Validation: which tasks work here

`leonardo_laion_validate.sbatch` runs the oracle agent (must score 1) and the nop agent (must score 0), 3 attempts each, through
the bridge.

| Run | Compatible |
|---|---|
| 3-task smoke test | 3 of 3 |
| 100-task overfit subset | 79 of 100 |
| Full set (4 chunks) | **741 of 984** |

The merged list is `laion-validation/rts-1k-leonardo-compatible-all.jsonl` on Leonardo. The failures are mostly missing images or
sandbox start errors.

## 4. Rolling out a model

`scripts/leonardo_laion_agent_rollout.sbatch` serves the policy with vLLM (CUDA container, `--data-parallel-size 4`, one replica per
GPU) and runs Harbor's **Terminus-2** agent against it inside the sandboxes, then scores each trial with the task's verifier.
`scripts/analyze_rollout.py` classifies the outcomes.

Problems found and fixed, in order:

1. vLLM's spawned workers imported OpenCV and needed `libxcb` → run workers in fork mode.
2. Terminus-2 refuses `hosted_vllm` models without `model_info` → pass token limits and costs.
3. `tmux` in `ubuntu:20.04` and `debian:bullseye` images is too old for Terminus-2's session start (`unknown option -- e`) → exclude
   those bases (20 compatible tasks).
4. The vLLM startup wait (10 min) was shorter than loading 17 GB on 4 GPUs at once → wait up to 35 min.
5. The task agent timeout (600 to 1,800 s) was too short for 100-turn trials → timeout multiplier 3.

## 5. Results with Prelude-9B-SFT-v2

Model: `marianna13/prelude-9b-sft-v2` (Qwen3 architecture, SFT only; its own reported scores are 3 % on Terminal-Bench 2.1 and 7.7 %
on TBLite).

| Run | Trials | Scored | Passed | Main finding |
|---|---:|---:|---:|---|
| Smoke, 20 turns | 16 | 15 | 0 | the loop works end to end |
| Debug, 30 turns | about 119 | about 110 | 2 | 87 % used all 30 turns without finishing; 5 % hit the 900 s agent timeout; 3 % tmux error |
| **100 turns** (100 tasks × 4) | 400 | 203 | **17** (8.4 % of scored) | **12 tasks with mixed outcomes**; 88 tasks passed none |

Distribution in the 100-turn run: 9 tasks passed 1 of 4 attempts, 1 task 2 of 4, 2 tasks 3 of 4, none 4 of 4. The mixed tasks are the
first candidates for RL training. Because 186 trials were never scored (below), the pass rate and the count of 12 are lower bounds.

**What the failures are** (400 trials): 17 passed, 77 used all 100 turns without finishing, 57 stopped early (declared done, verifier
disagreed), 54 hit the 1,200 s agent timeout, and **186 errored** before or outside scoring:

| Error among the 186 | Count |
|---|---:|
| `[Errno 28] No space left on device` at sandbox start | 154 |
| `only root user can use sandbox as overlay in setuid mode` | 25 |
| old `tmux` (`-e`) | 7 |

**The "no space" problem is not solved.** All 154 happen at sandbox start, take 0 seconds, and begin only late in the run, so
something accumulates per container until new sandboxes cannot start. Candidates are the 1 GB overlay image per instance, the copy of
the image's writable directories (`/opt`, `/srv`, `/usr/local/bin`, ...) made per trial, and a per-instance `/tmp` with no size limit
that a runaway agent could fill. Slurm refused every node-local disk request above the 10 GB default (`tmpfs` gres, 40 to 200 GB). I
moved sandbox staging to RAM-backed `/dev/shm` and halved the overlay as a stopgap, which does not stop the growth. The rerun now logs
staging usage by component every minute, so the next failure will show which one grows.

## 6. In progress

Job `64574537` (profile rerun: the same 100 tasks, 100 turns, 4 attempts, `/dev/shm` staging, usage logging) is queued on a 4-GPU
node. It should score almost every trial and show what fills the staging space. It asks for a whole node for up to 4 hours, which
may compete with other experiments on the account.

## 7. Other models and the hallucination benchmark

- **E8 step160** (`birgermoell/oellm-9b-256k-stage5-e8-search-rl-step160`, a search and honesty specialist): a failed test on the
  agent harness. 41 of 45 trials died on a vLLM `BadRequestError` ("Think tags in content; move reasoning to reasoning_content"), a
  harness incompatibility, so we have no pass rate for it. Serving without the reasoning parser, or stripping think tags, is the
  likely fix and is not done.
- **Hallucination benchmark** ([BirgerMoell/oellm-hallucination](https://github.com/BirgerMoell/oellm-hallucination), 500 questions in
  10 languages, judged by `gemma-4-31B-it`) on Prelude-9B-SFT-v2: **97.7 % fabrication on made-up subjects in default mode (93.6 %
  thinking), 2.3 % honest "I don't know" (6.4 %)**, 100 % wrong on obscure real people, 53.8 % on well-known people (that counts
  partial answers; fully correct is 16 of 80), 0 % over-refusal. Worse than every baseline in the repo (E10 64 %, Qwen3-8B 74 %,
  Apertus-8B 83 %; release gate under 10 %). It was SFT-trained without any honesty or abstention data.
- **On GitHub** in that repo (three commits on `main`): the Prelude rows in the README and results tables, the regenerated chart, the
  metrics JSON, and a detailed error analysis (`results/prelude-9b-sft-v2-analysis.md`).

## 8. The web-search gateway

`src/oellm_rlvr/web_gateway.py` is a security sidecar so an agent can search and fetch the live web through a controlled interface
(only `web_search` and `web_fetch`; provenance rules for URLs, HTTPS only, SSRF and IP checks, redirect re-validation, hidden-text and
injection filtering, per-session budgets, audit log, record/replay cache; Wikipedia, Brave and SearXNG backends). 38 local tests pass.
It has **never been run on Leonardo**: `scripts/leonardo_web_gateway.sbatch` needs a venv with fastapi, uvicorn and pyyaml, and the
tools are not registered in any rollout config. Open decisions: which search backend, and whether to relay through a small VPS so
Leonardo's IPs aren't the source of scraping. Another session built a separate live-Wikipedia relay for its own pilot (commit
`76216c1`) and did not use this gateway.

## 9. Git: what is committed and what is not

- **Repo and branch:** `BirgerMoell/oellm-rlvr`, branch `leonardo-laion-ot-datasets`, which exists **only locally** (not pushed).
- **Committed on the branch:** `ace5088` (first Leonardo LAION scripts) and `76216c1` (the E10/E11 session's live-web pilot, committed
  into this same checkout).
- **Uncommitted, 21 paths.** Modified: `leonardo_harbor_apptainer_bootstrap.sh`, `leonardo_laion_download.sh`,
  `leonardo_laion_sif_build.sbatch`, `leonardo_laion_validate.sbatch`, `summarize_laion_validation.py`. New:
  `scripts/laion_proxy.py`, `scripts/laion_rootless/`, `leonardo_laion_proxy.sbatch`, `leonardo_laion_sif_build_rootless.sbatch`,
  `laion_sif_build_offcluster.sh` (obsolete fallback), `leonardo_laion_agent_rollout.sbatch`, `analyze_rollout.py`,
  `leonardo_halu_bench.sbatch`, `leonardo_web_gateway.sbatch`, `src/oellm_rlvr/web_gateway.py`,
  `tests/test_summarize_laion_validation.py`, `configs/web-gateway.example.yaml`, and three files in `docs/` (this write-up, the
  hallucination error analysis, and `halu-prelude-sft-v2-results.patch`, which is obsolete now that the rows are pushed). `data/`
  is also untracked and should not be committed.
- **A sensible split:** (a) rootless builds, proxy and validation; (b) agent rollout, analyzer and hallucination-benchmark scripts;
  (c) the web gateway with its tests. Nothing has been pushed from this repo.

## 10. Next steps

1. **Wait for the rerun (64574537)** and read its staging log. Fix whatever component fills the space (limit or drop the per-instance
   `/tmp`, shrink or skip the writable-prefix copy, clean up leaked staging), then re-run if needed.
2. **Decide the training pool.** Take the tasks with mixed outcomes. If the rerun still finds fewer than a few dozen, widen the pool:
   more attempts on more tasks, easier datasets such as `nemotron-terminal-easy-clean1k` or `tasktrove-curriculum-easy`, or a stronger
   starting checkpoint.
3. **Port the SkyRL training step** with the apptainer environment: bridge and workers inside the training job, CUDA and A100
   settings, 64 GB memory limits.
4. **E8 step160:** fix the think-tag error and re-test if that model matters for this work.
5. **Gateway:** pick a backend, build the venv, test on a serial node, register the tools.
6. **Housekeeping:** commit the uncommitted work in the three groups above, delete the obsolete files, and push the branch.

## 11. Gotchas worth remembering

- The Leonardo SSH certificate lasts about 12 hours. Renew it with `step ssh login <CINECA email> --provisioner cineca-hpc`.
- System `python3` on Leonardo is 3.6. Use the harbor venv's Python 3.12 for anything newer.
- `sbatch` copies the script at submit time, so editing a script does not change jobs already queued.
- The debug QoS (30 min) is the fastest way to get a node but is too short for a 4-GPU load plus 100-turn trials.
- Compute nodes have only a 10 GB node-local `/tmp`, and Slurm refuses a larger `tmpfs` request.
