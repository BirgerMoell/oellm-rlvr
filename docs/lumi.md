# LUMI runbook

## 1. Stage immutable inputs

Use shared project or scratch storage for the control repository, the pinned TMAX checkout, model snapshots, datasets, task images, outputs, Hugging Face cache, and the Python venv. Record each Git commit and dataset/model revision. Compute nodes are offline, so a missing artifact will fail after allocation.

The default runtime is:

```text
/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/
  lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
```

It provides PyTorch 2.10 ROCm 7, gfx90a vLLM 0.22.1 with native weight transfer, Transformers 5.10, DeepSpeed, PyArrow, and LUMI's Slingshot/RCCL integration. The bootstrap adds Ray and missing backend libraries in a venv while retaining these system packages.

## 2. Bootstrap once

Run `scripts/bootstrap_lumi_env.sh` on a login node with network access. Do not use the backend's CUDA-oriented `uv sync`. Re-run bootstrap when the base SIF or pinned backend commit changes; do not mutate a venv used by an active run.

After bootstrapping, stage `hamishivi/Qwen3.5-2B` on shared storage. Keep the canonical repository ID in
`model.name_or_path` and set `model.local_path` to the staged snapshot. The control-plane launcher resolves
TMAX, Transformers, and vLLM to that local directory, so the offline compute job does not need a token or a
complete Hugging Face cache layout.

The LUMI profiles enable guarded, lazy compatibility hooks in every driver and Ray worker.
The hooks run only when the relevant vLLM module is imported, so Ray's generic prestarted workers remain cheap:

- `OELLM_PATCH_VLLM_MAMBA_ENUM=1` normalizes the unused `CUSTOM=None` Mamba backend enum value that
  vLLM's own `msgspec` IPC decoder rejects for Qwen3.5.
- `OELLM_PATCH_VLLM_WEIGHT_UPDATE=1` wraps TMAX's packed update in the `start_weight_update` /
  `finish_weight_update` transaction newly required by vLLM 0.22.1.
- Math profiles use `OELLM_PATCH_MATH_EQUIV_THREADS=1` because the pinned verifier invokes symbolic LaTeX
  comparison in an executor thread while its timeout tries to install a main-thread-only `SIGALRM` handler.
  The hook skips that signal operation off the main thread and rejects extracted expressions over 512
  characters before symbolic parsing.
- Code profiles use `OELLM_PATCH_SWERL_APPTAINER_KWARGS=1` to register the LUMI `slurm_apptainer`
  backend and filter prepared-backend defaults before pinned TMAX constructs a plain Apptainer backend.

Remove each flag after upgrading TMAX or vLLM to a pair that implements the corresponding behavior natively.

## 3. Qualify in layers

The one-node smoke profiles target LUMI's `dev-g` partition; the four-node production profile targets
`standard-g`. Change the partition only when local allocation policy requires it.

Run these in order:

1. `oellm-rlvr validate` and `topology` locally.
2. `doctor` on LUMI to resolve paths and the pinned Git commit.
3. A one-node preflight allocation that imports torch/Ray/vLLM/Open-Instruct and sees eight GCDs.
4. The bounded math smoke, proving rollout → verifier → learner step → weight broadcast.
5. The code smoke, additionally proving Apptainer reset, bash interaction, deferred tests, and reward parsing.
6. A signal-qualification run with active sampling and nonzero within-group reward variance.
7. A two-node weight-sync run before the four-node production profile.

The committed one-node smoke profiles disable active sampling and zero-standard-deviation filtering so they
finish even when a weak base policy receives identical rewards. The four-node profile enables active sampling.
Do not scale a run that has all-equal rewards, zero gradients, near-total truncation, high verifier errors, or
unbounded policy lag.

The LUMI profiles currently use `sequence_parallel_size: 1`. In the pinned August 2026 container, FLA's
context-parallel `chunk_delta_h` Triton kernel can fail AMD MLIR compilation on variable real-rollout shapes
even though the fixed-shape dummy step passes. Raise sequence parallelism only after a real learner-step smoke
passes in the exact replacement container/FLA/Triton combination.

## 4. Code sandbox image

The trainer container and task container have different roles. The LUMI AI Factory SIF runs PyTorch, Ray,
and vLLM. Nested setuid Apptainer is unavailable inside that SIF, while unprivileged nested execution depends
on user namespaces that may be exhausted or disabled. The `slurm_apptainer` backend therefore launches the
host Singularity runtime in a same-node overlapping Slurm step. It binds only an isolated per-episode
workspace, output, logs, tests, root, and temporary directory into the read-only task image.

Build the smoke task sandbox once on a login node:

```bash
bash scripts/build_lumi_task_sandbox.sh \
  /scratch/project_465002530/users/bmoell/sandboxes/python-3.12-slim
```

The smoke image contains only Python 3.12 and its standard library. Build a task-specific image with the
required runtimes and test dependencies for broader code training; generated code does not need the training
stack.

Each task directory contains:

```text
task-data/<task-id>/
├── instruction.md
├── image.txt
├── environment/seeds/...
└── tests/test.sh
```

The backend copies only seeds to `/workspace` during reset. At submission it uploads `tests/`, executes `/tests/test.sh`, and reads `/logs/verifier/reward.txt`. A test script must always write a reward on success and should leave no partial positive reward on failure.

For repeated task images, use `prepared_apptainer` after the basic `apptainer` smoke succeeds. Prepared mode amortizes setup and image work but has more cache/state invariants; keep its cache on fast shared or node-local storage, never `$HOME`.

## 5. Ray and networking

The generated job starts one Ray node per Slurm node under a background `srun --overlap`. It advertises all eight GCDs explicitly and keeps Ray's Unix sockets in short node-local `/tmp` paths. The driver runs once on the Ray head. `NCCL_SOCKET_IFNAME` selects LUMI's `hsn` interfaces in the cross-node qualification profile.

The LAIF `lumi-multitorch-full` image contains a matched ROCm/ROCr userspace and the Libfabric RCCL network plugin. The generated ROCm launch intentionally does not pass Singularity's `--rocm`: that flag replaces image libraries with host libraries and can produce a node-dependent HSA ABI mismatch. CUDA profiles still use `--nv`.

Qualify the native transfer communicator before loading a multi-billion-parameter model:

```bash
# Proven cross-node control: trainer rank 0 plus one independently masked engine.
sbatch --export=ALL,PROBE_WORLD_SIZE=2 scripts/lumi_weight_transfer_probe.sbatch

# Intended scale-up: trainer rank 0 plus eight independently masked engines.
sbatch --export=ALL,PROBE_WORLD_SIZE=9 scripts/lumi_weight_transfer_probe.sbatch
```

The probe uses vLLM's production `NCCLWeightTransferEngine`, initializes the same stateless communicator, and broadcasts a sentinel GPU tensor. It emits one JSON record per rank and fails after three minutes. On 2026-08-26 the two-rank topology completed over `NET/Libfabric` in about 5.5 seconds, while the nine-rank topology stalled while building local `P2P/IPC` links among the eight engine processes. The real-checkpoint qualification therefore uses one rollout engine until a wider probe completes. Do not scale the engine count merely because model loading succeeds.

If Ray fails to join, check name/IP resolution and Slurm step overlap before changing training code. If native weight transfer fails, verify that the LUMI vLLM is still the system build and that pip did not replace torch or vLLM.

## 6. Monitoring and promotion

Track at least:

- mean and per-domain reward;
- fraction of prompt groups with zero reward standard deviation;
- response length and clipped/truncated fraction;
- tool parser, environment reset, sandbox timeout, and test failure counts;
- entropy and KL/divergence metrics;
- rollout policy version, learner version, and maximum lag;
- generation, verification, weight-sync, and learner step throughput.

The defaults reject more than 80% zero-signal groups, 15% truncation, 2% verifier/system errors, or policy lag above four learner versions. Task failures are valid reward-zero outcomes; infrastructure failures are not and must remain separately labeled.

## 7. Recovery

Outputs, trainer checkpoints, traces, and rollout shards must live on shared storage. Ray state is ephemeral. On preemption or node failure, allocate a fresh cluster and resume from the backend checkpoint state; never try to reuse a half-dead Ray cluster. Keep run YAML, rendered sbatch, Git revisions, SIF path/digest, dataset revisions, and logs together under the run directory.
