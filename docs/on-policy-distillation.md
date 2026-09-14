# On-policy distillation

`oellm-rlvr` supports on-policy distillation (OPD) as an opt-in `verl_opd` backend. Existing configurations
continue to default to the pinned TMAX/Open-Instruct backend, so their commands and behavior do not change.

## What this implementation does

For each training step:

1. The current student policy samples its own completion with a colocated vLLM rollout engine.
2. A frozen teacher scores those exact sampled response tokens.
3. verl computes a token-level distillation loss from the student and teacher log-probabilities.
4. FSDP updates the student, and the next batch is sampled from that updated student.

The data therefore stays on the changing student's distribution. This is the important difference from offline
knowledge distillation, where a fixed set of teacher-generated answers can drift away from the student's actual
behavior during training.

The teacher does **not** have to be larger. It can be an equal-size specialist or even a smaller model. Capability
improves only where the teacher supplies a better conditional distribution on the chosen prompts, so an unrelated
or weaker teacher can also cause regressions. Use held-out capability and regression evaluations to decide whether
the transfer worked.

This direct logit implementation requires the student and every teacher to use exactly the same tokenizer and
token IDs. Architectures and parameter counts may differ. The OPD preflight authenticates checkpoint manifests
and rejects tokenizer mismatches before Ray starts. A different chat template is reported as a warning: verl sends
the student's token IDs directly, but the teacher may still have been tuned for a different prompt convention.

## Supported boundary

- Pure prompt OPD with one sample per prompt and no task reward.
- Hybrid single-turn math OPD+RLVR with at least two samples per prompt.
- One teacher, or multiple teachers routed by the dataset's `data_source` field.
- vLLM or SGLang teacher inference as supported by the pinned verl checkout. Student rollout currently uses vLLM.
- FSDP/FSDP2 training with synchronous, colocated student generation and a separate teacher GPU pool.

Code-agent OPD, asynchronous verl rollout, cross-tokenizer distillation, and a TMAX-native distillation loss are
not implemented. Configuration validation rejects the first two rather than silently dropping the settings.

## Install the backend

The adapter is pinned to verl commit `a9f2985159536a607211dcac730d3f5d55028950`, which contains the OPD trainer
used by this configuration:

```bash
git clone https://github.com/volcengine/verl.git /shared/oellm/verl
git -C /shared/oellm/verl checkout a9f2985159536a607211dcac730d3f5d55028950
cd /shared/oellm/verl
uv sync --frozen --all-packages --extra vllm --extra fsdp
uv pip install --python .venv/bin/python -e /shared/oellm/oellm-rlvr
```

That upstream lock is CUDA-specific. Do not install it unchanged on LUMI: it would replace the LAIF image's ROCm
PyTorch and vLLM packages with CUDA builds. The included LUMI profile is a runtime-qualification target for a
ROCm-compatible verl overlay, not a claim that upstream OPD is already qualified on MI250X. Build the overlay
without replacing the image's GPU packages, then require this generated job's import/GPU preflight to pass. verl's
published AMD results currently cover MI300-class hardware; LUMI MI250X must be qualified independently.

The backend environment must import `torch`, `ray`, `vllm`, `verl`, `verl.trainer.main_ppo`, and
`transfer_queue`. The generated Slurm job checks those imports on every allocated node.

## Prepare checkpoint contracts

Create and fully verify one immutable manifest per local checkpoint. Reuse a manifest only when student and teacher
really are the same checkpoint.

```bash
oellm-rlvr checkpoint-manifest \
  --model /shared/oellm/models/student \
  --model-id org/student \
  --revision IMMUTABLE_COMMIT \
  --output /shared/oellm/manifests/student.json

oellm-rlvr checkpoint-manifest \
  --model /shared/oellm/models/teacher \
  --model-id org/teacher \
  --revision IMMUTABLE_COMMIT \
  --output /shared/oellm/manifests/teacher.json

oellm-rlvr verify-checkpoint-manifest --manifest /shared/oellm/manifests/student.json
oellm-rlvr verify-checkpoint-manifest --manifest /shared/oellm/manifests/teacher.json
```

`verify-checkpoint-manifest` re-hashes all files and is the expensive integrity check. `opd-preflight` later checks
the signed manifest content, configured root/revision, and student-teacher tokenizer contract without re-reading
every weight shard.

## Prepare prompts

Input rows may be JSON, JSONL, or Parquet and need a non-empty `messages` or `prompt` list containing only system
and user messages. Do not include teacher answers or assistant messages: the student must generate the trajectory.

```json
{"id":"p1","messages":[{"role":"user","content":"Explain why the sky is blue."}]}
```

Convert them to verl's schema:

```bash
oellm-rlvr prepare-opd-data \
  --source prompts.jsonl \
  --output opd-train.parquet \
  --data-source general-teacher
```

For hybrid math OPD+RLVR, add `ground_truth` to every source row and pass `--require-ground-truth`. The adapter
then selects the project-owned exact/numeric math reward instead of the neutral reward.

With multiple teachers, run the converter once per routed shard. Each `--data-source` value must equal the `key`
of exactly one teacher, then list all prepared shards under `datasets`.

## Configure the run

Start from the
[`configs/cuda-opd-qwen35-2b-contract-smoke.yaml`](../configs/cuda-opd-qwen35-2b-contract-smoke.yaml) profile for
the upstream CUDA environment. The
[`configs/lumi-opd-qwen35-2b-contract-smoke.yaml`](../configs/lumi-opd-qwen35-2b-contract-smoke.yaml) profile is the
corresponding MI250X qualification target. Replace backend, dataset, checkpoint, revision, manifest, and output
paths in either profile.

The important topology invariant is:

```text
physical GPUs = learner GPUs + teacher GPUs
teacher GPUs = sum(num_replicas × TP × DP × PP for every teacher)
student rollout GPUs = learner GPUs, because rollout is colocated
```

For one teacher, verl reserves the name `teacher_model`. For multiple teachers, use distinct names such as
`math_teacher` and `reasoning_teacher`; `teacher_model` is deliberately rejected because upstream verl treats it
as a special default entry.

The initial recommended loss is:

```yaml
distillation:
  loss_mode: k1
  use_policy_gradient: true
  use_task_rewards: false
```

This treats the sampled reverse-KL estimate as a token reward. `forward_kl_topk` is also available with
`use_policy_gradient: false`; it carries richer distributional information but requires the teacher's top-k logits
and more memory. `coefficient` affects hybrid OPD+RLVR; pure OPD uses the distillation loss as its full signal.

## Run the contract smoke first

Use the same frozen checkpoint as student and teacher. This is not a capability experiment; it tests the entire
token/log-probability/resource path. The distillation loss should be finite and close to zero, allowing for
precision and inference-engine differences.

```bash
CONFIG=configs/cuda-opd-qwen35-2b-contract-smoke.yaml
oellm-rlvr validate --config "$CONFIG"
oellm-rlvr topology --config "$CONFIG"
oellm-rlvr opd-preflight --config "$CONFIG"
oellm-rlvr backend-command --config "$CONFIG"
oellm-rlvr render-slurm \
  --config "$CONFIG" \
  --output jobs/opd-contract-smoke.sbatch
sbatch jobs/opd-contract-smoke.sbatch
```

Only after this passes should the teacher path be changed to a different same-tokenizer checkpoint. Compare the
result with both the frozen parent and a no-distillation control on held-out tasks. Track verl's
`distillation/abs_loss`, total distillation loss, response length, throughput, and task reward where enabled.

## Artifacts and gates

verl writes checkpoints to `output.directory` and decoded rollout JSONL to `output.rollout_directory`. Console and
W&B logging contain its native `distillation/*` optimizer metrics. `DistillationTrace` extends this repository's
append-only trajectory contract with aligned token IDs, policy/teacher log-probabilities, masks, optional top-k
values, teacher identity, and sampled reverse KL. The generic `gate` command accepts such enriched records and
checks distillation coverage; upstream verl's decoded rollout dump alone does not contain all token-level fields,
so do not claim that gate from decoded JSONL without an exporter that preserves those tensors.

Always keep the frozen checkpoint manifests, rendered command, resolved YAML, backend commit, scheduler log, verl
metrics, and held-out evaluation outputs together. A falling distillation loss proves imitation, not capability.
