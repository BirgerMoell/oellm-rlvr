#!/usr/bin/env bash
# Real SkyRL -> vLLM -> Terminus-2 -> Singularity -> Harbor verifier rollout.
# This is generation-only by design: it must pass before learner updates are
# enabled. The default canary exercises one project-owned repository repair;
# set TASK_GLOB='repo-*' only after that strict gate passes.
#SBATCH --job-name=oellm-agent-rollout
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
ulimit -c 0
module purge
module load Local-LAIF
module load lumi-aif-singularity-bindings

: "${MODEL:=/scratch/project_465002530/users/bmoell/models/Qwen3.5-2B}"
: "${CONTROL_ROOT:=$SLURM_SUBMIT_DIR}"
: "${SOURCE_ROOT:=/scratch/project_465002530/users/bmoell/rl-sources}"
: "${SKYRL_ROOT:=$SOURCE_ROOT/SkyRL-v0.3.0}"
: "${HARBOR_ROOT:=$SOURCE_ROOT/harbor-v0.22.0}"
: "${PYTHON_BIN:=/scratch/project_465002530/users/bmoell/venvs/oellm-skyrl-v030/bin/python}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-rlvr/harbor-agent/$SLURM_JOB_ID}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_TOKENS_PER_TURN:=768}"
: "${MAX_TURNS:=6}"
: "${TEMPERATURE:=0.7}"
: "${TASK_GLOB:=repo-repair-clamp}"
: "${TOTAL_GPUS:=2}"

PACK="$RUN_ROOT/harbor-task-pack-v0"
ROLLOUT_PACK="$RUN_ROOT/code-rollout-pack"
JOBS="$RUN_ROOT/jobs"
CACHE_ROOT="$RUN_ROOT/cache"
HARBOR_LUMI_CACHE_ROOT="$RUN_ROOT/launcher-cache"
NODE_TMP="/tmp/oellm-agent-$SLURM_JOB_ID"
RAY_NODE_TMP="/tmp/oellm-agent-ray-$SLURM_JOB_ID"
mkdir -p logs "$RUN_ROOT"/{logs,ray,compatibility,manifests,checkpoints,exports} "$JOBS" "$ROLLOUT_PACK" \
  "$CACHE_ROOT"/{hf,xdg,triton,torchinductor,vllm,tmp,miopen} \
  "$HARBOR_LUMI_CACHE_ROOT"/{singularity,tmp} "$NODE_TMP" "$RAY_NODE_TMP"

preserve_ray_logs() {
  cp -a "$RAY_NODE_TMP"/. "$RUN_ROOT/ray"/ 2>/dev/null || true
}
trap preserve_ray_logs EXIT

test -r "$MODEL/config.json"
test -r "$CONTAINER"
test -r "$(dirname "$PYTHON_BIN")/../pyvenv.cfg"
test -x "$CONTROL_ROOT/scripts/lumi-host-launcher-bin/singularity"
singularity exec -B /pfs,/scratch,/appl "$CONTAINER" test -x "$PYTHON_BIN"

# Harbor itself lives in LAIF. Its Singularity subprocess is replaced by the
# project launcher, which opens an overlapping same-node Slurm step and invokes
# LUMI's supported host runtime. The task sandbox never receives GPU devices.
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd,/boot,/tmp
BIND+=,/usr/bin/srun:/usr/local/bin/srun,/usr/lib64:/hostlib64,/usr/lib64/slurm:/usr/lib64/slurm
BIND+=,/etc/slurm:/etc/slurm,/etc/passwd:/etc/passwd,/etc/group:/etc/group,/run/munge:/run/munge
BIND+=,/usr/share/lua:/usr/share/lua
VENV_SITE="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages"
LAUNCHER_PATH="$CONTROL_ROOT/scripts/lumi-host-launcher-bin"
PYTHONPATH="$SKYRL_ROOT:$HARBOR_ROOT/src:$CONTROL_ROOT/src:$VENV_SITE:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

export PYTHONPATH CONTROL_ROOT SOURCE_ROOT SKYRL_ROOT HARBOR_ROOT PYTHON_BIN MODEL CONTAINER RUN_ROOT
export PACK ROLLOUT_PACK JOBS MAX_MODEL_LEN MAX_TOKENS_PER_TURN MAX_TURNS TEMPERATURE
export HARBOR_LUMI_CACHE_ROOT HARBOR_SINGULARITY_DISABLE_FAKEROOT=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_LOGGING_LEVEL=DEBUG VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset VLLM_USE_V1 VLLM_ENABLE_V1_MULTIPROCESSING
export OELLM_PATCH_RAY_ROCM_VISIBILITY=1 OELLM_PATCH_VLLM_MAMBA_ENUM=1
export OELLM_PATCH_VLLM_WEIGHT_UPDATE=1 OELLM_PATCH_MATH_EQUIV_THREADS=1
export OELLM_PATCH_HARBOR_VLLM_TOKENS=1
export NCCL_SOCKET_IFNAME=hsn NCCL_CUMEM_ENABLE=0 HWLOC_COMPONENTS=-rsmi RAY_DEDUP_LOGS=0
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" XDG_CONFIG_HOME="$CACHE_ROOT/xdg-config"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen" HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets" PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TMPDIR="$NODE_TMP" RAY_TMPDIR="$RAY_NODE_TMP"

inside() {
  singularity exec -B "$BIND" "$CONTAINER" env -u BASH_ENV \
    PATH="$LAUNCHER_PATH:$(dirname "$PYTHON_BIN"):/opt/venv/bin:/usr/bin:/bin" \
    PYTHONPATH="$PYTHONPATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$@"
}

inside "$PYTHON_BIN" -m oellm_rlvr.cli build-harbor-pack --output "$PACK" --sif "$CONTAINER"
inside "$PYTHON_BIN" -m oellm_rlvr.cli validate-harbor-pack --pack "$PACK"
inside "$PYTHON_BIN" "$CONTROL_ROOT/scripts/probe_harbor_litellm_vllm.py" \
  --output "$RUN_ROOT/compatibility/harbor-litellm-vllm.json"
mapfile -t SELECTED_TASKS < <(find "$PACK" -mindepth 1 -maxdepth 1 -type d -name "$TASK_GLOB" -print | sort)
EXPECTED_TRIALS="${#SELECTED_TASKS[@]}"
if [[ "$EXPECTED_TRIALS" -lt 1 ]]; then
  echo "TASK_GLOB=$TASK_GLOB selected no Harbor tasks" >&2
  exit 1
fi
POLICY_GPUS="${POLICY_GPUS:-$EXPECTED_TRIALS}"
NUM_ENGINES="${NUM_ENGINES:-$EXPECTED_TRIALS}"
if [[ "$POLICY_GPUS" -gt 4 ]]; then POLICY_GPUS=4; fi
if [[ "$NUM_ENGINES" -gt 4 ]]; then NUM_ENGINES=4; fi
if [[ "$POLICY_GPUS" -lt 1 || "$NUM_ENGINES" -lt 1 ]]; then
  echo "POLICY_GPUS and NUM_ENGINES must both be positive" >&2
  exit 1
fi
if [[ $((POLICY_GPUS + NUM_ENGINES)) -gt "$TOTAL_GPUS" ]]; then
  echo "policy ($POLICY_GPUS) + inference ($NUM_ENGINES) exceeds allocation ($TOTAL_GPUS)" >&2
  exit 1
fi
HIP_FALLBACK="$(seq -s, 0 $((TOTAL_GPUS - 1)))"
cp -a "${SELECTED_TASKS[@]}" "$ROLLOUT_PACK"/
test "$(find "$ROLLOUT_PACK" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq "$EXPECTED_TRIALS"
printf '%s\n' "$TASK_GLOB" > "$RUN_ROOT/manifests/task-glob.txt"
printf '%s\n' "$EXPECTED_TRIALS" > "$RUN_ROOT/manifests/expected-trials.txt"
printf '%s\n' "$POLICY_GPUS" > "$RUN_ROOT/manifests/policy-gpus.txt"
printf '%s\n' "$NUM_ENGINES" > "$RUN_ROOT/manifests/inference-engines.txt"
export EXPECTED_TRIALS POLICY_GPUS NUM_ENGINES TOTAL_GPUS HIP_FALLBACK

git -C "$CONTROL_ROOT" rev-parse HEAD > "$RUN_ROOT/manifests/oellm-rlvr.commit"
git -C "$SKYRL_ROOT" rev-parse HEAD > "$RUN_ROOT/manifests/skyrl.commit"
git -C "$HARBOR_ROOT" rev-parse HEAD > "$RUN_ROOT/manifests/harbor.commit"
sha256sum "$CONTAINER" > "$RUN_ROOT/manifests/task-sif.sha256"
sha256sum "$MODEL/config.json" > "$RUN_ROOT/manifests/model-config.sha256"
env | LC_ALL=C sort > "$RUN_ROOT/manifests/environment.txt"

srun --label --cpu-bind=none --gpu-bind=none --gpus-per-task="$TOTAL_GPUS" singularity exec -B "$BIND" "$CONTAINER" \
  env -u BASH_ENV PATH="$LAUNCHER_PATH:$(dirname "$PYTHON_BIN"):/opt/venv/bin:/usr/bin:/bin" \
  PYTHONPATH="$PYTHONPATH" bash -c '
  set -euo pipefail
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$HIP_FALLBACK}"
  unset ROCR_VISIBLE_DEVICES
  "$PYTHON_BIN" "$CONTROL_ROOT/scripts/probe_skyrl_stack.py" \
    --skyrl-source "$SKYRL_ROOT" --harbor-source "$HARBOR_ROOT" --require-gpu \
    --expected-gpus "$TOTAL_GPUS" \
    --output "$RUN_ROOT/compatibility/gpu-imports.json"
  cd "$SKYRL_ROOT"
  "$PYTHON_BIN" -m examples.train_integrations.harbor.entrypoints.main_harbor_generate \
    data.train_data="['"$ROLLOUT_PACK"']" data.val_data="[]" \
    harbor_trial_config.trials_dir="$JOBS" \
    harbor_trial_config.agent.name=terminus-2 \
    harbor_trial_config.agent.override_timeout_sec=900 \
    harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
    harbor_trial_config.agent.kwargs.suppress_max_turns_warning=true \
    harbor_trial_config.agent.kwargs.enable_summarize=false \
    harbor_trial_config.agent.kwargs.collect_rollout_details=true \
    harbor_trial_config.agent.kwargs.record_terminal_session=false \
    harbor_trial_config.agent.kwargs.temperature="$TEMPERATURE" \
    harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
    harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_TOKENS_PER_TURN" \
    harbor_trial_config.agent.kwargs.model_info.input_cost_per_token=0.0 \
    harbor_trial_config.agent.kwargs.model_info.output_cost_per_token=0.0 \
    harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens="$MAX_TOKENS_PER_TURN" \
    harbor_trial_config.agent.kwargs.llm_kwargs.timeout=300 \
    harbor_trial_config.agent.kwargs.llm_kwargs.max_retries=0 \
    harbor_trial_config.environment.type=singularity \
    harbor_trial_config.environment.override_cpus=1 \
    harbor_trial_config.environment.override_memory_mb=4096 \
    harbor_trial_config.environment.override_storage_mb=4096 \
    harbor_trial_config.environment.suppress_override_warnings=true \
    harbor_trial_config.environment.kwargs.singularity_image_cache_dir="$HARBOR_LUMI_CACHE_ROOT/singularity" \
    harbor_trial_config.environment.kwargs.singularity_no_mount=home,tmp,bind-paths \
    trainer.policy.model.path="$MODEL" trainer.strategy=fsdp \
    trainer.policy.language_model_only=true trainer.ref.language_model_only=true \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=1 trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" \
    generator.inference_engine.served_model_name=oellm-agent \
    generator.inference_engine.num_engines="$NUM_ENGINES" generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.distributed_executor_backend=mp \
    generator.inference_engine.language_model_only=true \
    generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.enforce_eager=true \
    generator.inference_engine.gpu_memory_utilization=0.85 generator.inference_engine.max_num_seqs=4 \
    generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
    generator.step_wise_trajectories=true generator.merge_stepwise_output=true \
    generator.n_samples_per_prompt=1 generator.sampling_params.max_generate_length="$MAX_TOKENS_PER_TURN" \
    generator.rate_limit.enabled=true generator.rate_limit.trajectories_per_second="$NUM_ENGINES" \
    generator.rate_limit.max_concurrency="$NUM_ENGINES" \
    trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.use_kl_loss=false trainer.algorithm.use_kl_in_reward=false \
    trainer.train_batch_size="$EXPECTED_TRIALS" trainer.policy_mini_batch_size="$EXPECTED_TRIALS" \
    trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
    trainer.remove_microbatch_padding=false trainer.flash_attn=false \
    trainer.eval_before_train=false trainer.eval_interval=-1 trainer.logger=console \
    trainer.project_name=oellm-harbor-agentic trainer.run_name="lumi-$SLURM_JOB_ID" \
    trainer.resume_mode=null trainer.ckpt_path="$RUN_ROOT/checkpoints" \
    trainer.export_path="$RUN_ROOT/exports" trainer.log_path="$RUN_ROOT/logs"
'

inside "$PYTHON_BIN" -m oellm_rlvr.cli qualify-harbor-rollouts \
  --jobs-root "$JOBS" --index-output "$RUN_ROOT/campaign-index.jsonl" \
  --report-output "$RUN_ROOT/qualification.json" --run-id "$SLURM_JOB_ID" \
  --policy-version 0 --learner-version 0 --expected-trials "$EXPECTED_TRIALS" \
  --min-bash-commands-per-trial 1

echo "harbor_agentic_rollout_artifact=$RUN_ROOT"
