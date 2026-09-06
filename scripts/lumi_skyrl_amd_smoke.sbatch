#!/usr/bin/env bash
# Four-step SkyRL FSDP/vLLM smoke on one LUMI node. This uses a small local
# checkpoint to qualify the backend before loading the OpenEuroLLM 9B model.
#SBATCH --job-name=oellm-skyrl-amd
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
# Crashes in ROCm/Ray workers can otherwise leave multi-gigabyte core files in
# the pinned SkyRL source checkout and make the immutability probe fail later.
ulimit -c 0
module purge
module load Local-LAIF
module load lumi-aif-singularity-bindings

: "${MODEL:=/scratch/project_465002530/users/bmoell/models/Qwen3.5-2B}"
: "${DATA:=/scratch/project_465002530/users/bmoell/oellm-rlvr/data/skyrl-math-smoke/train.parquet}"
: "${CONTROL_ROOT:=$SLURM_SUBMIT_DIR}"
: "${SOURCE_ROOT:=/scratch/project_465002530/users/bmoell/rl-sources}"
: "${SKYRL_ROOT:=$SOURCE_ROOT/SkyRL-v0.3.0}"
: "${HARBOR_ROOT:=$SOURCE_ROOT/harbor-v0.22.0}"
: "${PYTHON_BIN:=/scratch/project_465002530/users/bmoell/venvs/oellm-skyrl-v030/bin/python}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${RUN_ROOT:=/scratch/project_465002530/users/bmoell/oellm-rlvr/skyrl-smoke/$SLURM_JOB_ID}"

BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd,/boot
VENV_SITE="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages"
PYTHONPATH="$SKYRL_ROOT:$HARBOR_ROOT/src:$CONTROL_ROOT/src:$VENV_SITE:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
CACHE_ROOT="$RUN_ROOT/cache"
mkdir -p logs "$RUN_ROOT"/{checkpoints,exports,logs,compatibility} \
  "$CACHE_ROOT"/{hf,xdg,triton,torchinductor,vllm,tmp,miopen}
NODE_TMP="/tmp/oellm-skyrl-$SLURM_JOB_ID"
mkdir -p "$NODE_TMP"
test -r "$MODEL/config.json"
test -r "$DATA"
test -r "$(dirname "$PYTHON_BIN")/../pyvenv.cfg"
singularity exec -B "$BIND" "$CONTAINER" test -x "$PYTHON_BIN"
export PYTHONPATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL DATA CONTROL_ROOT SOURCE_ROOT SKYRL_ROOT HARBOR_ROOT PYTHON_BIN RUN_ROOT CONTAINER
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_DISABLE_COMPILE_CACHE=1
export OELLM_PATCH_RAY_ROCM_VISIBILITY=1 OELLM_PATCH_VLLM_MAMBA_ENUM=1
export OELLM_PATCH_VLLM_WEIGHT_UPDATE=1 OELLM_PATCH_MATH_EQUIV_THREADS=1
export NCCL_SOCKET_IFNAME=hsn NCCL_CUMEM_ENABLE=0
# Ray already owns GPU placement. hwloc's AMD-SMI discovery crashes with
# SIGFPE in concurrent workers in the current LAIF image, so retain CPU/NUMA
# discovery while excluding only the redundant RSMI device component.
export HWLOC_COMPONENTS=-rsmi
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg-config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen"
export TMPDIR="$NODE_TMP" RAY_TMPDIR="$NODE_TMP/ray" PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export HF_HOME="$CACHE_ROOT/hf" HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"

srun --label --gpu-bind=none --gpus-per-task=8 singularity exec -B "$BIND" "$CONTAINER" \
  env -u BASH_ENV PYTHONPATH="$PYTHONPATH" bash -c '
  set -euo pipefail
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  unset ROCR_VISIBLE_DEVICES
  env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$CONTROL_ROOT/scripts/probe_skyrl_stack.py" \
    --skyrl-source "$SKYRL_ROOT" --harbor-source "$HARBOR_ROOT" --require-gpu \
    --output "$RUN_ROOT/compatibility/gpu-imports.json"
  cd "$SKYRL_ROOT"
  env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m skyrl.train.entrypoints.main_base \
    data.train_data="['$DATA']" data.val_data="[]" \
    trainer.policy.model.path="$MODEL" trainer.strategy=fsdp \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_nodes=1 trainer.placement.policy_num_gpus_per_node=4 \
    generator.inference_engine.num_engines=4 generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.enforce_eager=true \
    generator.inference_engine.gpu_memory_utilization=0.65 \
    generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
    generator.batched=true generator.n_samples_per_prompt=4 \
    generator.sampling_params.max_generate_length=512 \
    trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false trainer.max_training_steps=4 trainer.epochs=1 \
    trainer.train_batch_size=8 trainer.policy_mini_batch_size=8 \
    trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=1024 trainer.remove_microbatch_padding=false trainer.flash_attn=false \
    trainer.policy.optimizer_config.lr=1.0e-6 trainer.eval_before_train=false trainer.eval_interval=-1 \
    trainer.ckpt_interval=4 trainer.hf_save_interval=4 trainer.max_ckpts_to_keep=2 \
    trainer.logger=console trainer.project_name=oellm-skyrl-amd \
    trainer.run_name="lumi-$SLURM_JOB_ID" trainer.resume_mode=null \
    trainer.ckpt_path="$RUN_ROOT/checkpoints" trainer.export_path="$RUN_ROOT/exports" \
    trainer.log_path="$RUN_ROOT/logs" environment.env_class=gsm8k
'

echo "skyrl_smoke_artifacts=$RUN_ROOT"
