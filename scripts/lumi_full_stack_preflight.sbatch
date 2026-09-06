#!/usr/bin/env bash
# Standalone two-node acceptance test for the exact LUMI RL software surface.
# Submit from the oellm-rlvr checkout that produced the checkpoint manifest.
#SBATCH --job-name=oellm-rl-preflight
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
module purge
module load Local-LAIF
module load lumi-aif-singularity-bindings

: "${MODEL:=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/models/oellm-9b-256k-sft}"
: "${TOKENIZER:=$MODEL}"
: "${CONTROL_ROOT:=$SLURM_SUBMIT_DIR}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${PYTHON_BIN:=/scratch/project_465002530/users/bmoell/venvs/oellm-rlvr/bin/python}"
: "${RESULTS_ROOT:=/scratch/project_465002530/users/bmoell/oellm-rlvr/preflight}"
: "${HF_HOME:=/scratch/project_465002530/users/bmoell/hf_cache}"

BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd,/boot
RESULTS_DIR="$RESULTS_ROOT/$SLURM_JOB_ID"
CACHE_ROOT="$RESULTS_DIR/cache"
VENV_SITE="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages"
PYTHONPATH="$CONTROL_ROOT/src:$VENV_SITE:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
mapfile -t PREFLIGHT_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${PREFLIGHT_NODES[0]}"
RELAY_NODE="${PREFLIGHT_NODES[0]}"
TRAINER_NODE="${PREFLIGHT_NODES[1]}"
HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" getent hosts "$HEAD_NODE" | awk 'NR==1 {print $1}')"
TRAINER_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_NODE" getent hosts "$TRAINER_NODE" | awk 'NR==1 {print $1}')"
RAY_PORT="$((21000 + SLURM_JOB_ID % 10000))"
TRANSFER_PORT="$((32000 + SLURM_JOB_ID % 10000))"
mkdir -p logs "$RESULTS_DIR" "$CACHE_ROOT"/{xdg,triton,torchinductor,vllm,tmp,miopen}
NODE_TMP="/tmp/oellm-preflight-$SLURM_JOB_ID"
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 mkdir -p "$NODE_TMP"
test -r "$MODEL/config.json"
test -r "$TOKENIZER/tokenizer_config.json"
test -r "$CONTAINER"
test -r "$(dirname "$PYTHON_BIN")/../pyvenv.cfg"
singularity exec -B "$BIND" "$CONTAINER" test -x "$PYTHON_BIN"
export MODEL TOKENIZER CONTROL_ROOT CONTAINER PYTHON_BIN RESULTS_DIR HF_HOME PYTHONPATH
export NCCL_SOCKET_IFNAME=hsn NCCL_CUMEM_ENABLE=0 PYTHONUNBUFFERED=1
export RAY_USAGE_STATS_ENABLED=0 RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CONFIG_HOME="$CACHE_ROOT/xdg-config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen"
export TMPDIR="$NODE_TMP" PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"

cleanup() {
  if [[ -n "${RAY_STEP_PID:-}" ]]; then
    kill "$RAY_STEP_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 15); do
      if ! kill -0 "$RAY_STEP_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -KILL "$RAY_STEP_PID" >/dev/null 2>&1 || true
    wait "$RAY_STEP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[1/5] Per-node GPU and import checks"
srun --label --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --gpu-bind=none \
  singularity exec -B "$BIND" "$CONTAINER" env -u BASH_ENV PYTHONPATH="$PYTHONPATH" bash -c '
    set -euo pipefail
    env -u ROCR_VISIBLE_DEVICES PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" \
      "$CONTROL_ROOT/scripts/preflight_gpu.py" --accelerator rocm --gpus 8 \
      > "$RESULTS_DIR/node-${SLURMD_NODENAME}.json"
  '

echo "[2/5] Immutable package, container, and driver inventory"
singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m pip freeze \
  > "$RESULTS_DIR/package-lock.txt"
{
  sha256sum "$CONTAINER"
  echo "container=$CONTAINER"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "nodes=${PREFLIGHT_NODES[*]}"
  singularity exec -B "$BIND" "$CONTAINER" rocminfo | sed -n '1,120p'
} > "$RESULTS_DIR/container-and-driver-digests.txt"

echo "[3/5] Two-node Ray resource and placement check"
export RAY_TMPDIR="/tmp/oellm-ray-$SLURM_JOB_ID"
export RAY_BIN="$(dirname "$PYTHON_BIN")/ray"
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --gpu-bind=none \
  singularity exec -B "$BIND" "$CONTAINER" env -u ROCR_VISIBLE_DEVICES \
  PYTHONPATH="$PYTHONPATH" RAY_TMPDIR="$RAY_TMPDIR" RAY_BIN="$RAY_BIN" \
  "$CONTROL_ROOT/scripts/ray_node.sh" "$HEAD_IP" "$RAY_PORT" 8 &
RAY_STEP_PID=$!
for _ in $(seq 1 60); do
  if singularity exec -B "$BIND" "$CONTAINER" env -u BASH_ENV -u ROCR_VISIBLE_DEVICES \
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -c \
    "import ray; ray.init(address='$HEAD_IP:$RAY_PORT', logging_level='ERROR'); ray.shutdown()" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
singularity exec -B "$BIND" "$CONTAINER" env -u BASH_ENV -u ROCR_VISIBLE_DEVICES \
  PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" \
  "$CONTROL_ROOT/scripts/probe_ray_cluster.py" --address "$HEAD_IP:$RAY_PORT" \
  --expected-nodes 2 --expected-gpus 16 > "$RESULTS_DIR/ray-cluster.json"
cleanup
unset RAY_STEP_PID

echo "[4/5] Cross-node RCCL plus local fan-out"
srun --label --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=8 --gpu-bind=none \
  singularity exec -B "$BIND" "$CONTAINER" env -u ROCR_VISIBLE_DEVICES \
  /opt/venv/bin/python "$CONTROL_ROOT/scripts/probe_hierarchical_weight_transfer.py" \
  --trainer-address "$TRAINER_IP" --relay-address "$HEAD_IP" --master-port "$TRANSFER_PORT" \
  --trainer-host "$TRAINER_NODE" --leaf-count 7 --timeout 480 \
  | tee "$RESULTS_DIR/hierarchical-transfer.log"

echo "[5/5] Offline BF16 load and one-prompt vLLM generation"
srun --label --nodes=1 --ntasks=1 -w "$HEAD_NODE" --gpus-per-task=1 --gpu-bind=none \
  singularity exec -B "$BIND" "$CONTAINER" env -u BASH_ENV PYTHONPATH="$PYTHONPATH" bash -c '
    set -euo pipefail
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
      export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
      unset ROCR_VISIBLE_DEVICES
    fi
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export VLLM_USE_V1=1 VLLM_DISABLE_COMPILE_CACHE=1 TOKENIZERS_PARALLELISM=false
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$CONTROL_ROOT/scripts/probe_model_generation.py" \
      --model "$MODEL" --tokenizer "$TOKENIZER" --output "$RESULTS_DIR/model-generation.json" \
      > "$RESULTS_DIR/model-generation.log"
  '

singularity exec -B "$BIND" "$CONTAINER" env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" \
  "$CONTROL_ROOT/scripts/assemble_lumi_preflight.py" --results-dir "$RESULTS_DIR"
echo "preflight_artifact=$RESULTS_DIR/preflight.json"
