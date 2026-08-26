#!/usr/bin/env bash
# Isolate the 1 learner + 8 rollout native RCCL topology without loading a model.
# Submit this file from the oellm-rlvr repository root.
#SBATCH --job-name=rlvr-transfer-probe
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
module purge
module load Local-LAIF
module load lumi-aif-singularity-bindings

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
CONTROL_ROOT="$SLURM_SUBMIT_DIR"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd,/boot

mapfile -t PROBE_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
TRAINER_NODE="${PROBE_NODES[1]}"
TRAINER_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_NODE" getent hosts "$TRAINER_NODE" | awk 'NR==1 {print $1}')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
PROBE_WORLD_SIZE="${PROBE_WORLD_SIZE:-9}"
PROBE_UNMASKED_WORKERS="${PROBE_UNMASKED_WORKERS:-0}"
PROBE_EXTRA_ARGS=()
if [[ "$PROBE_UNMASKED_WORKERS" == 1 ]]; then
  PROBE_EXTRA_ARGS+=(--unmasked-workers)
fi

export NCCL_SOCKET_IFNAME=hsn
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export PYTHONUNBUFFERED=1

echo "Native transfer probe: trainer=${TRAINER_NODE}/${TRAINER_IP}, workers=${PROBE_NODES[0]}, port=${MASTER_PORT}, world_size=${PROBE_WORLD_SIZE}, unmasked_workers=${PROBE_UNMASKED_WORKERS}"
srun --label --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpu-bind=none \
  singularity exec -B "$BIND" "$CONTAINER" \
  env -u ROCR_VISIBLE_DEVICES /opt/venv/bin/python "$CONTROL_ROOT/scripts/probe_nccl_weight_transfer.py" \
  --master-address "$TRAINER_IP" \
  --master-port "$MASTER_PORT" \
  --world-size "$PROBE_WORLD_SIZE" \
  --trainer-host "$TRAINER_NODE" \
  --timeout 180 \
  "${PROBE_EXTRA_ARGS[@]}"
