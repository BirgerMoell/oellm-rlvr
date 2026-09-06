#!/usr/bin/env bash
# Validate the 16-task OpenEuroLLM Harbor pack on a LUMI CPU node. Set SOAK=1
# to run 112 launches (four oracle plus three nop attempts for every task).
#SBATCH --job-name=oellm-harbor-contract
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
module purge
module load Local-LAIF
module load lumi-aif-singularity-bindings

: "${CONTROL_ROOT:=$SLURM_SUBMIT_DIR}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${VENV:=/scratch/project_465002530/users/bmoell/venvs/oellm-skyrl-v030}"
: "${RESULTS_ROOT:=/scratch/project_465002530/users/bmoell/oellm-rlvr/harbor-contract}"
: "${SOAK:=0}"
: "${N_CONCURRENT:=4}"

ORACLE_ATTEMPTS=1
NOP_ATTEMPTS=1
if [[ "$SOAK" == "1" ]]; then
  ORACLE_ATTEMPTS=4
  NOP_ATTEMPTS=3
fi

RESULTS_DIR="$RESULTS_ROOT/$SLURM_JOB_ID"
PACK="$RESULTS_DIR/harbor-task-pack-v0"
JOBS="$RESULTS_DIR/jobs"
mkdir -p logs "$RESULTS_DIR" "$JOBS"
test -r "$CONTAINER"
test -r "$VENV/pyvenv.cfg"
singularity exec -B /pfs,/scratch,/appl "$CONTAINER" test -x "$VENV/bin/python"
test -x "$CONTROL_ROOT/scripts/lumi-host-launcher-bin/singularity"

# Harbor runs inside LAIF, but its sandbox command is the launcher above. The
# launcher uses the host srun plus host Slurm libraries, then starts the task
# SIF outside LAIF on this same node. /hostlib64 is visible only to srun.
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd,/tmp
BIND+=,/usr/bin/srun:/usr/local/bin/srun,/usr/lib64:/hostlib64,/usr/lib64/slurm:/usr/lib64/slurm
BIND+=,/etc/slurm:/etc/slurm,/etc/passwd:/etc/passwd,/etc/group:/etc/group,/run/munge:/run/munge
BIND+=,/usr/share/lua:/usr/share/lua
VENV_SITE="$VENV/lib/python3.12/site-packages"
PYTHONPATH="$CONTROL_ROOT/src:$VENV_SITE:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
LAUNCHER_PATH="$CONTROL_ROOT/scripts/lumi-host-launcher-bin"
export PYTHONPATH HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

inside() {
  singularity exec -B "$BIND" "$CONTAINER" env \
    PATH="$LAUNCHER_PATH:$VENV/bin:/opt/venv/bin:/usr/bin:/bin" \
    PYTHONPATH="$PYTHONPATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$@"
}

inside "$VENV/bin/python" -m oellm_rlvr.cli build-harbor-pack --output "$PACK" --sif "$CONTAINER"
inside "$VENV/bin/python" -m oellm_rlvr.cli validate-harbor-pack --pack "$PACK"
sha256sum "$CONTAINER" > "$RESULTS_DIR/task-sif.sha256"

inside "$VENV/bin/harbor" run -p "$PACK" -a oracle -e singularity \
  -n "$N_CONCURRENT" -k "$ORACLE_ATTEMPTS" -o "$JOBS" --job-name oracle -y -q
inside "$VENV/bin/python" "$CONTROL_ROOT/scripts/validate_harbor_trials.py" \
  --root "$JOBS/oracle" --agent oracle --reward 1 --tasks 16 --attempts "$ORACLE_ATTEMPTS" \
  --output "$RESULTS_DIR/harbor-oracle-results.json"

inside "$VENV/bin/harbor" run -p "$PACK" -a nop -e singularity \
  -n "$N_CONCURRENT" -k "$NOP_ATTEMPTS" -o "$JOBS" --job-name nop -y -q
inside "$VENV/bin/python" "$CONTROL_ROOT/scripts/validate_harbor_trials.py" \
  --root "$JOBS/nop" --agent nop --reward 0 --tasks 16 --attempts "$NOP_ATTEMPTS" \
  --output "$RESULTS_DIR/harbor-failure-results.json"

echo "harbor_contract_artifact=$RESULTS_DIR"
