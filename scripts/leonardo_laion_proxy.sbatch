#!/bin/bash
# Token-protected build proxy on an internet-enabled node, for rootless image builds on offline compute nodes.
# Writes host:port to $PROXY_HOSTFILE (the build wrapper reads it on every build, so a restarted proxy is picked up).
#   source leonardo.env; sbatch scripts/leonardo_laion_proxy.sbatch
#SBATCH --job-name=laion-proxy
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${PROXY_DIR:=$OELLM_SCRATCH/oellm-rl/proxy}"
: "${HARBOR_LAION_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion}"
mkdir -p "$PROXY_DIR" logs; chmod 700 "$PROXY_DIR"
"$HARBOR_LAION_VENV/bin/python" "${SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-.}/scripts}/laion_proxy.py" 0 "$PROXY_DIR/proxy.host" "$PROXY_DIR/proxy.token" 14000
