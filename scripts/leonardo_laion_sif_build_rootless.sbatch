#!/bin/bash
# Build x86_64 SIFs for one LAION dataset on a boost COMPUTE node (user namespaces on) via scripts/laion_rootless/singularity,
# using the build proxy from leonardo_laion_proxy.sbatch for internet. See that wrapper for how rootless builds work here.
#   source leonardo.env; sbatch scripts/leonardo_laion_proxy.sbatch          # first, and keep it running
#   DATASET=rts-1k-rootless-apptainer CONTEXT_HASH_FILE=canary-hashes.txt sbatch --qos=boost_qos_dbg --time=00:30:00 scripts/leonardo_laion_sif_build_rootless.sbatch
#SBATCH --job-name=laion-sif-rootless
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
set -euo pipefail
: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${DATASET:?set DATASET=<repo name without laion/>}"
: "${LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/data/laion-ot}"
: "${HARBOR_LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/src/harbor-laion}"
: "${HARBOR_LAION_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion}"
: "${SIF_CACHE:=$OELLM_SCRATCH/oellm-rl/sif-cache-x86_64}"
: "${PROXY_DIR:=$OELLM_SCRATCH/oellm-rl/proxy}"
: "${SIF_BUILD_WORKERS:=6}"
: "${SIF_BUILD_TIMEOUT_SEC:=1800}"
: "${CONTEXT_HASH_FILE:=}"
HERE="${SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-.}/scripts}"
TASKS="$LAION_ROOT/$DATASET/tasks"; [[ -d "$TASKS" ]] || { echo "no tasks at $TASKS" >&2; exit 1; }
[[ -s "$PROXY_DIR/proxy.host" ]] || { echo "no proxy: sbatch leonardo_laion_proxy.sbatch first" >&2; exit 1; }

export REAL_SINGULARITY="$(command -v singularity)" SHIM_SO="$PROXY_DIR/fakeown.so"
export PROXY_HOSTFILE="$PROXY_DIR/proxy.host" PROXY_TOKENFILE="$PROXY_DIR/proxy.token"
[[ -f "$SHIM_SO" ]] || gcc -shared -fPIC -O2 -o "$SHIM_SO" "$HERE/laion_rootless/fakeown.c"
export PATH="$HERE/laion_rootless:$PATH"
# Loopback relay: adds the proxy token so build tools see an unauthenticated 127.0.0.1 proxy (Java ignores URL credentials).
RELAY_PORT_FILE="$OELLM_SCRATCH/oellm-rl/apptainer-tmp/relay.${SLURM_JOB_ID:-manual}.port"; mkdir -p "$(dirname "$RELAY_PORT_FILE")"; rm -f "$RELAY_PORT_FILE"
"$HARBOR_LAION_VENV/bin/python" "$HERE/laion_rootless/local_relay.py" "$PROXY_HOSTFILE" "$PROXY_TOKENFILE" "$RELAY_PORT_FILE" 43000 &
RELAY_PID=$!; trap 'kill $RELAY_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do [[ -s "$RELAY_PORT_FILE" ]] && break; sleep 1; done
[[ -s "$RELAY_PORT_FILE" ]] || { echo "relay did not start" >&2; exit 1; }
export PROXY_LOCAL="http://127.0.0.1:$(cat "$RELAY_PORT_FILE")"
export SINGULARITY_CACHEDIR="$OELLM_SCRATCH/oellm-rl/apptainer-cache" SINGULARITY_TMPDIR="$OELLM_SCRATCH/oellm-rl/apptainer-tmp/${SLURM_JOB_ID:-manual}"
export APPTAINER_PREBUILD_RETRY_WITHOUT_FAKEROOT=0 PYTHONUNBUFFERED=1
mkdir -p "$SIF_CACHE" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs

args=("$TASKS" --sif-cache "$SIF_CACHE" --max-workers "$SIF_BUILD_WORKERS" --build-timeout-sec "$SIF_BUILD_TIMEOUT_SEC")
[[ -n "$CONTEXT_HASH_FILE" ]] && args+=(--context-hash-file "$CONTEXT_HASH_FILE")
echo "node=$(hostname) which singularity=$(command -v singularity) proxy=$(cat "$PROXY_HOSTFILE")"
cd "$HARBOR_LAION_ROOT"
PYTHONPATH="$HARBOR_LAION_ROOT/src" "$HARBOR_LAION_VENV/bin/python" -m harbor.environments.apptainer.prebuild_sifs "${args[@]}"
echo "sifs in cache: $(find "$SIF_CACHE" -maxdepth 1 -name '*.sif' | wc -l)"; du -sh "$SIF_CACHE"
