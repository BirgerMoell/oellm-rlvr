#!/bin/bash
# Web gateway sidecar with LIVE web search, on an internet-enabled Leonardo node (compute nodes are offline).
# Rollout nodes call it over the cluster network with a bearer token; the agent never gets a network of its own.
#   one-time:  uv venv --python 3.12 $OELLM_SCRATCH/oellm-rl/venvs/web-gateway && \
#              uv pip install --python $OELLM_SCRATCH/oellm-rl/venvs/web-gateway/bin/python fastapi uvicorn pyyaml
#   run:       source leonardo.env; GATEWAY_CONFIG=configs/web-gateway.example.yaml sbatch scripts/leonardo_web_gateway.sbatch
# The job writes host:port to $GATEWAY_DIR/gateway.host and a 600 token file to $GATEWAY_DIR/gateway.token; point the
# web tools' api_endpoint at http://<host:port>/web_search and send "Authorization: Bearer <token>".
# BRAVE_API_KEY (if the config uses brave) is read from $GATEWAY_DIR/brave.key, never from the config or the agent.
# Record once with mode: live + cache_dir, then run training with mode: replay (works on offline nodes, deterministic).
#SBATCH --job-name=web-gateway
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
set -euo pipefail
: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${GATEWAY_CONFIG:?set GATEWAY_CONFIG=<yaml>}"
: "${GATEWAY_DIR:=$OELLM_SCRATCH/oellm-rl/gateway}"
: "${GATEWAY_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/web-gateway}"
: "${GATEWAY_PORT:=0}"
SRC="${SLURM_SUBMIT_DIR:-.}"
mkdir -p "$GATEWAY_DIR" logs; chmod 700 "$GATEWAY_DIR"
[[ -f "$GATEWAY_DIR/gateway.token" ]] || { umask 077; "$GATEWAY_VENV/bin/python" -c 'import secrets;print(secrets.token_urlsafe(32))' > "$GATEWAY_DIR/gateway.token"; }
[[ -f "$GATEWAY_DIR/brave.key" ]] && export BRAVE_API_KEY="$(cat "$GATEWAY_DIR/brave.key")"
export OELLM_GATEWAY_CONFIG="$GATEWAY_CONFIG" OELLM_GATEWAY_TOKEN="$(cat "$GATEWAY_DIR/gateway.token")" PYTHONPATH="$SRC/src"
PORT="$GATEWAY_PORT"; [[ "$PORT" == 0 ]] && PORT=$(( 20000 + RANDOM % 20000 ))
echo "$(hostname):$PORT" > "$GATEWAY_DIR/gateway.host"
echo "gateway on $(hostname):$PORT config=$GATEWAY_CONFIG"
exec "$GATEWAY_VENV/bin/uvicorn" --factory oellm_rlvr.web_gateway:create_app_from_env --host 0.0.0.0 --port "$PORT"
