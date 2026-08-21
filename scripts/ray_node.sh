#!/usr/bin/env bash
set -euo pipefail

HEAD_IP="${1:?head ip}"
PORT="${2:?port}"
GPUS="${3:?gpus}"
RAY_BIN="${RAY_BIN:-ray}"
SELF_HOST="$(hostname)"
SELF_IP="$(getent hosts "$SELF_HOST" | awk 'NR==1 {print $1}')"
mkdir -p "${RAY_TMPDIR:?RAY_TMPDIR must be set}"
"$RAY_BIN" stop --force >/dev/null 2>&1 || true

if [[ "${SLURM_PROCID:-0}" == 0 ]]; then
  exec "$RAY_BIN" start --head --node-ip-address="$SELF_IP" --port="$PORT" --num-gpus="$GPUS" \
    --temp-dir="$RAY_TMPDIR" --dashboard-host=127.0.0.1 --block
fi

for _ in $(seq 1 60); do
  if "$RAY_BIN" start --address="${HEAD_IP}:${PORT}" --node-ip-address="$SELF_IP" --num-gpus="$GPUS" --block; then
    exit 0
  fi
  sleep 2
done
echo "Unable to join Ray head ${HEAD_IP}:${PORT}" >&2
exit 1
