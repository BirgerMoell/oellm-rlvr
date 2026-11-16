#!/bin/bash
# Real agentic rollout on Leonardo: a vLLM server (CUDA, 1 GPU) serves the policy; Harbor's Terminus-2 agent drives it inside
# apptainer sandboxes (bridge server + worker on this node, offline); Harbor's verifier scores each trial. Generation + reward only
# (no learner update): this is the gate before RL training and doubles as the pass-rate profile of the model on the tasks.
#
#   source leonardo.env
#   MODEL=$OELLM_SCRATCH/oellm-rl/models/prelude-9b-sft-v2 N_TASKS=16 ATTEMPTS=2 \
#     sbatch --qos=boost_qos_dbg --time=00:30:00 scripts/leonardo_laion_agent_rollout.sbatch
# TASK_LIST: file with {"task_id": ...} lines (default: the validated rts-1k compatible list). Tasks are taken in shuffled order (SEED).
#SBATCH --job-name=laion-agent-rollout
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G     # includes the memory-backed sandbox staging in /dev/shm
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
set -euo pipefail

: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${MODEL:?set MODEL=<local model dir>}"
: "${DATASET:=rts-1k-rootless-apptainer}"
: "${N_TASKS:=16}"
: "${ATTEMPTS:=2}"
: "${SEED:=0}"
: "${MAX_TURNS:=30}"
: "${AGENT_TIMEOUT_MULT:=3.0}"   # task agent timeouts are 600-1800 s; 100-turn trials under load need more
: "${TEMPERATURE:=0.6}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_TOKENS_PER_TURN:=4096}"
: "${CONCURRENT:=12}"
: "${AGENT_NAME:=terminus-2}"
: "${SERVED_NAME:=policy}"
: "${CONTROL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${ROOT:=$OELLM_SCRATCH/oellm-rl}"
: "${TASK_LIST:=$ROOT/laion-validation/rts-1k-leonardo-compatible-all.jsonl}"
: "${LAION_ROOT:=$ROOT/data/laion-ot}"
: "${HARBOR_LAION_ROOT:=$ROOT/src/harbor-laion}"
: "${HARBOR_LAION_VENV:=$ROOT/venvs/harbor-laion}"
: "${SIF_CACHE:=$ROOT/sif-cache-x86_64}"
: "${RESULTS_ROOT:=$ROOT/laion-rollouts}"
: "${VLLM_CONTAINER:=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif}"
: "${VLLM_PY:=$ROOT/tmax-reproduction/.venv/bin/python}"
: "${VLLM_PORT:=18100}"
: "${BRIDGE_PORT:=19920}"
: "${WORKERS_PER_NODE:=12}"
: "${BRIDGE_OVERLAY_SIZE_MB:=512}"
# Sandbox staging (per-trial overlay image + a copy of the image's writable prefixes): node-local /tmp is only 10 GB (Slurm refuses a larger
# tmpfs gres), which filled up at 64 concurrent sandboxes (Errno 28). /dev/shm is RAM-backed, counted against --mem.
: "${STAGING_ROOT:=/dev/shm}"
RUN_ID="${SLURM_JOB_ID:-$$}"
OUT="$RESULTS_ROOT/$DATASET-$RUN_ID"
STAGING_BASE="$STAGING_ROOT/harbor-apptainer-$RUN_ID"
PY="$HARBOR_LAION_VENV/bin/python"
SRC="$LAION_ROOT/$DATASET"
[[ -d "$SRC/tasks" ]] || { echo "no tasks at $SRC/tasks" >&2; exit 1; }
[[ -r "$MODEL/config.json" ]] || { echo "no model at $MODEL" >&2; exit 1; }
mkdir -p "$OUT/pack" "$OUT/jobs" "$OUT/service-logs" "$STAGING_BASE" logs

export BRIDGE_AUTH_TOKEN="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export BRIDGE_HOST=127.0.0.1 BRIDGE_PORT WORKERS_PER_NODE SIF_CACHE STAGING_BASE BRIDGE_OVERLAY_SIZE_MB
export BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT"
export APPTAINER_CONFIGDIR="$OUT/apptainer-config" PYTHONUNBUFFERED=1
export PYTHONPATH="$HARBOR_LAION_ROOT/src:$CONTROL_ROOT/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# litellm talks to the local vLLM server; the key is a placeholder (the server has none, and nothing leaves the node).
export HOSTED_VLLM_API_KEY=local OPENAI_API_KEY=local LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_LOG=ERROR
mkdir -p "$APPTAINER_CONFIGDIR"

"$PY" - "$SRC" "$TASK_LIST" "$OUT/pack" "$N_TASKS" "$SEED" <<'PYEOF'
import json, os, random, sys
src, lst, pack, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ids = [json.loads(l)["task_id"] for l in open(lst) if l.strip()]
# Old base images ship a tmux without the `-e` option Terminus-2 uses to start its session (ubuntu:20.04 = 3.0a, debian:bullseye = 3.1c).
skip = set(filter(None, os.environ.get("EXCLUDE_BASE", "ubuntu:20.04,ubuntu:18.04,debian:bullseye,debian:buster").split(",")))
def base(t):
    try:
        return next((l.split()[1] for l in open(os.path.join(src, "tasks", t, "environment", "Dockerfile")) if l.startswith("FROM ")), "")
    except OSError:
        return ""
ids = [t for t in ids if base(t) not in skip]
random.Random(seed).shuffle(ids)
ids = ids[:n] if n > 0 else ids
for t in ids:
    os.symlink(os.path.join(src, "tasks", t), os.path.join(pack, t))
print(f"pack: {len(ids)} tasks")
PYEOF

server_pid=""; worker_pid=""; vllm_pid=""; sampler_pid=""
cleanup() { trap - EXIT INT TERM
            for p in "$sampler_pid" "$vllm_pid" "$worker_pid" "$server_pid"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

# 1. policy server(s) (OpenAI-compatible; Qwen3 reasoning parser matches the Prelude chat template). DP_SIZE data-parallel
#    replicas behind one endpoint, one per GPU.
: "${DP_SIZE:=1}"
singularity exec --nv -B /leonardo_work,/leonardo_scratch "$VLLM_CONTAINER" \
  env VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=fork "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED_NAME" --host 127.0.0.1 --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --data-parallel-size "$DP_SIZE" \
  --enable-prefix-caching --reasoning-parser qwen3 \
  >"$OUT/service-logs/vllm.log" 2>&1 & vllm_pid=$!

# 2. Harbor apptainer bridge (server + worker) for the sandboxes.
APPT="$HARBOR_LAION_ROOT/src/harbor/environments/apptainer"
PYTHON_BIN="$PY" "$APPT/start_server.sh" >"$OUT/service-logs/server.log" 2>&1 & server_pid=$!
for _ in $(seq 1 30); do
  curl -fsS -H "Authorization: Bearer $BRIDGE_AUTH_TOKEN" "$BRIDGE_URL/status" >/dev/null 2>&1 && break; sleep 1
done
curl -fsS -H "Authorization: Bearer $BRIDGE_AUTH_TOKEN" "$BRIDGE_URL/status" >/dev/null \
  || { echo "bridge did not start; see $OUT/service-logs/server.log" >&2; exit 1; }
PYTHON_BIN="$PY" "$APPT/start_worker.sh" >"$OUT/service-logs/worker.log" 2>&1 & worker_pid=$!

# 3. wait for the policy server.
for i in $(seq 1 ${VLLM_WAIT_STEPS:-420}); do   # 5 s steps: 4 replicas read the weights from Lustre at once (~7 min)
  curl -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$vllm_pid" 2>/dev/null || { echo "vLLM exited early; see $OUT/service-logs/vllm.log" >&2; tail -n 40 "$OUT/service-logs/vllm.log" >&2; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { echo "vLLM did not become ready" >&2; exit 1; }
echo "vllm ready after ~$((i*5))s; model=$MODEL tasks=$(ls "$OUT/pack" | wc -l) attempts=$ATTEMPTS"

# 3b. staging-usage sampler: what grows in the sandbox staging (overlay.img, per-instance tmp, pre-copied image prefixes)?
#     Written to service-logs/staging-usage.log every 60 s (aggregated by entry name over all live instances).
( while true; do
    { echo "== $(date +%T) instances=$(ls -d "$STAGING_BASE"/*/ 2>/dev/null | wc -l)"; df -h "$STAGING_ROOT" | tail -n 1
      du -sk "$STAGING_BASE"/*/* 2>/dev/null | awk '{n=$2; sub(".*/","",n); s[n]+=$1} END{for(k in s) printf "%8d MB  %s\n", s[k]/1024, k}' | sort -rn | head -n 6
      echo "-- 3 largest instances:"; du -sk "$STAGING_BASE"/*/ 2>/dev/null | sort -rn | head -n 3
    } >> "$OUT/service-logs/staging-usage.log"
    sleep 60
  done ) & sampler_pid=$!

# 4. rollouts + verification.
"$PY" -m harbor.cli.main run --path "$OUT/pack" --agent "$AGENT_NAME" --model "hosted_vllm/$SERVED_NAME" \
  --agent-kwarg "api_base=http://127.0.0.1:$VLLM_PORT/v1" --agent-kwarg "temperature=$TEMPERATURE" \
  --agent-kwarg "max_turns=$MAX_TURNS" \
  --agent-kwarg "model_info={\"max_input_tokens\": $MAX_MODEL_LEN, \"max_output_tokens\": $MAX_TOKENS_PER_TURN, \"input_cost_per_token\": 0.0, \"output_cost_per_token\": 0.0}" \
  --env apptainer --environment-kwarg "bridge_url=$BRIDGE_URL" --environment-kwarg "sif_cache=$SIF_CACHE" \
  --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
  --jobs-dir "$OUT/jobs" --job-name rollout --n-concurrent "$CONCURRENT" -k "$ATTEMPTS" --yes \
  || echo "[warn] harbor run exited non-zero"

# 5. summary: per-task mean reward + exceptions.
"$PY" - "$OUT/jobs/rollout" "$OUT/rollout-summary.json" <<'PYEOF'
import json, os, sys, collections
jobdir, out = sys.argv[1], sys.argv[2]
per, exc = collections.defaultdict(list), collections.Counter()
for d in sorted(os.listdir(jobdir)) if os.path.isdir(jobdir) else []:
    f = os.path.join(jobdir, d, "result.json")
    if not os.path.isfile(f): continue
    r = json.load(open(f))
    name = os.path.basename((r.get("task_id") or {}).get("path", d)) or d
    reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if r.get("exception_info"):
        exc[str(r["exception_info"].get("exception_type", "exception"))] += 1
    per[name].append(reward)
trials = [x for v in per.values() for x in v]
scored = [x for x in trials if x is not None]
summary = {"tasks": len(per), "trials": len(trials), "scored": len(scored),
           "mean_reward": (sum(scored) / len(scored)) if scored else None,
           "pass_rate": (sum(1 for x in scored if x >= 1.0) / len(scored)) if scored else None,
           "tasks_with_any_pass": sum(1 for v in per.values() if any((x or 0) >= 1.0 for x in v)),
           "exceptions": dict(exc), "per_task": {k: v for k, v in per.items()}}
json.dump(summary, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
PYEOF
echo "rollout_artifact=$OUT"
