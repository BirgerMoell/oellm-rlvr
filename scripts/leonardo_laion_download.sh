#!/bin/bash
# Download and unpack the LAION "OT validated RL datasets" collection for Harbor agentic RL on Leonardo.
# https://huggingface.co/collections/laion/ot-validated-rl-datasets
#
# Run on a LOGIN node (compute nodes are offline). Idempotent: finished datasets are skipped.
#   source leonardo.env && scripts/leonardo_laion_download.sh [dataset-name ...]
# With no arguments every dataset in DATASETS below is fetched. Names are the repo names without "laion/".
#
# Layout: $LAION_ROOT/<name>/{SHA256SUMS,manifests/,tasks.tar.zst,...} and $LAION_ROOT/<name>/tasks/ (extracted).
# Each dataset's commit hash is recorded in $LAION_ROOT/<name>/.revision so a run config can pin it.
#
# Caveat: the published SIF caches are aarch64 (Jupiter) and are NOT shipped here; build x86_64 SIFs separately and
# re-validate oracle/nop on Leonardo before trusting the "validated" label.
set -euo pipefail

: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/data/laion-ot}"

DATASETS=(
  swegym-1k-rootless-apptainer
  endless-terminals-rootless-apptainer
  rts-1k-rootless-apptainer
  tasktrove-bugsinpy-v4-oracle17-apptainer-v1
  openthoughts-tblite-dev-apptainer-v1
  nemotron-terminal-easy-clean1k-rootless-apptainer
  tasktrove-curriculum-medium-v2-rootless-apptainer
  tasktrove-curriculum-easy-rootless-apptainer
  inferredbugs-rootless-apptainer
  r2egym-1k-rootless-apptainer
  tmax-legacy-1k-rootless-apptainer
  terminal-bench-2-1-apptainer-v1
)
if [[ $# -gt 0 ]]; then DATASETS=("$@"); fi

if command -v hf >/dev/null 2>&1; then HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then HF=(huggingface-cli download)
else echo "need the hf CLI (pip install -U huggingface_hub)" >&2; exit 1; fi
command -v zstd >/dev/null 2>&1 || { echo "need zstd on PATH (module load zstd?)" >&2; exit 1; }

mkdir -p "$LAION_ROOT"
for name in "${DATASETS[@]}"; do
  dest="$LAION_ROOT/$name"
  if [[ -f "$dest/.complete" ]]; then echo "[skip] $name"; continue; fi
  echo "[fetch] laion/$name -> $dest"
  "${HF[@]}" "laion/$name" --repo-type dataset --local-dir "$dest"

  if [[ -f "$dest/SHA256SUMS" ]]; then
    (cd "$dest" && sha256sum --quiet -c SHA256SUMS) || { echo "[FAIL] checksum mismatch: $name" >&2; exit 1; }
  else
    echo "[warn] $name has no SHA256SUMS; integrity not checked"
  fi

  if [[ -f "$dest/tasks.tar.zst" ]]; then
    mkdir -p "$dest/tasks"
    tar --use-compress-program=zstd -xf "$dest/tasks.tar.zst" -C "$dest/tasks"
  else
    echo "[warn] $name has no tasks.tar.zst; inspect its layout before use"
  fi

  # Pin the revision (the cache metadata written by hf download, when present).
  rev=$(cat "$dest"/.cache/huggingface/download/*.metadata 2>/dev/null | head -n1 || true)
  echo "${rev:-unknown}" > "$dest/.revision"
  touch "$dest/.complete"
done

echo "done. root=$LAION_ROOT"
du -sh "$LAION_ROOT"/* 2>/dev/null || true
