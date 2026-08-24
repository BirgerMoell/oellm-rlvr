#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:?usage: build_lumi_task_sandbox.sh OUTPUT [OCI_IMAGE]}
OCI_IMAGE=${2:-docker://python:3.12-slim}

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing task sandbox: $OUTPUT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
singularity build --sandbox "$OUTPUT" "$OCI_IMAGE"
echo "$OUTPUT"
