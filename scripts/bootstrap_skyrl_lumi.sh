#!/usr/bin/env bash
# Run on a LUMI login node after loading Local-LAIF and
# lumi-aif-singularity-bindings. Network access is required only here.
set -euo pipefail

: "${CONTROL_ROOT:=${1:-$PWD}}"
: "${SOURCE_ROOT:=/scratch/project_465002530/users/bmoell/rl-sources}"
: "${SKYRL_ROOT:=$SOURCE_ROOT/SkyRL-v0.3.0}"
: "${HARBOR_ROOT:=$SOURCE_ROOT/harbor-v0.22.0}"
: "${VENV:=/scratch/project_465002530/users/bmoell/venvs/oellm-skyrl-v030}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${BUILD_CACHE:=/scratch/project_465002530/users/bmoell/oellm-rlvr/build-cache}"

SKYRL_URL=https://github.com/NovaSky-AI/SkyRL.git
SKYRL_COMMIT=f5bc3b78dfddfb352870d5d7430cd226e5785838
HARBOR_URL=https://github.com/harbor-framework/harbor.git
HARBOR_COMMIT=4407eb5227a2ff4f0d3f16b2eb48849382fdf276
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
export PIP_CACHE_DIR="$BUILD_CACHE/pip" TMPDIR="$BUILD_CACHE/tmp"
# Some LUMI project roots contain administrator-owned Git metadata. Legacy
# setuptools sdists walk upwards from their temporary directory, so prevent
# that irrelevant repository discovery without changing ~/.gitconfig.
export GIT_CEILING_DIRECTORIES="$BUILD_CACHE"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

clone_at_commit() {
  local url="$1" commit="$2" destination="$3"
  if [[ ! -d "$destination/.git" ]]; then
    mkdir -p "$(dirname "$destination")"
    git clone --filter=blob:none "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
}

test -r "$CONTAINER"
test -r "$CONTROL_ROOT/containers/lumi-skyrl-overlay-requirements.txt"
test -r "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-no-fakeroot.patch"
test -r "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-unprivileged-server.patch"
test -r "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-unprivileged-server.patch"
clone_at_commit "$SKYRL_URL" "$SKYRL_COMMIT" "$SKYRL_ROOT"
clone_at_commit "$HARBOR_URL" "$HARBOR_COMMIT" "$HARBOR_ROOT"
# Harbor v0.22.0 unconditionally requests --fakeroot, which forces a user
# namespace. LUMI provides a supported setuid Singularity runtime instead, so
# make fakeroot conditional and disable it only in the LUMI batch environment.
if git -C "$HARBOR_ROOT" apply --check "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-no-fakeroot.patch"; then
  git -C "$HARBOR_ROOT" apply "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-no-fakeroot.patch"
elif ! git -C "$HARBOR_ROOT" apply --reverse --check \
  "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-no-fakeroot.patch"; then
  echo "Harbor LUMI fakeroot patch is neither applicable nor already applied" >&2
  exit 1
fi
if git -C "$HARBOR_ROOT" apply --check \
  "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-unprivileged-server.patch"; then
  git -C "$HARBOR_ROOT" apply \
    "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-unprivileged-server.patch"
elif ! git -C "$HARBOR_ROOT" apply --reverse --check \
  "$CONTROL_ROOT/patches/harbor-v0.22.0-lumi-unprivileged-server.patch"; then
  echo "Harbor LUMI server patch is neither applicable nor already applied" >&2
  exit 1
fi
# Harbor's upstream bootstrap tries to install packages from inside each
# compute-node sandbox. Replace only that file with the audited offline LUMI
# bootstrap; probe_skyrl_stack.py records and enforces the downstream patch.
cp "$CONTROL_ROOT/scripts/harbor_singularity_bootstrap_lumi.sh" \
  "$HARBOR_ROOT/src/harbor/environments/singularity/bootstrap.sh"

mkdir -p "$(dirname "$VENV")"
singularity exec -B "$BIND" "$CONTAINER" python -m venv --system-site-packages "$VENV"

run_python() {
  singularity exec -B "$BIND" "$CONTAINER" env \
    PIP_CACHE_DIR="$PIP_CACHE_DIR" TMPDIR="$TMPDIR" \
    GIT_CEILING_DIRECTORIES="$GIT_CEILING_DIRECTORIES" \
    PYTHONPATH="$SKYRL_ROOT:$HARBOR_ROOT/src:$CONTROL_ROOT/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}" \
    "$VENV/bin/python" "$@"
}

run_python -m pip install --upgrade 'pip<27' 'setuptools>=77.0.3,<80' wheel
run_python -m pip install -r "$CONTROL_ROOT/containers/lumi-skyrl-overlay-requirements.txt"
# The router wheel is architecture-neutral, but its dependency metadata would
# otherwise replace the LUMI-built vLLM. Install the pinned router alone.
run_python -m pip install --no-deps vllm-router==0.1.14
# Source installs are deliberately dependency-free. Their CUDA-oriented
# metadata must never drive resolution inside this ROCm overlay.
run_python -m pip install --no-deps -e "$SKYRL_ROOT"
run_python -m pip install --no-deps -e "$HARBOR_ROOT"
run_python -m pip install --no-deps -e "$CONTROL_ROOT[data,math]"

REPORT_ROOT="${REPORT_ROOT:-$SOURCE_ROOT/compatibility}"
mkdir -p "$REPORT_ROOT"
run_python "$CONTROL_ROOT/scripts/probe_skyrl_stack.py" \
  --skyrl-source "$SKYRL_ROOT" --harbor-source "$HARBOR_ROOT" \
  --output "$REPORT_ROOT/skyrl-harbor-login-imports.json"
run_python -m pip freeze > "$REPORT_ROOT/skyrl-harbor-overlay.freeze.txt"
echo "skyrl_root=$SKYRL_ROOT"
echo "harbor_root=$HARBOR_ROOT"
echo "venv=$VENV"
echo "compatibility_report=$REPORT_ROOT/skyrl-harbor-login-imports.json"
