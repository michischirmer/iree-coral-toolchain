#!/usr/bin/env bash
# Shared helpers for the IREE Coral Edge TPU toolchain scripts.
#
# Source this from every script:
#
#   TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#   . "$TOOLCHAIN_ROOT/scripts/lib.sh"
#
# It resolves paths, loads versions.lock and provides small logging helpers.
# Nothing here builds anything; it only makes the other scripts uniform.
#
# This file deliberately sets no `set -e`/`-u`/`-o pipefail`: a sourced library
# must not change the caller's error handling.  Every script sets its own flags
# before sourcing this file.

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Path independence: every path below is derived from TOOLCHAIN_ROOT or from an
# explicit environment override.  Nothing is derived from the current working
# directory, the hostname, the user name or the home directory, so the scripts
# work from any checkout in any location.
#
# Canonical overrides (recommended on a fresh machine):
#   IREE_CORAL_SOURCE          the IREE fork checkout
#   IREE_CORAL_BUILD_DIR       root for all build trees
#   IREE_CORAL_INSTALL_PREFIX  runtime install prefix
#   IREE_CORAL_WORKSPACE       umbrella root for fork + builds when the
#                              per-tree overrides are not used
#
# Individual trees can also be overridden on their own
# (IREE_CORAL_COMPILER_BUILD, IREE_CORAL_RUNTIME_INSTALL_BUILD, ...), so one
# build tree can live elsewhere without moving the others.

# Umbrella workspace: defaults to the toolchain's parent (the conventional
# layout has the fork and the builds beside the toolchain).
WORKSPACE="${IREE_CORAL_WORKSPACE:-$(cd "$TOOLCHAIN_ROOT/.." && pwd)}"

# Root for build trees.  IREE_CORAL_BUILD_DIR is the canonical override; the
# default keeps build trees beside the toolchain.
BUILD_ROOT="${IREE_CORAL_BUILD_DIR:-$WORKSPACE}"

# Load pinned versions.  The fork and Eigen archive URLs may be overridden from
# the environment (for a mirror); versions.lock supplies the defaults for a
# fresh machine.  Hashes always stay pinned.
_IREE_CORAL_REPO_OVERRIDE="${IREE_CORAL_REPO:-}"
_EIGEN_URL_OVERRIDE="${EIGEN_URL:-}"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/versions.lock"
IREE_CORAL_REPO="${_IREE_CORAL_REPO_OVERRIDE:-$IREE_CORAL_REPO}"
EIGEN_URL="${_EIGEN_URL_OVERRIDE:-$EIGEN_URL}"
unset _IREE_CORAL_REPO_OVERRIDE _EIGEN_URL_OVERRIDE

# The IREE fork checkout.  Resolve, in order: an explicit source override,
# IREE_CORAL_DIR, a sibling named by the pin, a sibling named "iree", then the
# configured default name.  A directory qualifies only if it actually
# contains the Coral compiler plugin, so a stray directory is never mistaken for
# the fork.
looks_like_fork() { [[ -d "$1/compiler/plugins/target/Coral" ]]; }

if [[ -n "${IREE_CORAL_SOURCE:-}" ]]; then
  IREE_DIR="$IREE_CORAL_SOURCE"
elif [[ -n "${IREE_CORAL_DIR:-}" ]]; then
  IREE_DIR="$IREE_CORAL_DIR"
else
  IREE_DIR=""
  for candidate in \
      "$WORKSPACE/$(basename "$IREE_CORAL_DEFAULT_DIR")" \
      "$WORKSPACE/iree-coral-fork" \
      "$WORKSPACE/iree"; do
    if looks_like_fork "$candidate"; then IREE_DIR="$candidate"; break; fi
  done
  [[ -n "$IREE_DIR" ]] || IREE_DIR="$WORKSPACE/$(basename "$IREE_CORAL_DEFAULT_DIR")"
fi

# Build trees (all regenerable).
COMPILER_BUILD="${IREE_CORAL_COMPILER_BUILD:-$BUILD_ROOT/iree-compiler-build}"
RUNTIME_BUILD="${IREE_CORAL_RUNTIME_BUILD:-$BUILD_ROOT/iree-runtime-build}"
RUNTIME_INSTALL="${IREE_CORAL_INSTALL_PREFIX:-${IREE_CORAL_RUNTIME_INSTALL:-$BUILD_ROOT/iree-runtime-install}}"
RUNTIME_INSTALL_BUILD="${IREE_CORAL_RUNTIME_INSTALL_BUILD:-$BUILD_ROOT/iree-runtime-install-build}"
TFLITE_SRC="${TFLITE_SRC:-$WORKSPACE/tensorflow-source}"
TFLITE_BUILD="${TFLITE_BUILD:-$BUILD_ROOT/tflite-build}"
# Verified Eigen archive, reused by every TFLite configure instead of cloning
# from GitLab (see scripts/fetch_third_party.sh).
EIGEN_ARCHIVE="${EIGEN_ARCHIVE:-$WORKSPACE/eigen-${EIGEN_COMMIT}.tar.gz}"
EDGETPU_SRC="${EDGETPU_SRC:-$WORKSPACE/edgetpu}"
# The TFLite-2.7-compatible libedgetpu.so.1.0 comes from the pycoral repository
# (libedgetpu_bin/), not from edgetpu (whose libedgetpu/ build is a newer ABI
# the TFLite 2.7 delegate rejects).  The edgetpu checkout provides the headers
# and the proprietary edgetpu_compiler.
PYCORAL_SRC="${PYCORAL_SRC:-$WORKSPACE/pycoral}"
LIBEDGETPU_DIR="${LIBEDGETPU_DIR:-$WORKSPACE/third-party/libedgetpu/lib}"
LIBEDGETPU_HEADERS="${LIBEDGETPU_HEADERS:-$WORKSPACE/libedgetpu/tflite/public}"
EDGETPU_COMPILER="${IREE_CORAL_EDGETPU_COMPILER:-$EDGETPU_SRC/compiler/x86_64/edgetpu_compiler}"

IREE_COMPILE="$COMPILER_BUILD/tools/iree-compile"
IREE_RUN_MODULE="$RUNTIME_INSTALL_BUILD/tools/iree-run-module"
# The native Coral shim.  Built into the toolchain's runtime/ directory by
# default (where verify_installation.sh and the runtime build look for it);
# override IREE_CORAL_NATIVE_BUILD to place the artifact elsewhere.
CORAL_NATIVE_BUILD="${IREE_CORAL_NATIVE_BUILD:-$TOOLCHAIN_ROOT/runtime}"
CORAL_NATIVE_SO="$CORAL_NATIVE_BUILD/libcoral_native.so"

VENV="${IREE_CORAL_VENV:-$BUILD_ROOT/.venv}"
PY="$VENV/bin/python"

# Compiler build-time intermediates (the Coral generation pass writes here).
GENERATED="${IREE_CORAL_GENERATED_ROOT:-$BUILD_ROOT/generated}"

# The fork's Coral generation pass shells out to this repository's generator
# scripts and the proprietary edgetpu_compiler, and writes intermediates under
# GENERATED. Those locations live here, not beside the build tree, so export
# them for any iree-compile invocation in this shell.
export IREE_CORAL_TOOLS_ROOT="${IREE_CORAL_TOOLS_ROOT:-$TOOLCHAIN_ROOT/tools}"
export IREE_CORAL_PYTHON="${IREE_CORAL_PYTHON:-$PY}"
export IREE_CORAL_GENERATED_ROOT="$GENERATED"
export IREE_CORAL_EDGETPU_COMPILER="$EDGETPU_COMPILER"

# The fork's Coral generation pass looks for its generator scripts under
# <workspace>/tools unless IREE_CORAL_TOOLS_ROOT says otherwise (see
# compiler/plugins/target/Coral/CoralGenerationPass.cpp).  The scripts live in
# this repository, which sits inside the workspace, so publish them under the
# name the pass expects.  The link is relative, which keeps the workspace
# relocatable, and it is created at most once.
publish_workspace_tools() {
  local scripts="$TOOLCHAIN_ROOT/tools"
  local link="$WORKSPACE/tools"
  [[ -d "$scripts" ]] || return 0
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$scripts")" ]] && return 0
    rm -f "$link" || return 0
  elif [[ -e "$link" ]]; then
    return 0 # a real directory already provides what the pass looks for
  fi
  local rel
  case "$scripts" in
    "$WORKSPACE"/*) rel="${scripts#"$WORKSPACE"/}" ;;
    *) return 0     # the toolchain lives outside the workspace
  esac
  ln -sfn "$rel" "$link" 2>/dev/null || true
}
publish_workspace_tools

# Build parallelism.  Two is deliberately conservative: the IREE build is large
# and two jobs completes on a machine with limited RAM.  Override with JOBS=<n>.
JOBS="${JOBS:-2}"

log()  { printf '%s\n' "$*" >&2; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  local c
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || die "required command not found: $c"
  done
}

require_file() {
  local f
  for f in "$@"; do
    [[ -f "$f" ]] || die "required file not found: $f"
  done
}

# Version comparison: returns 0 when $1 >= $2 (dotted numeric).
version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -1)" == "$1" ]]
}
