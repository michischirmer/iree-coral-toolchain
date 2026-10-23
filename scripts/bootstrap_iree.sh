#!/usr/bin/env bash
# Bootstrap the IREE fork: clone it if absent, then check out the pinned Coral
# commit.
#
#   scripts/bootstrap_iree.sh [--repo URL] [--dir PATH]
#
# The fork is expected as a sibling of the toolchain (see versions.lock
# IREE_CORAL_DEFAULT_DIR).  When it is absent, the script clones
# IREE_CORAL_REPO (the fork URL pinned in versions.lock) at the pinned
# `coral-edge-tpu` branch, so a fresh machine bootstraps without arguments once
# that branch is published.  Use --repo (or IREE_CORAL_REPO) for a mirror.
#
# A fresh clone leaves IREE's third_party submodules empty; this script
# initializes them (shallowly first, for speed).
#
# This script never contacts the network unless it has to clone/create the
# checkout, and it never pushes.
set -euo pipefail

TOOLCHAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
. "$TOOLCHAIN_ROOT/scripts/lib.sh"

REPO="${IREE_CORAL_REPO:-}"
DIR="$IREE_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dir)  DIR="$2";  shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

log "=== bootstrap IREE fork ==="
log "  target dir : $DIR"
log "  repo       : ${REPO:-<existing checkout>}"
log "  branch     : $IREE_CORAL_BRANCH"
log "  commit     : $IREE_CORAL_COMMIT"

if [[ -d "$DIR/.git" ]]; then
  log "  checkout already present"
else
  [[ -n "$REPO" ]] || die "no fork checkout at $DIR and no --repo given; pass the fork's URL"
  require_cmd git
  mkdir -p "$(dirname "$DIR")"
  log "  cloning $REPO -> $DIR"
  git clone --branch "$IREE_CORAL_BRANCH" "$REPO" "$DIR" || \
    die "cannot clone branch $IREE_CORAL_BRANCH from $REPO (is it published?)"
fi

log ""
log "=== check out the pinned commit ==="
# The pinned commit is the verified revision and the only one bootstrap builds.
# A fresh clone normally contains it; otherwise fetch it by hash.  If the
# checkout is at a different revision (for example a branch that moved), it is
# moved to the pin instead of building whatever the branch head holds.
if ! git -C "$DIR" cat-file -e "$IREE_CORAL_COMMIT^{commit}" 2>/dev/null; then
  origin="$(git -C "$DIR" remote get-url origin 2>/dev/null || true)"
  fetch_url="${REPO:-$origin}"
  [[ -n "$fetch_url" ]] || \
    die "commit $IREE_CORAL_COMMIT not found in $DIR and no remote to fetch it from"
  log "  fetching pinned commit $IREE_CORAL_COMMIT from $fetch_url"
  git -C "$DIR" fetch "$fetch_url" "$IREE_CORAL_COMMIT" || \
    die "commit $IREE_CORAL_COMMIT is not available from $fetch_url"
fi
if [[ "$(git -C "$DIR" rev-parse HEAD)" == "$IREE_CORAL_COMMIT" ]]; then
  log "  already at $IREE_CORAL_COMMIT"
else
  git -C "$DIR" checkout --detach "$IREE_CORAL_COMMIT"
fi
[[ "$(git -C "$DIR" rev-parse HEAD)" == "$IREE_CORAL_COMMIT" ]] || \
  die "failed to check out $IREE_CORAL_COMMIT"

log ""
log "=== initialize git submodules ==="
# IREE's CMake refuses to configure while any submodule is uninitialized
# (IREE_ERROR_ON_MISSING_SUBMODULES), and a plain clone leaves all of them
# empty.  Try a shallow fetch first (llvm-project is ~3 GiB); fall back to a
# full fetch when a pinned commit is not reachable shallowly.
if [[ "$(git -C "$DIR" submodule status | grep -c '^-')" -gt 0 ]]; then
  git -C "$DIR" submodule sync --recursive
  if ! git -C "$DIR" submodule update --init --recursive --depth 1; then
    warn "shallow submodule fetch failed; retrying with full history"
    git -C "$DIR" submodule update --init --recursive
  fi
  log "  submodules initialized"
else
  log "  already initialized"
fi

log ""
log "=== verify the Coral sources are present ==="
missing=0
for p in $IREE_REQUIRED_PATHS; do
  if [[ -f "$DIR/$p" ]]; then
    log "  OK       $p"
  else
    log "  MISSING  $p"
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || die "fork is missing Coral sources; wrong branch or commit?"

log ""
log "IREE fork ready at $DIR ($(git -C "$DIR" rev-parse --short HEAD))"
