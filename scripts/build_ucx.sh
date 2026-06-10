#!/usr/bin/env bash
# Build UCX (+ gdrcopy userspace lib) from source into a prefix, with the same
# flags BuildUCX.cmake uses. Handy to pre-build UCX once and then point commux
# (or snapy) at it via UCX_ROOT, instead of having CMake build it each time.
#
# Usage:
#   scripts/build_ucx.sh [PREFIX] [UCX_VERSION] [GDRCOPY_VERSION]
# Defaults:
#   PREFIX=$HOME/ucx-install  UCX_VERSION=1.18.0  GDRCOPY_VERSION=2.4.1
#
# Then:  export UCX_ROOT=$PREFIX   (commux/snapy find it via FindUCX.cmake)
#
# Requires: C/C++ compiler, make, wget/curl, tar, and (for GPU) the CUDA
# toolkit. The gdrdrv kernel module is a separate host prerequisite for gdr_copy
# to engage at runtime; without it UCX uses cuda_copy/cuda_ipc.
set -euo pipefail

PREFIX="${1:-$HOME/ucx-install}"
UCX_VERSION="${2:-1.18.0}"
GDRCOPY_VERSION="${3:-2.4.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
JOBS="$(nproc 2>/dev/null || echo 4)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$PREFIX"

fetch() {  # url dest
  if command -v wget >/dev/null 2>&1; then wget -qO "$2" "$1";
  else curl -fsSL -o "$2" "$1"; fi
}

GDR_FLAG="--without-gdrcopy"
if [ -d "$CUDA_HOME" ]; then
  echo "==> building gdrcopy ${GDRCOPY_VERSION} (userspace lib)"
  fetch "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz" "$WORK/gdr.tgz"
  tar -xzf "$WORK/gdr.tgz" -C "$WORK"
  make -C "$WORK/gdrcopy-${GDRCOPY_VERSION}" CUDA="$CUDA_HOME" lib
  make -C "$WORK/gdrcopy-${GDRCOPY_VERSION}" prefix="$PREFIX" CUDA="$CUDA_HOME" lib_install
  GDR_FLAG="--with-gdrcopy=$PREFIX"
  CUDA_FLAG="--with-cuda=$CUDA_HOME"
else
  echo "==> CUDA not found at $CUDA_HOME; building CPU-only UCX"
  CUDA_FLAG="--without-cuda"
fi

echo "==> building UCX ${UCX_VERSION}"
fetch "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" "$WORK/ucx.tgz"
tar -xzf "$WORK/ucx.tgz" -C "$WORK"
cd "$WORK/ucx-${UCX_VERSION}"
./configure --prefix="$PREFIX" --enable-mt "$CUDA_FLAG" "$GDR_FLAG" \
  --without-java --disable-doxygen-doc
make -j"$JOBS"
make install

echo
echo "UCX installed to: $PREFIX"
echo "  export UCX_ROOT=$PREFIX"
"$PREFIX/bin/ucx_info" -v 2>/dev/null | head -3 || true
