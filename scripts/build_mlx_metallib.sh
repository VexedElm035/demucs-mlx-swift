#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/build_mlx_metallib.sh [debug|release]

Builds MLX's Metal shader library (mlx.metallib) and copies it into the SwiftPM output directory.

If you see "missing Metal Toolchain", run:
  xcodebuild -downloadComponent MetalToolchain
USAGE
}

CONFIG="${1:-release}"
if [[ "$CONFIG" != "release" && "$CONFIG" != "debug" ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/.build"
MLX_SWIFT_DIR="$BUILD_DIR/checkouts/mlx-swift"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "error: $BUILD_DIR not found (run swift build first)" >&2
  exit 1
fi

if [[ ! -d "$KERNELS_DIR" ]]; then
  echo "error: MLX metal kernels not found at $KERNELS_DIR" >&2
  echo "hint: run swift build once to fetch dependencies" >&2
  exit 1
fi

OUT_DIR="$BUILD_DIR/$CONFIG"
if [[ ! -d "$OUT_DIR" ]]; then
  OUT_DIR="$(find "$BUILD_DIR" -maxdepth 3 -type d -path "*/$CONFIG" | head -n 1 || true)"
fi
if [[ -z "${OUT_DIR:-}" || ! -d "$OUT_DIR" ]]; then
  echo "error: could not find SwiftPM output directory for config=$CONFIG" >&2
  exit 1
fi

mapfile -t METAL_SRCS < <(find "$KERNELS_DIR" -type f -name '*.metal' ! -name '*_nax.metal' | LC_ALL=C sort)
if [[ "${#METAL_SRCS[@]}" -eq 0 ]]; then
  echo "error: no .metal files found under $KERNELS_DIR" >&2
  exit 1
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlx-metallib.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

METAL_FLAGS=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
)

echo "Compiling ${#METAL_SRCS[@]} Metal sources..."
AIR_FILES=()
for SRC in "${METAL_SRCS[@]}"; do
  REL="${SRC#"$KERNELS_DIR/"}"
  KEY="$(printf '%s' "$REL" | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
  OUT_AIR="$TMP/$KEY.air"

  if ! xcrun -sdk macosx metal "${METAL_FLAGS[@]}" -c "$SRC" -I"$KERNELS_DIR" -I"$MLX_SWIFT_DIR/Source/Cmlx/mlx" -o "$OUT_AIR" 2>"$TMP/metal.err"; then
    if grep -q "missing Metal Toolchain" "$TMP/metal.err" 2>/dev/null; then
      echo "error: missing Metal Toolchain" >&2
      echo "run: xcodebuild -downloadComponent MetalToolchain" >&2
    fi
    cat "$TMP/metal.err" >&2
    exit 1
  fi

  AIR_FILES+=("$OUT_AIR")
done

OUT_METALLIB="$OUT_DIR/mlx.metallib"
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$OUT_METALLIB"
echo "OK: wrote $OUT_METALLIB"
