#!/bin/bash
set -euo pipefail

# ============================================================================
# Benchmark all three Demucs implementations:
#   1. Original PyTorch Demucs
#   2. Python MLX Demucs (demucs-mlx)
#   3. Swift MLX Demucs (demucs-mlx-swift)
#
# Usage:
#   ./scripts/benchmark_all.sh <input_audio.mp3> [model_name]
#
# Prerequisites:
#   - Python with demucs and demucs-mlx installed
#   - Swift project built: swift build -c release
#   - Model checkpoints exported: python scripts/export_all_models.py
# ============================================================================

INPUT="${1:?Usage: $0 <input_audio> [model_name]}"
MODEL="${2:-htdemucs}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Output directories
OUT_ROOT="$PROJECT_DIR/.scratch/benchmark"
OUT_PYTORCH="$OUT_ROOT/pytorch_${MODEL}"
OUT_MLX_PY="$OUT_ROOT/mlx_python_${MODEL}"
OUT_SWIFT="$OUT_ROOT/swift_${MODEL}"

mkdir -p "$OUT_ROOT"

echo "============================================"
echo "Benchmark: $MODEL"
echo "Input: $INPUT"
echo "============================================"

# ----------------------------------------------------------
# 1. Original PyTorch Demucs
# ----------------------------------------------------------
echo ""
echo "--- [1/3] PyTorch Demucs ---"
PYTORCH_START=$(python3 -c "import time; print(time.time())")
python3 -m demucs --name "$MODEL" --out "$OUT_PYTORCH" "$INPUT" 2>&1 || echo "  (PyTorch failed or not installed)"
PYTORCH_END=$(python3 -c "import time; print(time.time())")
PYTORCH_TIME=$(python3 -c "print(f'{$PYTORCH_END - $PYTORCH_START:.2f}')")
echo "  PyTorch time: ${PYTORCH_TIME}s"

# ----------------------------------------------------------
# 2. Python MLX Demucs
# ----------------------------------------------------------
echo ""
echo "--- [2/3] Python MLX Demucs ---"
MLX_PY_START=$(python3 -c "import time; print(time.time())")
python3 -m demucs_mlx --model "$MODEL" --out "$OUT_MLX_PY" "$INPUT" 2>&1 || echo "  (demucs-mlx failed or not installed)"
MLX_PY_END=$(python3 -c "import time; print(time.time())")
MLX_PY_TIME=$(python3 -c "print(f'{$MLX_PY_END - $MLX_PY_START:.2f}')")
echo "  Python MLX time: ${MLX_PY_TIME}s"

# ----------------------------------------------------------
# 3. Swift MLX Demucs
# ----------------------------------------------------------
echo ""
echo "--- [3/3] Swift MLX Demucs ---"

# Build release if not already built
if [ ! -f "$PROJECT_DIR/.build/release/demucs-mlx-swift" ]; then
    echo "  Building release binary..."
    (cd "$PROJECT_DIR" && swift build -c release 2>&1 | tail -1)
fi

SWIFT_BIN="$PROJECT_DIR/.build/release/demucs-mlx-swift"
SWIFT_START=$(python3 -c "import time; print(time.time())")
"$SWIFT_BIN" --name "$MODEL" --out "$OUT_SWIFT" "$INPUT" 2>&1 || echo "  (Swift failed)"
SWIFT_END=$(python3 -c "import time; print(time.time())")
SWIFT_TIME=$(python3 -c "print(f'{$SWIFT_END - $SWIFT_START:.2f}')")
echo "  Swift MLX time: ${SWIFT_TIME}s"

# ----------------------------------------------------------
# Compare outputs
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "Output Quality Comparison"
echo "============================================"

# Find stem directories
find_stems() {
    local dir="$1"
    # Look for directory containing bass.wav, drums.wav, etc.
    find "$dir" -name "bass.wav" -exec dirname {} \; 2>/dev/null | head -1
}

PYTORCH_STEMS=$(find_stems "$OUT_PYTORCH" 2>/dev/null || echo "")
MLX_PY_STEMS=$(find_stems "$OUT_MLX_PY" 2>/dev/null || echo "")
SWIFT_STEMS=$(find_stems "$OUT_SWIFT" 2>/dev/null || echo "")

if [ -n "$PYTORCH_STEMS" ] && [ -n "$MLX_PY_STEMS" ]; then
    echo ""
    echo "--- PyTorch vs Python MLX ---"
    python3 "$SCRIPT_DIR/compare_stems.py" --ref "$PYTORCH_STEMS" --cand "$MLX_PY_STEMS" --name "PyTorch vs Python MLX" 2>&1 || true
fi

if [ -n "$PYTORCH_STEMS" ] && [ -n "$SWIFT_STEMS" ]; then
    echo ""
    echo "--- PyTorch vs Swift MLX ---"
    python3 "$SCRIPT_DIR/compare_stems.py" --ref "$PYTORCH_STEMS" --cand "$SWIFT_STEMS" --name "PyTorch vs Swift MLX" 2>&1 || true
fi

if [ -n "$MLX_PY_STEMS" ] && [ -n "$SWIFT_STEMS" ]; then
    echo ""
    echo "--- Python MLX vs Swift MLX ---"
    python3 "$SCRIPT_DIR/compare_stems.py" --ref "$MLX_PY_STEMS" --cand "$SWIFT_STEMS" --name "Python MLX vs Swift MLX" 2>&1 || true
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================"
echo "Speed Summary ($MODEL)"
echo "============================================"
echo "PyTorch:    ${PYTORCH_TIME}s"
echo "Python MLX: ${MLX_PY_TIME}s"
echo "Swift MLX:  ${SWIFT_TIME}s"
echo "============================================"
