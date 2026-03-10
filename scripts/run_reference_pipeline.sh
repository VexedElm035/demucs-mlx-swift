#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-Sample Files/09 Still Into You.m4a}"
VENV="${VENV:-$ROOT/.venv311}"

if [[ ! -f "$INPUT" ]]; then
  echo "error: input not found: $INPUT" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "error: venv not found: $VENV" >&2
  echo "create with: python3.11 -m venv .venv311" >&2
  exit 1
fi

source "$VENV/bin/activate"

echo "[1/5] Running demucs reference"
python -m demucs.separate -n htdemucs -o reference_outputs/demucs "$INPUT"

echo "[2/5] Running demucs-mlx reference"
demucs-mlx -n htdemucs -o reference_outputs/demucs_mlx "$INPUT"

echo "[3/5] Building Swift CLI"
swift build -c release --disable-sandbox

echo "[4/5] Running Swift output"
.build/release/demucs-mlx-swift "$INPUT" -o reference_outputs/swift_current --segment 8.0 --overlap 0.25 --shifts 1 --batch-size 8

echo "[5/5] Comparing outputs"
python scripts/compare_stems.py --name 'demucs-mlx vs demucs' --ref reference_outputs/demucs --cand reference_outputs/demucs_mlx
python scripts/compare_stems.py --name 'swift_current vs demucs' --ref reference_outputs/demucs --cand reference_outputs/swift_current
python scripts/compare_stems.py --name 'swift_current vs demucs-mlx' --ref reference_outputs/demucs_mlx --cand reference_outputs/swift_current
