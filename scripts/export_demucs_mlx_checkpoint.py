#!/usr/bin/env python3
"""
Export demucs-mlx pickle checkpoint to flat safetensors + JSON metadata.

This is a preparation step for native Swift/MLX loading.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any
from fractions import Fraction

import mlx.core as mx


def flatten_tree(node: Any, prefix: str = "") -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}

    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_tree(v, key))
        return out

    if isinstance(node, (list, tuple)):
        for idx, v in enumerate(node):
            key = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(flatten_tree(v, key))
        return out

    # MLX array leaf
    if isinstance(node, mx.array):
        out[prefix] = node
        return out

    # Non-array leaf in state tree: ignore.
    return out


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=os.path.expanduser("~/.cache/demucs-mlx/htdemucs_mlx.pkl"),
        help="Path to demucs-mlx pickle checkpoint",
    )
    ap.add_argument(
        "--out-dir",
        default="./Models/htdemucs",
        help="Output directory",
    )
    ap.add_argument(
        "--name",
        default="htdemucs",
        help="Output model basename",
    )
    args = ap.parse_args()

    ck_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with ck_path.open("rb") as f:
        checkpoint = pickle.load(f)

    if "state" not in checkpoint:
        raise ValueError(f"No 'state' key in checkpoint: {ck_path}")

    flat = flatten_tree(checkpoint["state"])
    if not flat:
        raise ValueError("No MLX arrays found while flattening state tree")

    safetensors_path = out_dir / f"{args.name}.safetensors"
    config_path = out_dir / f"{args.name}_config.json"

    mx.save_safetensors(str(safetensors_path), flat)

    metadata = {
        "model_name": checkpoint.get("model_name"),
        "model_class": checkpoint.get("model_class"),
        "sub_model_class": checkpoint.get("sub_model_class"),
        "num_models": checkpoint.get("num_models"),
        "weights": checkpoint.get("weights"),
        "args": to_builtin(checkpoint.get("args", [])),
        "kwargs": to_builtin(checkpoint.get("kwargs", {})),
        "mlx_version": checkpoint.get("mlx_version"),
        "tensor_count": len(flat),
        "tensors": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
            for k, v in flat.items()
        },
    }

    with config_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"wrote {safetensors_path}")
    print(f"wrote {config_path}")
    print(f"tensors: {len(flat)}")


if __name__ == "__main__":
    main()
