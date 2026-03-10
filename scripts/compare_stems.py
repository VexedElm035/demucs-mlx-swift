#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import torch
import torchaudio


def load(path: Path):
    wav, sr = torchaudio.load(str(path))
    return wav, sr


def align(a: torch.Tensor, b: torch.Tensor):
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def metrics(ref: torch.Tensor, pred: torch.Tensor):
    ref, pred = align(ref, pred)
    diff = pred - ref
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    max_abs = diff.abs().max().item()

    signal_power = (ref ** 2).mean().item()
    error_power = (diff ** 2).mean().item()
    sdr = 10.0 * math.log10((signal_power + 1e-12) / (error_power + 1e-12))

    ref_flat = ref.reshape(-1)
    pred_flat = pred.reshape(-1)
    ref_center = ref_flat - ref_flat.mean()
    pred_center = pred_flat - pred_flat.mean()
    denom = (ref_center.norm() * pred_center.norm()).item() + 1e-12
    corr = (ref_center @ pred_center).item() / denom

    return {
        "mae": mae,
        "mse": mse,
        "max_abs": max_abs,
        "sdr_db": sdr,
        "corr": corr,
    }


def find_stem_dir(root: Path) -> Path:
    # Accept direct stem dir or one level wrapper (like htdemucs/<track>)
    stems = {"bass.wav", "drums.wav", "other.wav", "vocals.wav"}
    if root.is_dir() and stems.issubset({p.name for p in root.glob("*.wav")}):
        return root
    matches = []
    for p in root.rglob("*"):
        if p.is_dir() and stems.issubset({x.name for x in p.glob("*.wav")}):
            matches.append(p)
    if not matches:
        raise FileNotFoundError(f"No stem folder found under {root}")
    # pick shortest path depth
    matches.sort(key=lambda x: len(x.parts))
    return matches[0]


def compare(name: str, ref_dir: Path, cand_dir: Path):
    print(f"\n== {name} ==")
    print(f"ref:  {ref_dir}")
    print(f"cand: {cand_dir}")
    print("stem      mae        mse        sdr_db    corr      max_abs")
    print("--------- ---------- ---------- --------- --------- ----------")

    stems = ["bass", "drums", "other", "vocals"]
    agg = {k: 0.0 for k in ["mae", "mse", "sdr_db", "corr", "max_abs"]}

    for stem in stems:
        ref_wav, ref_sr = load(ref_dir / f"{stem}.wav")
        cand_wav, cand_sr = load(cand_dir / f"{stem}.wav")
        if ref_sr != cand_sr:
            cand_wav = torchaudio.functional.resample(cand_wav, cand_sr, ref_sr)
        m = metrics(ref_wav, cand_wav)
        print(f"{stem:<9} {m['mae']:<10.6f} {m['mse']:<10.6f} {m['sdr_db']:<9.3f} {m['corr']:<9.5f} {m['max_abs']:<10.6f}")
        for k in agg:
            agg[k] += m[k]

    n = len(stems)
    print("--------- ---------- ---------- --------- --------- ----------")
    print(f"avg       {agg['mae']/n:<10.6f} {agg['mse']/n:<10.6f} {agg['sdr_db']/n:<9.3f} {agg['corr']/n:<9.5f} {agg['max_abs']/n:<10.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--cand", required=True, type=Path)
    ap.add_argument("--name", default="comparison")
    args = ap.parse_args()

    ref = find_stem_dir(args.ref)
    cand = find_stem_dir(args.cand)
    compare(args.name, ref, cand)


if __name__ == "__main__":
    main()
