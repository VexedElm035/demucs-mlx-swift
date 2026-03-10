#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


STEMS = ["drums", "bass", "other", "vocals"]


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    y, out_sr = librosa.load(str(path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    return y, out_sr


def align_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-9) -> float:
    ref_f = ref.reshape(-1).astype(np.float64)
    est_f = est.reshape(-1).astype(np.float64)
    alpha = np.dot(est_f, ref_f) / (np.dot(ref_f, ref_f) + eps)
    target = alpha * ref_f
    noise = est_f - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / denom)


def spectral_lsd_db(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    # Left channel diagnostic metric.
    a0 = a[0]
    b0 = b[0]
    sa = np.abs(librosa.stft(a0, n_fft=2048, hop_length=512))
    sb = np.abs(librosa.stft(b0, n_fft=2048, hop_length=512))
    m = min(sa.shape[1], sb.shape[1])
    sa = sa[:, :m]
    sb = sb[:, :m]
    la = 20.0 * np.log10(sa + 1e-9)
    lb = 20.0 * np.log10(sb + 1e-9)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def evaluate(original: np.ndarray, stems_sum: np.ndarray, sr: int) -> dict:
    original, stems_sum = align_len(original, stems_sum)
    residual = original - stems_sum

    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    peak_abs = float(np.max(np.abs(residual)))

    signal_power = float(np.mean(original**2))
    error_power = float(np.mean(residual**2))
    sdr = float(10.0 * np.log10((signal_power + 1e-12) / (error_power + 1e-12)))

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "peak_abs_residual": peak_abs,
        "sdr_db": sdr,
        "si_sdr_db": si_sdr(original, stems_sum),
        "correlation": corrcoef(original, stems_sum),
        "lsd_db": spectral_lsd_db(original, stems_sum, sr),
        "signal_power": signal_power,
        "error_power": error_power,
        "error_to_signal_ratio": float(error_power / (signal_power + 1e-12)),
    }
    return metrics


def plot_residual(original: np.ndarray, reconstructed: np.ndarray, sr: int, out: Path, title: str):
    original, reconstructed = align_len(original, reconstructed)
    residual = original - reconstructed
    t = np.arange(original.shape[-1]) / sr

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(t, original[0], label="original", alpha=0.8, linewidth=0.8)
    axes[0].plot(t, reconstructed[0], label="reconstructed", alpha=0.7, linewidth=0.8)
    axes[0].set_title(f"{title}: Original vs Reconstructed (left channel)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    spec = librosa.amplitude_to_db(np.abs(librosa.stft(residual[0], n_fft=2048, hop_length=512)) + 1e-9, ref=np.max)
    librosa.display.specshow(spec, sr=sr, hop_length=512, x_axis="time", y_axis="log", cmap="RdBu_r", ax=axes[1], vmin=-40, vmax=40)
    axes[1].set_title("Residual Spectrogram (left channel, dB)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Hz")

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def sum_stems(folder: Path, target_sr: int) -> tuple[np.ndarray, int]:
    stems = []
    sr = target_sr
    for stem in STEMS:
        p = folder / f"{stem}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Missing stem: {p}")
        y, _ = load_audio(p, sr=target_sr)
        stems.append(y)
    out = stems[0].copy()
    for s in stems[1:]:
        out, s = align_len(out, s)
        out = out + s
    return out, sr


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality by summing stems and comparing to original mix.")
    parser.add_argument("--original", type=Path, required=True, help="Path to original mix file")
    parser.add_argument("--stems-dir", type=Path, required=True, help="Directory containing drums/bass/other/vocals wav stems")
    parser.add_argument("--name", type=str, required=True, help="Label for report")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    original, sr = load_audio(args.original, sr=None)
    reconstructed, _ = sum_stems(args.stems_dir, sr)

    metrics = evaluate(original, reconstructed, sr)
    payload = {
        "name": args.name,
        "original": str(args.original.resolve()),
        "stems_dir": str(args.stems_dir.resolve()),
        "sample_rate": sr,
        "metrics": metrics,
    }

    json_path = args.out_dir / f"{args.name}_reconstruction_metrics.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    plot_path = args.out_dir / f"{args.name}_reconstruction_residual.png"
    plot_residual(original, reconstructed, sr, plot_path, args.name)

    print(f"Wrote metrics: {json_path}")
    print(f"Wrote residual plot: {plot_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
