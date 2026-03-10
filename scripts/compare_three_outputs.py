#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np


STEMS = ["bass", "drums", "other", "vocals"]
PAIRS = [("demucs", "demucs_mlx"), ("demucs", "swift"), ("demucs_mlx", "swift")]


def load_stereo(path: Path, target_sr: int | None = None):
    y, sr = librosa.load(path, sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    return y, sr


def align_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    n = min(a.shape[-1], b.shape[-1], c.shape[-1])
    return a[..., :n], b[..., :n], c[..., :n]


def mag_db(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
    d = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(d) + 1e-10, ref=np.max)


def save_waveform_overlay(stem: str, data: dict[str, np.ndarray], sr: int, out: Path):
    # Use left channel for display; metrics rely on full stereo elsewhere.
    y0 = data["demucs"][0]
    y1 = data["demucs_mlx"][0]
    y2 = data["swift"][0]
    t = np.arange(y0.shape[-1]) / sr

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, y0, label="demucs", alpha=0.8, linewidth=0.8)
    ax.plot(t, y1, label="demucs-mlx", alpha=0.7, linewidth=0.8)
    ax.plot(t, y2, label="demucs-mlx-swift", alpha=0.7, linewidth=0.8)
    ax.set_title(f"{stem}: Waveform Overlay (left channel)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def save_spectrogram_grid(stem: str, data: dict[str, np.ndarray], sr: int, out: Path):
    specs = {
        "demucs": mag_db(data["demucs"][0], sr),
        "demucs-mlx": mag_db(data["demucs_mlx"][0], sr),
        "demucs-mlx-swift": mag_db(data["swift"][0], sr),
    }

    vmin = min(np.min(v) for v in specs.values())
    vmax = max(np.max(v) for v in specs.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (name, spec) in zip(axes, specs.items()):
        img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(name)
    # Reserve a dedicated margin on the right so the colorbar never overlays plots.
    fig.subplots_adjust(top=0.86, right=0.92, wspace=0.10)
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", fraction=0.03, pad=0.02)
    cbar.set_label("dB")
    fig.suptitle(f"{stem}: Spectrograms (left channel)", y=0.98)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def save_pairwise_diff(stem: str, data: dict[str, np.ndarray], sr: int, out: Path):
    # Waveform and spectrogram differences for each pair.
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))

    for i, (a_name, b_name) in enumerate(PAIRS):
        a = data[a_name][0]
        b = data[b_name][0]
        n = min(a.shape[-1], b.shape[-1])
        a = a[:n]
        b = b[:n]
        t = np.arange(n) / sr
        diff = a - b

        axes[i, 0].plot(t, diff, linewidth=0.7)
        axes[i, 0].axhline(0, linestyle="--", linewidth=0.8)
        axes[i, 0].set_title(f"Waveform diff: {a_name} - {b_name}")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(alpha=0.3)

        sa = mag_db(a, sr)
        sb = mag_db(b, sr)
        m = min(sa.shape[1], sb.shape[1])
        sdiff = sa[:, :m] - sb[:, :m]
        img = librosa.display.specshow(sdiff, sr=sr, x_axis="time", y_axis="log", ax=axes[i, 1], cmap="RdBu_r", vmin=-20, vmax=20)
        axes[i, 1].set_title(f"Spec diff (dB): {a_name} - {b_name}")
        fig.colorbar(img, ax=axes[i, 1], format="%+2.0f dB")

    fig.suptitle(f"{stem}: Pairwise Differences", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare demucs vs demucs-mlx vs demucs-mlx-swift outputs.")
    parser.add_argument("--demucs", type=Path, required=True)
    parser.add_argument("--demucs-mlx", dest="demucs_mlx", type=Path, required=True)
    parser.add_argument("--swift", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for stem in STEMS:
        paths = {
            "demucs": args.demucs / f"{stem}.wav",
            "demucs_mlx": args.demucs_mlx / f"{stem}.wav",
            "swift": args.swift / f"{stem}.wav",
        }
        for name, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing {name} stem: {p}")

        d0, sr0 = load_stereo(paths["demucs"])
        d1, sr1 = load_stereo(paths["demucs_mlx"], target_sr=sr0)
        d2, sr2 = load_stereo(paths["swift"], target_sr=sr0)
        if sr1 != sr0 or sr2 != sr0:
            raise RuntimeError("Sample rate mismatch after load/resample")

        d0, d1, d2 = align_three(d0, d1, d2)
        data = {"demucs": d0, "demucs_mlx": d1, "swift": d2}

        save_waveform_overlay(stem, data, sr0, args.out / f"{stem}_waveform_overlay.png")
        save_spectrogram_grid(stem, data, sr0, args.out / f"{stem}_spectrograms.png")
        save_pairwise_diff(stem, data, sr0, args.out / f"{stem}_pairwise_differences.png")

    print(f"Wrote comparison images to: {args.out}")


if __name__ == "__main__":
    # Lazy import to avoid hard dependency when module is inspected without plotting.
    import librosa.display  # noqa: F401

    main()
