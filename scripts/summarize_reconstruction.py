#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path):
    with path.open() as f:
        return json.load(f)["metrics"]


def main():
    root = Path("/Users/kylehowells/Developer/Github/demucs-mlx-swift/reference_outputs/reconstruction_eval/latest")
    files = {
        "demucs": root / "demucs_reconstruction_metrics.json",
        "demucs-mlx": root / "demucs_mlx_reconstruction_metrics.json",
        "demucs-mlx-swift": root / "demucs_mlx_swift_reconstruction_metrics.json",
    }

    data = {k: load_metrics(v) for k, v in files.items()}

    summary = {}
    for k, m in data.items():
        summary[k] = {
            "mae": m["mae"],
            "rmse": m["rmse"],
            "sdr_db": m["sdr_db"],
            "si_sdr_db": m["si_sdr_db"],
            "corr": m["correlation"],
            "err_ratio": m["error_to_signal_ratio"],
        }

    out_json = root / "reconstruction_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    names = list(summary.keys())
    mae = [summary[n]["mae"] for n in names]
    rmse = [summary[n]["rmse"] for n in names]
    sdr = [summary[n]["sdr_db"] for n in names]
    sisdr = [summary[n]["si_sdr_db"] for n in names]

    x = np.arange(len(names))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - w / 2, mae, width=w, label="MAE")
    axes[0].bar(x + w / 2, rmse, width=w, label="RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=10)
    axes[0].set_title("Reconstruction Error (lower better)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - w / 2, sdr, width=w, label="SDR (dB)")
    axes[1].bar(x + w / 2, sisdr, width=w, label="SI-SDR (dB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=10)
    axes[1].set_title("Reconstruction Quality (higher better)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_png = root / "reconstruction_summary_chart.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Wrote summary JSON: {out_json}")
    print(f"Wrote summary chart: {out_png}")


if __name__ == "__main__":
    main()
