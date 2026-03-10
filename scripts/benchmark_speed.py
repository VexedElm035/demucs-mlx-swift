#!/usr/bin/env python3
import argparse
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt


def run_cmd(cmd: list[str], cwd: Path) -> float:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    end = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return end - start


def benchmark_one(
    name: str,
    cmd: list[str],
    root: Path,
    runs: int,
    warmup: int,
    output_root: Path,
) -> list[float]:
    times: list[float] = []
    for i in range(warmup + runs):
        run_dir = output_root / name / f"run_{i+1}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        patched = []
        for token in cmd:
            if token == "{OUT}":
                patched.append(str(run_dir))
            else:
                patched.append(token)

        elapsed = run_cmd(patched, root)
        if i >= warmup:
            times.append(elapsed)
            print(f"[{name}] run {i - warmup + 1}/{runs}: {elapsed:.2f}s")
        else:
            print(f"[{name}] warmup {i + 1}/{warmup}: {elapsed:.2f}s")
    return times


def chart(results: dict, out_path: Path):
    names = list(results.keys())
    means = [results[n]["mean_s"] for n in names]
    stds = [results[n]["stdev_s"] for n in names]
    rtfs = [results[n]["rtf"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(names, means, yerr=stds, capsize=6)
    axes[0].set_ylabel("Seconds (lower is faster)")
    axes[0].set_title("Wall Time")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(means):
        axes[0].text(i, v, f"{v:.1f}s", ha="center", va="bottom")

    axes[1].bar(names, rtfs)
    axes[1].set_ylabel("x realtime (audio_duration / processing_time)")
    axes[1].set_title("Realtime Factor")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(rtfs):
        axes[1].text(i, v, f"{v:.2f}x", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark demucs vs demucs-mlx vs demucs-mlx-swift.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--swift-bin", type=Path, default=Path(".build/release/demucs-mlx-swift"))
    parser.add_argument("--model-dir", type=Path, default=Path(".scratch/models/htdemucs"))
    parser.add_argument("--out-dir", type=Path, default=Path("reference_outputs/benchmarks/latest"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_sec = librosa.get_duration(path=str(args.input))
    print(f"Input duration: {audio_sec:.2f}s")

    jobs = {
        "demucs": [
            "python",
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "-o",
            "{OUT}",
            str(args.input),
        ],
        "demucs-mlx": [
            "demucs-mlx",
            "-n",
            "htdemucs",
            "-o",
            "{OUT}",
            str(args.input),
        ],
        "demucs-mlx-swift": [
            str((root / args.swift_bin).resolve()),
            "--model-dir",
            str((root / args.model_dir).resolve()),
            "--out",
            "{OUT}",
            "--batch-size",
            "1",
            str(args.input),
        ],
    }

    results = {}
    raw_times = {}
    for name, cmd in jobs.items():
        times = benchmark_one(name, cmd, root, args.runs, args.warmup, out_dir / "raw_runs")
        mean_s = statistics.mean(times)
        stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
        rtf = audio_sec / mean_s
        results[name] = {
            "mean_s": mean_s,
            "stdev_s": stdev_s,
            "rtf": rtf,
        }
        raw_times[name] = times

    payload = {
        "input": str(args.input.resolve()),
        "audio_duration_s": audio_sec,
        "runs": args.runs,
        "warmup": args.warmup,
        "summary": results,
        "times_s": raw_times,
    }

    json_path = out_dir / "benchmark_results.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    png_path = out_dir / "benchmark_chart.png"
    chart(results, png_path)

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote chart: {png_path}")


if __name__ == "__main__":
    main()
