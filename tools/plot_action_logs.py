#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_dims(dims_str: str):
    if not dims_str:
        return None
    s = dims_str.strip().lower()
    if s == "left_arm":
        return list(range(0, 7))
    if s == "right_arm":
        return list(range(8, 15))
    if s == "grippers":
        return [7, 15]
    dims = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            st, ed = part.split(":", 1)
            dims.extend(list(range(int(st), int(ed))))
        else:
            dims.append(int(part))
    return sorted(set(dims))


def read_metadata(log_dir: Path):
    meta_path = log_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {log_dir}")
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_chunks(log_dir: Path, rows, kind: str):
    selected = [r for r in rows if r.get("kind") == kind]
    selected.sort(key=lambda r: (r.get("chunk_id", 0), r.get("timestamp", 0)))
    chunks = []
    for r in selected:
        file_name = r.get("file")
        if not file_name:
            continue
        p = log_dir / file_name
        if not p.exists():
            continue
        arr = np.load(p)
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks.append((r, arr))
    return chunks


def pick_dims(action_dim: int, dims):
    if dims is None:
        return list(range(action_dim))
    return [d for d in dims if 0 <= d < action_dim]


def plot_time_series(chunks, dims, save_path: Path):
    lengths = [arr.shape[0] for _, arr in chunks]
    offsets = np.cumsum([0] + lengths[:-1])

    plt.figure(figsize=(14, 6))
    for (meta, arr), off in zip(chunks, offsets):
        x = np.arange(arr.shape[0]) + off
        for d in dims:
            plt.plot(x, arr[:, d], linewidth=1.0, alpha=0.8)
        plt.axvline(x=x[-1], color="gray", linestyle="--", alpha=0.2)

    plt.title("Action time-series across chunks")
    plt.xlabel("Global step index")
    plt.ylabel("Action value")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_boundary_zoom(chunks, dims, save_path: Path, tail: int = 10, head: int = 10):
    if len(chunks) < 2:
        return
    n_boundaries = len(chunks) - 1
    fig, axes = plt.subplots(n_boundaries, 1, figsize=(12, 3 * n_boundaries), squeeze=False)

    for i in range(n_boundaries):
        prev_meta, prev_arr = chunks[i]
        next_meta, next_arr = chunks[i + 1]
        ax = axes[i, 0]
        prev_tail = prev_arr[max(0, prev_arr.shape[0] - tail):, :]
        next_head = next_arr[:head, :]

        x_prev = np.arange(-prev_tail.shape[0], 0)
        x_next = np.arange(0, next_head.shape[0])
        for d in dims:
            ax.plot(x_prev, prev_tail[:, d], color="tab:blue", alpha=0.8)
            ax.plot(x_next, next_head[:, d], color="tab:orange", alpha=0.8)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(
            f"Boundary {i}: chunk {prev_meta.get('chunk_id')} -> {next_meta.get('chunk_id')}"
        )
        ax.set_xlabel("Boundary-local step")
        ax.set_ylabel("Action")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_jump_magnitude(chunks, save_path: Path):
    intra = []
    boundary = []
    for i, (_, arr) in enumerate(chunks):
        if arr.shape[0] > 1:
            diffs = arr[1:] - arr[:-1]
            intra.extend(np.linalg.norm(diffs, axis=1).tolist())
        if i > 0:
            prev_arr = chunks[i - 1][1]
            jump = np.linalg.norm(arr[0] - prev_arr[-1])
            boundary.append(jump)

    plt.figure(figsize=(12, 5))
    if intra:
        plt.plot(intra, label="intra-chunk step jump", alpha=0.7)
    if boundary:
        bx = np.linspace(0, max(len(intra) - 1, 1), num=len(boundary))
        plt.scatter(bx, boundary, label="boundary jump", color="red")
    plt.title("Jump magnitude (L2 norm)")
    plt.xlabel("Step index")
    plt.ylabel("L2 jump")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_histogram(chunks, dims, save_path: Path):
    stacked = np.concatenate([arr for _, arr in chunks], axis=0)
    plt.figure(figsize=(12, 5))
    for d in dims:
        plt.hist(stacked[:, d], bins=60, alpha=0.35, label=f"dim {d}")
    plt.title("Action value histogram (secondary diagnostic)")
    plt.xlabel("Action value")
    plt.ylabel("Count")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot action logs for cross-chunk diagnostics")
    parser.add_argument("--log_dir", required=True, help="Task log directory containing metadata.jsonl and npy files")
    parser.add_argument("--kind", default="raw_chunk", choices=["raw_chunk", "sync_smoothed_chunk", "rtc_fused_chunk", "executed_step"])
    parser.add_argument("--dims", default="0:16", help="Dims spec, e.g. '0:7', '8:15', '7,15', 'left_arm', 'right_arm', 'grippers'")
    parser.add_argument("--boundary_only", action="store_true", help="Only render boundary zoom and jump plots")
    parser.add_argument("--save_dir", default=None, help="Directory to save png files")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = read_metadata(log_dir)
    chunks = collect_chunks(log_dir, rows, args.kind)
    if not chunks:
        raise RuntimeError(f"No chunks found for kind={args.kind} in {log_dir}")

    action_dim = chunks[0][1].shape[1]
    dims = pick_dims(action_dim, parse_dims(args.dims))
    if not dims:
        raise RuntimeError("No valid dims selected")

    save_dir = Path(args.save_dir) if args.save_dir else (log_dir / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.boundary_only:
        plot_time_series(chunks, dims, save_dir / f"{args.kind}_timeseries.png")
        plot_histogram(chunks, dims, save_dir / f"{args.kind}_hist.png")
    plot_boundary_zoom(chunks, dims, save_dir / f"{args.kind}_boundary_zoom.png")
    plot_jump_magnitude(chunks, save_dir / f"{args.kind}_jump.png")

    print(f"Saved plots to: {save_dir}")


if __name__ == "__main__":
    main()
