import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate valid GigaHands training windows and plot the distribution "
            "of mean 3D hand-joint speed for each window."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the GigaHands dataset root.")
    parser.add_argument(
        "--train-text-file",
        type=Path,
        default=Path("runs/hand_motion_splits/train.jsonl"),
        help="Training JSONL file. Defaults to runs/hand_motion_splits/train.jsonl.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help=(
            "Fallback annotation JSONL when --train-text-file does not exist "
            "(default: <dataset_root>/annotations_v2.jsonl)."
        ),
    )
    parser.add_argument("--window-len", type=int, default=20, help="Number of sampled frames per window.")
    parser.add_argument(
        "--time-stride-aug-max",
        type=int,
        default=2,
        help="Enumerate training windows for strides 1..N; default matches torch_ode2vae_hand2.sh.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate used to convert frame gaps to seconds.")
    parser.add_argument(
        "--speed-unit",
        type=str,
        default="m/s",
        choices=("m/s", "mm/s"),
        help="Unit used in the CSV, JSON summary, and plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gigahands_window_speed"),
        help="Directory for the plot and tabular outputs.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional debug limit on the number of annotation rows scanned.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def sequence_name_from_row(row: Dict[str, object]) -> str:
    sequence = row["sequence"]
    if isinstance(sequence, list):
        return str(sequence[0])
    return str(sequence)


def resolve_keypoints_dir(dataset_root: Path, row: Dict[str, object]) -> Optional[Path]:
    scene = str(row["scene"])
    sequence_name = sequence_name_from_row(row)
    keypoints_dir = dataset_root / "hand_poses" / scene / "keypoints_3d" / sequence_name
    if (keypoints_dir / "chosen_frames_right.json").exists() and (keypoints_dir / "right.jsonl").exists():
        return keypoints_dir
    return None


def load_right_hand_joints(keypoints_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    with (keypoints_dir / "chosen_frames_right.json").open("r", encoding="utf-8") as file:
        frame_ids = np.asarray(json.load(file), dtype=np.int64)
    with (keypoints_dir / "right.jsonl").open("r", encoding="utf-8") as file:
        joints = np.asarray([json.loads(line) for line in file if line.strip()], dtype=np.float32)

    if len(frame_ids) != len(joints):
        raise ValueError(
            f"Frame/joint count mismatch in {keypoints_dir}: "
            f"{len(frame_ids)} frame ids vs {len(joints)} joint frames."
        )
    if joints.ndim != 3 or joints.shape[-1] < 3:
        raise ValueError(f"Expected joints shaped [T, J, >=3] in {keypoints_dir}, got {joints.shape}.")
    order = np.argsort(frame_ids)
    return frame_ids[order], joints[order, :, :3]


def select_and_densify_clip(
    frame_ids: np.ndarray,
    joints: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_range = frame_ids >= start_frame
    if end_frame != -1:
        in_range &= frame_ids <= end_frame

    frame_ids = frame_ids[in_range]
    joints = joints[in_range]
    if len(frame_ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool), np.empty((0, 0, 3), dtype=np.float32)

    dense_end = int(end_frame) if end_frame != -1 else int(frame_ids[-1])
    dense_frame_ids = np.arange(start_frame, dense_end + 1, dtype=np.int64)
    dense_joints = np.zeros((len(dense_frame_ids), joints.shape[1], 3), dtype=np.float32)
    mask = np.zeros(len(dense_frame_ids), dtype=bool)

    positions = frame_ids - start_frame
    dense_joints[positions] = joints
    mask[positions] = True
    return dense_frame_ids, mask, dense_joints


def valid_sampling_options(mask: np.ndarray, target_len: int, max_stride: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("--window-len must be positive.")
    if max_stride <= 0:
        raise ValueError("--time-stride-aug-max must be positive.")

    target_offsets = np.arange(target_len, dtype=np.int64)
    options: List[np.ndarray] = []
    total_len = mask.shape[0]
    for stride in range(1, max_stride + 1):
        span = 1 + (target_len - 1) * stride
        if span > total_len:
            break
        starts = np.arange(total_len - span + 1, dtype=np.int64)
        idx = starts[:, None] + stride * target_offsets[None, :]
        valid_starts = starts[mask[idx].all(axis=1)]
        if valid_starts.size == 0:
            continue
        stride_column = np.full((valid_starts.shape[0], 1), stride, dtype=np.int64)
        options.append(np.concatenate([valid_starts[:, None], stride_column], axis=1))

    if not options:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(options, axis=0)


def window_mean_joint_speed(
    frame_ids: np.ndarray,
    joints: np.ndarray,
    start: int,
    stride: int,
    window_len: int,
    fps: float,
) -> float:
    idx = start + stride * np.arange(window_len, dtype=np.int64)
    window_joints = joints[idx]
    window_frame_ids = frame_ids[idx]
    dt = np.diff(window_frame_ids).astype(np.float32) / float(fps)
    if np.any(dt <= 0):
        raise ValueError("Window frame ids must be strictly increasing.")

    distances = np.linalg.norm(np.diff(window_joints, axis=0), axis=-1)
    speeds = distances / dt[:, None]
    return float(speeds.mean())


def make_window_csv_writer(file: TextIO) -> csv.DictWriter:
    fieldnames = [
        "row_index",
        "scene",
        "sequence",
        "window_start_index",
        "window_stride",
        "start_frame_id",
        "end_frame_id",
        "mean_joint_speed",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def make_sequence_csv_writer(file: TextIO) -> csv.DictWriter:
    fieldnames = [
        "row_index",
        "scene",
        "sequence",
        "window_count",
        "mean_window_speed",
        "median_window_speed",
        "min_window_speed",
        "max_window_speed",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def plot_distribution(
    path: Path,
    speeds: np.ndarray,
    bins: int,
    unit: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=160)
    ax.hist(speeds, bins=bins, color="#2f6f73", edgecolor="white", linewidth=0.45, alpha=0.9)
    ax.axvline(float(np.mean(speeds)), color="#c44536", linewidth=1.8, label=f"mean={np.mean(speeds):.4g}")
    ax.axvline(float(np.median(speeds)), color="#f2a541", linewidth=1.8, label=f"median={np.median(speeds):.4g}")
    ax.set_title(title)
    ax.set_xlabel(f"Mean joint speed ({unit})")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def finite_percentiles(values: np.ndarray) -> Dict[str, float]:
    percentiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    keys = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]
    return {key: float(value) for key, value in zip(keys, percentiles)}


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        **finite_percentiles(values),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    train_text_file = args.train_text_file.expanduser().resolve()
    annotation_file = (
        args.annotation_file.expanduser().resolve()
        if args.annotation_file is not None
        else dataset_root / "annotations_v2.jsonl"
    )
    text_file = train_text_file if train_text_file.exists() else annotation_file

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not text_file.exists():
        raise FileNotFoundError(f"Training/annotation file does not exist: {text_file}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive when provided.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(load_jsonl(text_file))
    if args.max_sequences is not None:
        rows = rows[: args.max_sequences]

    speeds: List[float] = []
    sequence_mean_speeds: List[float] = []
    scanned_sequences = 0
    used_sequences = 0
    skipped_sequences = 0
    csv_path = output_dir / "window_mean_joint_speeds.csv"
    sequence_csv_path = output_dir / "sequence_mean_window_speeds.csv"

    iterator: Iterable[Tuple[int, Dict[str, object]]] = enumerate(rows)
    if not args.quiet:
        iterator = tqdm(list(iterator), desc="Enumerating windows", unit="seq")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file, sequence_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as sequence_csv_file:
        csv_writer = make_window_csv_writer(csv_file)
        sequence_csv_writer = make_sequence_csv_writer(sequence_csv_file)
        for row_index, row in iterator:
            scanned_sequences += 1
            keypoints_dir = resolve_keypoints_dir(dataset_root, row)
            if keypoints_dir is None:
                skipped_sequences += 1
                continue

            frame_ids, joints = load_right_hand_joints(keypoints_dir)
            start_frame = int(row.get("start_frame_id", 0))
            end_frame = int(row.get("end_frame_id", -1))
            dense_frame_ids, mask, dense_joints = select_and_densify_clip(
                frame_ids=frame_ids,
                joints=joints,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            options = valid_sampling_options(
                mask=mask,
                target_len=args.window_len,
                max_stride=args.time_stride_aug_max,
            )
            if options.size == 0:
                skipped_sequences += 1
                continue

            used_sequences += 1
            scene = str(row["scene"])
            sequence = sequence_name_from_row(row)
            current_sequence_speeds: List[float] = []
            for start, stride in options:
                speed = window_mean_joint_speed(
                    frame_ids=dense_frame_ids,
                    joints=dense_joints,
                    start=int(start),
                    stride=int(stride),
                    window_len=args.window_len,
                    fps=args.fps,
                )
                if args.speed_unit == "mm/s":
                    speed *= 1000.0
                speeds.append(speed)
                current_sequence_speeds.append(speed)
                idx = int(start) + int(stride) * np.arange(args.window_len, dtype=np.int64)
                csv_writer.writerow(
                    {
                        "row_index": row_index,
                        "scene": scene,
                        "sequence": sequence,
                        "window_start_index": int(start),
                        "window_stride": int(stride),
                        "start_frame_id": int(dense_frame_ids[idx[0]]),
                        "end_frame_id": int(dense_frame_ids[idx[-1]]),
                        "mean_joint_speed": speed,
                    }
                )
            current_sequence_array = np.asarray(current_sequence_speeds, dtype=np.float64)
            sequence_mean_speed = float(np.mean(current_sequence_array))
            sequence_mean_speeds.append(sequence_mean_speed)
            sequence_csv_writer.writerow(
                {
                    "row_index": row_index,
                    "scene": scene,
                    "sequence": sequence,
                    "window_count": int(current_sequence_array.size),
                    "mean_window_speed": sequence_mean_speed,
                    "median_window_speed": float(np.median(current_sequence_array)),
                    "min_window_speed": float(np.min(current_sequence_array)),
                    "max_window_speed": float(np.max(current_sequence_array)),
                }
            )

    if not speeds:
        raise RuntimeError(
            f"No valid windows found for window_len={args.window_len}, "
            f"time_stride_aug_max={args.time_stride_aug_max} from {text_file}."
        )

    speed_array = np.asarray(speeds, dtype=np.float64)
    sequence_speed_array = np.asarray(sequence_mean_speeds, dtype=np.float64)
    npy_path = output_dir / "window_mean_joint_speeds.npy"
    sequence_npy_path = output_dir / "sequence_mean_window_speeds.npy"
    plot_path = output_dir / "window_mean_joint_speed_distribution.png"
    sequence_plot_path = output_dir / "sequence_mean_window_speed_distribution.png"
    summary_path = output_dir / "summary.json"

    np.save(npy_path, speed_array)
    np.save(sequence_npy_path, sequence_speed_array)
    plot_distribution(
        plot_path,
        speed_array,
        args.bins,
        args.speed_unit,
        title="GigaHands Training Window Mean Joint Speed Distribution",
        ylabel="Window count",
    )
    plot_distribution(
        sequence_plot_path,
        sequence_speed_array,
        args.bins,
        args.speed_unit,
        title="GigaHands Per-Sequence Mean Window Speed Distribution",
        ylabel="Sequence count",
    )

    summary = {
        "dataset_root": str(dataset_root),
        "text_file": str(text_file),
        "window_len": int(args.window_len),
        "time_stride_aug_max": int(args.time_stride_aug_max),
        "fps": float(args.fps),
        "speed_unit": args.speed_unit,
        "scanned_sequences": int(scanned_sequences),
        "used_sequences": int(used_sequences),
        "skipped_sequences": int(skipped_sequences),
        "window_count": int(speed_array.size),
        **summarize_values(speed_array),
        "per_sequence": {
            "sequence_count": int(sequence_speed_array.size),
            **summarize_values(sequence_speed_array),
        },
        "outputs": {
            "csv": str(csv_path),
            "npy": str(npy_path),
            "plot": str(plot_path),
            "sequence_csv": str(sequence_csv_path),
            "sequence_npy": str(sequence_npy_path),
            "sequence_plot": str(sequence_plot_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps({**summary, "summary_json": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
