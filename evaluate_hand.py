import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from hand_dataset import GigaHandDataset
from torch_ode2vae_hand import ODE2VAEHand, build_mano_right_layer, device


def load_checkpoint(checkpoint_path: Path) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")
    if "args" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain training args: {checkpoint_path}")
    return checkpoint


def build_test_loader(
    dataset_root: str,
    checkpoint_args: Dict,
    num_workers: int,
    text_file: Optional[str] = None,
) -> Tuple[GigaHandDataset, data.DataLoader]:
    dataset_split = "all" if text_file is not None else "test"
    dataset = GigaHandDataset(
        dataset_root=dataset_root,
        split=dataset_split,
        text_file=text_file,
        random_mask=False,
        fps=float(checkpoint_args.get("fps", 30.0)),
        history_len=int(checkpoint_args.get("history_len", 10)),
        horizon=int(checkpoint_args.get("horizon", 5)),
    )
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return dataset, loader


def build_model(dataset: GigaHandDataset, checkpoint_args: Dict, checkpoint: Dict) -> ODE2VAEHand:
    model = ODE2VAEHand(
        input_dim=dataset.motion_dim,
        q=int(checkpoint_args["q"]),
        hidden_dim=int(checkpoint_args["hidden_dim"]),
        history_len=int(checkpoint_args.get("history_len", 10)),
        horizon=int(checkpoint_args.get("horizon", 5)),
        dynamics_damping=float(checkpoint_args.get("dynamics_damping", 0.05)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_all_subsequences(
    batch: Dict[str, torch.Tensor],
    subseq_len: int,
    stride: int,
    anchor_offset: int,
) -> Optional[Dict[str, torch.Tensor]]:
    total_len = int(batch["pose"].shape[1])
    if total_len < subseq_len:
        return None

    starts: List[int] = []
    for start in range(0, total_len - subseq_len + 1, stride):
        anchor = start + anchor_offset
        if float(batch["mask"][0, anchor].item()) <= 0:
            continue
        if float(batch["mask"][0, anchor + 1 : start + subseq_len].sum().item()) <= 0:
            continue
        starts.append(start)
    if not starts:
        return None

    stacked: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == total_len:
            windows = [value[:, start : start + subseq_len] for start in starts]
            stacked[key] = torch.cat(windows, dim=0)
        else:
            stacked[key] = value
    stacked["times"] = stacked["times"] - stacked["times"][:, :1]
    return stacked


def move_tensor_dict_to_device(batch: Dict[str, torch.Tensor], target_device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(target_device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def accumulate_curve(
    errors_sum: List[float],
    counts: List[int],
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    joint_err = torch.linalg.norm(pred_joints - gt_joints, dim=-1).mean(dim=-1)
    valid_mask = mask > 0
    max_frames = valid_mask.shape[1]
    while len(errors_sum) < max_frames:
        errors_sum.append(0.0)
        counts.append(0)
    for frame_idx in range(max_frames):
        frame_valid = valid_mask[:, frame_idx]
        if not torch.any(frame_valid):
            continue
        frame_err = joint_err[frame_valid, frame_idx]
        relative_idx = frame_idx
        errors_sum[relative_idx] += float(frame_err.sum().item())
        counts[relative_idx] += int(frame_valid.sum().item())


def accumulate_dynamics_curves(
    pred_joints: torch.Tensor,
    mask: torch.Tensor,
    times: torch.Tensor,
    speed_sums: List[float],
    speed_counts: List[int],
    acc_sums: List[float],
    acc_counts: List[int],
    jerk_sums: List[float],
    jerk_counts: List[int],
) -> None:
    if pred_joints.shape[1] < 2:
        return
    dt = times[:, 1:] - times[:, :-1]
    valid_vel = (mask[:, 1:] > 0) & (mask[:, :-1] > 0) & (dt > 1e-8)
    if torch.any(valid_vel):
        vel_dt = dt[:, :, None, None]
        pred_vel = (pred_joints[:, 1:] - pred_joints[:, :-1]) / vel_dt
        pred_speed = torch.linalg.norm(pred_vel, dim=-1).mean(dim=-1)
        max_speed_frames = pred_speed.shape[1]
        while len(speed_sums) < max_speed_frames:
            speed_sums.append(0.0)
            speed_counts.append(0)
        for frame_idx in range(max_speed_frames):
            frame_valid = valid_vel[:, frame_idx]
            if not torch.any(frame_valid):
                continue
            frame_speed = pred_speed[frame_valid, frame_idx]
            speed_sums[frame_idx] += float(frame_speed.sum().item())
            speed_counts[frame_idx] += int(frame_valid.sum().item())

        if pred_joints.shape[1] >= 3:
            valid_acc = valid_vel[:, 1:] & valid_vel[:, :-1]
            if torch.any(valid_acc):
                acc_dt = ((dt[:, 1:] + dt[:, :-1]) * 0.5)[:, :, None, None]
                pred_acc = (pred_vel[:, 1:] - pred_vel[:, :-1]) / acc_dt
                pred_acc_mag = torch.linalg.norm(pred_acc, dim=-1).mean(dim=-1)
                max_acc_frames = pred_acc_mag.shape[1]
                while len(acc_sums) < max_acc_frames:
                    acc_sums.append(0.0)
                    acc_counts.append(0)
                for frame_idx in range(max_acc_frames):
                    frame_valid = valid_acc[:, frame_idx]
                    if not torch.any(frame_valid):
                        continue
                    frame_acc = pred_acc_mag[frame_valid, frame_idx]
                    acc_sums[frame_idx] += float(frame_acc.sum().item())
                    acc_counts[frame_idx] += int(frame_valid.sum().item())

                if pred_joints.shape[1] >= 4:
                    valid_jerk = valid_acc[:, 1:] & valid_acc[:, :-1]
                    if torch.any(valid_jerk):
                        jerk_dt = ((acc_dt[:, 1:] + acc_dt[:, :-1]) * 0.5)
                        pred_jerk = (pred_acc[:, 1:] - pred_acc[:, :-1]) / jerk_dt
                        pred_jerk_mag = torch.linalg.norm(pred_jerk, dim=-1).mean(dim=-1)
                        max_jerk_frames = pred_jerk_mag.shape[1]
                        while len(jerk_sums) < max_jerk_frames:
                            jerk_sums.append(0.0)
                            jerk_counts.append(0)
                        for frame_idx in range(max_jerk_frames):
                            frame_valid = valid_jerk[:, frame_idx]
                            if not torch.any(frame_valid):
                                continue
                            frame_jerk = pred_jerk_mag[frame_valid, frame_idx]
                            jerk_sums[frame_idx] += float(frame_jerk.sum().item())
                            jerk_counts[frame_idx] += int(frame_valid.sum().item())


def summarize_curve(values: np.ndarray, prefix: str) -> Dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(data.mean()),
        f"{prefix}_p95": float(np.percentile(data, 95.0)),
        f"{prefix}_max": float(data.max()),
    }


def curve_pairs(sums: List[float], counts: List[int], start_frame: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [(idx + start_frame, sums[idx] / counts[idx]) for idx in range(len(sums)) if counts[idx] > 0]
    if not pairs:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)
    return (
        np.asarray([item[0] for item in pairs], dtype=np.int32),
        np.asarray([item[1] for item in pairs], dtype=np.float64),
    )


def save_curve(
    frame_numbers: np.ndarray,
    mpjpe_mm: np.ndarray,
    output_dir: Path,
    metadata: Dict,
    stem: str = "future_mpjpe",
    label: str = "Future MPJPE",
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "mpjpe_mm"])
        for frame, mpjpe in zip(frame_numbers.tolist(), mpjpe_mm.tolist()):
            writer.writerow([frame, mpjpe])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "frame_numbers": frame_numbers.tolist(),
                "mpjpe_mm": mpjpe_mm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plt.figure(figsize=(9, 5))
    plt.plot(frame_numbers, mpjpe_mm, color="#d04a35", linewidth=2.2)
    plt.scatter(frame_numbers, mpjpe_mm, color="#1f3b73", s=14)
    plt.xlabel("Prediction Step")
    plt.ylabel(f"{label} (mm)")
    plt.title(f"GigaHands Test Set: {label} Curve")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return csv_path, json_path, png_path


def save_scalar_curve(
    frame_numbers: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    ylabel: str,
    title: str,
) -> Path:
    plt.figure(figsize=(9, 5))
    if values.size > 0:
        plt.plot(frame_numbers, values, color="#2d6a4f", linewidth=2.2)
        plt.scatter(frame_numbers, values, color="#264653", s=14)
    plt.xlabel("Frame Number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_comparison_curve(
    frame_numbers: np.ndarray,
    model_values: np.ndarray,
    baseline_frame_numbers: np.ndarray,
    baseline_values: np.ndarray,
    output_path: Path,
    ylabel: str,
    title: str,
) -> Path:
    plt.figure(figsize=(9, 5))
    if model_values.size > 0:
        plt.plot(frame_numbers, model_values, color="#d04a35", linewidth=2.2, label="Model")
        plt.scatter(frame_numbers, model_values, color="#1f3b73", s=14)
    if baseline_values.size > 0:
        plt.plot(
            baseline_frame_numbers,
            baseline_values,
            color="#2d6a4f",
            linewidth=2.2,
            linestyle="--",
            label="Last-frame baseline",
        )
        plt.scatter(baseline_frame_numbers, baseline_values, color="#264653", s=14)
    plt.xlabel("Prediction Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def evaluate(args: argparse.Namespace) -> Dict:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint["args"]

    dataset_root = args.dataset_root or checkpoint_args.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root is missing. Please pass it explicitly or keep it in the checkpoint args.")

    test_text_file = args.test_text_file or checkpoint_args.get("test_text_file")
    dataset, loader = build_test_loader(dataset_root, checkpoint_args, args.num_workers, text_file=test_text_file)
    model = build_model(dataset, checkpoint_args, checkpoint)
    mano_right = build_mano_right_layer(checkpoint_args.get("mano_model_path"))
    method = args.method or checkpoint_args.get("method", "rk4")
    history_len = model.history_len
    horizon = model.horizon
    subseq_len = history_len + horizon

    errors_sum: List[float] = []
    counts: List[int] = []
    speed_sums: List[float] = []
    speed_counts: List[int] = []
    acc_sums: List[float] = []
    acc_counts: List[int] = []
    jerk_sums: List[float] = []
    jerk_counts: List[int] = []
    baseline_errors_sum: List[float] = []
    baseline_counts: List[int] = []
    baseline_speed_sums: List[float] = []
    baseline_speed_counts: List[int] = []
    baseline_acc_sums: List[float] = []
    baseline_acc_counts: List[int] = []
    baseline_jerk_sums: List[float] = []
    baseline_jerk_counts: List[int] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating", unit="seq")
        for batch in progress:
            subsequences = extract_all_subsequences(
                batch,
                subseq_len=subseq_len,
                stride=args.stride,
                anchor_offset=history_len - 1,
            )
            if subsequences is None:
                continue
            num_windows = int(subsequences["pose"].shape[0])
            for start in range(0, num_windows, args.eval_batch_size):
                end = min(start + args.eval_batch_size, num_windows)
                clip_batch = {
                    "motion": subsequences["motion"][start:end],
                    "mask": subsequences["mask"][start:end],
                    "times": subsequences["times"][start:end],
                    "pose": subsequences["pose"][start:end],
                    "Rh": subsequences["Rh"][start:end],
                    "Th": subsequences["Th"][start:end],
                    "shape": subsequences["shape"][start:end],
                }
                clip_batch = move_tensor_dict_to_device(clip_batch, device)
                outputs = model(
                    clip_batch,
                    mano_right=mano_right,
                    history_len=history_len,
                    horizon=horizon,
                    sample=False,
                    method=method,
                    future_discount=1.0,
                    need_verts=False,
                )
                if outputs is None:
                    continue
                pred_joints = outputs["pred_joints_future"]
                gt_joints = outputs["gt_joints_future"]
                future_mask = outputs["future_mask"]
                future_times = clip_batch["times"][:, history_len : history_len + horizon]
                targets = model._build_local_targets(
                    pose=clip_batch["pose"],
                    rh=clip_batch["Rh"],
                    th=clip_batch["Th"],
                    shape=clip_batch["shape"],
                    mask=clip_batch["mask"],
                    history_len=history_len,
                    mano_right=mano_right,
                    need_verts=False,
                )
                baseline_joints = targets["joints"][:, history_len - 1 : history_len]
                baseline_joints = baseline_joints.expand(-1, horizon, -1, -1)
                accumulate_curve(errors_sum, counts, pred_joints, gt_joints, future_mask)
                accumulate_curve(
                    baseline_errors_sum,
                    baseline_counts,
                    baseline_joints,
                    gt_joints,
                    future_mask,
                )
                accumulate_dynamics_curves(
                    pred_joints,
                    future_mask,
                    future_times,
                    speed_sums,
                    speed_counts,
                    acc_sums,
                    acc_counts,
                    jerk_sums,
                    jerk_counts,
                )
                accumulate_dynamics_curves(
                    baseline_joints,
                    future_mask,
                    future_times,
                    baseline_speed_sums,
                    baseline_speed_counts,
                    baseline_acc_sums,
                    baseline_acc_counts,
                    baseline_jerk_sums,
                    baseline_jerk_counts,
                )

    frame_numbers, mpjpe_m = curve_pairs(errors_sum, counts, start_frame=1)
    if frame_numbers.size == 0:
        raise RuntimeError("No valid frame statistics were accumulated from the test split.")

    mpjpe_mm = mpjpe_m * 1000.0
    baseline_frame_numbers, baseline_mpjpe_m = curve_pairs(baseline_errors_sum, baseline_counts, start_frame=1)
    baseline_mpjpe_mm = baseline_mpjpe_m * 1000.0
    speed_frame_numbers, speed_values = curve_pairs(speed_sums, speed_counts, start_frame=2)
    acc_frame_numbers, acc_values = curve_pairs(acc_sums, acc_counts, start_frame=3)
    jerk_frame_numbers, jerk_values = curve_pairs(jerk_sums, jerk_counts, start_frame=4)
    baseline_speed_frame_numbers, baseline_speed_values = curve_pairs(
        baseline_speed_sums,
        baseline_speed_counts,
        start_frame=2,
    )
    baseline_acc_frame_numbers, baseline_acc_values = curve_pairs(
        baseline_acc_sums,
        baseline_acc_counts,
        start_frame=3,
    )
    baseline_jerk_frame_numbers, baseline_jerk_values = curve_pairs(
        baseline_jerk_sums,
        baseline_jerk_counts,
        start_frame=4,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    dynamics_summary: Dict[str, float] = {}
    dynamics_summary.update(summarize_curve(speed_values, "speed"))
    dynamics_summary.update(summarize_curve(acc_values, "acc"))
    dynamics_summary.update(summarize_curve(jerk_values, "jerk"))
    baseline_dynamics_summary: Dict[str, float] = {}
    baseline_dynamics_summary.update(summarize_curve(baseline_speed_values, "speed"))
    baseline_dynamics_summary.update(summarize_curve(baseline_acc_values, "acc"))
    baseline_dynamics_summary.update(summarize_curve(baseline_jerk_values, "jerk"))
    metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "split": "test",
        "num_sequences": len(dataset),
        "method": method,
        "history_len": history_len,
        "horizon": horizon,
        "subseq_len": subseq_len,
        "stride": args.stride,
        "counts": [counts[idx] for idx in range(len(errors_sum)) if counts[idx] > 0],
        "dynamics": dynamics_summary,
        "baseline_counts": [
            baseline_counts[idx] for idx in range(len(baseline_errors_sum)) if baseline_counts[idx] > 0
        ],
        "baseline_dynamics": baseline_dynamics_summary,
        "baseline_frame_numbers": baseline_frame_numbers.tolist(),
        "baseline_mpjpe_mm": baseline_mpjpe_mm.tolist(),
        "speed_frame_numbers": speed_frame_numbers.tolist(),
        "speed_values": speed_values.tolist(),
        "acc_frame_numbers": acc_frame_numbers.tolist(),
        "acc_values": acc_values.tolist(),
        "jerk_frame_numbers": jerk_frame_numbers.tolist(),
        "jerk_values": jerk_values.tolist(),
        "baseline_speed_frame_numbers": baseline_speed_frame_numbers.tolist(),
        "baseline_speed_values": baseline_speed_values.tolist(),
        "baseline_acc_frame_numbers": baseline_acc_frame_numbers.tolist(),
        "baseline_acc_values": baseline_acc_values.tolist(),
        "baseline_jerk_frame_numbers": baseline_jerk_frame_numbers.tolist(),
        "baseline_jerk_values": baseline_jerk_values.tolist(),
    }
    csv_path, json_path, png_path = save_curve(frame_numbers, mpjpe_mm, output_dir, metadata)
    baseline_csv_path, baseline_json_path, baseline_png_path = save_curve(
        baseline_frame_numbers,
        baseline_mpjpe_mm,
        output_dir,
        metadata,
        stem="last_frame_baseline_mpjpe",
        label="Last-Frame Baseline MPJPE",
    )
    comparison_png_path = save_comparison_curve(
        frame_numbers,
        mpjpe_mm,
        baseline_frame_numbers,
        baseline_mpjpe_mm,
        output_dir / "future_mpjpe_comparison.png",
        "Future MPJPE (mm)",
        "GigaHands Test Set: Model vs Last-Frame Baseline",
    )
    speed_png_path = save_scalar_curve(
        speed_frame_numbers,
        speed_values,
        output_dir / "speed_magnitude_curve.png",
        "Speed Magnitude (m/s)",
        "GigaHands Test Set: Speed Magnitude Curve",
    )
    acc_png_path = save_scalar_curve(
        acc_frame_numbers,
        acc_values,
        output_dir / "acc_magnitude_curve.png",
        "Acceleration Magnitude (m/s^2)",
        "GigaHands Test Set: Acceleration Magnitude Curve",
    )
    jerk_png_path = save_scalar_curve(
        jerk_frame_numbers,
        jerk_values,
        output_dir / "jerk_magnitude_curve.png",
        "Jerk Magnitude (m/s^3)",
        "GigaHands Test Set: Jerk Magnitude Curve",
    )
    baseline_speed_png_path = save_scalar_curve(
        baseline_speed_frame_numbers,
        baseline_speed_values,
        output_dir / "last_frame_baseline_speed_magnitude_curve.png",
        "Speed Magnitude (m/s)",
        "GigaHands Test Set: Last-Frame Baseline Speed Magnitude Curve",
    )
    baseline_acc_png_path = save_scalar_curve(
        baseline_acc_frame_numbers,
        baseline_acc_values,
        output_dir / "last_frame_baseline_acc_magnitude_curve.png",
        "Acceleration Magnitude (m/s^2)",
        "GigaHands Test Set: Last-Frame Baseline Acceleration Magnitude Curve",
    )
    baseline_jerk_png_path = save_scalar_curve(
        baseline_jerk_frame_numbers,
        baseline_jerk_values,
        output_dir / "last_frame_baseline_jerk_magnitude_curve.png",
        "Jerk Magnitude (m/s^3)",
        "GigaHands Test Set: Last-Frame Baseline Jerk Magnitude Curve",
    )

    return {
        "frame_numbers": frame_numbers,
        "mpjpe_mm": mpjpe_mm,
        "baseline_frame_numbers": baseline_frame_numbers,
        "baseline_mpjpe_mm": baseline_mpjpe_mm,
        "dynamics": dynamics_summary,
        "baseline_dynamics": baseline_dynamics_summary,
        "speed_frame_numbers": speed_frame_numbers,
        "speed_values": speed_values,
        "acc_frame_numbers": acc_frame_numbers,
        "acc_values": acc_values,
        "jerk_frame_numbers": jerk_frame_numbers,
        "jerk_values": jerk_values,
        "baseline_speed_frame_numbers": baseline_speed_frame_numbers,
        "baseline_speed_values": baseline_speed_values,
        "baseline_acc_frame_numbers": baseline_acc_frame_numbers,
        "baseline_acc_values": baseline_acc_values,
        "baseline_jerk_frame_numbers": baseline_jerk_frame_numbers,
        "baseline_jerk_values": baseline_jerk_values,
        "csv_path": csv_path,
        "json_path": json_path,
        "png_path": png_path,
        "baseline_csv_path": baseline_csv_path,
        "baseline_json_path": baseline_json_path,
        "baseline_png_path": baseline_png_path,
        "comparison_png_path": comparison_png_path,
        "speed_png_path": speed_png_path,
        "acc_png_path": acc_png_path,
        "jerk_png_path": jerk_png_path,
        "baseline_speed_png_path": baseline_speed_png_path,
        "baseline_acc_png_path": baseline_acc_png_path,
        "baseline_jerk_png_path": baseline_jerk_png_path,
        "num_sequences": len(dataset),
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate future-step MPJPE on all history+horizon windows from the GigaHands test split."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ode2vae_hand_bnn_elbo/best_model.pt")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-text-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/ode2vae_hand_bnn_elbo/eval_test")
    parser.add_argument("--method", type=str, default=None, help="ODE solver used for mean reconstruction.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    result = evaluate(args)
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"num_sequences: {result['num_sequences']}")
    print(f"num_frames_in_curve: {len(result['frame_numbers'])}")
    print(f"frame_1_mpjpe_mm: {result['mpjpe_mm'][0]:.3f}")
    print(f"mean_mpjpe_mm: {float(np.mean(result['mpjpe_mm'])):.3f}")
    print(f"last_frame_mpjpe_mm: {result['mpjpe_mm'][-1]:.3f}")
    print(f"baseline_frame_1_mpjpe_mm: {result['baseline_mpjpe_mm'][0]:.3f}")
    print(f"baseline_mean_mpjpe_mm: {float(np.mean(result['baseline_mpjpe_mm'])):.3f}")
    print(f"baseline_last_frame_mpjpe_mm: {result['baseline_mpjpe_mm'][-1]:.3f}")
    print(
        f"speed_m_per_s: mean={result['dynamics']['speed_mean']:.4f}, "
        f"p95={result['dynamics']['speed_p95']:.4f}, max={result['dynamics']['speed_max']:.4f}"
    )
    print(
        f"acc_m_per_s2: mean={result['dynamics']['acc_mean']:.4f}, "
        f"p95={result['dynamics']['acc_p95']:.4f}, max={result['dynamics']['acc_max']:.4f}"
    )
    print(
        f"jerk_m_per_s3: mean={result['dynamics']['jerk_mean']:.4f}, "
        f"p95={result['dynamics']['jerk_p95']:.4f}, max={result['dynamics']['jerk_max']:.4f}"
    )
    print(f"csv: {result['csv_path']}")
    print(f"json: {result['json_path']}")
    print(f"png: {result['png_path']}")
    print(f"baseline_csv: {result['baseline_csv_path']}")
    print(f"baseline_json: {result['baseline_json_path']}")
    print(f"baseline_png: {result['baseline_png_path']}")
    print(f"comparison_png: {result['comparison_png_path']}")
    print(f"speed_png: {result['speed_png_path']}")
    print(f"acc_png: {result['acc_png_path']}")
    print(f"jerk_png: {result['jerk_png_path']}")
    print(f"baseline_speed_png: {result['baseline_speed_png_path']}")
    print(f"baseline_acc_png: {result['baseline_acc_png_path']}")
    print(f"baseline_jerk_png: {result['baseline_jerk_png_path']}")


if __name__ == "__main__":
    main()
