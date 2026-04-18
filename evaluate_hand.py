import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def build_test_loader(dataset_root: str, checkpoint_args: Dict, num_workers: int) -> Tuple[GigaHandDataset, data.DataLoader]:
    dataset = GigaHandDataset(
        dataset_root=dataset_root,
        split="test",
        normalize_trans=bool(checkpoint_args.get("normalize_trans", False)),
        use_global_rot=not bool(checkpoint_args.get("disable_global_rot", False)),
        random_mask=False,
        fps=float(checkpoint_args.get("fps", 30.0))
    )
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return dataset, loader


def build_model(dataset: GigaHandDataset, checkpoint_args: Dict, checkpoint: Dict) -> ODE2VAEHand:
    model = ODE2VAEHand(
        input_dim=dataset.motion_dim,
        q=int(checkpoint_args["q"]),
        hidden_dim=int(checkpoint_args["hidden_dim"]),
        n_init_obs=int(checkpoint_args["n_init_obs"]),
        use_global_rot=not bool(checkpoint_args.get("disable_global_rot", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruct_joints(
    model: ODE2VAEHand,
    mano_right,
    pred_motion: torch.Tensor,
    gt_pose: torch.Tensor,
    gt_rh: torch.Tensor,
    gt_th: torch.Tensor,
    gt_shape: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, t, _ = pred_motion.shape
    pred_pose, pred_rh, pred_th, pred_shape = model._motion_to_mano_params(
        pred_motion,
        gt_shape,
        gt_pose,
        gt_rh,
    )
    gt_pose_axis = model._rot6d_to_axis_angle(gt_pose.reshape(n * t, -1, 6)).view(n * t, -1, 3).reshape(n * t, -1)
    gt_rh_axis = model._rot6d_to_axis_angle(gt_rh.reshape(n * t, 6)).view(n * t, 3)
    gt_th_flat = gt_th.reshape(n * t, 3)
    gt_shape_flat = gt_shape.reshape(n * t, -1)

    pred_joints = mano_right(
        poses=pred_pose,
        shapes=pred_shape,
        Rh=pred_rh,
        Th=pred_th,
        return_verts=False,
        return_tensor=True,
    ).view(n, t, -1, 3)
    gt_joints = mano_right(
        poses=gt_pose_axis,
        shapes=gt_shape_flat,
        Rh=gt_rh_axis,
        Th=gt_th_flat,
        return_verts=False,
        return_tensor=True,
    ).view(n, t, -1, 3)
    return pred_joints, gt_joints


def first_valid_frame_indices(mask: torch.Tensor) -> torch.Tensor:
    valid_any = (mask > 0).any(dim=1)
    if not torch.all(valid_any):
        raise RuntimeError("Encountered a subsequence without any valid frames in the test set.")
    return torch.argmax((mask > 0).to(torch.int64), dim=1)


def to_first_frame_relative(joints: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first_valid = first_valid_frame_indices(mask)
    batch_indices = torch.arange(joints.shape[0], device=joints.device)
    anchors = joints[batch_indices, first_valid].unsqueeze(1)
    return joints - anchors


def extract_all_subsequences(
    batch: Dict[str, torch.Tensor],
    subseq_len: int,
    stride: int,
) -> Optional[Dict[str, torch.Tensor]]:
    total_len = int(batch["motion"].shape[1])
    if total_len < subseq_len:
        return None

    starts: List[int] = []
    for start in range(0, total_len - subseq_len + 1, stride):
        if float(batch["mask"][0, start].item()) <= 0:
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
    pred_rel_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    gt_rel_joints = to_first_frame_relative(gt_joints, mask)
    joint_err = torch.linalg.norm(pred_rel_joints - gt_rel_joints, dim=-1).mean(dim=-1)
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


def save_curve(
    frame_numbers: np.ndarray,
    mpjpe_mm: np.ndarray,
    output_dir: Path,
    metadata: Dict,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "first_frame_relative_mpjpe.csv"
    json_path = output_dir / "first_frame_relative_mpjpe.json"
    png_path = output_dir / "first_frame_relative_mpjpe.png"

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
    plt.xlabel("Frame Number")
    plt.ylabel("First-Frame-Relative MPJPE (mm)")
    plt.title("GigaHands Test Set: First-Frame-Relative MPJPE Curve")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return csv_path, json_path, png_path


def evaluate(args: argparse.Namespace) -> Dict:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint["args"]

    dataset_root = args.dataset_root or checkpoint_args.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root is missing. Please pass it explicitly or keep it in the checkpoint args.")

    dataset, loader = build_test_loader(dataset_root, checkpoint_args, args.num_workers)
    model = build_model(dataset, checkpoint_args, checkpoint)
    mano_right = build_mano_right_layer(checkpoint_args.get("mano_model_path"))
    method = args.method or checkpoint_args.get("method", "rk4")

    errors_sum: List[float] = []
    counts: List[int] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating", unit="seq")
        for batch in progress:
            subsequences = extract_all_subsequences(batch, args.subseq_len, args.stride)
            if subsequences is None:
                continue
            num_windows = int(subsequences["motion"].shape[0])
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
                motion = clip_batch["motion"]
                mask = clip_batch["mask"]
                times = clip_batch["times"]
                pose = clip_batch["pose"]
                rh = clip_batch["Rh"]
                th = clip_batch["Th"]
                shape = clip_batch["shape"]
                pred_motion, _ = model.mean_rec(
                    motion,
                    times,
                    mask=mask,
                    shape=shape,
                    pose=pose,
                    Rh=rh,
                    Th=th,
                    mano_right=mano_right,
                    method=method,
                )
                pred_joints, gt_joints = reconstruct_joints(model, mano_right, pred_motion, pose, rh, th, shape)
                pred_rel_joints = to_first_frame_relative(pred_joints, mask)
                accumulate_curve(errors_sum, counts, pred_rel_joints, gt_joints, mask)

    valid_pairs = [(idx + 1, errors_sum[idx] / counts[idx]) for idx in range(len(errors_sum)) if counts[idx] > 0]
    if not valid_pairs:
        raise RuntimeError("No valid frame statistics were accumulated from the test split.")

    frame_numbers = np.asarray([item[0] for item in valid_pairs], dtype=np.int32)
    mpjpe_m = np.asarray([item[1] for item in valid_pairs], dtype=np.float64)
    mpjpe_mm = mpjpe_m * 1000.0

    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "split": "test",
        "num_sequences": len(dataset),
        "method": method,
        "subseq_len": args.subseq_len,
        "stride": args.stride,
        "counts": [counts[idx] for idx in range(len(errors_sum)) if counts[idx] > 0],
    }
    csv_path, json_path, png_path = save_curve(frame_numbers, mpjpe_mm, output_dir, metadata)

    return {
        "frame_numbers": frame_numbers,
        "mpjpe_mm": mpjpe_mm,
        "csv_path": csv_path,
        "json_path": json_path,
        "png_path": png_path,
        "num_sequences": len(dataset),
        "checkpoint_path": checkpoint_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate first-frame-relative MPJPE on all length-N subsequences from the GigaHands test split."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ode2vae_hand4/best_model.pt")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/ode2vae_hand4/eval_test")
    parser.add_argument("--method", type=str, default=None, help="ODE solver used for mean reconstruction.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subseq-len", type=int, default=150)
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
    print(f"csv: {result['csv_path']}")
    print(f"json: {result['json_path']}")
    print(f"png: {result['png_path']}")


if __name__ == "__main__":
    main()
