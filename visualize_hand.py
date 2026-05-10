import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D

from evaluate_hand import build_model, load_checkpoint
from hand_dataset import GigaHandDataset
from torch_ode2vae_hand import build_mano_right_layer, device


HAND_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGERTIPS: Tuple[int, ...] = (4, 8, 12, 16, 20)
TIP_NAMES: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
TIP_COLORS: Tuple[str, ...] = ("#d04a35", "#1f77b4", "#2a9d8f", "#7b2cbf", "#f4a261")
SEQUENCE_COLORS: Tuple[str, ...] = (
    "#1f3b73",
    "#2a9d8f",
    "#d04a35",
    "#f77f00",
    "#7b2cbf",
    "#4d908e",
    "#c1121f",
    "#577590",
)
AXIS_MODE_CHOICES: Tuple[str, ...] = (
    "xyz",
    "xzy_neg_y",
    "x_negz_y",
    "negx_zy",
    "yzx",
    "zxy",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_tensor_dict_to_device(batch: Dict[str, torch.Tensor], target_device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(target_device) if torch.is_tensor(value) else value for key, value in batch.items()}


def checkpoint_arg(checkpoint_args: object, key: str, default=None):
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key, default)
    return getattr(checkpoint_args, key, default)


def build_visualization_dataset(
    args: argparse.Namespace,
    dataset_root: str,
    checkpoint_args: object,
) -> GigaHandDataset:
    split_text_keys = {
        "train": "train_text_file",
        "val": "val_text_file",
        "test": "test_text_file",
        "all": "text_file",
    }
    text_file = args.test_text_file or checkpoint_arg(
        checkpoint_args,
        split_text_keys[args.split],
    )
    dataset_split = "all" if text_file is not None else args.split
    dataset = GigaHandDataset(
        dataset_root=dataset_root,
        split=dataset_split,
        text_file=text_file,
        random_mask=False,
        fps=float(checkpoint_arg(checkpoint_args, "fps", 30.0)),
        max_sequences=args.max_sequences,
        history_len=int(checkpoint_arg(checkpoint_args, "history_len", 10)),
        horizon=int(checkpoint_arg(checkpoint_args, "horizon", 5)),
        time_stride_aug_max=1,
        verbose=not args.quiet_dataset,
    )
    dataset.return_full_sequence = True
    dataset.records = dataset.all_records
    return dataset


def batchify_sample(sample: Dict[str, object]) -> Dict[str, object]:
    batch: Dict[str, object] = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.ndim >= 1:
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return batch


def select_window(
    batch: Dict[str, torch.Tensor],
    window_len: int,
    history_len: int,
    window_stride: int,
    start_frame: Optional[int],
    start_frame_id: Optional[int],
) -> Dict[str, torch.Tensor]:
    if window_stride <= 0:
        raise ValueError("window_stride must be positive.")
    total_len = int(batch["pose"].shape[1])
    span = 1 + (window_len - 1) * window_stride
    if total_len < span:
        raise RuntimeError(
            f"Sequence length {total_len} is shorter than required window span {span} "
            f"(window_len={window_len}, window_stride={window_stride})."
        )

    mask = batch["mask"][0]
    frame_ids = batch["frame_ids"][0]
    valid_starts: List[int] = []
    future_offsets = window_stride * torch.arange(history_len, window_len, device=mask.device)
    for start in range(0, total_len - span + 1):
        anchor = start + (history_len - 1) * window_stride
        if float(mask[anchor].item()) <= 0:
            continue
        if float(mask[start + future_offsets].sum().item()) <= 0:
            continue
        valid_starts.append(start)
    if not valid_starts:
        raise RuntimeError("Selected sequence does not contain a valid history+horizon window.")

    valid_start_set = set(valid_starts)
    valid_start_hint = f"Valid start examples: {valid_starts[:10]}"
    start = valid_starts[0]
    if start_frame is not None:
        if start_frame < 0 or start_frame + span > total_len:
            raise ValueError(
                f"--start-frame {start_frame} does not allow a full window span of {span} frames."
            )
        start = int(start_frame)
        if start not in valid_start_set:
            raise ValueError(
                f"--start-frame {start_frame} does not select a valid history+horizon window. "
                f"{valid_start_hint}"
            )
    if start_frame_id is not None:
        matches = torch.nonzero(frame_ids == int(start_frame_id), as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(
                f"--start-frame-id {start_frame_id} is not present in this sequence. "
                f"Valid frame_id range: [{int(frame_ids[0].item())}, {int(frame_ids[-1].item())}]"
            )
        start = int(matches[0].item())
        if start + span > total_len:
            raise ValueError(
                f"--start-frame-id {start_frame_id} does not allow a full window span of {span} frames."
            )
        if start not in valid_start_set:
            raise ValueError(
                f"--start-frame-id {start_frame_id} maps to start index {start}, "
                f"which is not a valid history+horizon window. {valid_start_hint}"
            )

    indices = start + window_stride * torch.arange(window_len, device=frame_ids.device)
    clipped: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == total_len:
            clipped[key] = value.index_select(1, indices)
        else:
            clipped[key] = value
    clipped["times"] = clipped["times"] - clipped["times"][:, :1]
    return clipped


def predict_outputs(
    model,
    clip_batch: Dict[str, torch.Tensor],
    mano_right,
    method: str,
    sample_initial: bool,
    sample_dynamics: bool,
    need_verts: bool,
) -> Dict[str, torch.Tensor]:
    history_len = model.history_len
    horizon = model.horizon
    targets, stats = model._prepare_forward_context(
        batch=clip_batch,
        mano_right=mano_right,
        history_len=history_len,
        horizon=horizon,
        need_verts=need_verts,
    )
    init_state = model._sample_initial_state(stats, sample=sample_initial)
    rollout = model._rollout(
        init_state,
        future_dt=targets["future_dt"],
        method=method,
        sample_dynamics=sample_dynamics,
    )
    outputs = model._compute_losses(
        init_state=init_state,
        stats=stats,
        rollout=rollout,
        targets=targets,
        mano_right=mano_right,
        history_len=history_len,
        horizon=horizon,
        future_discount=1.0,
        need_verts=need_verts,
    )
    outputs["targets"] = targets
    return outputs


def split_pred_motion(model, pred_motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = pred_motion[..., :6]
    pose = pred_motion[..., 6 : 6 + model.pose_dim]
    trans = pred_motion[..., 6 + model.pose_dim : 6 + model.pose_dim + 3]
    return pose, root, trans


def pred_vertices_from_motion(model, mano_right, pred_motion: torch.Tensor, anchor_shape: torch.Tensor) -> torch.Tensor:
    pose, root, trans = split_pred_motion(model, pred_motion)
    shape = anchor_shape.unsqueeze(1).expand(-1, pred_motion.shape[1], -1)
    _, verts = model._mano_forward(pose, root, trans, shape, mano_right, return_verts=True)
    if verts is None:
        raise RuntimeError("Expected predicted vertices, got None.")
    return verts


def combine_history_future(history: torch.Tensor, future: torch.Tensor, history_len: int) -> torch.Tensor:
    return torch.cat([history[:, :history_len], future], dim=1)


def remap_axes_array(joints: np.ndarray, mode: str) -> np.ndarray:
    if mode == "xyz":
        return joints.copy()
    if mode == "xzy_neg_y":
        return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)
    if mode == "x_negz_y":
        return np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)
    if mode == "negx_zy":
        return np.stack([-joints[..., 0], joints[..., 2], joints[..., 1]], axis=-1)
    if mode == "yzx":
        return np.stack([joints[..., 1], joints[..., 2], joints[..., 0]], axis=-1)
    if mode == "zxy":
        return np.stack([joints[..., 2], joints[..., 0], joints[..., 1]], axis=-1)
    raise ValueError(f"Unsupported axis mode: {mode}")


def remap_axes_tensor(joints: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "xyz":
        return joints.clone()
    if mode == "xzy_neg_y":
        return torch.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], dim=-1)
    if mode == "x_negz_y":
        return torch.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], dim=-1)
    if mode == "negx_zy":
        return torch.stack([-joints[..., 0], joints[..., 2], joints[..., 1]], dim=-1)
    if mode == "yzx":
        return torch.stack([joints[..., 1], joints[..., 2], joints[..., 0]], dim=-1)
    if mode == "zxy":
        return torch.stack([joints[..., 2], joints[..., 0], joints[..., 1]], dim=-1)
    raise ValueError(f"Unsupported axis mode: {mode}")


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_hand(ax, joints: np.ndarray, color: str, title: str) -> None:
    for parent, child in HAND_BONES:
        seg = joints[[parent, child]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=2.0, alpha=0.95)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=10, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=22, azim=-58)


def compute_speed_curve(joints: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    dt = times[1:] - times[:-1]
    valid = (mask[1:] > 0) & (mask[:-1] > 0) & (dt > 1e-8)
    speed = torch.full((joints.shape[0] - 1,), float("nan"), device=joints.device)
    if torch.any(valid):
        vel = (joints[1:] - joints[:-1]) / dt[:, None, None]
        speed_vals = torch.linalg.norm(vel, dim=-1).mean(dim=-1)
        speed[valid] = speed_vals[valid]
    return times[1:].detach().cpu().numpy(), speed.detach().cpu().numpy()


def summarize_sample_errors(
    sample_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = torch.linalg.norm(sample_joints - gt_joints.unsqueeze(0), dim=-1).mean(dim=-1)
    errors[:, ~valid_mask] = float("nan")
    mean_err = torch.nanmean(errors, dim=0)
    p10 = torch.nanquantile(errors, q=0.10, dim=0)
    p90 = torch.nanquantile(errors, q=0.90, dim=0)
    return mean_err.detach().cpu().numpy(), p10.detach().cpu().numpy(), p90.detach().cpu().numpy()


def compute_diversity_curve(sample_joints: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    sample_count, num_frames, _, _ = sample_joints.shape
    diversity = torch.full((num_frames,), float("nan"), device=sample_joints.device)
    if sample_count < 2:
        return diversity.detach().cpu().numpy()
    pairwise = []
    for i in range(sample_count):
        for j in range(i + 1, sample_count):
            pairwise.append(torch.linalg.norm(sample_joints[i] - sample_joints[j], dim=-1).mean(dim=-1))
    pairwise_tensor = torch.stack(pairwise, dim=0)
    diversity[valid_mask] = pairwise_tensor[:, valid_mask].mean(dim=0)
    return diversity.detach().cpu().numpy()


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max())


def summarize_trajectory_band(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmean(curves, axis=0),
        np.nanquantile(curves, 0.10, axis=0),
        np.nanquantile(curves, 0.90, axis=0),
    )


def save_metric_figure(
    future_times: np.ndarray,
    mean_error: np.ndarray,
    sample_mean_error: np.ndarray,
    sample_p10_error: np.ndarray,
    sample_p90_error: np.ndarray,
    diversity: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(future_times, mean_error * 1000.0, color="#1f3b73", linewidth=2.0, label="mean prediction")
    axes[0].plot(future_times, sample_mean_error * 1000.0, color="#d04a35", linewidth=2.0, label="sample mean")
    axes[0].fill_between(
        future_times,
        sample_p10_error * 1000.0,
        sample_p90_error * 1000.0,
        color="#f2b3a9",
        alpha=0.45,
        label="10%-90% range",
    )
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("Future Prediction Error")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(future_times, diversity * 1000.0, color="#2a9d8f", linewidth=2.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pairwise Dist. (mm)")
    axes[1].set_title("Sample Diversity")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_motion_figure(
    gt_speed_time: np.ndarray,
    gt_speed: np.ndarray,
    mean_speed: np.ndarray,
    sample_speed_curves: np.ndarray,
    gt_joints: np.ndarray,
    mean_joints: np.ndarray,
    sample_joints: np.ndarray,
    valid_frames: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    axes[0].plot(gt_speed_time, gt_speed, color="#1f3b73", linewidth=1.8, linestyle="--", label="gt speed")
    axes[0].plot(gt_speed_time, mean_speed, color="#2a9d8f", linewidth=2.0, label="mean speed")
    if sample_speed_curves.size > 0:
        sample_speed_mean, sample_speed_p10, sample_speed_p90 = summarize_trajectory_band(sample_speed_curves)
        axes[0].plot(gt_speed_time, sample_speed_mean, color="#d04a35", linewidth=2.1, label="sample mean speed")
        axes[0].fill_between(gt_speed_time, sample_speed_p10, sample_speed_p90, color="#f2b3a9", alpha=0.45)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Motion Plausibility: Speed Profile")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")

    gt_tip = gt_joints[valid_frames][:, FINGERTIPS]
    mean_tip = mean_joints[valid_frames][:, FINGERTIPS]
    for tip_idx, tip_name, color in zip(FINGERTIPS, TIP_NAMES, TIP_COLORS):
        fingertip_order = FINGERTIPS.index(tip_idx)
        axes[1].plot(
            gt_tip[:, fingertip_order, 0] * 1000.0,
            gt_tip[:, fingertip_order, 2] * 1000.0,
            color=color,
            linewidth=2.0,
            label=f"gt {tip_name}",
        )
        axes[1].plot(
            mean_tip[:, fingertip_order, 0] * 1000.0,
            mean_tip[:, fingertip_order, 2] * 1000.0,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
        )

    for sample_idx in range(min(3, sample_joints.shape[0])):
        sample_tip = sample_joints[sample_idx, valid_frames][:, FINGERTIPS]
        for fingertip_order, color in enumerate(TIP_COLORS):
            axes[1].plot(
                sample_tip[:, fingertip_order, 0] * 1000.0,
                sample_tip[:, fingertip_order, 2] * 1000.0,
                color=color,
                linewidth=1.0,
                alpha=0.25 + 0.2 * (sample_idx == 0),
            )

    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Z (mm)")
    axes[1].set_title("Fingertip Trajectories on X-Z Plane")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_sequence_animation(
    sequences: Sequence[Tuple[str, np.ndarray, str]],
    valid_frames: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    fps: int,
    trail_length: int,
) -> Path:
    if valid_frames.size == 0:
        raise RuntimeError("No valid frames available for animation.")
    if len(sequences) == 0:
        raise RuntimeError("No sequences available for animation.")

    num_sequences = len(sequences)
    num_cols = int(np.ceil(np.sqrt(num_sequences)))
    num_rows = int(np.ceil(num_sequences / num_cols))
    fig = plt.figure(figsize=(4.4 * num_cols, 4.0 * num_rows))

    all_points = np.concatenate([item[1][valid_frames].reshape(-1, 3) for item in sequences], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.6, 1e-3)

    bone_lines = []
    joint_scatters = []
    wrist_traces = []
    for idx, (name, _, color) in enumerate(sequences):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1, projection="3d")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.view_init(elev=20, azim=-58)
        ax.set_title(name, fontsize=11)
        lines = []
        for _ in HAND_BONES:
            line, = ax.plot([], [], [], color=color, linewidth=2.2, alpha=0.98)
            lines.append(line)
        scatter = ax.scatter([], [], [], color=color, s=12, alpha=0.98)
        trace, = ax.plot([], [], [], color=color, linewidth=1.3, alpha=0.35)
        bone_lines.append(lines)
        joint_scatters.append(scatter)
        wrist_traces.append(trace)

    legend_handles = [Line2D([0], [0], color=color, lw=2.5, label=name) for name, _, color in sequences]
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)), frameon=False)

    def update(frame_cursor: int):
        frame_idx = int(valid_frames[frame_cursor])
        time_value = float(times[frame_idx])
        trail_start_cursor = max(0, frame_cursor - trail_length + 1)
        trail_frames = valid_frames[trail_start_cursor : frame_cursor + 1]
        artists = []
        for seq_idx, (_, joints_seq, _) in enumerate(sequences):
            joints = joints_seq[frame_idx]
            for line, (parent, child) in zip(bone_lines[seq_idx], HAND_BONES):
                segment = joints[[parent, child]]
                line.set_data(segment[:, 0], segment[:, 1])
                line.set_3d_properties(segment[:, 2])
                artists.append(line)
            scatter = joint_scatters[seq_idx]
            scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
            artists.append(scatter)
            wrist_points = joints_seq[trail_frames, 0]
            trace = wrist_traces[seq_idx]
            trace.set_data(wrist_points[:, 0], wrist_points[:, 1])
            trace.set_3d_properties(wrist_points[:, 2])
            artists.append(trace)

        fig.suptitle(
            f"3D Hand Skeleton Sequence Comparison | frame={frame_idx} | t={time_value:.3f}s",
            fontsize=14,
            y=0.97,
        )
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(valid_frames),
        interval=max(int(1000 / max(fps, 1)), 1),
        blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=160)
    plt.close(fig)
    return output_path


def save_axis_mode_comparison(
    gt_joints: np.ndarray,
    mean_joints: np.ndarray,
    sample_joints: np.ndarray,
    valid_frames: np.ndarray,
    output_path: Path,
    axis_modes: Sequence[str],
) -> Path:
    if valid_frames.size == 0:
        raise RuntimeError("No valid frames available for axis comparison.")

    frame_idx = int(valid_frames[len(valid_frames) // 2])
    num_rows = len(axis_modes)
    num_cols = 3
    fig = plt.figure(figsize=(4.2 * num_cols, 4.0 * num_rows))

    for row, axis_mode in enumerate(axis_modes):
        gt_points = remap_axes_array(gt_joints[frame_idx], axis_mode)
        mean_points = remap_axes_array(mean_joints[frame_idx], axis_mode)
        sample_points = remap_axes_array(sample_joints[0, frame_idx], axis_mode)
        all_points = np.concatenate([gt_points, mean_points, sample_points], axis=0)
        entries = (
            ("Ground Truth", gt_points, "#1f3b73"),
            ("Mean", mean_points, "#2a9d8f"),
            ("Sample 1", sample_points, "#d04a35"),
        )
        for col, (title, points, color) in enumerate(entries):
            ax = fig.add_subplot(num_rows, num_cols, row * num_cols + col + 1, projection="3d")
            draw_hand(ax, points, color, f"{title} | {axis_mode}")
            set_equal_3d_axes(ax, all_points)

    fig.suptitle(f"Axis Mode Comparison at frame={frame_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_snapshot_grid(
    gt_joints: np.ndarray,
    mean_joints: np.ndarray,
    sample_joints: np.ndarray,
    valid_frames: np.ndarray,
    output_path: Path,
    visible_samples: int,
) -> Path:
    if valid_frames.size == 0:
        raise RuntimeError("No valid frames available for snapshot visualization.")

    num_cols = min(6, valid_frames.size)
    chosen_indices = np.linspace(0, valid_frames.size - 1, num=num_cols, dtype=int)
    frame_ids = valid_frames[chosen_indices]
    row_count = 2 + min(visible_samples, sample_joints.shape[0])
    fig = plt.figure(figsize=(2.8 * num_cols, 2.8 * row_count))

    all_points = [gt_joints[frame_ids].reshape(-1, 3), mean_joints[frame_ids].reshape(-1, 3)]
    for sample_idx in range(row_count - 2):
        all_points.append(sample_joints[sample_idx, frame_ids].reshape(-1, 3))
    all_points_np = np.concatenate(all_points, axis=0)

    for col, frame_idx in enumerate(frame_ids):
        ax = fig.add_subplot(row_count, num_cols, col + 1, projection="3d")
        draw_hand(ax, gt_joints[frame_idx], "#202c59", f"GT t={frame_idx}")
        set_equal_3d_axes(ax, all_points_np)
        ax = fig.add_subplot(row_count, num_cols, num_cols + col + 1, projection="3d")
        draw_hand(ax, mean_joints[frame_idx], "#2a9d8f", f"Mean t={frame_idx}")
        set_equal_3d_axes(ax, all_points_np)

    for row in range(2, row_count):
        for col, frame_idx in enumerate(frame_ids):
            ax = fig.add_subplot(row_count, num_cols, row * num_cols + col + 1, projection="3d")
            draw_hand(ax, sample_joints[row - 2, frame_idx], "#d04a35", f"Sample {row - 1} t={frame_idx}")
            set_equal_3d_axes(ax, all_points_np)

    fig.suptitle("Ground Truth vs Mean Prediction vs Random Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in vertices.tolist():
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for i, j, k in faces.tolist():
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def export_obj_sequences(
    output_dir: Path,
    frame_ids: np.ndarray,
    valid_frames: np.ndarray,
    faces: np.ndarray,
    mesh_sequences: Sequence[Tuple[str, np.ndarray]],
    frame_step: int,
) -> Dict[str, str]:
    export_paths: Dict[str, str] = {}
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if valid_frames.size == 0:
        return export_paths
    chosen_frames = valid_frames[::frame_step]
    if chosen_frames.size == 0 or chosen_frames[-1] != valid_frames[-1]:
        chosen_frames = np.unique(np.concatenate([chosen_frames, valid_frames[-1:]], axis=0))
    obj_root = output_dir / "objs"
    obj_root.mkdir(parents=True, exist_ok=True)
    for name, vertices_seq in mesh_sequences:
        seq_dir = obj_root / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in chosen_frames.tolist():
            frame_id = int(frame_ids[frame_idx])
            write_obj(seq_dir / f"frame_{frame_idx:04d}_fid_{frame_id:06d}.obj", vertices_seq[frame_idx], faces)
        export_paths[name] = str(seq_dir)
    return export_paths


def evaluate_sequence(args: argparse.Namespace) -> Dict[str, object]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint["args"]
    dataset_root = args.dataset_root or checkpoint_arg(checkpoint_args, "dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root is missing. Please pass it explicitly or keep it in the checkpoint args.")

    dataset = build_visualization_dataset(args, dataset_root, checkpoint_args)
    if args.sequence_index < 0 or args.sequence_index >= len(dataset):
        raise IndexError(f"--sequence-index must be in [0, {len(dataset) - 1}], got {args.sequence_index}")

    model = build_model(dataset, checkpoint_args, checkpoint)
    mano_right = build_mano_right_layer(checkpoint_arg(checkpoint_args, "mano_model_path"))
    method = args.method or checkpoint_arg(checkpoint_args, "method", "rk4")
    history_len = model.history_len
    horizon = model.horizon
    window_len = history_len + horizon

    batch = batchify_sample(dataset[args.sequence_index])
    full_frame_ids = batch["frame_ids"][0].detach().cpu().numpy()
    batch = select_window(
        batch,
        window_len=window_len,
        history_len=history_len,
        window_stride=args.window_stride,
        start_frame=args.start_frame,
        start_frame_id=args.start_frame_id,
    )
    sample_name = batch["sample_name"]
    frame_ids = batch["frame_ids"][0].detach().cpu().numpy()
    window_start_index = int(np.where(full_frame_ids == frame_ids[0])[0][0])
    clip_batch = move_tensor_dict_to_device(
        {
            "motion": batch["motion"],
            "mask": batch["mask"],
            "times": batch["times"],
            "pose": batch["pose"],
            "Rh": batch["Rh"],
            "Th": batch["Th"],
            "shape": batch["shape"],
        },
        device,
    )

    with torch.no_grad():
        mean_outputs = predict_outputs(
            model,
            clip_batch,
            mano_right,
            method=method,
            sample_initial=False,
            sample_dynamics=False,
            need_verts=args.export_obj,
        )
        targets = mean_outputs["targets"]
        gt_joints = targets["joints"]
        gt_vertices = targets["verts"]
        mean_future_joints = mean_outputs["pred_joints_future"]
        mean_joints = combine_history_future(gt_joints, mean_future_joints, history_len)
        sample_joints_list: List[torch.Tensor] = []
        sample_vertices_list: List[torch.Tensor] = []
        sample_pred_motion_list: List[torch.Tensor] = []
        for _ in range(args.sample_count):
            sample_outputs = predict_outputs(
                model,
                clip_batch,
                mano_right,
                method=method,
                sample_initial=args.sample_z0,
                sample_dynamics=True,
                need_verts=args.export_obj,
            )
            sample_joints_list.append(combine_history_future(gt_joints, sample_outputs["pred_joints_future"], history_len)[0])
            sample_pred_motion_list.append(sample_outputs["pred_motion"][0])
            if args.export_obj:
                sample_vertices = pred_vertices_from_motion(
                    model,
                    mano_right,
                    sample_outputs["pred_motion"],
                    targets["anchor_shape"],
                )
                sample_vertices_list.append(combine_history_future(gt_vertices, sample_vertices, history_len)[0])

        sample_joints = torch.stack(sample_joints_list, dim=0)
        sample_pred_motion = torch.stack(sample_pred_motion_list, dim=0)
        if args.export_obj:
            mean_future_vertices = pred_vertices_from_motion(
                model,
                mano_right,
                mean_outputs["pred_motion"],
                targets["anchor_shape"],
            )
            mean_vertices = combine_history_future(gt_vertices, mean_future_vertices, history_len)
            sample_vertices = torch.stack(sample_vertices_list, dim=0)
        else:
            mean_vertices = None
            sample_vertices = None

    mask = clip_batch["mask"][0]
    times = clip_batch["times"][0]
    valid_mask = mask > 0
    future_valid_mask = mask[history_len : history_len + horizon] > 0
    valid_frames = torch.nonzero(valid_mask, as_tuple=False).flatten().detach().cpu().numpy()
    future_times = times[history_len : history_len + horizon]

    gt_joints_seq = gt_joints[0]
    mean_joints_seq = mean_joints[0]
    sample_future_joints = sample_joints[:, history_len : history_len + horizon]
    gt_future_joints = gt_joints_seq[history_len : history_len + horizon]
    mean_future_error = torch.linalg.norm(mean_future_joints[0] - gt_future_joints, dim=-1).mean(dim=-1)
    mean_future_error[~future_valid_mask] = float("nan")
    sample_mean_error, sample_p10_error, sample_p90_error = summarize_sample_errors(
        sample_future_joints,
        gt_future_joints,
        future_valid_mask,
    )
    diversity_curve = compute_diversity_curve(sample_future_joints, future_valid_mask)

    gt_speed_time, gt_speed = compute_speed_curve(gt_joints_seq, mask, times)
    _, mean_speed = compute_speed_curve(mean_joints_seq, mask, times)
    sample_speed_curves = []
    for sample_idx in range(sample_joints.shape[0]):
        _, sample_speed = compute_speed_curve(sample_joints[sample_idx], mask, times)
        sample_speed_curves.append(sample_speed)
    sample_speed_curves_np = np.stack(sample_speed_curves, axis=0) if sample_speed_curves else np.empty((0, 0))

    gt_joints_vis = remap_axes_tensor(gt_joints_seq, args.axis_mode)
    mean_joints_vis = remap_axes_tensor(mean_joints_seq, args.axis_mode)
    sample_joints_vis = remap_axes_tensor(sample_joints, args.axis_mode)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_png = save_metric_figure(
        future_times.detach().cpu().numpy(),
        mean_future_error.detach().cpu().numpy(),
        sample_mean_error,
        sample_p10_error,
        sample_p90_error,
        diversity_curve,
        output_dir / "sample_metrics.png",
    )
    motion_png = save_motion_figure(
        gt_speed_time,
        gt_speed,
        mean_speed,
        sample_speed_curves_np,
        gt_joints_vis.detach().cpu().numpy(),
        mean_joints_vis.detach().cpu().numpy(),
        sample_joints_vis.detach().cpu().numpy(),
        valid_frames,
        output_dir / "sample_motion.png",
    )
    snapshot_png = save_snapshot_grid(
        gt_joints_vis.detach().cpu().numpy(),
        mean_joints_vis.detach().cpu().numpy(),
        sample_joints_vis.detach().cpu().numpy(),
        valid_frames,
        output_dir / "sample_snapshots.png",
        visible_samples=args.visible_samples,
    )
    animation_sequences: List[Tuple[str, np.ndarray, str]] = [
        ("Ground Truth", gt_joints_vis.detach().cpu().numpy(), SEQUENCE_COLORS[0]),
        ("Mean", mean_joints_vis.detach().cpu().numpy(), SEQUENCE_COLORS[1]),
    ]
    for sample_idx in range(min(args.visible_samples, sample_joints_vis.shape[0])):
        animation_sequences.append(
            (
                f"Sample {sample_idx + 1}",
                sample_joints_vis[sample_idx].detach().cpu().numpy(),
                SEQUENCE_COLORS[(sample_idx + 2) % len(SEQUENCE_COLORS)],
            )
        )
    sequence_gif = save_sequence_animation(
        animation_sequences,
        valid_frames,
        times.detach().cpu().numpy(),
        output_dir / "skeleton_sequences.gif",
        fps=args.gif_fps,
        trail_length=args.trail_length,
    )
    axis_compare_png = save_axis_mode_comparison(
        gt_joints_seq.detach().cpu().numpy(),
        mean_joints_seq.detach().cpu().numpy(),
        sample_joints.detach().cpu().numpy(),
        valid_frames,
        output_dir / "axis_mode_comparison.png",
        axis_modes=args.compare_axis_modes,
    )

    obj_dirs: Dict[str, str] = {}
    if args.export_obj:
        mesh_sequences: List[Tuple[str, np.ndarray]] = [
            ("gt", gt_vertices[0].detach().cpu().numpy()),
            ("mean", mean_vertices[0].detach().cpu().numpy()),
        ]
        max_sample_exports = min(args.obj_max_samples, sample_vertices.shape[0])
        for sample_idx in range(max_sample_exports):
            mesh_sequences.append((f"sample_{sample_idx + 1:02d}", sample_vertices[sample_idx].detach().cpu().numpy()))
        obj_dirs = export_obj_sequences(
            output_dir,
            frame_ids,
            valid_frames,
            mano_right.faces_tensor.detach().cpu().numpy(),
            mesh_sequences,
            frame_step=args.obj_frame_step,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "sample_name": sample_name,
        "sequence_index": args.sequence_index,
        "history_len": history_len,
        "horizon": horizon,
        "window_len": window_len,
        "window_stride": args.window_stride,
        "window_start_index": window_start_index,
        "valid_frame_count": int(valid_mask.sum().item()),
        "future_valid_frame_count": int(future_valid_mask.sum().item()),
        "frame_id_start": int(frame_ids[0]),
        "frame_id_end": int(frame_ids[-1]),
        "sample_count": args.sample_count,
        "method": method,
        "sample_z0": bool(args.sample_z0),
        "axis_mode": args.axis_mode,
        "compare_axis_modes": list(args.compare_axis_modes),
        "mean_curve_mpjpe_mm": finite_mean(mean_future_error.detach().cpu().numpy() * 1000.0),
        "sample_curve_mpjpe_mm": finite_mean(sample_mean_error * 1000.0),
        "sample_curve_mpjpe_mm_max": finite_max(sample_mean_error * 1000.0),
        "diversity_mm": finite_mean(diversity_curve * 1000.0),
        "diversity_mm_max": finite_max(diversity_curve * 1000.0),
        "gt_speed_mps": finite_mean(gt_speed),
        "mean_speed_mps": finite_mean(mean_speed),
        "sample_speed_mps": finite_mean(np.nanmean(sample_speed_curves_np, axis=0)) if sample_speed_curves_np.size > 0 else float("nan"),
        "sample_pred_motion_change": finite_mean(
            torch.linalg.norm(sample_pred_motion - sample_pred_motion[:, :1], dim=-1).detach().cpu().numpy()
        ),
        "outputs": {
            "axis_compare_png": str(axis_compare_png),
            "sequence_gif": str(sequence_gif),
            "snapshot_png": str(snapshot_png),
            "metric_png": str(metric_png),
            "motion_png": str(motion_png),
            "obj_dirs": obj_dirs,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {**summary, "summary_json": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ODE2VAE hand future predictions by comparing mean rollout and random BNN samples."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ode2vae_hand_bnn_elbo/best_model.pt")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-text-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/ode2vae_hand_bnn_elbo/visualize")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--quiet-dataset", action="store_true", help="Disable dataset loading progress output.")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None, help="Start from this in-sequence frame index.")
    parser.add_argument("--start-frame-id", type=int, default=None, help="Start from this original dataset frame_id.")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Select every Nth frame inside the visualization window.",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--visible-samples", type=int, default=4)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--trail-length", type=int, default=8)
    parser.add_argument("--axis-mode", type=str, default="xyz", choices=AXIS_MODE_CHOICES)
    parser.add_argument(
        "--compare-axis-modes",
        type=str,
        nargs="+",
        default=("xyz", "xzy_neg_y", "x_negz_y", "negx_zy"),
        help="Axis modes to render in the axis comparison figure.",
    )
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Kept for CLI compatibility; visualization loads samples in-process.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-z0", action="store_true", help="Sample the latent initial state in addition to BNN weights.")
    parser.add_argument("--export-obj", action="store_true", help="Export MANO meshes as OBJ sequences for inspection.")
    parser.add_argument("--obj-frame-step", type=int, default=10, help="Export one OBJ every N valid frames.")
    parser.add_argument("--obj-max-samples", type=int, default=3, help="Maximum number of sampled sequences to export as OBJ.")
    args = parser.parse_args()

    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.visible_samples <= 0:
        raise ValueError("--visible-samples must be positive")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive")
    if args.trail_length <= 0:
        raise ValueError("--trail-length must be positive")
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive")
    if args.max_sequences is not None and args.max_sequences <= 0:
        raise ValueError("--max-sequences must be positive when provided")
    if args.start_frame is not None and args.start_frame_id is not None:
        raise ValueError("Please specify only one of --start-frame or --start-frame-id")
    if args.obj_frame_step <= 0:
        raise ValueError("--obj-frame-step must be positive")
    if args.obj_max_samples <= 0:
        raise ValueError("--obj-max-samples must be positive")
    invalid_modes = [mode for mode in args.compare_axis_modes if mode not in AXIS_MODE_CHOICES]
    if invalid_modes:
        raise ValueError(f"Unsupported axis modes in --compare-axis-modes: {invalid_modes}")

    set_seed(args.seed)
    result = evaluate_sequence(args)
    print(f"sample_name: {result['sample_name']}")
    print(f"sequence_index: {result['sequence_index']}")
    print(f"history_len: {result['history_len']}")
    print(f"horizon: {result['horizon']}")
    print(f"window_len: {result['window_len']}")
    print(f"window_stride: {result['window_stride']}")
    print(f"valid_frame_count: {result['valid_frame_count']}")
    print(f"future_valid_frame_count: {result['future_valid_frame_count']}")
    print(f"sample_count: {result['sample_count']}")
    print(f"axis_mode: {result['axis_mode']}")
    print(f"mean_curve_mpjpe_mm: {result['mean_curve_mpjpe_mm']:.3f}")
    print(f"sample_curve_mpjpe_mm: {result['sample_curve_mpjpe_mm']:.3f}")
    print(f"sample_curve_mpjpe_mm_max: {result['sample_curve_mpjpe_mm_max']:.3f}")
    print(f"diversity_mm: {result['diversity_mm']:.3f}")
    print(f"diversity_mm_max: {result['diversity_mm_max']:.3f}")
    print(f"gt_speed_mps: {result['gt_speed_mps']:.4f}")
    print(f"mean_speed_mps: {result['mean_speed_mps']:.4f}")
    print(f"sample_speed_mps: {result['sample_speed_mps']:.4f}")
    print(f"axis_compare_png: {result['outputs']['axis_compare_png']}")
    print(f"sequence_gif: {result['outputs']['sequence_gif']}")
    print(f"snapshot_png: {result['outputs']['snapshot_png']}")
    print(f"metric_png: {result['outputs']['metric_png']}")
    print(f"motion_png: {result['outputs']['motion_png']}")
    if result["outputs"]["obj_dirs"]:
        for name, path in result["outputs"]["obj_dirs"].items():
            print(f"obj_dir_{name}: {path}")
    print(f"summary_json: {result['summary_json']}")


if __name__ == "__main__":
    main()
