import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.lines import Line2D
from torchdiffeq import odeint

from evaluate_hand import (
    build_model,
    build_test_loader,
    load_checkpoint,
    reconstruct_joints,
    to_first_frame_relative,
)
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
    return {
        key: value.to(target_device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def select_window(
    batch: Dict[str, torch.Tensor],
    window_len: Optional[int],
    start_frame: Optional[int],
    start_frame_id: Optional[int],
) -> Dict[str, torch.Tensor]:
    total_len = int(batch["motion"].shape[1])
    mask = batch["mask"][0]
    frame_ids = batch["frame_ids"][0]
    valid_starts = torch.nonzero(mask > 0, as_tuple=False).flatten()
    if valid_starts.numel() == 0:
        raise RuntimeError("Selected sequence does not contain any valid frames.")

    default_start = int(valid_starts[0].item())
    start = default_start
    if start_frame is not None:
        if start_frame < 0 or start_frame >= total_len:
            raise ValueError(f"--start-frame must be in [0, {total_len - 1}], got {start_frame}")
        start = int(start_frame)
    if start_frame_id is not None:
        matches = torch.nonzero(frame_ids == int(start_frame_id), as_tuple=False).flatten()
        if matches.numel() == 0:
            raise ValueError(
                f"--start-frame-id {start_frame_id} is not present in this sequence. "
                f"Valid frame_id range: [{int(frame_ids[0].item())}, {int(frame_ids[-1].item())}]"
            )
        start = int(matches[0].item())
    if window_len is None or window_len <= 0 or total_len <= window_len:
        end = total_len
    else:
        max_start = max(total_len - window_len, 0)
        start = min(start, max_start)
        end = start + window_len

    clipped: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == total_len:
            clipped[key] = value[:, start:end]
        else:
            clipped[key] = value
    clipped["times"] = clipped["times"] - clipped["times"][:, :1]
    return clipped


def sample_motion_sequences(
    model,
    motion: torch.Tensor,
    mask: torch.Tensor,
    times: torch.Tensor,
    sample_count: int,
    method: str,
    sample_z0: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, t, d = motion.shape
    times_1d = model._prepare_times(times, t, motion.device)
    frame_h = model.frame_encoder(motion.contiguous().view(n * t, d)).view(n, t, -1)
    qz0_m, qz0_logv = model._encode_initial_state(frame_h, motion, mask, times_1d)

    samples: List[torch.Tensor] = []
    latent_trajectories: List[torch.Tensor] = []
    for _ in range(sample_count):
        if sample_z0:
            eps = torch.randn_like(qz0_m)
            z0 = qz0_m + eps * torch.exp(qz0_logv)
        else:
            z0 = qz0_m

        f = model.bnn.draw_f(mean=False)

        def rhs(_, state):
            q = state.shape[1] // 2
            dv = f(state)
            ds = state[:, :q]
            return torch.cat([dv, ds], dim=1)

        zt = odeint(rhs, z0, times_1d, method=method).permute(1, 0, 2)
        st = zt[:, :, model.q :]
        sample_motion = model.decoder(st.contiguous().view(n * t, model.q)).view(n, t, d)
        samples.append(sample_motion[0])
        latent_trajectories.append(zt[0])

    return torch.stack(samples, dim=0), qz0_m, torch.stack(latent_trajectories, dim=0)


def compute_mean_latent_trajectory(
    model,
    motion: torch.Tensor,
    mask: torch.Tensor,
    times: torch.Tensor,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, t, d = motion.shape
    times_1d = model._prepare_times(times, t, motion.device)
    frame_h = model.frame_encoder(motion.contiguous().view(n * t, d)).view(n, t, -1)
    qz0_m, _ = model._encode_initial_state(frame_h, motion, mask, times_1d)

    def rhs(_, state):
        q = state.shape[1] // 2
        dv = model.bnn.draw_f(mean=True)(state)
        ds = state[:, :q]
        return torch.cat([dv, ds], dim=1)

    zt = odeint(rhs, qz0_m, times_1d, method=method).permute(1, 0, 2)
    return qz0_m, zt[0]


def compute_speed_curve(joints: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    rel_joints = to_first_frame_relative(joints.unsqueeze(0), mask.unsqueeze(0))[0]
    dt = times[1:] - times[:-1]
    valid = (mask[1:] > 0) & (mask[:-1] > 0) & (dt > 1e-8)
    speed = torch.full((rel_joints.shape[0] - 1,), float("nan"), device=rel_joints.device)
    if torch.any(valid):
        vel = (rel_joints[1:] - rel_joints[:-1]) / dt[:, None, None]
        speed_vals = torch.linalg.norm(vel, dim=-1).mean(dim=-1)
        speed[valid] = speed_vals[valid]
    return times[1:].detach().cpu().numpy(), speed.detach().cpu().numpy()


def rot6d_rowmajor_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    flat = rot6d.reshape(-1, 6)
    a1 = torch.stack([flat[:, 0], flat[:, 2], flat[:, 4]], dim=1)
    a2 = torch.stack([flat[:, 1], flat[:, 3], flat[:, 5]], dim=1)
    b1 = torch.nn.functional.normalize(a1, dim=1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    rotmat = torch.stack([b1, b2, b3], dim=-1)
    return rotmat.view(*rot6d.shape[:-1], 3, 3)


def split_motion_components(
    model,
    motion: torch.Tensor,
    pose_ref: torch.Tensor,
    rh_ref: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pose_dim = pose_ref.shape[-1]
    if model.use_global_rot:
        rh_6d = motion[..., :6]
        pose_6d = motion[..., 6 : 6 + pose_dim]
    else:
        rh_6d = rh_ref
        pose_6d = motion[..., :pose_dim]
    return pose_6d, rh_6d


def compute_pose_change_curve_from_pose6d(pose_6d: torch.Tensor) -> np.ndarray:
    num_joints = pose_6d.shape[-1] // 6
    pose_rot = rot6d_rowmajor_to_rotmat(pose_6d.view(pose_6d.shape[0], num_joints, 6))
    ref_rot = pose_rot[:1]
    rel_rot = torch.matmul(pose_rot, ref_rot.transpose(-1, -2))
    trace = rel_rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.rad2deg(torch.arccos(cos_theta))
    return theta.mean(dim=-1).detach().cpu().numpy()


def compute_local_joint_deformation_curve(joints: torch.Tensor, rh_6d: torch.Tensor) -> np.ndarray:
    wrist_centered = joints - joints[:, :1]
    rh_rot = rot6d_rowmajor_to_rotmat(rh_6d)
    local_joints = torch.matmul(wrist_centered, rh_rot)
    ref_local = local_joints[:1]
    local_delta = torch.linalg.norm(local_joints - ref_local, dim=-1)
    return local_delta[:, 1:].mean(dim=-1).detach().cpu().numpy()


def reconstruct_vertices(
    model,
    mano_right,
    pred_motion: torch.Tensor,
    gt_pose: torch.Tensor,
    gt_rh: torch.Tensor,
    gt_th: torch.Tensor,
    gt_shape: torch.Tensor,
) -> torch.Tensor:
    n, t, _ = pred_motion.shape
    pred_pose, pred_rh, pred_th, pred_shape = model._motion_to_mano_params(
        pred_motion,
        gt_shape,
        gt_pose,
        gt_rh,
    )
    pred_vertices = mano_right(
        poses=pred_pose,
        shapes=pred_shape,
        Rh=pred_rh,
        Th=pred_th,
        return_verts=True,
        return_tensor=True,
    ).view(n, t, -1, 3)
    return pred_vertices


def reconstruct_gt_vertices(
    model,
    mano_right,
    gt_pose: torch.Tensor,
    gt_rh: torch.Tensor,
    gt_th: torch.Tensor,
    gt_shape: torch.Tensor,
) -> torch.Tensor:
    n, t, _ = gt_pose.shape
    gt_pose_axis = model._rot6d_to_axis_angle(gt_pose.reshape(n * t, -1, 6)).view(n * t, -1, 3).reshape(n * t, -1)
    gt_rh_axis = model._rot6d_to_axis_angle(gt_rh.reshape(n * t, 6)).view(n * t, 3)
    gt_vertices = mano_right(
        poses=gt_pose_axis,
        shapes=gt_shape.reshape(n * t, -1),
        Rh=gt_rh_axis,
        Th=gt_th.reshape(n * t, 3),
        return_verts=True,
        return_tensor=True,
    ).view(n, t, -1, 3)
    return gt_vertices


def compute_diversity_curve(sample_rel_joints: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    sample_count, num_frames, _, _ = sample_rel_joints.shape
    diversity = torch.full((num_frames,), float("nan"), device=sample_rel_joints.device)
    if sample_count < 2:
        return diversity.detach().cpu().numpy()

    pairwise = []
    for i in range(sample_count):
        for j in range(i + 1, sample_count):
            dist = torch.linalg.norm(sample_rel_joints[i] - sample_rel_joints[j], dim=-1).mean(dim=-1)
            pairwise.append(dist)
    pairwise_tensor = torch.stack(pairwise, dim=0)
    diversity[valid_mask] = pairwise_tensor[:, valid_mask].mean(dim=0)
    return diversity.detach().cpu().numpy()


def summarize_sample_errors(
    sample_rel_joints: torch.Tensor,
    gt_rel_joints: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = torch.linalg.norm(sample_rel_joints - gt_rel_joints.unsqueeze(0), dim=-1).mean(dim=-1)
    errors[:, ~valid_mask] = float("nan")
    mean_err = torch.nanmean(errors, dim=0)
    p10 = torch.nanquantile(errors, q=0.10, dim=0)
    p90 = torch.nanquantile(errors, q=0.90, dim=0)
    return (
        mean_err.detach().cpu().numpy(),
        p10.detach().cpu().numpy(),
        p90.detach().cpu().numpy(),
    )


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

    axes = []
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
        axes.append(ax)
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
    gt_rel_joints: np.ndarray,
    mean_rel_joints: np.ndarray,
    sample_rel_joints: np.ndarray,
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
        gt_points = remap_axes_array(gt_rel_joints[frame_idx], axis_mode)
        mean_points = remap_axes_array(mean_rel_joints[frame_idx], axis_mode)
        sample_points = remap_axes_array(sample_rel_joints[0, frame_idx], axis_mode)
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


def save_snapshot_grid(
    gt_rel_joints: np.ndarray,
    sample_rel_joints: np.ndarray,
    valid_frames: np.ndarray,
    output_path: Path,
    visible_samples: int,
) -> Path:
    if valid_frames.size == 0:
        raise RuntimeError("No valid frames available for snapshot visualization.")

    num_cols = min(6, valid_frames.size)
    chosen_indices = np.linspace(0, valid_frames.size - 1, num=num_cols, dtype=int)
    frame_ids = valid_frames[chosen_indices]
    row_count = 1 + min(visible_samples, sample_rel_joints.shape[0])
    fig = plt.figure(figsize=(2.8 * num_cols, 2.8 * row_count))

    all_points = [gt_rel_joints[frame_ids].reshape(-1, 3)]
    for sample_idx in range(row_count - 1):
        all_points.append(sample_rel_joints[sample_idx, frame_ids].reshape(-1, 3))
    all_points_np = np.concatenate(all_points, axis=0)

    for col, frame_idx in enumerate(frame_ids):
        ax = fig.add_subplot(row_count, num_cols, col + 1, projection="3d")
        draw_hand(ax, gt_rel_joints[frame_idx], "#202c59", f"GT t={frame_idx}")
        set_equal_3d_axes(ax, all_points_np)

    for row in range(1, row_count):
        for col, frame_idx in enumerate(frame_ids):
            ax = fig.add_subplot(row_count, num_cols, row * num_cols + col + 1, projection="3d")
            draw_hand(ax, sample_rel_joints[row - 1, frame_idx], "#d04a35", f"Sample {row} t={frame_idx}")
            set_equal_3d_axes(ax, all_points_np)

    fig.suptitle("Ground Truth vs Random BNN Samples (First-Frame-Relative Joints)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_figure(
    times: np.ndarray,
    gt_error: np.ndarray,
    mean_error: np.ndarray,
    p10_error: np.ndarray,
    p90_error: np.ndarray,
    diversity: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(times, mean_error * 1000.0, color="#d04a35", linewidth=2.2, label="sample mean error")
    axes[0].fill_between(
        times,
        p10_error * 1000.0,
        p90_error * 1000.0,
        color="#f2b3a9",
        alpha=0.45,
        label="10%-90% range",
    )
    axes[0].plot(times, gt_error * 1000.0, color="#1f3b73", linewidth=1.8, linestyle="--", label="mean trajectory error")
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title("Reasonableness: Sample-to-GT Error Envelope")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(times, diversity * 1000.0, color="#2a9d8f", linewidth=2.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pairwise Dist. (mm)")
    axes[1].set_title("Diversity: Mean Pairwise Joint Distance Between Random Samples")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_motion_figure(
    gt_speed_time: np.ndarray,
    gt_speed: np.ndarray,
    sample_speed_curves: np.ndarray,
    gt_rel_joints: np.ndarray,
    sample_rel_joints: np.ndarray,
    valid_frames: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    if sample_speed_curves.size > 0:
        sample_speed_mean = np.nanmean(sample_speed_curves, axis=0)
        sample_speed_p10 = np.nanquantile(sample_speed_curves, 0.10, axis=0)
        sample_speed_p90 = np.nanquantile(sample_speed_curves, 0.90, axis=0)
        axes[0].plot(gt_speed_time, sample_speed_mean, color="#d04a35", linewidth=2.1, label="sample mean speed")
        axes[0].fill_between(
            gt_speed_time,
            sample_speed_p10,
            sample_speed_p90,
            color="#f2b3a9",
            alpha=0.45,
            label="10%-90% range",
        )
    axes[0].plot(gt_speed_time, gt_speed, color="#1f3b73", linewidth=1.8, linestyle="--", label="gt speed")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Motion Plausibility: Speed Profile")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")

    valid_frames = valid_frames[: max(len(valid_frames), 1)]
    gt_tip = gt_rel_joints[valid_frames][:, FINGERTIPS]
    for tip_idx, tip_name, color in zip(FINGERTIPS, TIP_NAMES, TIP_COLORS):
        fingertip_order = FINGERTIPS.index(tip_idx)
        axes[1].plot(
            gt_tip[:, fingertip_order, 0] * 1000.0,
            gt_tip[:, fingertip_order, 2] * 1000.0,
            color=color,
            linewidth=2.0,
            label=f"gt {tip_name}",
        )

    for sample_idx in range(min(3, sample_rel_joints.shape[0])):
        sample_tip = sample_rel_joints[sample_idx, valid_frames][:, FINGERTIPS]
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


def summarize_trajectory_band(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmean(curves, axis=0),
        np.nanquantile(curves, 0.10, axis=0),
        np.nanquantile(curves, 0.90, axis=0),
    )


def save_latent_diagnostic_figure(
    times: np.ndarray,
    mean_v_norm: np.ndarray,
    sample_v_curves: np.ndarray,
    mean_s_delta: np.ndarray,
    sample_s_curves: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    if sample_v_curves.size > 0:
        sample_v_mean, sample_v_p10, sample_v_p90 = summarize_trajectory_band(sample_v_curves)
        axes[0].plot(times, sample_v_mean, color="#d04a35", linewidth=2.2, label="sample mean")
        axes[0].fill_between(times, sample_v_p10, sample_v_p90, color="#f2b3a9", alpha=0.45, label="10%-90% range")
    axes[0].plot(times, mean_v_norm, color="#1f3b73", linewidth=1.8, linestyle="--", label="mean trajectory")
    axes[0].set_ylabel("||v(t)||")
    axes[0].set_title("Latent Velocity Magnitude")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")

    if sample_s_curves.size > 0:
        sample_s_mean, sample_s_p10, sample_s_p90 = summarize_trajectory_band(sample_s_curves)
        axes[1].plot(times, sample_s_mean, color="#2a9d8f", linewidth=2.2, label="sample mean")
        axes[1].fill_between(times, sample_s_p10, sample_s_p90, color="#b7e4dc", alpha=0.45, label="10%-90% range")
    axes[1].plot(times, mean_s_delta, color="#264653", linewidth=1.8, linestyle="--", label="mean trajectory")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("||s(t)-s(0)||")
    axes[1].set_title("Static Latent Drift from First Frame")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_motion_change_figure(
    times: np.ndarray,
    mean_motion_change: np.ndarray,
    sample_motion_change_curves: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.8))

    if sample_motion_change_curves.size > 0:
        sample_mean, sample_p10, sample_p90 = summarize_trajectory_band(sample_motion_change_curves)
        ax.plot(times, sample_mean, color="#d04a35", linewidth=2.2, label="sample mean")
        ax.fill_between(times, sample_p10, sample_p90, color="#f2b3a9", alpha=0.45, label="10%-90% range")
    ax.plot(times, mean_motion_change, color="#1f3b73", linewidth=1.8, linestyle="--", label="mean trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||x(t)-x(0)||")
    ax.set_title("Predicted Motion Change Relative to First Frame")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_articulation_diagnostic_figure(
    times: np.ndarray,
    gt_pose_change: np.ndarray,
    mean_pose_change: np.ndarray,
    sample_pose_change_curves: np.ndarray,
    gt_local_deform: np.ndarray,
    mean_local_deform: np.ndarray,
    sample_local_deform_curves: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    if sample_pose_change_curves.size > 0:
        sample_mean, sample_p10, sample_p90 = summarize_trajectory_band(sample_pose_change_curves)
        axes[0].plot(times, sample_mean, color="#d04a35", linewidth=2.2, label="sample mean")
        axes[0].fill_between(times, sample_p10, sample_p90, color="#f2b3a9", alpha=0.45, label="10%-90% range")
    axes[0].plot(times, mean_pose_change, color="#2a9d8f", linewidth=1.8, linestyle="--", label="mean trajectory")
    axes[0].plot(times, gt_pose_change, color="#1f3b73", linewidth=1.6, label="ground truth")
    axes[0].set_ylabel("Pose Delta (deg)")
    axes[0].set_title("Local MANO Pose Change Relative to First Frame")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper left")

    if sample_local_deform_curves.size > 0:
        sample_mean, sample_p10, sample_p90 = summarize_trajectory_band(sample_local_deform_curves)
        axes[1].plot(times, sample_mean * 1000.0, color="#d04a35", linewidth=2.2, label="sample mean")
        axes[1].fill_between(
            times,
            sample_p10 * 1000.0,
            sample_p90 * 1000.0,
            color="#f2b3a9",
            alpha=0.45,
            label="10%-90% range",
        )
    axes[1].plot(times, mean_local_deform * 1000.0, color="#2a9d8f", linewidth=1.8, linestyle="--", label="mean trajectory")
    axes[1].plot(times, gt_local_deform * 1000.0, color="#1f3b73", linewidth=1.6, label="ground truth")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Joint Deform (mm)")
    axes[1].set_title("Wrist- and Global-Rotation-Normalized Joint Deformation")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_sequence(args: argparse.Namespace) -> Dict[str, object]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint["args"]

    dataset_root = args.dataset_root or checkpoint_args.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root is missing. Please pass it explicitly or keep it in the checkpoint args.")

    test_text_file = args.test_text_file or checkpoint_args.get("test_text_file")
    dataset, loader = build_test_loader(dataset_root, checkpoint_args, args.num_workers, text_file=test_text_file)
    if args.sequence_index < 0 or args.sequence_index >= len(dataset):
        raise IndexError(f"--sequence-index must be in [0, {len(dataset) - 1}], got {args.sequence_index}")

    model = build_model(dataset, checkpoint_args, checkpoint)
    mano_right = build_mano_right_layer(checkpoint_args.get("mano_model_path"))
    method = args.method or checkpoint_args.get("method", "rk4")

    batch = dataset[args.sequence_index]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) and value.ndim >= 1 and key not in {"seq_id"} else value
        for key, value in batch.items()
    }
    full_frame_ids = batch["frame_ids"][0].detach().cpu().numpy()
    batch = select_window(batch, args.window_len, args.start_frame, args.start_frame_id)
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
        mean_motion, _ = model.mean_rec(
            clip_batch["motion"],
            clip_batch["times"],
            mask=clip_batch["mask"],
            shape=clip_batch["shape"],
            pose=clip_batch["pose"],
            Rh=clip_batch["Rh"],
            Th=clip_batch["Th"],
            mano_right=mano_right,
            method=method,
        )
        qz0_m_mean, mean_zt = compute_mean_latent_trajectory(
            model,
            clip_batch["motion"],
            clip_batch["mask"],
            clip_batch["times"],
            method=method,
        )
        sampled_motion, qz0_m, sampled_zt = sample_motion_sequences(
            model,
            clip_batch["motion"],
            clip_batch["mask"],
            clip_batch["times"],
            sample_count=args.sample_count,
            method=method,
            sample_z0=args.sample_z0,
        )
        mean_joints, gt_joints = reconstruct_joints(
            model,
            mano_right,
            mean_motion,
            clip_batch["pose"],
            clip_batch["Rh"],
            clip_batch["Th"],
            clip_batch["shape"],
        )
        mean_vertices = reconstruct_vertices(
            model,
            mano_right,
            mean_motion,
            clip_batch["pose"],
            clip_batch["Rh"],
            clip_batch["Th"],
            clip_batch["shape"],
        )
        gt_vertices = reconstruct_gt_vertices(
            model,
            mano_right,
            clip_batch["pose"],
            clip_batch["Rh"],
            clip_batch["Th"],
            clip_batch["shape"],
        )

        sample_joints_list: List[torch.Tensor] = []
        sample_vertices_list: List[torch.Tensor] = []
        for sample_idx in range(sampled_motion.shape[0]):
            pred_joints, _ = reconstruct_joints(
                model,
                mano_right,
                sampled_motion[sample_idx : sample_idx + 1],
                clip_batch["pose"],
                clip_batch["Rh"],
                clip_batch["Th"],
                clip_batch["shape"],
            )
            sample_joints_list.append(pred_joints[0])
            pred_vertices = reconstruct_vertices(
                model,
                mano_right,
                sampled_motion[sample_idx : sample_idx + 1],
                clip_batch["pose"],
                clip_batch["Rh"],
                clip_batch["Th"],
                clip_batch["shape"],
            )
            sample_vertices_list.append(pred_vertices[0])
        sample_joints = torch.stack(sample_joints_list, dim=0)
        sample_vertices = torch.stack(sample_vertices_list, dim=0)

    mask = clip_batch["mask"][0]
    times = clip_batch["times"][0]
    valid_mask = (mask > 0)
    valid_frames = torch.nonzero(valid_mask, as_tuple=False).flatten().detach().cpu().numpy()

    gt_rel_joints = to_first_frame_relative(gt_joints, clip_batch["mask"])[0]
    mean_rel_joints = to_first_frame_relative(mean_joints, clip_batch["mask"])[0]
    sample_rel_joints = torch.stack(
        [to_first_frame_relative(sample_joints[i : i + 1], clip_batch["mask"])[0] for i in range(sample_joints.shape[0])],
        dim=0,
    )
    gt_pose_6d, gt_rh_6d = split_motion_components(model, clip_batch["motion"][0], clip_batch["pose"][0], clip_batch["Rh"][0])
    mean_pose_6d, mean_rh_6d = split_motion_components(model, mean_motion[0], clip_batch["pose"][0], clip_batch["Rh"][0])
    sample_pose_6d_list = []
    sample_rh_6d_list = []
    for sample_idx in range(sampled_motion.shape[0]):
        sample_pose_6d, sample_rh_6d = split_motion_components(
            model,
            sampled_motion[sample_idx],
            clip_batch["pose"][0],
            clip_batch["Rh"][0],
        )
        sample_pose_6d_list.append(sample_pose_6d)
        sample_rh_6d_list.append(sample_rh_6d)
    sample_pose_6d_tensor = torch.stack(sample_pose_6d_list, dim=0)
    sample_rh_6d_tensor = torch.stack(sample_rh_6d_list, dim=0)
    gt_rel_joints_vis = remap_axes_tensor(gt_rel_joints, args.axis_mode)
    mean_rel_joints_vis = remap_axes_tensor(mean_rel_joints, args.axis_mode)
    sample_rel_joints_vis = remap_axes_tensor(sample_rel_joints, args.axis_mode)

    mean_error_curve = torch.linalg.norm(mean_rel_joints - gt_rel_joints, dim=-1).mean(dim=-1)
    mean_error_curve[~valid_mask] = float("nan")
    sample_mean_error, sample_p10_error, sample_p90_error = summarize_sample_errors(
        sample_rel_joints,
        gt_rel_joints,
        valid_mask,
    )
    diversity_curve = compute_diversity_curve(sample_rel_joints, valid_mask)
    gt_speed_time, gt_speed = compute_speed_curve(gt_joints[0], mask, times)
    sample_speed_curves = []
    for sample_idx in range(sample_joints.shape[0]):
        _, sample_speed = compute_speed_curve(sample_joints[sample_idx], mask, times)
        sample_speed_curves.append(sample_speed)
    sample_speed_curves_np = np.stack(sample_speed_curves, axis=0) if sample_speed_curves else np.empty((0, 0))
    gt_pose_change = compute_pose_change_curve_from_pose6d(gt_pose_6d)
    mean_pose_change = compute_pose_change_curve_from_pose6d(mean_pose_6d)
    sample_pose_change_curves_np = np.stack(
        [compute_pose_change_curve_from_pose6d(sample_pose_6d_tensor[i]) for i in range(sample_pose_6d_tensor.shape[0])],
        axis=0,
    ) if sample_pose_6d_tensor.shape[0] > 0 else np.empty((0, 0))
    gt_local_deform = compute_local_joint_deformation_curve(gt_joints[0], gt_rh_6d)
    mean_local_deform = compute_local_joint_deformation_curve(mean_joints[0], mean_rh_6d)
    sample_local_deform_curves_np = np.stack(
        [compute_local_joint_deformation_curve(sample_joints[i], sample_rh_6d_tensor[i]) for i in range(sample_joints.shape[0])],
        axis=0,
    ) if sample_joints.shape[0] > 0 else np.empty((0, 0))
    mean_v_norm = torch.linalg.norm(mean_zt[:, : model.q], dim=-1).detach().cpu().numpy()
    mean_s_delta = torch.linalg.norm(mean_zt[:, model.q :] - mean_zt[:1, model.q :], dim=-1).detach().cpu().numpy()
    sample_v_curves_np = torch.linalg.norm(sampled_zt[:, :, : model.q], dim=-1).detach().cpu().numpy()
    sample_s_curves_np = torch.linalg.norm(
        sampled_zt[:, :, model.q :] - sampled_zt[:, :1, model.q :],
        dim=-1,
    ).detach().cpu().numpy()
    mean_motion_change = torch.linalg.norm(
        mean_motion[0] - mean_motion[0, :1],
        dim=-1,
    ).detach().cpu().numpy()
    sample_motion_change_curves_np = torch.linalg.norm(
        sampled_motion - sampled_motion[:, :1],
        dim=-1,
    ).detach().cpu().numpy()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_png = save_metric_figure(
        times.detach().cpu().numpy(),
        mean_error_curve.detach().cpu().numpy(),
        sample_mean_error,
        sample_p10_error,
        sample_p90_error,
        diversity_curve,
        output_dir / "sample_metrics.png",
    )
    latent_diag_png = save_latent_diagnostic_figure(
        times.detach().cpu().numpy(),
        mean_v_norm,
        sample_v_curves_np,
        mean_s_delta,
        sample_s_curves_np,
        output_dir / "latent_diagnostics.png",
    )
    motion_change_png = save_motion_change_figure(
        times.detach().cpu().numpy(),
        mean_motion_change,
        sample_motion_change_curves_np,
        output_dir / "motion_change.png",
    )
    articulation_diag_png = save_articulation_diagnostic_figure(
        times.detach().cpu().numpy(),
        gt_pose_change,
        mean_pose_change,
        sample_pose_change_curves_np,
        gt_local_deform,
        mean_local_deform,
        sample_local_deform_curves_np,
        output_dir / "articulation_diagnostics.png",
    )
    motion_png = save_motion_figure(
        gt_speed_time,
        gt_speed,
        sample_speed_curves_np,
        gt_rel_joints_vis.detach().cpu().numpy(),
        sample_rel_joints_vis.detach().cpu().numpy(),
        valid_frames,
        output_dir / "sample_motion.png",
    )
    snapshot_png = save_snapshot_grid(
        gt_rel_joints_vis.detach().cpu().numpy(),
        sample_rel_joints_vis.detach().cpu().numpy(),
        valid_frames,
        output_dir / "sample_snapshots.png",
        visible_samples=args.visible_samples,
    )
    animation_sequences: List[Tuple[str, np.ndarray, str]] = [
        ("Ground Truth", gt_rel_joints_vis.detach().cpu().numpy(), SEQUENCE_COLORS[0]),
        ("Mean", mean_rel_joints_vis.detach().cpu().numpy(), SEQUENCE_COLORS[1]),
    ]
    for sample_idx in range(min(args.visible_samples, sample_rel_joints_vis.shape[0])):
        animation_sequences.append(
            (
                f"Sample {sample_idx + 1}",
                sample_rel_joints_vis[sample_idx].detach().cpu().numpy(),
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
        gt_rel_joints.detach().cpu().numpy(),
        mean_rel_joints.detach().cpu().numpy(),
        sample_rel_joints.detach().cpu().numpy(),
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
            mesh_sequences.append(
                (f"sample_{sample_idx + 1:02d}", sample_vertices[sample_idx].detach().cpu().numpy())
            )
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
        "window_len": int(clip_batch["motion"].shape[1]),
        "window_start_index": window_start_index,
        "valid_frame_count": int(valid_mask.sum().item()),
        "frame_id_start": int(frame_ids[0]),
        "frame_id_end": int(frame_ids[-1]),
        "sample_count": args.sample_count,
        "method": method,
        "sample_z0": bool(args.sample_z0),
        "axis_mode": args.axis_mode,
        "compare_axis_modes": list(args.compare_axis_modes),
        "latent_init_norm": float(torch.linalg.norm(qz0_m[0]).item()),
        "mean_latent_init_norm": float(torch.linalg.norm(qz0_m_mean[0]).item()),
        "mean_v_norm": finite_mean(mean_v_norm),
        "sample_v_norm": finite_mean(np.nanmean(sample_v_curves_np, axis=0)) if sample_v_curves_np.size > 0 else float("nan"),
        "mean_s_delta": finite_mean(mean_s_delta),
        "sample_s_delta": finite_mean(np.nanmean(sample_s_curves_np, axis=0)) if sample_s_curves_np.size > 0 else float("nan"),
        "mean_motion_change": finite_mean(mean_motion_change),
        "sample_motion_change": finite_mean(np.nanmean(sample_motion_change_curves_np, axis=0)) if sample_motion_change_curves_np.size > 0 else float("nan"),
        "gt_pose_change_deg": finite_mean(gt_pose_change),
        "mean_pose_change_deg": finite_mean(mean_pose_change),
        "sample_pose_change_deg": finite_mean(np.nanmean(sample_pose_change_curves_np, axis=0)) if sample_pose_change_curves_np.size > 0 else float("nan"),
        "gt_local_deform_mm": finite_mean(gt_local_deform * 1000.0),
        "mean_local_deform_mm": finite_mean(mean_local_deform * 1000.0),
        "sample_local_deform_mm": finite_mean(np.nanmean(sample_local_deform_curves_np, axis=0) * 1000.0) if sample_local_deform_curves_np.size > 0 else float("nan"),
        "mean_curve_mpjpe_mm": finite_mean(mean_error_curve.detach().cpu().numpy() * 1000.0),
        "sample_curve_mpjpe_mm": finite_mean(sample_mean_error * 1000.0),
        "sample_curve_mpjpe_mm_max": finite_max(sample_mean_error * 1000.0),
        "diversity_mm": finite_mean(diversity_curve * 1000.0),
        "diversity_mm_max": finite_max(diversity_curve * 1000.0),
        "gt_speed_mps": finite_mean(gt_speed),
        "sample_speed_mps": finite_mean(np.nanmean(sample_speed_curves_np, axis=0)) if sample_speed_curves_np.size > 0 else float("nan"),
        "outputs": {
            "axis_compare_png": str(axis_compare_png),
            "latent_diag_png": str(latent_diag_png),
            "motion_change_png": str(motion_change_png),
            "articulation_diag_png": str(articulation_diag_png),
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

    return {
        **summary,
        "summary_json": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ODE2VAE hand motion diversity by comparing random BNN samples against a real sequence."
    )
    parser.add_argument("--checkpoint", type=str, default="runs/ode2vae_hand5/best_model.pt")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--test-text-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/ode2vae_hand5/visualize")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--window-len", type=int, default=150)
    parser.add_argument("--start-frame", type=int, default=None, help="Start from this in-sequence frame index before cropping.")
    parser.add_argument("--start-frame-id", type=int, default=None, help="Start from this original dataset frame_id before cropping.")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-z0", action="store_true", help="Sample the latent initial state in addition to the BNN weights.")
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
    if args.window_len == 0:
        raise ValueError("--window-len must be positive or omitted")
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
    print(f"window_len: {result['window_len']}")
    print(f"valid_frame_count: {result['valid_frame_count']}")
    print(f"sample_count: {result['sample_count']}")
    print(f"axis_mode: {result['axis_mode']}")
    print(f"mean_v_norm: {result['mean_v_norm']:.4f}")
    print(f"sample_v_norm: {result['sample_v_norm']:.4f}")
    print(f"mean_s_delta: {result['mean_s_delta']:.4f}")
    print(f"sample_s_delta: {result['sample_s_delta']:.4f}")
    print(f"mean_motion_change: {result['mean_motion_change']:.4f}")
    print(f"sample_motion_change: {result['sample_motion_change']:.4f}")
    print(f"gt_pose_change_deg: {result['gt_pose_change_deg']:.3f}")
    print(f"mean_pose_change_deg: {result['mean_pose_change_deg']:.3f}")
    print(f"sample_pose_change_deg: {result['sample_pose_change_deg']:.3f}")
    print(f"gt_local_deform_mm: {result['gt_local_deform_mm']:.3f}")
    print(f"mean_local_deform_mm: {result['mean_local_deform_mm']:.3f}")
    print(f"sample_local_deform_mm: {result['sample_local_deform_mm']:.3f}")
    print(f"mean_curve_mpjpe_mm: {result['mean_curve_mpjpe_mm']:.3f}")
    print(f"sample_curve_mpjpe_mm: {result['sample_curve_mpjpe_mm']:.3f}")
    print(f"sample_curve_mpjpe_mm_max: {result['sample_curve_mpjpe_mm_max']:.3f}")
    print(f"diversity_mm: {result['diversity_mm']:.3f}")
    print(f"diversity_mm_max: {result['diversity_mm_max']:.3f}")
    print(f"gt_speed_mps: {result['gt_speed_mps']:.4f}")
    print(f"sample_speed_mps: {result['sample_speed_mps']:.4f}")
    print(f"axis_compare_png: {result['outputs']['axis_compare_png']}")
    print(f"latent_diag_png: {result['outputs']['latent_diag_png']}")
    print(f"motion_change_png: {result['outputs']['motion_change_png']}")
    print(f"articulation_diag_png: {result['outputs']['articulation_diag_png']}")
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
