import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

from easymocap.smplmodel.body_model import SMPLlayer
from hand_dataset import GigaHandDataset
from torch_bnn import BNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _identity_rot6d(batch_size: int, target_device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=target_device, dtype=dtype)
    return base.unsqueeze(0).expand(batch_size, -1)


def _skew(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            zeros,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            zeros,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            zeros,
        ],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def rot6d_rowmajor_to_matrix(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, 6)
    a1 = torch.stack([x[:, 0], x[:, 2], x[:, 4]], dim=1)
    a2 = torch.stack([x[:, 1], x[:, 3], x[:, 5]], dim=1)
    b1 = F.normalize(a1, dim=1)
    proj = (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    mats = torch.stack([b1, b2, b3], dim=-1)
    return mats.reshape(*orig_shape, 3, 3)


def matrix_to_rot6d_rowmajor(mat: torch.Tensor) -> torch.Tensor:
    return mat[..., :, :2].contiguous().reshape(*mat.shape[:-2], 6)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = theta2.sqrt()
    k = _skew(axis_angle)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.view(*([1] * (axis_angle.ndim - 1)), 3, 3)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)

    theta4 = theta2 * theta2
    a = torch.where(
        theta2 > 1e-8,
        torch.sin(theta) / theta.clamp_min(1e-8),
        1.0 - theta2 / 6.0 + theta4 / 120.0,
    )
    b = torch.where(
        theta2 > 1e-8,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8),
        0.5 - theta2 / 24.0 + theta4 / 720.0,
    )
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def matrix_to_axis_angle(mat: torch.Tensor) -> torch.Tensor:
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta).unsqueeze(-1)
    vee = torch.stack(
        [
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ],
        dim=-1,
    )
    small = theta < 1e-4
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-6))
    rotvec = scale * vee
    rotvec_small = 0.5 * vee
    return torch.where(small.expand_as(rotvec), rotvec_small, rotvec)


def geodesic_distance_from_matrices(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = pred.transpose(-1, -2) @ target
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.dtype)
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def weighted_mean_per_sample(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.dtype)
    batch_size = values.shape[0]
    weighted = (values * weights).reshape(batch_size, -1)
    denom = weights.reshape(batch_size, -1).sum(dim=1).clamp_min(1e-6)
    return weighted.sum(dim=1) / denom


def mse_feature_mean(pred: torch.Tensor, target: torch.Tensor, dim) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="none").mean(dim=dim)


def smooth_l1_feature_mean(pred: torch.Tensor, target: torch.Tensor, dim) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="none", beta=0.001).mean(dim=dim)


def path_differences(
    x: torch.Tensor,
    valid: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] < 2:
        empty = torch.zeros(*x.shape[:1], 0, *x.shape[2:], device=x.device, dtype=x.dtype)
        empty_mask = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)
        return empty, empty_mask
    scale = dt.view(dt.shape[0], dt.shape[1], *([1] * (x.ndim - 2))).clamp_min(1e-6)
    diff = (x[:, 1:] - x[:, :-1]) / scale
    pair_valid = ((valid[:, 1:] > 0) & (valid[:, :-1] > 0)).to(x.dtype)
    diff = diff * pair_valid.view(pair_valid.shape[0], pair_valid.shape[1], *([1] * (x.ndim - 2)))
    return diff, pair_valid


class FrameEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=2 ** i) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        for conv, norm in zip(self.layers, self.norms):
            pad = (self.kernel_size - 1) * conv.dilation[0]
            residual = y
            z = F.pad(y, (pad, 0))
            z = conv(z).transpose(1, 2)
            z = F.silu(norm(z)).transpose(1, 2)
            y = residual + z
        return y.transpose(1, 2)


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean(x), self.logvar(x).clamp(-8.0, 4.0)


class BayesianDynamics(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, damping: float = 0.05) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.damping = damping
        self.bnn = BNN(
            input_dim,
            latent_dim + 6,
            n_hid_layers=2,
            n_hidden=hidden_dim,
            act="celu",
            layer_norm=True,
            bnn=True,
        )

    def _split_output(
        self,
        out: torch.Tensor,
        hdot: torch.Tensor,
        nu: torch.Tensor,
        omega: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ddh = out[..., : self.latent_dim] - self.damping * hdot
        dnu = out[..., self.latent_dim : self.latent_dim + 3] - self.damping * nu
        domega = out[..., self.latent_dim + 3 : self.latent_dim + 6] - self.damping * omega
        return ddh, dnu, domega

    def draw_dynamics(self, mean: bool = False):
        f = self.bnn.draw_f(mean=mean)

        def dynamics(
            h: torch.Tensor,
            hdot: torch.Tensor,
            nu: torch.Tensor,
            omega: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = torch.cat([h, hdot, nu, omega], dim=-1)
            return self._split_output(f(x), hdot, nu, omega)

        return dynamics

    def forward(
        self,
        h: torch.Tensor,
        hdot: torch.Tensor,
        nu: torch.Tensor,
        omega: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.draw_dynamics(mean=not self.training)(h, hdot, nu, omega)

    def kl(self) -> torch.Tensor:
        return self.bnn.kl()


class InternalStateODEFunc(nn.Module):
    def __init__(self, model: "ODE2VAEHand", future_dt: torch.Tensor, dynamics_fn) -> None:
        super().__init__()
        self.model = model
        self.future_dt = future_dt
        self.dynamics_fn = dynamics_fn

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        horizon = self.future_dt.shape[1]
        interval_idx = int(torch.floor(t.detach()).clamp(0, horizon - 1).item())
        h, hdot, nu, omega = self.model._split_internal_state(z)
        ddh, dnu, domega = self.dynamics_fn(h, hdot, nu, omega)
        dz = torch.cat([hdot, ddh, dnu, domega], dim=-1)
        return dz * self.future_dt[:, interval_idx].unsqueeze(-1)


class ODE2VAEHand(nn.Module):
    def __init__(
        self,
        input_dim: int,
        q: int = 16,
        hidden_dim: int = 256,
        history_len: int = 10,
        horizon: int = 5,
        num_joints: int = 21,
        dynamics_damping: float = 0.05,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.num_joints = num_joints
        self.latent_dim = q

        self.pose_dim = input_dim - 9
        if self.pose_dim <= 0 or self.pose_dim % 6 != 0:
            raise ValueError(f"Invalid pose_dim inferred from input_dim={input_dim}. Expected motion = Rh(6) + pose(6*k) + Th(3).")
        self.num_pose_joints = self.pose_dim // 6
        self.dpose_dim = 3 * self.num_pose_joints
        self.state_dim = self.latent_dim
        self.vel_dim = self.latent_dim + 6
        self.feature_dim = (
            self.pose_dim
            + self.dpose_dim
            + 3 * self.num_joints
            + 3 * self.num_joints
            + 1
        )

        self.frame_encoder = FrameEncoder(self.feature_dim, hidden_dim)
        self.temporal_encoder = TemporalConvEncoder(hidden_dim)
        self.state_head = GaussianHead(hidden_dim, self.state_dim)
        self.vel_head = GaussianHead(hidden_dim, self.vel_dim)
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.pose_dim),
        )
        self.dpose_decoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dpose_dim),
        )
        self.dynamics = BayesianDynamics(
            self.latent_dim * 2 + 6,
            hidden_dim,
            self.latent_dim,
            damping=dynamics_damping,
        )

    @staticmethod
    def _rot6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
        mats = rot6d_rowmajor_to_matrix(x)
        return matrix_to_axis_angle(mats)

    def _sequence_to_axis_angle(self, pose6d: torch.Tensor, root6d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pose_axis = self._rot6d_to_axis_angle(pose6d.reshape(-1, 6)).reshape(-1, self.num_pose_joints, 3)
        root_axis = self._rot6d_to_axis_angle(root6d.reshape(-1, 6)).reshape(-1, 3)
        return pose_axis.reshape(pose6d.shape[0], -1), root_axis

    def _mano_forward(
        self,
        pose6d: torch.Tensor,
        root6d: torch.Tensor,
        trans: torch.Tensor,
        shape: torch.Tensor,
        mano_right: SMPLlayer,
        return_verts: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n, t, _ = pose6d.shape
        pose_axis, root_axis = self._sequence_to_axis_angle(
            pose6d.reshape(n * t, -1),
            root6d.reshape(n * t, 6),
        )
        trans_flat = trans.reshape(n * t, 3)
        shape_flat = shape.reshape(n * t, -1)
        joints = mano_right(
            poses=pose_axis,
            shapes=shape_flat,
            Rh=root_axis,
            Th=trans_flat,
            return_verts=False,
            return_tensor=True,
        ).reshape(n, t, -1, 3)
        verts = None
        if return_verts:
            verts = mano_right(
                poses=pose_axis,
                shapes=shape_flat,
                Rh=root_axis,
                Th=trans_flat,
                return_verts=True,
                return_tensor=True,
            ).reshape(n, t, -1, 3)
        return joints, verts

    def _build_local_targets(
        self,
        pose: torch.Tensor,
        rh: torch.Tensor,
        th: torch.Tensor,
        shape: torch.Tensor,
        mask: torch.Tensor,
        history_len: int,
        mano_right: SMPLlayer,
        need_verts: bool,
    ) -> Dict[str, torch.Tensor]:
        n, total_len, _ = pose.shape
        anchor_idx = history_len - 1
        root_mats = rot6d_rowmajor_to_matrix(rh.reshape(-1, 6)).reshape(n, total_len, 3, 3)
        anchor_rot = root_mats[:, anchor_idx]
        anchor_inv = anchor_rot.transpose(-1, -2)
        root_rel = torch.matmul(anchor_inv.unsqueeze(1), root_mats)
        root_rel6d = matrix_to_rot6d_rowmajor(root_rel)

        trans_centered = th - th[:, anchor_idx : anchor_idx + 1]
        trans_rel = torch.matmul(anchor_inv.unsqueeze(1), trans_centered.unsqueeze(-1)).squeeze(-1)

        anchor_shape = shape[:, anchor_idx]
        full_shape = anchor_shape.unsqueeze(1).expand(-1, total_len, -1)
        joints, verts = self._mano_forward(pose, root_rel6d, trans_rel, full_shape, mano_right, return_verts=need_verts)
        local_root6d = _identity_rot6d(n * total_len, pose.device, pose.dtype).reshape(n, total_len, 6)
        local_trans = torch.zeros(n, total_len, 3, device=pose.device, dtype=pose.dtype)
        joints_rel, _ = self._mano_forward(
            pose,
            local_root6d,
            local_trans,
            full_shape,
            mano_right,
            return_verts=False,
        )
        joint_delta = torch.zeros_like(joints_rel)
        if total_len > 1:
            prev_root_inv = root_mats[:, :-1].transpose(-1, -2)
            step_root = prev_root_inv @ root_mats[:, 1:]
            step_root6d = matrix_to_rot6d_rowmajor(step_root)
            step_trans = (prev_root_inv @ (th[:, 1:] - th[:, :-1]).unsqueeze(-1)).squeeze(-1)

            prev_root6d = _identity_rot6d(n * (total_len - 1), pose.device, pose.dtype).reshape(n, total_len - 1, 6)
            prev_trans = torch.zeros(n, total_len - 1, 3, device=pose.device, dtype=pose.dtype)
            prev_joints, _ = self._mano_forward(
                pose[:, :-1],
                prev_root6d,
                prev_trans,
                full_shape[:, :-1],
                mano_right,
                return_verts=False,
            )
            next_joints, _ = self._mano_forward(
                pose[:, 1:],
                step_root6d,
                step_trans,
                full_shape[:, 1:],
                mano_right,
                return_verts=False,
            )
            joint_delta[:, 1:] = next_joints - prev_joints

        return {
            "root_rel6d": root_rel6d,
            "root_rel_mat": root_rel,
            "trans_rel": trans_rel,
            "joints": joints,
            "joints_rel": joints_rel,
            "joint_delta": joint_delta,
            "verts": verts,
            "anchor_shape": anchor_shape,
            "mask": mask,
        }

    @staticmethod
    def _root_velocity_targets(
        root_rel_mat: torch.Tensor,
        trans_rel: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, total_len = mask.shape
        if total_len < 2:
            zeros = torch.zeros(n, total_len, 3, device=trans_rel.device, dtype=trans_rel.dtype)
            return zeros, zeros
        dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-6)
        pair_valid = ((mask[:, 1:] > 0) & (mask[:, :-1] > 0)).to(trans_rel.dtype)

        trans_step = (trans_rel[:, 1:] - trans_rel[:, :-1]) / dt.unsqueeze(-1)
        nu = (root_rel_mat[:, :-1].transpose(-1, -2) @ trans_step.unsqueeze(-1)).squeeze(-1)
        nu = nu * pair_valid.unsqueeze(-1)

        rel_step = root_rel_mat[:, :-1].transpose(-1, -2) @ root_rel_mat[:, 1:]
        omega = matrix_to_axis_angle(rel_step) / dt.unsqueeze(-1)
        omega = omega * pair_valid.unsqueeze(-1)

        zero = torch.zeros(n, 1, 3, device=trans_rel.device, dtype=trans_rel.dtype)
        return torch.cat([zero, nu], dim=1), torch.cat([zero, omega], dim=1)

    def _pose_angular_velocity(
        self,
        pose: torch.Tensor,
        dt: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        n = pose.shape[0]
        omega, _ = self._pose_angular_velocity_pairs(pose, dt, valid)
        zero = torch.zeros(n, 1, self.dpose_dim, device=pose.device, dtype=pose.dtype)
        return torch.cat([zero, omega], dim=1)

    def _pose_angular_velocity_targets(
        self,
        pose: torch.Tensor,
        dt: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = pose.shape[0]
        omega, pair_valid = self._pose_angular_velocity_pairs(pose, dt, valid)
        zero = torch.zeros(n, 1, self.dpose_dim, device=pose.device, dtype=pose.dtype)
        zero_mask = torch.zeros(n, 1, device=pose.device, dtype=pose.dtype)
        return torch.cat([zero, omega], dim=1), torch.cat([zero_mask, pair_valid], dim=1)

    def _pose_angular_velocity_pairs(
        self,
        pose: torch.Tensor,
        dt: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, total_len, _ = pose.shape
        if total_len < 2:
            omega = torch.zeros(n, 0, self.dpose_dim, device=pose.device, dtype=pose.dtype)
            pair_valid = torch.zeros(n, 0, device=pose.device, dtype=pose.dtype)
            return omega, pair_valid
        pose_mat = rot6d_rowmajor_to_matrix(pose.reshape(n, total_len, self.num_pose_joints, 6))
        rel_step = pose_mat[:, :-1].transpose(-1, -2) @ pose_mat[:, 1:]
        omega = matrix_to_axis_angle(rel_step) / dt.view(n, total_len - 1, 1, 1).clamp_min(1e-6)
        pair_valid = ((valid[:, 1:] > 0) & (valid[:, :-1] > 0)).to(pose.dtype)
        omega = omega * pair_valid.unsqueeze(-1).unsqueeze(-1)
        return omega.reshape(n, total_len - 1, self.dpose_dim), pair_valid

    def _history_features(
        self,
        pose: torch.Tensor,
        joints_rel: torch.Tensor,
        joint_delta: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        history_len: int,
    ) -> Dict[str, torch.Tensor]:
        hist_pose = pose[:, :history_len]
        hist_joints = joints_rel[:, :history_len]
        hist_joint_delta = joint_delta[:, :history_len]
        hist_mask = mask[:, :history_len]
        hist_times = times[:, :history_len]
        dt = (hist_times[:, 1:] - hist_times[:, :-1]).clamp_min(1e-6)

        dpose = self._pose_angular_velocity(hist_pose, dt, hist_mask)
        djoints = torch.zeros_like(hist_joints)
        if history_len > 1:
            pair_valid = ((hist_mask[:, 1:] > 0) & (hist_mask[:, :-1] > 0)).to(hist_joints.dtype)
            djoints[:, 1:] = hist_joint_delta[:, 1:] / dt.view(dt.shape[0], dt.shape[1], 1, 1).clamp_min(1e-6)
            djoints[:, 1:] = djoints[:, 1:] * pair_valid.unsqueeze(-1).unsqueeze(-1)

        feature = torch.cat(
            [
                hist_pose,
                dpose,
                hist_joints.reshape(hist_joints.shape[0], history_len, -1),
                djoints.reshape(hist_joints.shape[0], history_len, -1),
                hist_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        feature = feature.clone()
        feature[:, :, :-1] = feature[:, :, :-1] * hist_mask.unsqueeze(-1)
        return {
            "feature": feature,
            "hist_mask": hist_mask,
        }

    def _encode_initial_distribution(
        self,
        history_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n, history_len, d = history_feature.shape
        frame_h = self.frame_encoder(history_feature.reshape(n * history_len, d)).reshape(n, history_len, -1)
        temporal_h = self.temporal_encoder(frame_h)
        summary = temporal_h[:, -1]

        state_mu, state_logvar = self.state_head(summary)
        vel_mu, vel_logvar = self.vel_head(summary)
        return {
            "state_mu": state_mu,
            "state_logvar": state_logvar,
            "vel_mu": vel_mu,
            "vel_logvar": vel_logvar,
        }

    def _sample_initial_state(
        self,
        stats: Dict[str, torch.Tensor],
        sample: bool,
    ) -> Dict[str, torch.Tensor]:
        if sample:
            eps_state = torch.randn_like(stats["state_mu"])
            eps_vel = torch.randn_like(stats["vel_mu"])
            state = stats["state_mu"] + eps_state * torch.exp(0.5 * stats["state_logvar"])
            vel = stats["vel_mu"] + eps_vel * torch.exp(0.5 * stats["vel_logvar"])
        else:
            state = stats["state_mu"]
            vel = stats["vel_mu"]
        h0 = state
        hdot0 = vel[:, : self.latent_dim]
        nu0 = vel[:, self.latent_dim : self.latent_dim + 3]
        omega0 = vel[:, self.latent_dim + 3 : self.latent_dim + 6]
        pose0 = self._decode_pose(h0, hdot0)
        root0 = _identity_rot6d(state.shape[0], state.device, state.dtype)
        return {
            "h0": h0,
            "hdot0": hdot0,
            "root0_6d": root0,
            "root0_mat": rot6d_rowmajor_to_matrix(root0),
            "pose0": pose0,
            "nu0": nu0,
            "omega0": omega0,
        }

    def _decode_pose(self, h: torch.Tensor, hdot: torch.Tensor) -> torch.Tensor:
        return self.pose_decoder(torch.cat([h, hdot], dim=-1))

    def _decode_dpose(self, h: torch.Tensor, hdot: torch.Tensor) -> torch.Tensor:
        return self.dpose_decoder(torch.cat([h, hdot], dim=-1))

    def _pack_internal_state(
        self,
        h: torch.Tensor,
        hdot: torch.Tensor,
        nu: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([h, hdot, nu, omega], dim=-1)

    def _split_internal_state(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = z[..., : self.latent_dim]
        hdot = z[..., self.latent_dim : 2 * self.latent_dim]
        nu = z[..., 2 * self.latent_dim : 2 * self.latent_dim + 3]
        omega = z[..., 2 * self.latent_dim + 3 : 2 * self.latent_dim + 6]
        return h, hdot, nu, omega

    def _rollout(
        self,
        init_state: Dict[str, torch.Tensor],
        future_dt: torch.Tensor,
        method: str,
        sample_dynamics: bool,
    ) -> Dict[str, torch.Tensor]:
        horizon = future_dt.shape[1]
        z0 = self._pack_internal_state(init_state["h0"], init_state["hdot0"], init_state["nu0"], init_state["omega0"])
        t_eval = 0.5 * torch.arange(2 * horizon + 1, device=future_dt.device, dtype=future_dt.dtype)
        dynamics_fn = self.dynamics.draw_dynamics(mean=False if self.training else not sample_dynamics)
        ode_func = InternalStateODEFunc(self, future_dt, dynamics_fn)
        z_dense = odeint(
            ode_func,
            z0,
            t_eval,
            method=method,
        ).transpose(0, 1)
        z_traj = z_dense[:, ::2]  # [B, H + 1, 2 * latent_dim + 6]
        z_mid = z_dense[:, 1::2]  # [B, H, 2 * latent_dim + 6]

        h_future, hdot_future, _, _ = self._split_internal_state(z_traj[:, 1:])
        h_interval, hdot_interval, nu_interval, omega_interval = self._split_internal_state(z_traj[:, :-1])
        del h_interval, hdot_interval
        _, _, nu_mid, omega_mid = self._split_internal_state(z_mid)
        _, _, nu_end, omega_end = self._split_internal_state(z_traj[:, 1:])

        pose_future = self._decode_pose(
            h_future.reshape(-1, self.latent_dim),
            hdot_future.reshape(-1, self.latent_dim),
        ).reshape(z0.shape[0], horizon, self.pose_dim)
        dpose_future = self._decode_dpose(
            h_future.reshape(-1, self.latent_dim),
            hdot_future.reshape(-1, self.latent_dim),
        ).reshape(z0.shape[0], horizon, self.dpose_dim)

        root_seq: List[torch.Tensor] = []
        delta_p_seq: List[torch.Tensor] = []
        root_mat = init_state["root0_mat"]
        delta_p = torch.zeros_like(init_state["nu0"])

        for step in range(horizon):
            dt_col = future_dt[:, step].unsqueeze(-1)
            half_dt = 0.5 * dt_col
            nu0_step = nu_interval[:, step]
            nu_mid_step = nu_mid[:, step]
            nu1_step = nu_end[:, step]
            omega0_step = omega_interval[:, step]
            omega_mid_step = omega_mid[:, step]
            omega1_step = omega_end[:, step]

            root_k2 = root_mat @ axis_angle_to_matrix(omega0_step * half_dt)
            root_k3 = root_mat @ axis_angle_to_matrix(omega_mid_step * half_dt)
            root_k4 = root_mat @ axis_angle_to_matrix(omega_mid_step * dt_col)
            k1 = (root_mat @ nu0_step.unsqueeze(-1)).squeeze(-1)
            k2 = (root_k2 @ nu_mid_step.unsqueeze(-1)).squeeze(-1)
            k3 = (root_k3 @ nu_mid_step.unsqueeze(-1)).squeeze(-1)
            k4 = (root_k4 @ nu1_step.unsqueeze(-1)).squeeze(-1)
            delta_p = delta_p + (dt_col / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            omega_rk4 = (omega0_step + 2.0 * omega_mid_step + 2.0 * omega_mid_step + omega1_step) / 6.0
            root_mat = root_mat @ axis_angle_to_matrix(omega_rk4 * dt_col)
            root_seq.append(matrix_to_rot6d_rowmajor(root_mat))
            delta_p_seq.append(delta_p)

        return {
            "z_path": z_traj,
            "z_future": z_traj[:, 1:],
            "root_future": torch.stack(root_seq, dim=1),
            "pose_future": pose_future,
            "dpose_future": dpose_future,
            "nu_interval": nu_interval,
            "omega_interval": omega_interval,
            "delta_p_future": torch.stack(delta_p_seq, dim=1),
        }

    @staticmethod
    def _latent_stats(stats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = torch.cat([stats["state_mu"], stats["vel_mu"]], dim=-1)
        logvar = torch.cat([stats["state_logvar"], stats["vel_logvar"]], dim=-1)
        return mu, logvar

    def _kl_z_loss(self, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._kl_z_loss_per_sample(stats).mean()

    def _kl_z_loss_per_sample(self, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, logvar = self._latent_stats(stats)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1)

    def _kl_w_loss(self) -> torch.Tensor:
        total = next(self.parameters()).new_tensor(0.0)
        kl_fn = getattr(self.dynamics, "kl", None)
        if callable(kl_fn):
            kl_value = kl_fn()
            if torch.is_tensor(kl_value):
                total = total + kl_value.sum()
        return total

    @staticmethod
    def _gaussian_kl_to_unit(mu: torch.Tensor, logvar: torch.Tensor, target_mu: torch.Tensor) -> torch.Tensor:
        return 0.5 * (logvar.exp() + (mu - target_mu).pow(2) - 1.0 - logvar).sum(dim=-1)

    def _inst_kl_loss(
        self,
        rollout: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        history_len: int,
        horizon: int,
        future_discount: float,
    ) -> torch.Tensor:
        return self._inst_kl_loss_per_sample(
            rollout=rollout,
            targets=targets,
            history_len=history_len,
            horizon=horizon,
            future_discount=future_discount,
        ).mean()

    def _inst_kl_loss_per_sample(
        self,
        rollout: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        history_len: int,
        horizon: int,
        future_discount: float,
    ) -> torch.Tensor:
        if horizon <= 0:
            return rollout["z_path"].new_zeros(rollout["z_path"].shape[0])

        inst_mu = []
        inst_logvar = []
        for step in range(1, horizon + 1):
            start = step
            end = start + history_len
            history = self._history_features(
                pose=targets["pose"][:, start:end],
                joints_rel=targets["joints_rel"][:, start:end],
                joint_delta=targets["joint_delta"][:, start:end],
                mask=targets["mask"][:, start:end],
                times=targets["times"][:, start:end],
                history_len=history_len,
            )
            step_stats = self._encode_initial_distribution(history["feature"])
            step_mu, step_logvar = self._latent_stats(step_stats)
            inst_mu.append(step_mu)
            inst_logvar.append(step_logvar)

        inst_mu_t = torch.stack(inst_mu, dim=1)
        inst_logvar_t = torch.stack(inst_logvar, dim=1)
        ode_z = rollout["z_future"]
        inst_kl = self._gaussian_kl_to_unit(inst_mu_t, inst_logvar_t, ode_z)

        future_mask = targets["mask"][:, history_len : history_len + horizon]
        future_weights = future_discount ** torch.arange(horizon, device=future_mask.device, dtype=future_mask.dtype)
        future_weights = future_weights.view(1, -1) * future_mask
        return weighted_mean_per_sample(inst_kl, future_weights)

    def _pose_geodesic(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        pred_mat = rot6d_rowmajor_to_matrix(pred_pose.reshape(-1, 6)).reshape(
            pred_pose.shape[0], pred_pose.shape[1], self.num_pose_joints, 3, 3
        )
        gt_mat = rot6d_rowmajor_to_matrix(gt_pose.reshape(-1, 6)).reshape(
            gt_pose.shape[0], gt_pose.shape[1], self.num_pose_joints, 3, 3
        )
        return geodesic_distance_from_matrices(pred_mat, gt_mat).mean(dim=2)

    def _compute_losses(
        self,
        init_state: Dict[str, torch.Tensor],
        stats: Dict[str, torch.Tensor],
        rollout: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mano_right: SMPLlayer,
        history_len: int,
        horizon: int,
        future_discount: float,
        need_verts: bool,
    ) -> Dict[str, torch.Tensor]:
        anchor_idx = history_len - 1
        future_mask = targets["mask"][:, history_len : history_len + horizon]
        future_weights = future_discount ** torch.arange(horizon, device=future_mask.device, dtype=future_mask.dtype)
        future_weights = future_weights.view(1, -1) * future_mask

        gt_root_future = targets["root_rel6d"][:, history_len : history_len + horizon]
        gt_pose_future = targets["pose"][:, history_len : history_len + horizon]
        gt_delta_p_future = targets["trans_rel"][:, history_len : history_len + horizon]
        gt_nu_future = targets["root_nu"][:, history_len : history_len + horizon]
        gt_omega_future = targets["root_omega"][:, history_len : history_len + horizon]
        gt_dpose_future = targets["dpose"][:, history_len : history_len + horizon]
        dpose_mask_future = targets["dpose_mask"][:, history_len : history_len + horizon]
        gt_joints_future = targets["joints"][:, history_len : history_len + horizon]
        gt_verts_future = None if targets["verts"] is None else targets["verts"][:, history_len : history_len + horizon]

        pred_root_future = rollout["root_future"]
        pred_pose_future = rollout["pose_future"]
        pred_dpose_future = rollout["dpose_future"]
        pred_delta_p_future = rollout["delta_p_future"]
        pred_nu_future = rollout["nu_interval"]
        pred_omega_future = rollout["omega_interval"]

        pred_shape_future = targets["anchor_shape"].unsqueeze(1).expand(-1, horizon, -1)
        pred_joints_future, pred_verts_future = self._mano_forward(
            pred_pose_future,
            pred_root_future,
            pred_delta_p_future,
            pred_shape_future,
            mano_right,
            return_verts=need_verts,
        )

        gt_pose0 = targets["pose"][:, anchor_idx]
        vel_valid = ((targets["mask"][:, anchor_idx] > 0) & (targets["mask"][:, anchor_idx - 1] > 0)).to(gt_pose0.dtype)
        gt_nu0 = targets["root_nu"][:, anchor_idx]
        gt_omega0 = targets["root_omega"][:, anchor_idx]

        init_pose_loss = self._pose_geodesic(init_state["pose0"].unsqueeze(1), gt_pose0.unsqueeze(1)).squeeze(1)
        init_state_weight = (targets["mask"][:, anchor_idx] > 0).to(gt_pose0.dtype)
        init_state_loss_per_sample = weighted_mean_per_sample(init_pose_loss.unsqueeze(1), init_state_weight.unsqueeze(1))
        init_state_loss = weighted_mean(init_pose_loss, init_state_weight)

        init_vel_error = (
            smooth_l1_feature_mean(init_state["nu0"], gt_nu0, dim=-1)
            + smooth_l1_feature_mean(init_state["omega0"], gt_omega0, dim=-1)
        )
        init_vel_loss_per_sample = weighted_mean_per_sample(init_vel_error.unsqueeze(1), vel_valid.unsqueeze(1))
        init_vel_loss = weighted_mean(init_vel_error, vel_valid)

        root_rot_error = geodesic_distance_from_matrices(
            rot6d_rowmajor_to_matrix(pred_root_future.reshape(-1, 6)).reshape(-1, horizon, 3, 3),
            rot6d_rowmajor_to_matrix(gt_root_future.reshape(-1, 6)).reshape(-1, horizon, 3, 3),
        )
        root_rot_loss_per_sample = weighted_mean_per_sample(root_rot_error, future_weights)
        root_rot_loss = weighted_mean(root_rot_error, future_weights)
        pose_error = self._pose_geodesic(pred_pose_future, gt_pose_future)
        pose_loss_per_sample = weighted_mean_per_sample(pose_error, future_weights)
        pose_loss = weighted_mean(pose_error, future_weights)
        delta_p_error = smooth_l1_feature_mean(pred_delta_p_future, gt_delta_p_future, dim=-1)
        delta_p_loss_per_sample = weighted_mean_per_sample(delta_p_error, future_weights)
        delta_p_loss = weighted_mean(delta_p_error, future_weights)
        nu_error = smooth_l1_feature_mean(pred_nu_future, gt_nu_future, dim=-1)
        nu_loss_per_sample = weighted_mean_per_sample(nu_error, future_weights)
        nu_loss = weighted_mean(nu_error, future_weights)
        omega_error = smooth_l1_feature_mean(pred_omega_future, gt_omega_future, dim=-1)
        omega_loss_per_sample = weighted_mean_per_sample(omega_error, future_weights)
        omega_loss = weighted_mean(omega_error, future_weights)
        dpose_weights = future_discount ** torch.arange(horizon, device=dpose_mask_future.device, dtype=dpose_mask_future.dtype)
        dpose_weights = dpose_weights.view(1, -1) * dpose_mask_future
        dpose_error = smooth_l1_feature_mean(pred_dpose_future, gt_dpose_future, dim=-1)
        dpose_loss_per_sample = weighted_mean_per_sample(dpose_error, dpose_weights)
        dpose_loss = weighted_mean(dpose_error, dpose_weights)

        joint_error = smooth_l1_feature_mean(pred_joints_future, gt_joints_future, dim=(-1, -2))
        joint_loss_per_sample = weighted_mean_per_sample(joint_error, future_weights)
        joint_loss = weighted_mean(joint_error, future_weights)

        vert_loss_per_sample = torch.zeros(gt_pose0.shape[0], device=pred_pose_future.device, dtype=pred_pose_future.dtype)
        vert_loss = pred_pose_future.new_tensor(0.0)
        if need_verts and pred_verts_future is not None and gt_verts_future is not None:
            vert_loss_per_sample = weighted_mean_per_sample(
                smooth_l1_feature_mean(pred_verts_future, gt_verts_future, dim=(-1, -2)),
                future_weights,
            )
            vert_loss = weighted_mean(
                smooth_l1_feature_mean(pred_verts_future, gt_verts_future, dim=(-1, -2)),
                future_weights,
            )

        pred_path_root = torch.cat([init_state["root0_6d"].unsqueeze(1), pred_root_future], dim=1)
        pred_path_pose = torch.cat([init_state["pose0"].unsqueeze(1), pred_pose_future], dim=1)
        pred_path_trans = torch.cat([torch.zeros_like(pred_delta_p_future[:, :1]), pred_delta_p_future], dim=1)

        gt_path_pose = targets["pose"][:, anchor_idx : anchor_idx + horizon + 1]
        gt_path_trans = targets["trans_rel"][:, anchor_idx : anchor_idx + horizon + 1]
        gt_path_joints = targets["joints"][:, anchor_idx : anchor_idx + horizon + 1]
        path_mask = targets["mask"][:, anchor_idx : anchor_idx + horizon + 1]
        future_dt = targets["future_dt"]
        pred_visual_path_joints = torch.cat([gt_path_joints[:, :1], pred_joints_future], dim=1)

        # Keep this consistency term fully differentiable through both decoder branches:
        # pred_pose shapes the SO(3) finite-difference target, and pred_dpose must match it.
        dpose_from_pose, dpose_cons_mask = self._pose_angular_velocity_pairs(pred_path_pose, future_dt, path_mask)
        dpose_cons_weights = future_discount ** torch.arange(horizon, device=dpose_cons_mask.device, dtype=dpose_cons_mask.dtype)
        dpose_cons_weights = dpose_cons_weights.view(1, -1) * dpose_cons_mask
        dpose_cons_error = smooth_l1_feature_mean(pred_dpose_future, dpose_from_pose, dim=-1)
        dpose_cons_loss_per_sample = weighted_mean_per_sample(dpose_cons_error, dpose_cons_weights)
        dpose_cons_loss = weighted_mean(dpose_cons_error, dpose_cons_weights)

        pred_pose_vel, pose_pair_mask = path_differences(pred_path_pose, path_mask, future_dt)
        gt_pose_vel, _ = path_differences(gt_path_pose, path_mask, future_dt)
        pred_trans_vel, trans_pair_mask = path_differences(pred_path_trans, path_mask, future_dt)
        gt_trans_vel, _ = path_differences(gt_path_trans, path_mask, future_dt)
        pred_joint_vel, joint_pair_mask = path_differences(pred_visual_path_joints, path_mask, future_dt)
        gt_joint_vel, _ = path_differences(gt_path_joints, path_mask, future_dt)
        pred_joint_speed = torch.linalg.norm(pred_joint_vel, dim=-1).mean(dim=-1)
        gt_joint_speed = torch.linalg.norm(gt_joint_vel, dim=-1).mean(dim=-1)
        speed_mag_error = F.smooth_l1_loss(
            pred_joint_speed,
            gt_joint_speed,
            reduction="none",
            beta=0.001,
        )
        speed_mag_loss_per_sample = weighted_mean_per_sample(speed_mag_error, joint_pair_mask)
        speed_mag_loss = weighted_mean(speed_mag_error, joint_pair_mask)

        boundary_dt = future_dt[:, :1].clamp_min(1e-6)
        boundary_mask = path_mask[:, :2]
        boundary_valid = ((boundary_mask[:, 0] > 0) & (boundary_mask[:, 1] > 0)).to(pred_joints_future.dtype)
        pred_boundary_joint_vel = (pred_joints_future[:, 0] - gt_path_joints[:, 0]) / boundary_dt.unsqueeze(-1)
        gt_boundary_joint_vel = (gt_joints_future[:, 0] - gt_path_joints[:, 0]) / boundary_dt.unsqueeze(-1)
        boundary_vel_error = smooth_l1_feature_mean(
            pred_boundary_joint_vel,
            gt_boundary_joint_vel,
            dim=(-1, -2),
        )
        boundary_vel_loss_per_sample = weighted_mean_per_sample(
            boundary_vel_error.unsqueeze(1),
            boundary_valid.unsqueeze(1),
        )
        boundary_vel_loss = weighted_mean(boundary_vel_error, boundary_valid)

        vel_loss_per_sample = (
            weighted_mean_per_sample(smooth_l1_feature_mean(pred_pose_vel, gt_pose_vel, dim=-1), pose_pair_mask)
            + weighted_mean_per_sample(smooth_l1_feature_mean(pred_trans_vel, gt_trans_vel, dim=-1), trans_pair_mask)
            + weighted_mean_per_sample(smooth_l1_feature_mean(pred_joint_vel, gt_joint_vel, dim=(-1, -2)), joint_pair_mask)
        )
        vel_loss = (
            weighted_mean(smooth_l1_feature_mean(pred_pose_vel, gt_pose_vel, dim=-1), pose_pair_mask)
            + weighted_mean(smooth_l1_feature_mean(pred_trans_vel, gt_trans_vel, dim=-1), trans_pair_mask)
            + weighted_mean(smooth_l1_feature_mean(pred_joint_vel, gt_joint_vel, dim=(-1, -2)), joint_pair_mask)
        )

        kl_z_per_sample = self._kl_z_loss_per_sample(stats)
        kl_z = kl_z_per_sample.mean()
        kl_w = self._kl_w_loss()
        inst_KL_per_sample = self._inst_kl_loss_per_sample(
            rollout=rollout,
            targets=targets,
            history_len=history_len,
            horizon=horizon,
            future_discount=future_discount,
        )
        inst_KL = inst_KL_per_sample.mean()
        kl_loss = kl_z + kl_w + inst_KL

        return {
            "init_state_loss": init_state_loss,
            "init_state_loss_per_sample": init_state_loss_per_sample,
            "init_vel_loss": init_vel_loss,
            "init_vel_loss_per_sample": init_vel_loss_per_sample,
            "delta_p_loss": delta_p_loss,
            "delta_p_loss_per_sample": delta_p_loss_per_sample,
            "root_rot_loss": root_rot_loss,
            "root_rot_loss_per_sample": root_rot_loss_per_sample,
            "nu_loss": nu_loss,
            "nu_loss_per_sample": nu_loss_per_sample,
            "omega_loss": omega_loss,
            "omega_loss_per_sample": omega_loss_per_sample,
            "dpose_loss": dpose_loss,
            "dpose_loss_per_sample": dpose_loss_per_sample,
            "dpose_cons_loss": dpose_cons_loss,
            "dpose_cons_loss_per_sample": dpose_cons_loss_per_sample,
            "pose_loss": pose_loss,
            "pose_loss_per_sample": pose_loss_per_sample,
            "joint_loss": joint_loss,
            "joint_loss_per_sample": joint_loss_per_sample,
            "vert_loss": vert_loss,
            "vert_loss_per_sample": vert_loss_per_sample,
            "boundary_vel_loss": boundary_vel_loss,
            "boundary_vel_loss_per_sample": boundary_vel_loss_per_sample,
            "speed_mag_loss": speed_mag_loss,
            "speed_mag_loss_per_sample": speed_mag_loss_per_sample,
            "vel_loss": vel_loss,
            "vel_loss_per_sample": vel_loss_per_sample,
            "kl_z": kl_z,
            "kl_z_per_sample": kl_z_per_sample,
            "kl_w": kl_w,
            "inst_KL": inst_KL,
            "inst_KL_per_sample": inst_KL_per_sample,
            "kl_loss": kl_loss,
            "future_mask_mean": future_mask.mean(),
            "pred_joint_error": joint_loss.detach(),
            "pred_motion": torch.cat([pred_root_future, pred_pose_future, pred_delta_p_future], dim=-1),
            "pred_dpose": pred_dpose_future,
            "dpose_gt": gt_dpose_future,
            "dpose_mask": dpose_mask_future,
            "pred_joints_future": pred_joints_future,
            "gt_joints_future": gt_joints_future,
            "future_mask": future_mask,
        }

    def _prepare_forward_context(
        self,
        batch: Dict[str, torch.Tensor],
        mano_right: SMPLlayer,
        history_len: int,
        horizon: int,
        need_verts: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        total_len = history_len + horizon

        pose = batch["pose"]
        rh = batch["Rh"]
        th = batch["Th"]
        shape = batch["shape"]
        mask = batch["mask"]
        times = batch["times"]
        actual_len = int(pose.shape[1])
        if actual_len != total_len:
            raise ValueError(
                f"Expected batch sequence length {total_len} (= history_len {history_len} + horizon {horizon}), "
                f"got {actual_len}. Ensure the training window sampler passes exactly the required length."
            )

        targets = self._build_local_targets(
            pose=pose,
            rh=rh,
            th=th,
            shape=shape,
            mask=mask,
            history_len=history_len,
            mano_right=mano_right,
            need_verts=need_verts,
        )
        targets["pose"] = pose
        targets["times"] = times
        targets["root_nu"], targets["root_omega"] = self._root_velocity_targets(
            targets["root_rel_mat"],
            targets["trans_rel"],
            mask,
            times,
        )
        full_dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-6)
        targets["dpose"], targets["dpose_mask"] = self._pose_angular_velocity_targets(
            pose=pose,
            dt=full_dt,
            valid=mask,
        )
        history = self._history_features(
            pose=pose,
            joints_rel=targets["joints_rel"],
            joint_delta=targets["joint_delta"],
            mask=mask,
            times=times,
            history_len=history_len,
        )
        targets["future_dt"] = (
            times[:, history_len : history_len + horizon] - times[:, history_len - 1 : history_len + horizon - 1]
        ).clamp_min(1e-6)

        stats = self._encode_initial_distribution(
            history_feature=history["feature"],
        )
        return targets, stats

    def forward_samples(
        self,
        batch: Dict[str, torch.Tensor],
        mano_right: SMPLlayer,
        num_samples: int,
        history_len: Optional[int] = None,
        horizon: Optional[int] = None,
        sample: bool = True,
        method: str = "midpoint",
        future_discount: float = 1.0,
        need_verts: bool = False,
        sample_dynamics: Optional[bool] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        history_len = self.history_len if history_len is None else history_len
        horizon = self.horizon if horizon is None else horizon
        sample_dynamics = sample if sample_dynamics is None else sample_dynamics
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        targets, stats = self._prepare_forward_context(
            batch=batch,
            mano_right=mano_right,
            history_len=history_len,
            horizon=horizon,
            need_verts=need_verts,
        )
        outputs = []
        for _ in range(num_samples):
            init_state = self._sample_initial_state(stats, sample=sample)
            rollout = self._rollout(
                init_state,
                future_dt=targets["future_dt"],
                method=method,
                sample_dynamics=sample_dynamics,
            )
            outputs.append(
                self._compute_losses(
                    init_state=init_state,
                    stats=stats,
                    rollout=rollout,
                    targets=targets,
                    mano_right=mano_right,
                    history_len=history_len,
                    horizon=horizon,
                    future_discount=future_discount,
                    need_verts=need_verts,
                )
            )
        return outputs

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mano_right: SMPLlayer,
        history_len: Optional[int] = None,
        horizon: Optional[int] = None,
        sample: bool = True,
        method: str = "midpoint",
        future_discount: float = 1.0,
        need_verts: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        history_len = self.history_len if history_len is None else history_len
        horizon = self.horizon if horizon is None else horizon
        targets, stats = self._prepare_forward_context(
            batch=batch,
            mano_right=mano_right,
            history_len=history_len,
            horizon=horizon,
            need_verts=need_verts,
        )
        init_state = self._sample_initial_state(stats, sample=sample)
        rollout = self._rollout(
            init_state,
            future_dt=targets["future_dt"],
            method=method,
            sample_dynamics=sample,
        )
        return self._compute_losses(
            init_state=init_state,
            stats=stats,
            rollout=rollout,
            targets=targets,
            mano_right=mano_right,
            history_len=history_len,
            horizon=horizon,
            future_discount=future_discount,
            need_verts=need_verts,
        )

    def mean_rec(
        self,
        X: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor = None,
        shape: torch.Tensor = None,
        pose: torch.Tensor = None,
        Rh: torch.Tensor = None,
        Th: torch.Tensor = None,
        mano_right: SMPLlayer = None,
        method: str = "midpoint",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del X
        if any(v is None for v in [mask, shape, pose, Rh, Th, mano_right]):
            raise ValueError("mask, shape, pose, Rh, Th, and mano_right are required.")
        outputs = self.forward(
            {
                "mask": mask,
                "shape": shape,
                "pose": pose,
                "Rh": Rh,
                "Th": Th,
                "times": times,
            },
            mano_right=mano_right,
            history_len=self.history_len,
            horizon=self.horizon,
            sample=False,
            method=method,
            future_discount=1.0,
            need_verts=False,
        )
        if outputs is None:
            raise RuntimeError("No valid validation windows available for mean reconstruction.")
        joint_error = weighted_mean(
            torch.linalg.norm(outputs["pred_joints_future"] - outputs["gt_joints_future"], dim=-1).mean(dim=-1),
            outputs["future_mask"],
        )
        return outputs["pred_motion"], joint_error

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("mano_right.")}
        return super().load_state_dict(filtered, strict=strict)


def build_mano_right_layer(mano_model_path: str = None) -> SMPLlayer:
    if mano_model_path is None:
        mano_model_path = Path(__file__).resolve().parent / "EasyMocap" / "data" / "smplx"
    mano_model_path = str(mano_model_path)
    return SMPLlayer(
        os.path.join(mano_model_path, "smplh", "MANO_RIGHT.pkl"),
        model_type="mano",
        gender="neutral",
        device=device,
        regressor_path=os.path.join(mano_model_path, "J_regressor_mano_RIGHT.txt"),
        num_pca_comps=6,
        use_pose_blending=True,
        use_shape_blending=True,
        use_pca=False,
        use_flat_mean=False,
    )


def build_dataloaders(args):
    train_text_file = args.train_text_file or args.text_file
    val_text_file = args.val_text_file or args.text_file
    train_split = "all" if args.train_text_file is not None else "train"
    val_split = "all" if args.val_text_file is not None else "val"
    train_dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        split=train_split,
        text_file=train_text_file,
        random_mask=args.random_mask,
        random_mask_prob=args.random_mask_prob,
        fps=args.fps,
        max_sequences=args.max_sequences,
        history_len=args.history_len,
        horizon=min(args.warmup_horizon, args.horizon),
        time_stride_aug_max=args.time_stride_aug_max,
    )
    val_dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        split=val_split,
        text_file=val_text_file,
        random_mask=False,
        fps=args.fps,
        max_sequences=args.max_sequences,
        history_len=args.history_len,
        horizon=args.horizon,
        time_stride_aug_max=1,
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_dataset, val_dataset, train_loader, val_loader


def extract_all_subsequences(
    batch: Dict[str, torch.Tensor],
    subseq_len: int,
    stride: int,
    anchor_offset: int,
) -> Optional[Dict[str, torch.Tensor]]:
    total_len = int(batch["pose"].shape[1])
    if total_len < subseq_len:
        return None

    starts = []
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
    return {key: value.to(target_device) if torch.is_tensor(value) else value for key, value in batch.items()}


def scheduled_horizon(epoch: int, args) -> int:
    warmup_horizon = min(args.warmup_horizon, args.horizon)
    if epoch < args.warmup_epochs:
        return warmup_horizon
    if args.curriculum_epochs <= 0 or warmup_horizon >= args.horizon:
        return args.horizon
    progress = min(epoch - args.warmup_epochs + 1, args.curriculum_epochs)
    ratio = progress / args.curriculum_epochs
    scheduled = warmup_horizon + math.ceil((args.horizon - warmup_horizon) * ratio)
    return min(scheduled, args.horizon)


RECON_LOSS_KEYS = (
    "init_state_loss",
    "init_vel_loss",
    "delta_p_loss",
    "root_rot_loss",
    "nu_loss",
    "omega_loss",
    "dpose_loss",
    "dpose_cons_loss",
    "pose_loss",
    "joint_loss",
    "vert_loss",
    "boundary_vel_loss",
    "speed_mag_loss",
    "vel_loss",
)

RECON_LOSS_ARG_NAMES = {
    "init_state_loss": "lambda_init_state",
    "init_vel_loss": "lambda_init_vel",
    "delta_p_loss": "lambda_delta_p",
    "root_rot_loss": "lambda_root_rot",
    "nu_loss": "lambda_nu",
    "omega_loss": "lambda_omega",
    "dpose_loss": "lambda_dpose",
    "dpose_cons_loss": "lambda_dpose_cons",
    "pose_loss": "lambda_pose",
    "joint_loss": "lambda_joint",
    "vert_loss": "lambda_vert",
    "boundary_vel_loss": "lambda_boundary_vel",
    "speed_mag_loss": "lambda_speed_mag",
    "vel_loss": "lambda_vel",
}


def compute_reconstruction_nll(losses: Dict[str, torch.Tensor], args) -> torch.Tensor:
    total = next(iter(losses.values())).new_tensor(0.0)
    for key in RECON_LOSS_KEYS:
        total = total + getattr(args, RECON_LOSS_ARG_NAMES[key]) * losses[key]
    return total


def compute_reconstruction_nll_per_sample(losses: Dict[str, torch.Tensor], args) -> torch.Tensor:
    total = None
    for key in RECON_LOSS_KEYS:
        value = getattr(args, RECON_LOSS_ARG_NAMES[key]) * losses[f"{key}_per_sample"]
        total = value if total is None else total + value
    if total is None:
        raise RuntimeError("No reconstruction loss terms were configured.")
    return total


def compute_elbo_terms(losses: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    recon_nll = compute_reconstruction_nll(losses, args)
    recon_lhood = -recon_nll
    kl_term = args.beta_kl * (losses["kl_z"] + losses["kl_w"] + losses["inst_KL"])
    elbo = recon_lhood - kl_term
    return {
        "recon_nll": recon_nll,
        "recon_lhood": recon_lhood,
        "kl_term": kl_term,
        "elbo": elbo,
        "loss": -elbo,
    }


def compute_best_of_k_elbo_terms(
    sample_losses: List[Dict[str, torch.Tensor]],
    args,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not sample_losses:
        raise ValueError("sample_losses must contain at least one sample.")

    recon_per_sample = torch.stack(
        [compute_reconstruction_nll_per_sample(losses, args) for losses in sample_losses],
        dim=0,
    )
    best_indices = torch.argmin(recon_per_sample.detach(), dim=0)
    gather_index = best_indices.view(1, -1)
    best_recon_per_sample = recon_per_sample.gather(0, gather_index).squeeze(0)
    mean_recon_per_sample = recon_per_sample.mean(dim=0)

    combined: Dict[str, torch.Tensor] = {}
    for key in RECON_LOSS_KEYS:
        stacked = torch.stack([losses[f"{key}_per_sample"] for losses in sample_losses], dim=0)
        selected = stacked.gather(0, gather_index).squeeze(0)
        combined[f"{key}_per_sample"] = selected
        combined[key] = selected.mean()

    kl_z = torch.stack([losses["kl_z"] for losses in sample_losses]).mean()
    kl_w = torch.stack([losses["kl_w"] for losses in sample_losses]).mean()
    inst_kl = torch.stack([losses["inst_KL"] for losses in sample_losses]).mean()
    combined["kl_z"] = kl_z
    combined["kl_w"] = kl_w
    combined["inst_KL"] = inst_kl
    combined["kl_loss"] = kl_z + kl_w + inst_kl
    combined["future_mask_mean"] = torch.stack([losses["future_mask_mean"] for losses in sample_losses]).mean()

    best_recon_nll = best_recon_per_sample.mean()
    mean_recon_nll = mean_recon_per_sample.mean()
    aux_weight = float(args.best_of_k_aux_weight)
    recon_nll = best_recon_nll + aux_weight * mean_recon_nll
    recon_lhood = -recon_nll
    kl_term = args.beta_kl * (kl_z + kl_w + inst_kl)
    elbo = recon_lhood - kl_term

    combined["best_of_k_best_recon_nll"] = best_recon_nll.detach()
    combined["best_of_k_mean_recon_nll"] = mean_recon_nll.detach()
    combined["best_of_k_aux_weight"] = best_recon_nll.new_tensor(aux_weight)
    combined["best_of_k_mean_index"] = best_indices.to(best_recon_nll.dtype).mean()

    return combined, {
        "recon_nll": recon_nll,
        "recon_lhood": recon_lhood,
        "kl_term": kl_term,
        "elbo": elbo,
        "loss": -elbo,
        "best_recon_nll": best_recon_nll,
        "mean_recon_nll": mean_recon_nll,
        "aux_recon_nll": aux_weight * mean_recon_nll,
    }

def main():
    parser = argparse.ArgumentParser(description="Short-horizon ODE2VAE hand dynamics prior with translation increment rollout.")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--train-text-file", type=str, default=None)
    parser.add_argument("--val-text-file", type=str, default=None)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--warmup-horizon", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--curriculum-epochs", type=int, default=6)
    parser.add_argument("--time-stride-aug-max", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dynamics-damping", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-decay-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--method", type=str, default="midpoint")
    parser.add_argument("--random-mask", action="store_true")
    parser.add_argument("--random-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--mano-model-path", type=str, default=None)
    parser.add_argument("--future-discount", type=float, default=1.0)
    parser.add_argument("--lambda-init-state", type=float, default=1.0)
    parser.add_argument("--lambda-init-vel", type=float, default=0.5)
    parser.add_argument("--lambda-delta-p", type=float, default=2.0)
    parser.add_argument("--lambda-root-rot", type=float, default=1.0)
    parser.add_argument("--lambda-nu", type=float, default=1.0)
    parser.add_argument("--lambda-omega", type=float, default=1.0)
    parser.add_argument("--lambda-dpose", type=float, default=0.1)
    parser.add_argument("--lambda-dpose-cons", type=float, default=0.05)
    parser.add_argument("--lambda-pose", type=float, default=1.0)
    parser.add_argument("--lambda-joint", type=float, default=1.0)
    parser.add_argument("--lambda-vert", type=float, default=0.0)
    parser.add_argument("--lambda-boundary-vel", type=float, default=0.0)
    parser.add_argument("--lambda-speed-mag", type=float, default=0.0)
    parser.add_argument("--lambda-vel", type=float, default=0.25)
    parser.add_argument("--best-of-k", type=int, default=1, help="Number of stochastic trajectories sampled per batch for best-of-K training.")
    parser.add_argument(
        "--best-of-k-aux-weight",
        type=float,
        default=0.1,
        help="Weight for the auxiliary mean reconstruction loss over all K samples when --best-of-k > 1.",
    )
    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--logdir", type=str, default="runs/ode2vae_hand")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a saved checkpoint to resume training from.")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Only load model weights from --resume-checkpoint; keep optimizer, scheduler, epoch, and step fresh.",
    )
    args = parser.parse_args()

    if args.history_len < 2:
        raise ValueError("--history-len must be at least 2 so initial velocity supervision can be formed.")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.val_stride <= 0:
        raise ValueError("--val-stride must be positive.")
    if args.val_every_steps < 0:
        raise ValueError("--val-every-steps must be non-negative.")
    if args.time_stride_aug_max <= 0:
        raise ValueError("--time-stride-aug-max must be positive.")
    if args.dynamics_damping < 0:
        raise ValueError("--dynamics-damping must be non-negative.")
    if args.best_of_k <= 0:
        raise ValueError("--best-of-k must be positive.")
    if args.best_of_k_aux_weight < 0:
        raise ValueError("--best-of-k-aux-weight must be non-negative.")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    logdir = Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    board_dir = logdir / "board"
    board_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = logdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "log.txt"
    best_model_path = logdir / "best_model.pt"
    writer = SummaryWriter(log_dir=str(board_dir))
    log_mode = "a" if args.resume_checkpoint else "w"
    log_file = log_path.open(log_mode, encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    mano_right = build_mano_right_layer(args.mano_model_path)
    model = ODE2VAEHand(
        input_dim=train_dataset.motion_dim,
        q=args.q,
        hidden_dim=args.hidden_dim,
        history_len=args.history_len,
        horizon=args.horizon,
        dynamics_damping=args.dynamics_damping,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.min_lr,
    )

    global_step = 0
    start_epoch = 0
    best_val_joint_err = float("inf")
    best_epoch = -1
    last_val_joint_err = None

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume-checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
            checkpoint = {}
        model.load_state_dict(model_state_dict)

        if not args.resume_model_only:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = int(checkpoint.get("global_step", 0))
            start_epoch = int(checkpoint.get("next_epoch", int(checkpoint.get("epoch", -1)) + 1))
            best_val_joint_err = float(checkpoint.get("best_val_joint_err", checkpoint.get("val_joint_err", float("inf"))))
            best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1)))
            if "val_joint_err" in checkpoint:
                last_val_joint_err = float(checkpoint["val_joint_err"])

        log(
            f"Resumed from {resume_path} "
            f"(model_only={args.resume_model_only}, start_epoch={start_epoch}, global_step={global_step}, "
            f"best_val_joint_err={best_val_joint_err:.6f}, best_epoch={best_epoch})"
        )

    def build_checkpoint(epoch: int, val_joint_err: float) -> dict:
        current_lr = optimizer.param_groups[0]["lr"]
        return {
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "global_step": global_step,
            "val_joint_err": val_joint_err,
            "best_val_joint_err": best_val_joint_err,
            "best_epoch": best_epoch,
            "lr": current_lr,
            "optimizer_lrs": [group["lr"] for group in optimizer.param_groups],
            "args": vars(args).copy(),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

    def run_validation(epoch: int, trigger_step: int) -> float:
        model.eval()
        val_joint_err_sum = 0.0
        val_window_count = 0
        val_start = time.time()
        with torch.no_grad():
            for batch in val_loader:
                subsequences = extract_all_subsequences(
                    batch,
                    subseq_len=args.history_len + args.horizon,
                    stride=args.val_stride,
                    anchor_offset=args.history_len - 1,
                )
                if subsequences is None:
                    continue
                num_windows = int(subsequences["pose"].shape[0])
                for start in range(0, num_windows, args.batch_size):
                    end = min(start + args.batch_size, num_windows)
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
                    _, val_joint_err = model.mean_rec(
                        clip_batch["motion"],
                        clip_batch["times"],
                        mask=clip_batch["mask"],
                        shape=clip_batch["shape"],
                        pose=clip_batch["pose"],
                        Rh=clip_batch["Rh"],
                        Th=clip_batch["Th"],
                        mano_right=mano_right,
                        method=args.method,
                    )
                    window_count = end - start
                    val_joint_err_sum += val_joint_err.item() * window_count
                    val_window_count += window_count

        if val_window_count == 0:
            raise RuntimeError("Validation produced zero valid windows. Check the split or window configuration.")

        val_joint_err = val_joint_err_sum / val_window_count
        scheduler.step(val_joint_err)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("val/joint_err", val_joint_err, trigger_step)
        writer.add_scalar("train/lr_val", current_lr, trigger_step)
        writer.add_scalar("val/runtime_sec", time.time() - val_start, trigger_step)
        return val_joint_err

    try:
        log("\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())))
        writer.add_text("run/config", "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())))

        for epoch in range(start_epoch, args.epochs):
            model.train()
            current_horizon = scheduled_horizon(epoch, args)
            if current_horizon != train_dataset.current_horizon:
                train_dataset.set_training_horizon(current_horizon)
            epoch_start = time.time()
            avg_batch_time = None
            num_batches = len(train_loader)
            for step, raw_batch in enumerate(train_loader):
                batch_start = time.time()
                batch = move_tensor_dict_to_device(raw_batch, device)

                if args.best_of_k > 1:
                    sample_outputs = model.forward_samples(
                        batch,
                        mano_right=mano_right,
                        num_samples=args.best_of_k,
                        history_len=args.history_len,
                        horizon=current_horizon,
                        sample=True,
                        method=args.method,
                        future_discount=args.future_discount,
                        need_verts=args.lambda_vert > 0.0,
                        sample_dynamics=False,
                    )
                    outputs, elbo_terms = compute_best_of_k_elbo_terms(sample_outputs, args)
                else:
                    outputs = model(
                        batch,
                        mano_right=mano_right,
                        history_len=args.history_len,
                        horizon=current_horizon,
                        sample=True,
                        method=args.method,
                        future_discount=args.future_discount,
                        need_verts=args.lambda_vert > 0.0,
                    )
                    if outputs is None:
                        continue
                    elbo_terms = compute_elbo_terms(outputs, args)
                loss = elbo_terms["loss"]
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_time = time.time() - batch_start
                avg_batch_time = batch_time if avg_batch_time is None else 0.9 * avg_batch_time + 0.1 * batch_time
                remaining_batches = max(num_batches - step - 1, 0)
                eta_seconds = remaining_batches * avg_batch_time
                epoch_elapsed = time.time() - epoch_start

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/elbo", elbo_terms["elbo"].item(), global_step)
                writer.add_scalar("train/recon_lhood", elbo_terms["recon_lhood"].item(), global_step)
                writer.add_scalar("train/recon_nll", elbo_terms["recon_nll"].item(), global_step)
                writer.add_scalar("train/kl_term", elbo_terms["kl_term"].item(), global_step)
                writer.add_scalar("train/init_state_loss", outputs["init_state_loss"].item(), global_step)
                writer.add_scalar("train/init_vel_loss", outputs["init_vel_loss"].item(), global_step)
                writer.add_scalar("train/delta_p_loss", outputs["delta_p_loss"].item(), global_step)
                writer.add_scalar("train/root_rot_loss", outputs["root_rot_loss"].item(), global_step)
                writer.add_scalar("train/nu_loss", outputs["nu_loss"].item(), global_step)
                writer.add_scalar("train/omega_loss", outputs["omega_loss"].item(), global_step)
                writer.add_scalar("train/loss_dpose", outputs["dpose_loss"].item(), global_step)
                writer.add_scalar("train/loss_dpose_cons", outputs["dpose_cons_loss"].item(), global_step)
                writer.add_scalar("train/pose_loss", outputs["pose_loss"].item(), global_step)
                writer.add_scalar("train/joint_loss", outputs["joint_loss"].item(), global_step)
                writer.add_scalar("train/vert_loss", outputs["vert_loss"].item(), global_step)
                writer.add_scalar("train/boundary_vel_loss", outputs["boundary_vel_loss"].item(), global_step)
                writer.add_scalar("train/speed_mag_loss", outputs["speed_mag_loss"].item(), global_step)
                writer.add_scalar("train/vel_loss", outputs["vel_loss"].item(), global_step)
                writer.add_scalar("train/kl_z", outputs["kl_z"].item(), global_step)
                writer.add_scalar("train/kl_w", outputs["kl_w"].item(), global_step)
                writer.add_scalar("train/inst_KL", outputs["inst_KL"].item(), global_step)
                writer.add_scalar("train/kl_loss", outputs["kl_loss"].item(), global_step)
                writer.add_scalar("train/future_mask_mean", outputs["future_mask_mean"].item(), global_step)
                writer.add_scalar("train/horizon", current_horizon, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/batch_time", batch_time, global_step)
                if args.best_of_k > 1:
                    writer.add_scalar("train/best_of_k_best_recon_nll", elbo_terms["best_recon_nll"].item(), global_step)
                    writer.add_scalar("train/best_of_k_mean_recon_nll", elbo_terms["mean_recon_nll"].item(), global_step)
                    writer.add_scalar("train/best_of_k_aux_recon_nll", elbo_terms["aux_recon_nll"].item(), global_step)
                    writer.add_scalar("train/best_of_k_mean_index", outputs["best_of_k_mean_index"].item(), global_step)

                best_of_k_msg = ""
                if args.best_of_k > 1:
                    best_of_k_msg = (
                        f" K:{args.best_of_k} best_rec:{elbo_terms['best_recon_nll'].item():7.4f} "
                        f"mean_rec:{elbo_terms['mean_recon_nll'].item():7.4f} "
                        f"aux:{elbo_terms['aux_recon_nll'].item():7.4f}"
                    )

                log(
                    f"Epoch:{epoch:03d} Step:{step:04d} H:{current_horizon:02d} "
                    f"loss:{loss.item():8.4f} elbo:{elbo_terms['elbo'].item():8.4f} "
                    f"lhood:{elbo_terms['recon_lhood'].item():8.4f} dp:{outputs['delta_p_loss'].item():7.4f} "
                    f"root:{outputs['root_rot_loss'].item():7.4f} pose:{outputs['pose_loss'].item():7.4f} "
                    f"nu:{outputs['nu_loss'].item():7.4f} omega:{outputs['omega_loss'].item():7.4f} "
                    f"loss_dpose:{outputs['dpose_loss'].item():7.4f} loss_dpose_cons:{outputs['dpose_cons_loss'].item():7.4f} "
                    f"joint:{outputs['joint_loss'].item():7.4f} bvel:{outputs['boundary_vel_loss'].item():7.4f} "
                    f"smag:{outputs['speed_mag_loss'].item():7.4f} vel:{outputs['vel_loss'].item():7.4f} "
                    f"kl_z:{outputs['kl_z'].item():7.4f} kl_w:{outputs['kl_w'].item():7.4f} "
                    f"inst_KL:{outputs['inst_KL'].item():7.4f} kl_term:{elbo_terms['kl_term'].item():7.4f}"
                    f"{best_of_k_msg} "
                    f"batch_time:{batch_time:6.2f}s epoch_elapsed:{epoch_elapsed:7.2f}s "
                    f"epoch_eta:{eta_seconds:7.2f}s"
                )

                global_step += 1
                if args.val_every_steps > 0 and global_step % args.val_every_steps == 0:
                    val_joint_err = run_validation(epoch, global_step)
                    last_val_joint_err = val_joint_err
                    if val_joint_err < best_val_joint_err:
                        best_val_joint_err = val_joint_err
                        best_epoch = epoch
                    checkpoint = build_checkpoint(epoch, val_joint_err)
                    step_model_path = checkpoint_dir / f"step_{global_step:07d}.pt"
                    torch.save(checkpoint, step_model_path)
                    if best_val_joint_err == val_joint_err:
                        torch.save(checkpoint, best_model_path)
                        log(
                            f"Epoch:{epoch:03d} GlobalStep:{global_step:07d} "
                            f"Val joint_err:{val_joint_err:9.6f} lr:{optimizer.param_groups[0]['lr']:.6e} "
                            f"[saved to {step_model_path}] [best saved to {best_model_path}]"
                        )
                    else:
                        log(
                            f"Epoch:{epoch:03d} GlobalStep:{global_step:07d} "
                            f"Val joint_err:{val_joint_err:9.6f} lr:{optimizer.param_groups[0]['lr']:.6e} "
                            f"[saved to {step_model_path}] "
                            f"(best:{best_val_joint_err:9.6f} @ epoch {best_epoch:03d})"
                        )
                    model.train()

            writer.flush()

            if args.val_every_steps == 0:
                val_joint_err = run_validation(epoch, global_step)
                last_val_joint_err = val_joint_err
                if val_joint_err < best_val_joint_err:
                    best_val_joint_err = val_joint_err
                    best_epoch = epoch
                checkpoint = build_checkpoint(epoch, val_joint_err)
                epoch_model_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                torch.save(checkpoint, epoch_model_path)
                if best_val_joint_err == val_joint_err:
                    torch.save(checkpoint, best_model_path)
                    log(
                        f"Epoch:{epoch:03d} Val joint_err:{val_joint_err:9.6f} "
                        f"[saved to {epoch_model_path}] [best saved to {best_model_path}]"
                    )
                else:
                    log(
                        f"Epoch:{epoch:03d} Val joint_err:{val_joint_err:9.6f} "
                        f"[saved to {epoch_model_path}] (best:{best_val_joint_err:9.6f} @ epoch {best_epoch:03d})"
                    )

    finally:
        writer.close()
        if best_epoch >= 0:
            log(
                f"best epoch={best_epoch} "
                f"val_joint_err={best_val_joint_err:.6f} "
                f"model_path={best_model_path}"
            )
        else:
            last_val_str = "nan" if last_val_joint_err is None else f"{last_val_joint_err:.6f}"
            log(f"best epoch=-1 val_joint_err={last_val_str} model_path=None")
        log_file.close()


if __name__ == "__main__":
    main()
