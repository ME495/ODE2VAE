import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint
from scipy.spatial.transform import Rotation

from hand_dataset import GigaHandDataset
from torch_bnn import BNN
from easymocap.smplmodel.body_model import SMPLlayer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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


class StaticEncoder(nn.Module):
    def __init__(self, hidden_dim: int, q: int) -> None:
        super().__init__()
        self.fc_mean = nn.Linear(hidden_dim, q)
        self.fc_logv = nn.Linear(hidden_dim, q)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mean(h), self.fc_logv(h)


class VelocityEncoder(nn.Module):
    def __init__(self, hidden_dim: int, q: int, n_init_obs: int) -> None:
        super().__init__()
        self.n_init_obs = n_init_obs
        init_dim = n_init_obs * hidden_dim + n_init_obs
        self.net = nn.Sequential(
            nn.Linear(init_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, q)
        self.fc_logv = nn.Linear(hidden_dim, q)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mean(h), self.fc_logv(h)


class MLPDecoder(nn.Module):
    def __init__(self, q: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_content: torch.Tensor) -> torch.Tensor:
        return self.net(z_content)


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


class ODE2VAEHand(nn.Module):
    def __init__(
        self,
        input_dim: int,
        q: int = 16,
        hidden_dim: int = 256,
        n_init_obs: int = 4,
        use_global_rot: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.q = q
        self.n_init_obs = n_init_obs
        self.use_global_rot = use_global_rot
        self.frame_encoder = FrameEncoder(input_dim, hidden_dim)
        self.static_encoder = StaticEncoder(hidden_dim, q)
        self.velocity_encoder = VelocityEncoder(hidden_dim, q, n_init_obs)
        self.decoder = MLPDecoder(q, hidden_dim, input_dim)
        self.bnn = BNN(2 * q, q, n_hid_layers=2, n_hidden=hidden_dim, act="celu", layer_norm=True, bnn=True)
        self.beta = 1.0
        self.recon_logstd = nn.Parameter(torch.full((input_dim,), -1.5))

        self._zero_mean = torch.zeros(2 * q).to(device)
        self._eye_covar = torch.eye(2 * q).to(device)
        self.mvn = MultivariateNormal(self._zero_mean, self._eye_covar)

    def ode2vae_rhs(self, t, vs_logp, f):
        vs, logp = vs_logp
        q = vs.shape[1] // 2
        dv = f(vs)
        ds = vs[:, :q]
        dvs = torch.cat([dv, ds], dim=1)
        ddvi_dvi = torch.stack(
            [
                torch.autograd.grad(
                    dv[:, i],
                    vs,
                    torch.ones_like(dv[:, i]),
                    retain_graph=True,
                    create_graph=True,
                )[0][:, i]
                for i in range(q)
            ],
            dim=1,
        )
        tr_ddvi_dvi = ddvi_dvi.sum(dim=1)
        return dvs, -tr_ddvi_dvi

    def elbo(
        self,
        qz_m: torch.Tensor,
        qz_logv: torch.Tensor,
        zode_L: torch.Tensor,
        logpL: torch.Tensor,
        X: torch.Tensor,
        mask: torch.Tensor,
        XrecL: torch.Tensor,
        Ndata: int,
        qz_enc_m: torch.Tensor = None,
        qz_enc_logv: torch.Tensor = None,
        inst_mask: torch.Tensor = None,
    ):
        N, T, D = X.shape
        L = zode_L.shape[0]
        q = qz_m.shape[1] // 2

        log_pzt = self.mvn.log_prob(zode_L.contiguous().view(L * N * T, 2 * q)).view(L, N, T)
        kl_zt = logpL - log_pzt
        kl_z = kl_zt.sum(dim=2).mean(dim=0)
        kl_w = self.bnn.kl().sum()

        XL = X.unsqueeze(0).expand(L, -1, -1, -1)
        sigma = self.recon_logstd.exp().view(1, 1, 1, D) + 1e-4
        lhood_dim = Normal(XrecL, sigma).log_prob(XL)
        lhood_t = lhood_dim.sum(dim=3)
        maskL = mask.unsqueeze(0).expand(L, -1, -1)
        valid_counts = mask.sum(dim=1).clamp_min(1.0)
        lhood = (lhood_t * maskL).sum(dim=2) / valid_counts.unsqueeze(0)
        lhood = lhood.mean(dim=0)

        if qz_enc_m is not None:
            qz_enc_mL = qz_enc_m.repeat(L, 1)
            qz_enc_logvL = qz_enc_logv.repeat(L, 1)
            mean_ = qz_enc_mL.contiguous().view(-1)
            std_ = 1e-3 + qz_enc_logvL.exp().contiguous().view(-1)
            qenc_zt_ode = Normal(mean_, std_).log_prob(zode_L.contiguous().view(-1)).view(L, N, T, 2 * q)
            qenc_zt_ode = qenc_zt_ode.sum(dim=3)
            inst_enc_KL = logpL - qenc_zt_ode
            if inst_mask is None:
                inst_mask = mask
            inst_maskL = inst_mask.unsqueeze(0).expand(L, -1, -1)
            inst_counts = inst_mask.sum(dim=1).clamp_min(1.0)
            inst_enc_KL = (inst_enc_KL * inst_maskL).sum(dim=2) / inst_counts.unsqueeze(0)
            inst_enc_KL = inst_enc_KL.mean(dim=0)
            return Ndata * lhood.mean(), Ndata * kl_z.mean(), kl_w, Ndata * inst_enc_KL.mean()

        return Ndata * lhood.mean(), Ndata * kl_z.mean(), kl_w

    def _prepare_times(self, times: torch.Tensor, T: int, ref_device: torch.device) -> torch.Tensor:
        if times.ndim == 2:
            times = times[0]
        if times.ndim != 1 or times.shape[0] != T:
            raise ValueError(f"Expected times to have shape [T] or [N,T], got {tuple(times.shape)}")
        return times.to(ref_device)

    def _build_velocity_features(
        self,
        frame_h: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, T, H = frame_h.shape
        K = self.n_init_obs
        if start_positions.ndim == 1:
            start_positions = start_positions.unsqueeze(1)

        S = start_positions.shape[1]
        valid = mask > 0
        valid_long = valid.long()
        cum_valid = valid_long.cumsum(dim=1)

        start_counts = torch.gather(cum_valid, 1, start_positions)
        start_is_valid = torch.gather(valid_long, 1, start_positions)
        first_target = start_counts + (1 - start_is_valid)
        target_ordinals = first_target.unsqueeze(-1) + torch.arange(K, device=frame_h.device).view(1, 1, K)

        total_valid = cum_valid[:, -1:].unsqueeze(-1)
        full_window = target_ordinals[:, :, -1] <= total_valid.squeeze(-1)

        target_expanded = target_ordinals.unsqueeze(1)
        cum_expanded = cum_valid.unsqueeze(2).unsqueeze(3)
        valid_expanded = valid.unsqueeze(2).unsqueeze(3)
        first_match = ((cum_expanded >= target_expanded) & valid_expanded).long().argmax(dim=1)

        last_valid = (valid_long * torch.arange(T, device=frame_h.device).view(1, T)).max(dim=1).values
        last_valid = last_valid.view(N, 1, 1).expand_as(first_match)
        idx = torch.where(target_ordinals <= total_valid, first_match, last_valid)

        frame_h_expanded = frame_h.unsqueeze(1).expand(N, S, T, H)
        vel_h = torch.gather(frame_h_expanded, 2, idx.unsqueeze(-1).expand(N, S, K, H))
        time_grid = times.view(1, 1, T).expand(N, S, T)
        vel_t = torch.gather(time_grid, 2, idx)
        rel_t = vel_t - vel_t[:, :, :1]
        features = torch.cat([vel_h.reshape(N, S, K * H), rel_t.reshape(N, S, K)], dim=2)
        return features, full_window

    def _encode_initial_state(
        self, frame_h: torch.Tensor, X: torch.Tensor, mask: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, _, _ = X.shape
        valid = mask > 0
        has_valid = valid.any(dim=1)
        first_valid_idx = valid.float().argmax(dim=1).long()
        first_valid_idx = torch.where(
            has_valid,
            first_valid_idx,
            torch.zeros_like(first_valid_idx),
        )
        start_positions = first_valid_idx

        s0_h = frame_h[torch.arange(N, device=X.device), first_valid_idx]
        s0_mu, s0_logv = self.static_encoder(s0_h)
        v0_feat, _ = self._build_velocity_features(frame_h, mask, times, start_positions)
        v0_mu, v0_logv = self.velocity_encoder(v0_feat.squeeze(1))
        return torch.cat([v0_mu, s0_mu], dim=1), torch.cat([v0_logv, s0_logv], dim=1)

    def _encode_instant_state(
        self, frame_h: torch.Tensor, mask: torch.Tensor, times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, H = frame_h.shape
        s_mu, s_logv = self.static_encoder(frame_h.reshape(N * T, H))

        start_positions = torch.arange(T, device=frame_h.device).view(1, T).expand(N, T)
        all_v_feat, full_window = self._build_velocity_features(frame_h, mask, times, start_positions)
        inst_mask = ((mask > 0) & full_window).float()
        v_mu, v_logv = self.velocity_encoder(all_v_feat.reshape(N * T, -1))
        return torch.cat([v_mu, s_mu], dim=1), torch.cat([v_logv, s_logv], dim=1), inst_mask

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        Ndata: int,
        L: int = 1,
        inst_enc: bool = False,
        method: str = "dopri5",
    ):
        N, T, D = X.shape
        times = self._prepare_times(times, T, X.device)
        frame_h = self.frame_encoder(X.contiguous().view(N * T, D)).view(N, T, -1)

        # Match the original ODE2VAE factorization: v0 from the first K observations,
        # s0 from the first valid observation.
        qz0_m, qz0_logv = self._encode_initial_state(frame_h, X, mask, times)
        eps = torch.randn_like(qz0_m)
        z0 = qz0_m + eps * torch.exp(qz0_logv)
        logp0 = self.mvn.log_prob(eps)

        ztL, logpL = [], []
        for _ in range(L):
            f = self.bnn.draw_f()
            oderhs = lambda t, vs: self.ode2vae_rhs(t, vs, f)
            zt, logp = odeint(oderhs, (z0, logp0), times, method=method)
            ztL.append(zt.permute(1, 0, 2).unsqueeze(0))
            logpL.append(logp.permute(1, 0).unsqueeze(0))
        ztL = torch.cat(ztL, dim=0)
        logpL = torch.cat(logpL, dim=0)

        st_muL = ztL[:, :, :, self.q :]
        Xrec = self.decoder(st_muL.contiguous().view(L * N * T, self.q))
        Xrec = Xrec.view(L, N, T, D)

        if inst_enc:
            qz_enc_m, qz_enc_logv, inst_mask = self._encode_instant_state(frame_h, mask, times)
            lhood, kl_z, kl_w, inst_KL = self.elbo(
                qz0_m, qz0_logv, ztL, logpL, X, mask, Xrec, Ndata, qz_enc_m, qz_enc_logv, inst_mask
            )
            elbo = lhood - kl_z - inst_KL - self.beta * kl_w
        else:
            lhood, kl_z, kl_w = self.elbo(qz0_m, qz0_logv, ztL, logpL, X, mask, Xrec, Ndata)
            elbo = lhood - kl_z - self.beta * kl_w

        return Xrec, qz0_m, qz0_logv, ztL, elbo, lhood, kl_z, self.beta * kl_w

    @staticmethod
    def _rot6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 6)
        # `hand_dataset.py` stores the first two rotation-matrix columns after
        # slicing `rotmat[:, :2]` and flattening in NumPy/C row-major order:
        # [r00, r01, r10, r11, r20, r21]. Rebuild the two 3D column vectors
        # accordingly so GT MANO meshes match the official raw-axis-angle loader.
        a1 = torch.stack([x[:, 0], x[:, 2], x[:, 4]], dim=1)
        a2 = torch.stack([x[:, 1], x[:, 3], x[:, 5]], dim=1)
        a1 = F.normalize(a1, dim=1)
        b2 = F.normalize(a2 - (a1 * a2).sum(dim=1, keepdim=True) * a1, dim=1)
        b3 = torch.cross(a1, b2, dim=1)
        rotmat = torch.stack([a1, b2, b3], dim=-1).detach().cpu().numpy()
        axis = Rotation.from_matrix(rotmat).as_rotvec()
        return torch.from_numpy(axis).to(x.device, dtype=x.dtype)

    def _motion_to_mano_params(
        self,
        motion: torch.Tensor,
        shape: torch.Tensor,
        pose_ref: torch.Tensor,
        Rh_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, _ = motion.shape
        pose_dim = pose_ref.shape[-1]
        offset = 0
        if self.use_global_rot:
            Rh_6d = motion[:, :, :6]
            offset = 6
            Rh_axis = self._rot6d_to_axis_angle(Rh_6d.reshape(N * T, 6)).view(N, T, 3)
        else:
            Rh_axis = self._rot6d_to_axis_angle(Rh_ref.reshape(N * T, 6)).view(N, T, 3)
        pose_6d = motion[:, :, offset : offset + pose_dim]
        Th = motion[:, :, offset + pose_dim : offset + pose_dim + 3]
        pose_axis = self._rot6d_to_axis_angle(pose_6d.reshape(N * T, -1, 6)).view(N, T, -1, 3)
        pose_axis = pose_axis.reshape(N, T, -1)
        return (
            pose_axis.reshape(N * T, -1),
            Rh_axis.reshape(N * T, 3),
            Th.reshape(N * T, 3),
            shape.reshape(N * T, -1),
        )

    def _mano_joint_error(
        self,
        pred_motion: torch.Tensor,
        gt_pose: torch.Tensor,
        gt_Rh: torch.Tensor,
        gt_Th: torch.Tensor,
        gt_shape: torch.Tensor,
        mask: torch.Tensor,
        mano_right: SMPLlayer,
    ) -> torch.Tensor:
        N, T, _ = pred_motion.shape
        pred_pose, pred_Rh, pred_Th, pred_shape = self._motion_to_mano_params(
            pred_motion, gt_shape, gt_pose, gt_Rh
        )
        gt_pose_axis = self._rot6d_to_axis_angle(gt_pose.reshape(N * T, -1, 6)).view(N * T, -1, 3).reshape(N * T, -1)
        gt_Rh_axis = self._rot6d_to_axis_angle(gt_Rh.reshape(N * T, 6)).view(N * T, 3)
        gt_Th_flat = gt_Th.reshape(N * T, 3)
        gt_shape_flat = gt_shape.reshape(N * T, -1)

        pred_joints = mano_right(
            poses=pred_pose,
            shapes=pred_shape,
            Rh=pred_Rh,
            Th=pred_Th,
            return_verts=False,
            return_tensor=True,
        ).view(N, T, -1, 3)
        gt_joints = mano_right(
            poses=gt_pose_axis,
            shapes=gt_shape_flat,
            Rh=gt_Rh_axis,
            Th=gt_Th_flat,
            return_verts=False,
            return_tensor=True,
        ).view(N, T, -1, 3)
        joint_err = torch.norm(pred_joints - gt_joints, dim=3).mean(dim=2)
        return (joint_err * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

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
        method: str = "dopri5",
    ):
        N, T, D = X.shape
        times = self._prepare_times(times, T, X.device)
        if mask is None:
            mask = torch.ones(N, T, device=X.device)
        frame_h = self.frame_encoder(X.contiguous().view(N * T, D)).view(N, T, -1)
        qz0_m, _ = self._encode_initial_state(frame_h, X, mask, times)

        def ode2vae_mean_rhs(t, vs, f):
            q = vs.shape[1] // 2
            dv = f(vs)
            ds = vs[:, :q]
            return torch.cat([dv, ds], dim=1)

        f = self.bnn.draw_f(mean=True)
        odef = lambda t, vs: ode2vae_mean_rhs(t, vs, f)
        zt_mu = odeint(odef, qz0_m, times, method=method).permute(1, 0, 2)
        st_mu = zt_mu[:, :, self.q :]
        Xrec_mu = self.decoder(st_mu.contiguous().view(N * T, self.q)).view(N, T, D)
        if shape is None or pose is None or Rh is None or Th is None or mano_right is None:
            raise ValueError("shape, pose, Rh, Th, and mano_right are required to compute MANO joint error.")
        joint_error = self._mano_joint_error(Xrec_mu, pose, Rh, Th, shape, mask, mano_right)
        return Xrec_mu, joint_error.mean()

    def load_state_dict(self, state_dict, strict: bool = True):
        # Allow loading legacy checkpoints that embedded MANO buffers in the model state.
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("mano_right.")}
        return super().load_state_dict(filtered, strict=strict)


def build_dataloaders(args):
    train_text_file = args.train_text_file or args.text_file
    val_text_file = args.val_text_file or args.text_file
    train_split = "all" if args.train_text_file is not None else "train"
    val_split = "all" if args.val_text_file is not None else "val"
    train_dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        seq_len=args.seq_len,
        split=train_split,
        text_file=train_text_file,
        normalize_trans=args.normalize_trans,
        use_global_rot=not args.disable_global_rot,
        random_mask=args.random_mask,
        random_mask_prob=args.random_mask_prob,
        fps=args.fps,
        max_sequences=args.max_sequences,
    )
    val_dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        seq_len=args.seq_len,
        split=val_split,
        text_file=val_text_file,
        normalize_trans=args.normalize_trans,
        use_global_rot=not args.disable_global_rot,
        random_mask=False,
        fps=args.fps,
        max_sequences=args.max_sequences,
    )
    params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": args.num_workers}
    train_loader = data.DataLoader(train_dataset, **params)
    # Validation and test use full sequences, so batch_size=1 avoids collation errors
    # when clips have different lengths.
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_dataset, val_dataset, train_loader, val_loader


def extract_all_subsequences(
    batch: Dict[str, torch.Tensor],
    subseq_len: int,
    stride: int,
) -> Optional[Dict[str, torch.Tensor]]:
    total_len = int(batch["motion"].shape[1])
    if total_len < subseq_len:
        return None

    starts = []
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


def main():
    parser = argparse.ArgumentParser(description="Vector-sequence ODE2VAE for GigaHands MANO motion.")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--train-text-file", type=str, default=None)
    parser.add_argument("--val-text-file", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-init-obs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-decay-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--inst-enc", action="store_true")
    parser.add_argument("--normalize-trans", action="store_true")
    parser.add_argument("--disable-global-rot", action="store_true")
    parser.add_argument("--random-mask", action="store_true")
    parser.add_argument("--random-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--mano-model-path", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="runs/ode2vae_hand")
    args = parser.parse_args()
    if args.val_stride <= 0:
        raise ValueError("--val-stride must be positive")
    if args.val_every_steps < 0:
        raise ValueError("--val-every-steps must be non-negative")

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
    log_file = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

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
                subsequences = extract_all_subsequences(batch, args.seq_len, args.val_stride)
                if subsequences is None:
                    continue
                num_windows = int(subsequences["motion"].shape[0])
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
            raise RuntimeError("Validation produced zero windows. Check seq_len/val_stride or validation split.")

        val_joint_err = val_joint_err_sum / val_window_count
        scheduler.step(val_joint_err)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("val/joint_err", val_joint_err, trigger_step)
        writer.add_scalar("train/lr_val", current_lr, trigger_step)
        writer.add_scalar("val/runtime_sec", time.time() - val_start, trigger_step)
        return val_joint_err

    model = ODE2VAEHand(
        input_dim=train_dataset.motion_dim,
        q=args.q,
        hidden_dim=args.hidden_dim,
        n_init_obs=args.n_init_obs,
        use_global_rot=not args.disable_global_rot,
    ).to(device)
    mano_right = build_mano_right_layer(args.mano_model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.min_lr,
    )

    try:
        log("\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())))
        writer.add_text(
            "run/config",
            "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        )
        global_step = 0
        best_val_joint_err = float("inf")
        best_epoch = -1
        last_val_joint_err = None

        for epoch in range(args.epochs):
            model.train()
            # L = 1 if epoch < args.epochs // 2 else 5
            L = 1
            epoch_start = time.time()
            avg_batch_time = None
            num_batches = len(train_loader)
            for step, batch in enumerate(train_loader):
                batch_start = time.time()
                motion = batch["motion"].to(device)
                mask = batch["mask"].to(device)
                times = batch["times"].to(device)

                outputs = model(
                    motion,
                    mask,
                    times,
                    Ndata=len(train_dataset),
                    L=L,
                    inst_enc=args.inst_enc,
                    method=args.method,
                )
                elbo, lhood, kl_z, kl_w = outputs[4:]
                loss = -elbo
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_time = time.time() - batch_start
                if avg_batch_time is None:
                    avg_batch_time = batch_time
                else:
                    avg_batch_time = 0.9 * avg_batch_time + 0.1 * batch_time
                remaining_batches = max(num_batches - step - 1, 0)
                eta_seconds = remaining_batches * avg_batch_time
                epoch_elapsed = time.time() - epoch_start

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/elbo", elbo.item(), global_step)
                writer.add_scalar("train/lhood", lhood.item(), global_step)
                writer.add_scalar("train/kl_z", kl_z.item(), global_step)
                writer.add_scalar("train/kl_w", kl_w.item(), global_step)
                writer.add_scalar("train/batch_time", batch_time, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                log(
                    f"Epoch:{epoch:03d} Step:{step:04d} "
                    f"loss:{loss.item():9.2f} "
                    f"lhood:{lhood.item():9.2f} kl_z:{kl_z.item():9.2f} kl_w:{kl_w.item():9.2f} "
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
                    current_lr = optimizer.param_groups[0]["lr"]
                    checkpoint = build_checkpoint(epoch, val_joint_err)
                    step_model_path = checkpoint_dir / f"step_{global_step:07d}.pt"
                    torch.save(checkpoint, step_model_path)
                    if best_val_joint_err == val_joint_err:
                        torch.save(checkpoint, best_model_path)
                        log(
                            f"Epoch:{epoch:03d} GlobalStep:{global_step:07d} "
                            f"Val joint_err:{val_joint_err:9.6f} lr:{current_lr:.6e} "
                            f"[saved to {step_model_path}] [best saved to {best_model_path}]"
                        )
                    else:
                        log(
                            f"Epoch:{epoch:03d} GlobalStep:{global_step:07d} "
                            f"Val joint_err:{val_joint_err:9.6f} lr:{current_lr:.6e} "
                            f"[saved to {step_model_path}] "
                            f"(best:{best_val_joint_err:9.6f} @ epoch {best_epoch:03d})"
                        )
                    writer.add_scalar("val/joint_err", val_joint_err, global_step)
                    model.train()

            writer.flush()
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
