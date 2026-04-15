import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils import data
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


class ODE2VAEHand(nn.Module):
    def __init__(
        self,
        input_dim: int,
        q: int = 16,
        hidden_dim: int = 256,
        n_init_obs: int = 4,
        use_global_rot: bool = True,
        mano_model_path: str = None,
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
        if mano_model_path is None:
            mano_model_path = Path(__file__).resolve().parent / "EasyMocap" / "data" / "smplx"
        mano_model_path = str(mano_model_path)
        self.mano_right = SMPLlayer(
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
        a1 = F.normalize(x[:, 0:3], dim=1)
        a2 = x[:, 3:6]
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
    ) -> torch.Tensor:
        N, T, _ = pred_motion.shape
        pred_pose, pred_Rh, pred_Th, pred_shape = self._motion_to_mano_params(
            pred_motion, gt_shape, gt_pose, gt_Rh
        )
        gt_pose_axis = self._rot6d_to_axis_angle(gt_pose.reshape(N * T, -1, 6)).view(N * T, -1, 3).reshape(N * T, -1)
        gt_Rh_axis = self._rot6d_to_axis_angle(gt_Rh.reshape(N * T, 6)).view(N * T, 3)
        gt_Th_flat = gt_Th.reshape(N * T, 3)
        gt_shape_flat = gt_shape.reshape(N * T, -1)

        pred_joints = self.mano_right(
            poses=pred_pose,
            shapes=pred_shape,
            Rh=pred_Rh,
            Th=pred_Th,
            return_verts=False,
            return_tensor=True,
        ).view(N, T, -1, 3)
        gt_joints = self.mano_right(
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
        if shape is None or pose is None or Rh is None or Th is None:
            raise ValueError("shape, pose, Rh, and Th are required to compute MANO joint error.")
        joint_error = self._mano_joint_error(Xrec_mu, pose, Rh, Th, shape, mask)
        return Xrec_mu, joint_error.mean()


def build_dataloaders(args):
    train_dataset = GigaHandDataset(
        dataset_root=args.dataset_root,
        seq_len=args.seq_len,
        split="train",
        text_file=args.text_file,
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
        split="val",
        text_file=args.text_file,
        normalize_trans=args.normalize_trans,
        use_global_rot=not args.disable_global_rot,
        random_mask=False,
        fps=args.fps,
        max_sequences=args.max_sequences,
    )
    params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": args.num_workers}
    train_loader = data.DataLoader(train_dataset, **params)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataset, val_dataset, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Vector-sequence ODE2VAE for GigaHands MANO motion.")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-init-obs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--inst-enc", action="store_true")
    parser.add_argument("--normalize-trans", action="store_true")
    parser.add_argument("--disable-global-rot", action="store_true")
    parser.add_argument("--random-mask", action="store_true")
    parser.add_argument("--random-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args)
    model = ODE2VAEHand(
        input_dim=train_dataset.motion_dim,
        q=args.q,
        hidden_dim=args.hidden_dim,
        n_init_obs=args.n_init_obs,
        use_global_rot=not args.disable_global_rot,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            print(
                f"Epoch:{epoch:03d} Step:{step:04d} "
                f"loss:{loss.item():9.2f} "
                f"lhood:{lhood.item():9.2f} kl_z:{kl_z.item():9.2f} kl_w:{kl_w.item():9.2f} "
                f"batch_time:{batch_time:6.2f}s epoch_elapsed:{epoch_elapsed:7.2f}s "
                f"epoch_eta:{eta_seconds:7.2f}s"
            )

        model.eval()
        val_mse_sum = 0.0
        val_batches = 0
        for batch in val_loader:
            motion = batch["motion"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"].to(device)
            shape = batch["shape"].to(device)
            pose = batch["pose"].to(device)
            Rh = batch["Rh"].to(device)
            Th = batch["Th"].to(device)
            _, val_joint_err = model.mean_rec(
                motion,
                times,
                mask=mask,
                shape=shape,
                pose=pose,
                Rh=Rh,
                Th=Th,
                method=args.method,
            )
            val_mse_sum += val_joint_err.item()
            val_batches += 1

        if val_batches > 0:
            print(
                f"Epoch:{epoch:03d} Val "
                f"joint_err:{val_mse_sum / val_batches:9.6f}"
            )


if __name__ == "__main__":
    main()
