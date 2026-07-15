# Direction-aware D-optimal frame scorer (joint mode: geometry x change).
#
# Fixes the direction-blindness of the Beta coverage scorer (see
# experiments/paslcd_nbv_report.md §3): each Gaussian keeps a 3x3 mean-block
# information matrix H_g, a candidate frame contributes the viewing-geometry
# FIM increment
#     dH_g(c) = tau_g(c) / d_g^2 * (I - r_g r_g^T),   r_g = (mu_g - o_c)/d_g,
# (stage-8 proxy, responsibility-weighted), and the frame score is
#     S(c) = sum_g w_g * [logdet(H_g + dH_g(c)) - logdet(H_g)]
# with w_g the Beta ambiguity weight. Because (I - r r^T) is rank-2, a revisit
# from the SAME direction adds little once committed, while an orthogonal
# baseline fills the weak eigendirection — "look again, from a different
# angle, at what is not yet confirmed."

from __future__ import annotations

import torch

from target_nbv.change.counts import responsibilities

_TAU_EPS = 1e-3  # gaussians below this responsibility don't contribute


class DoptFrameState:
    """Per-Gaussian 3x3 mean-block information, damping applied once at init."""

    def __init__(self, n: int, damping: float = 1.0, device: str = "cuda"):
        if damping <= 0:
            raise ValueError("damping must be > 0")
        self.damping = float(damping)
        self.H = (damping * torch.eye(3, dtype=torch.float64, device=device)
                  ).expand(n, 3, 3).contiguous()

    def __len__(self) -> int:
        return self.H.shape[0]

    def append(self, k: int) -> None:
        new = (self.damping * torch.eye(3, dtype=torch.float64, device=self.H.device)
               ).expand(k, 3, 3).contiguous()
        self.H = torch.cat([self.H, new])

    def _increments(self, model, cam, tau: torch.Tensor):
        """(visible_index, dH) for gaussians with tau > eps."""
        vis = (tau > _TAU_EPS).nonzero(as_tuple=True)[0]
        if vis.numel() == 0:
            return vis, None
        mu = model.get_xyz.detach().double()[vis]
        o = cam.camera_center.double()
        diff = mu - o
        d2 = diff.pow(2).sum(-1).clamp_min(1e-8)
        r = diff / d2.sqrt().unsqueeze(-1)
        eye = torch.eye(3, dtype=torch.float64, device=mu.device)
        proj = eye - r.unsqueeze(-1) @ r.unsqueeze(-2)
        dH = (tau[vis] / d2).reshape(-1, 1, 1) * proj
        return vis, dH

    def score_frame(self, model, cam, w: torch.Tensor, pipe,
                    tau: torch.Tensor | None = None) -> float:
        """S(c) = sum_g w_g * dlogdet_g. tau may be passed if already computed."""
        if tau is None:
            tau = responsibilities(model, cam, pipe)
        vis, dH = self._increments(model, cam, tau)
        if dH is None:
            return 0.0
        H = self.H[vis]
        gain = torch.linalg.slogdet(H + dH).logabsdet - torch.linalg.slogdet(H).logabsdet
        return float((w.double()[vis] * gain.clamp_min(0.0)).sum())

    def commit(self, model, cam, tau: torch.Tensor) -> None:
        """Accumulate the OBSERVED frame's geometric information."""
        if tau.shape[0] != len(self):
            raise ValueError(f"tau size {tau.shape[0]} != state {len(self)}")
        vis, dH = self._increments(model, cam, tau)
        if dH is None:
            return
        self.H[vis] = self.H[vis] + dH
