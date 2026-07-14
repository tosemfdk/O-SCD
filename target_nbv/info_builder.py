# Target information state builder (stage 7, docs/target_gaussian_nbv.md §6).
# H_data = sum_v J_v^T diag(w_v) J_v accumulated per observed view; damping is
# applied EXACTLY ONCE, inside TargetInformationState.H_prior(). Scorers never
# add damping and receive a read-only array.

from __future__ import annotations

import numpy as np
import torch

from target_nbv.config import TargetNBVConfig
from target_nbv.types import TargetInformationState, TargetParameterSpec


def view_information(J: torch.Tensor, w: torch.Tensor | None = None) -> np.ndarray:
    """H_v = J^T diag(w) J, symmetrized, float64 numpy (D,D)."""
    J64 = J.double()
    Jw = J64 if w is None else J64 * w.double().reshape(-1, 1)
    H = (J64.T @ Jw).cpu().numpy()
    return 0.5 * (H + H.T)


class TargetInformationBuilder:
    def __init__(self, target_pid: int, spec: TargetParameterSpec,
                 cfg: TargetNBVConfig, model_version: str = ""):
        d = spec.dimension
        self.state = TargetInformationState(
            target_pid=target_pid, spec=spec,
            H_data=np.zeros((d, d), dtype=np.float64),
            absolute_damping=cfg.information.absolute_damping,
            relative_damping=cfg.information.relative_damping,
            model_version=model_version,
        )

    def add_view(self, view_id: str, J: torch.Tensor, w: torch.Tensor | None = None) -> bool:
        """Accumulate one observed view. Duplicate view_ids are ignored."""
        if view_id in self.state.observed_view_ids:
            return False
        H_v = view_information(J, w)
        self.state.per_view_cache[view_id] = H_v
        self.state.observed_view_ids.append(view_id)
        self.state.H_data = 0.5 * ((self.state.H_data + H_v) + (self.state.H_data + H_v).T)
        self.state.version += 1
        return True

    def recompute_from_cache(self) -> np.ndarray:
        d = self.state.spec.dimension
        H = np.zeros((d, d), dtype=np.float64)
        for view_id in self.state.observed_view_ids:
            H += self.state.per_view_cache[view_id]
        return 0.5 * (H + H.T)

    def H_prior(self) -> np.ndarray:
        return self.state.H_prior()

    def diagnostics(self) -> dict:
        H = self.state.H_prior()
        eig = np.linalg.eigvalsh(H)
        return {
            "eigenvalues": eig.tolist(),
            "condition_number": float(eig[-1] / max(eig[0], 1e-300)),
            "damping": self.state.damping(),
            "num_views": len(self.state.observed_view_ids),
        }

    def save(self, path: str) -> None:
        torch.save({
            "target_pid": self.state.target_pid,
            "parameter_names": self.state.spec.parameter_names,
            "H_data": self.state.H_data,
            "absolute_damping": self.state.absolute_damping,
            "relative_damping": self.state.relative_damping,
            "observed_view_ids": self.state.observed_view_ids,
            "per_view_cache": self.state.per_view_cache,
            "version": self.state.version,
            "model_version": self.state.model_version,
        }, path)

    @classmethod
    def load(cls, path: str, cfg: TargetNBVConfig,
             expected_model_version: str | None = None) -> "TargetInformationBuilder":
        data = torch.load(path, weights_only=False)
        spec = TargetParameterSpec(parameter_names=data["parameter_names"])
        builder = cls(data["target_pid"], spec, cfg, data["model_version"])
        if expected_model_version is not None and data["model_version"] != expected_model_version:
            raise ValueError(
                f"information state was built for model version {data['model_version']!r}, "
                f"current is {expected_model_version!r}; rebuild required")
        builder.state.H_data = data["H_data"]
        builder.state.absolute_damping = data["absolute_damping"]
        builder.state.relative_damping = data["relative_damping"]
        builder.state.observed_view_ids = data["observed_view_ids"]
        builder.state.per_view_cache = data["per_view_cache"]
        builder.state.version = data["version"]
        return builder
