"""Aggregate D1 PASLCD view-consistency diagnostics without ground truth.

The D1 runner writes one scene directory per ``instance/scene`` with a
``manifest.json`` and sparse per-frame NPZ shards.  This analysis accepts either
one scene directory or a full D1 output root containing many nested scene
manifests, keeps ``(instance, scene, gaussian_index)`` namespaces separate, and
writes one aggregate report directory.

Required outputs:
  summary.json, cohort_stats.csv, event_windows.csv, and five PNG plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EPS = 1.0e-12
KEEP_WINDOW = 8
EVENT_ACTIONS = {"OPEN", "CLOSE", "REOPEN"}
ACTION_ENUM = {
    0: "NONE",
    1: "OPEN",
    2: "KEEP",
    3: "CLOSE",
    4: "UNCERTAIN",
}
OUTPUT_FILES = (
    "summary.json",
    "cohort_stats.csv",
    "event_windows.csv",
    "mass_at_close_vs_keep.png",
    "view_delta_vs_transition.png",
    "q_volatility_vs_reopen.png",
    "transition_odds_at_events.png",
    "representative_gaussian_trajectories.png",
)


def _load_pyplot():
    """Import plotting support only when plots are actually requested.

    The detector and analysis helpers remain importable in the lean ``oscd``
    environment used by the repository test suite.  The full analysis command
    still fails explicitly at plot generation when matplotlib is unavailable.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except Exception as exc:  # pragma: no cover - depends on the runtime env
        raise RuntimeError("matplotlib is required to generate D1 diagnostic plots") from exc
    return pyplot


@dataclass(frozen=True)
class FrameInfo:
    instance: str
    scene: str
    index: int
    name: str
    path: Path
    view_delta: float
    translation_delta: float
    angular_delta_deg: float


@dataclass(frozen=True)
class SceneInfo:
    instance: str
    scene: str
    manifest_path: Path
    manifest: Mapping[str, Any]
    frames: tuple[FrameInfo, ...]
    gaussian_count: int | None


@dataclass
class DenseSceneState:
    instance: str
    scene: str
    size: int
    observed_count: np.ndarray
    first_frame: np.ndarray
    last_frame: np.ndarray
    ever_active: np.ndarray
    currently_active: np.ndarray
    open_count: np.ndarray
    close_count: np.ndarray
    reopen_count: np.ndarray
    keep_count: np.ndarray
    transition_count: np.ndarray
    q_sum: np.ndarray
    q_sq_sum: np.ndarray
    prev_q: np.ndarray
    prev_q_sign: np.ndarray
    q_abs_delta_sum: np.ndarray
    q_delta_count: np.ndarray
    q_sign_flip_count: np.ndarray
    keep_mass: np.ndarray
    keep_strength: np.ndarray
    keep_view_delta: np.ndarray
    keep_q: np.ndarray
    keep_pos: np.ndarray
    keep_seen: np.ndarray

    @classmethod
    def make(cls, instance: str, scene: str, size: int) -> "DenseSceneState":
        size = max(int(size), 1)
        return cls(
            instance=instance,
            scene=scene,
            size=size,
            observed_count=np.zeros(size, dtype=np.int64),
            first_frame=np.full(size, -1, dtype=np.int64),
            last_frame=np.full(size, -1, dtype=np.int64),
            ever_active=np.zeros(size, dtype=bool),
            currently_active=np.zeros(size, dtype=bool),
            open_count=np.zeros(size, dtype=np.int64),
            close_count=np.zeros(size, dtype=np.int64),
            reopen_count=np.zeros(size, dtype=np.int64),
            keep_count=np.zeros(size, dtype=np.int64),
            transition_count=np.zeros(size, dtype=np.int64),
            q_sum=np.zeros(size, dtype=np.float64),
            q_sq_sum=np.zeros(size, dtype=np.float64),
            prev_q=np.full(size, np.nan, dtype=np.float64),
            prev_q_sign=np.zeros(size, dtype=np.int8),
            q_abs_delta_sum=np.zeros(size, dtype=np.float64),
            q_delta_count=np.zeros(size, dtype=np.int64),
            q_sign_flip_count=np.zeros(size, dtype=np.int64),
            keep_mass=np.full((size, KEEP_WINDOW), np.nan, dtype=np.float32),
            keep_strength=np.full((size, KEEP_WINDOW), np.nan, dtype=np.float32),
            keep_view_delta=np.full((size, KEEP_WINDOW), np.nan, dtype=np.float32),
            keep_q=np.full((size, KEEP_WINDOW), np.nan, dtype=np.float32),
            keep_pos=np.zeros(size, dtype=np.int64),
            keep_seen=np.zeros(size, dtype=np.int64),
        )

    def ensure(self, max_index: int) -> None:
        if max_index < self.size:
            return
        new_size = max(max_index + 1, self.size * 2)

        def grow_1d(arr: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full(new_size, fill, dtype=arr.dtype)
            out[: self.size] = arr
            return out

        def grow_2d(arr: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full((new_size, arr.shape[1]), fill, dtype=arr.dtype)
            out[: self.size] = arr
            return out

        self.observed_count = grow_1d(self.observed_count, 0)
        self.first_frame = grow_1d(self.first_frame, -1)
        self.last_frame = grow_1d(self.last_frame, -1)
        self.ever_active = grow_1d(self.ever_active, False)
        self.currently_active = grow_1d(self.currently_active, False)
        self.open_count = grow_1d(self.open_count, 0)
        self.close_count = grow_1d(self.close_count, 0)
        self.reopen_count = grow_1d(self.reopen_count, 0)
        self.keep_count = grow_1d(self.keep_count, 0)
        self.transition_count = grow_1d(self.transition_count, 0)
        self.q_sum = grow_1d(self.q_sum, 0.0)
        self.q_sq_sum = grow_1d(self.q_sq_sum, 0.0)
        self.prev_q = grow_1d(self.prev_q, np.nan)
        self.prev_q_sign = grow_1d(self.prev_q_sign, 0)
        self.q_abs_delta_sum = grow_1d(self.q_abs_delta_sum, 0.0)
        self.q_delta_count = grow_1d(self.q_delta_count, 0)
        self.q_sign_flip_count = grow_1d(self.q_sign_flip_count, 0)
        self.keep_mass = grow_2d(self.keep_mass, np.nan)
        self.keep_strength = grow_2d(self.keep_strength, np.nan)
        self.keep_view_delta = grow_2d(self.keep_view_delta, np.nan)
        self.keep_q = grow_2d(self.keep_q, np.nan)
        self.keep_pos = grow_1d(self.keep_pos, 0)
        self.keep_seen = grow_1d(self.keep_seen, 0)
        self.size = new_size

    def observed_mask(self) -> np.ndarray:
        return self.observed_count > 0

    def q_mean(self) -> np.ndarray:
        out = np.full(self.size, np.nan, dtype=np.float64)
        m = self.observed_count > 0
        out[m] = self.q_sum[m] / self.observed_count[m]
        return out

    def q_variance(self) -> np.ndarray:
        out = np.zeros(self.size, dtype=np.float64)
        m = self.observed_count > 0
        mean = np.zeros(self.size, dtype=np.float64)
        mean[m] = self.q_sum[m] / self.observed_count[m]
        out[m] = np.maximum(0.0, self.q_sq_sum[m] / self.observed_count[m] - mean[m] * mean[m])
        return out

    def q_abs_delta_mean(self) -> np.ndarray:
        out = np.zeros(self.size, dtype=np.float64)
        m = self.q_delta_count > 0
        out[m] = self.q_abs_delta_sum[m] / self.q_delta_count[m]
        return out

    def cohort_codes(self) -> np.ndarray:
        # 0 reopen/high-chatter, 1 close-only, 2 stable-open,
        # 3 sufficiently observed stable-inactive, 4 insufficient observation.
        observed = self.observed_mask()
        codes = np.full(self.size, 4, dtype=np.int8)
        codes[observed & (self.observed_count >= 5)] = 3
        stable_open = observed & (self.ever_active | self.currently_active | (self.open_count > 0))
        codes[stable_open] = 2
        close_only = observed & (self.close_count > 0)
        codes[close_only] = 1
        repeated = observed & (
            (self.reopen_count > 0)
            | (self.open_count > 1)
            | (self.transition_count > 2)
        )
        codes[repeated] = 0
        codes[~observed] = -1
        return codes


def finite_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_div(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(num, dtype=float) / np.maximum(np.asarray(den, dtype=float), EPS)


def summarize(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "q05": None, "q25": None, "q50": None, "q75": None, "q95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def normalize_action_array(data: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    if "action_name" in data:
        arr = np.asarray(data["action_name"]).reshape(-1)
        if arr.size != n:
            raise ValueError(f"action_name length {arr.size} != row length {n}")
        return np.asarray([decode_string(v).upper() for v in arr], dtype=object)
    if "action" in data:
        arr = np.asarray(data["action"]).reshape(-1)
        if arr.size != n:
            raise ValueError(f"action length {arr.size} != row length {n}")
        out: list[str] = []
        for v in arr:
            if isinstance(v, (bytes, np.bytes_, str, np.str_)):
                out.append(decode_string(v).upper())
            else:
                out.append(ACTION_ENUM.get(int(v), f"ACTION_{int(v)}"))
        return np.asarray(out, dtype=object)
    return np.full(n, "NONE", dtype=object)


def decode_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def get_array(data: Mapping[str, np.ndarray], names: Sequence[str], n: int, default: float = np.nan) -> np.ndarray:
    for name in names:
        if name in data:
            arr = np.asarray(data[name]).reshape(-1)
            if arr.size == 1 and n != 1:
                return np.full(n, float(arr[0]), dtype=np.float64)
            if arr.size != n:
                raise ValueError(f"{name} length {arr.size} != row length {n}")
            return arr.astype(np.float64, copy=False)
    return np.full(n, default, dtype=np.float64)


def load_frame_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        if "gaussian_index" not in loaded:
            raise KeyError(f"{path} missing gaussian_index")
        idx = np.asarray(loaded["gaussian_index"]).reshape(-1).astype(np.int64, copy=False)
        n = int(idx.size)
        q = get_array(loaded, ("q",), n)
        strength = get_array(loaded, ("evidence_strength",), n)
        if np.all(~np.isfinite(q)) or np.all(~np.isfinite(strength)):
            delta_a = get_array(loaded, ("delta_a", "e_plus"), n)
            delta_b = get_array(loaded, ("delta_b", "e_minus"), n)
            if np.all(~np.isfinite(strength)):
                strength = delta_a + delta_b
            if np.all(~np.isfinite(q)):
                q = delta_a / np.maximum(delta_a + delta_b, EPS)
        fields: dict[str, np.ndarray] = {
            "gaussian_index": idx,
            "raw_mass": get_array(loaded, ("raw_mass", "total_mass", "mass"), n),
            "q": q,
            "evidence_strength": strength,
            "p_active_pre": get_array(loaded, ("p_active_pre", "p_active_before", "p_before"), n),
            "p_active_post": get_array(loaded, ("p_active_post", "p_active"), n),
            "p00": get_array(loaded, ("p00", "p_00"), n),
            "p01": get_array(loaded, ("p01", "p_01"), n),
            "p10": get_array(loaded, ("p10", "p_10"), n),
            "p11": get_array(loaded, ("p11", "p_11"), n),
            "pflip": get_array(loaded, ("pflip", "p_flip"), n),
            "old_slot": get_array(loaded, ("old_slot",), n, default=-1).astype(np.int64),
            "current_slot": get_array(loaded, ("current_slot", "new_current_slot"), n, default=-1).astype(np.int64),
            "visible_observation_count": get_array(loaded, ("visible_observation_count", "visible_observations"), n),
            "action_name": normalize_action_array(loaded, n),
        }
    return fields


def resolve_npz_path(raw: str, *, manifest_dir: Path, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (manifest_dir / path, Path.cwd() / path, root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def discover_scene_infos(root: Path, manifest_name: str = "manifest.json") -> list[SceneInfo]:
    root = root.resolve()
    candidates = sorted(root.rglob(manifest_name)) if root.is_dir() else [root]
    scenes: list[SceneInfo] = []
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        frame_rows = manifest.get("frames")
        if not isinstance(frame_rows, list) or not frame_rows:
            continue
        scene_summary = manifest.get("scene_summary") if isinstance(manifest.get("scene_summary"), Mapping) else {}
        first = frame_rows[0] if isinstance(frame_rows[0], Mapping) else {}
        instance = str(scene_summary.get("instance", first.get("instance", manifest_path.parent.parent.name)))
        scene = str(scene_summary.get("scene", first.get("scene", manifest_path.parent.name)))
        frames: list[FrameInfo] = []
        prev_center: np.ndarray | None = None
        prev_forward: np.ndarray | None = None
        for ordinal, row in enumerate(frame_rows):
            if not isinstance(row, Mapping) or "npz" not in row:
                continue
            path = resolve_npz_path(str(row["npz"]), manifest_dir=manifest_path.parent, root=root)
            trans = finite_float(row.get("translation_delta_from_previous"), np.nan)
            ang = finite_float(row.get("angular_delta_from_previous_degrees"), np.nan)
            if not math.isfinite(trans) or not math.isfinite(ang):
                center = vector3(row.get("camera_center"))
                forward = vector3(row.get("camera_forward"))
                if center is not None and prev_center is not None and not math.isfinite(trans):
                    trans = float(np.linalg.norm(center - prev_center))
                if forward is not None and prev_forward is not None and not math.isfinite(ang):
                    ang = angular_delta_deg(forward, prev_forward)
                if center is not None:
                    prev_center = center
                if forward is not None:
                    prev_forward = forward
            view_delta = combined_view_delta(trans, ang)
            frames.append(
                FrameInfo(
                    instance=instance,
                    scene=scene,
                    index=int(row.get("timestamp", row.get("frame_index", ordinal))),
                    name=str(row.get("frame_name", path.stem)),
                    path=path,
                    view_delta=view_delta,
                    translation_delta=trans,
                    angular_delta_deg=ang,
                )
            )
        if frames:
            gaussian_count = scene_summary.get("gaussian_count", manifest.get("gaussian_count"))
            scenes.append(
                SceneInfo(
                    instance=instance,
                    scene=scene,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    frames=tuple(sorted(frames, key=lambda f: (f.index, f.name))),
                    gaussian_count=None if gaussian_count is None else int(gaussian_count),
                )
            )
    # Avoid double-processing an aggregate root manifest plus scene manifests: only manifests
    # with frame NPZ entries qualify above, so this is just defensive de-duplication.
    seen: set[Path] = set()
    unique: list[SceneInfo] = []
    for scene in scenes:
        if scene.manifest_path not in seen:
            unique.append(scene)
            seen.add(scene.manifest_path)
    if not unique:
        raise RuntimeError(f"No D1 scene manifests with frame NPZ entries found under {root}")
    return unique


def vector3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def angular_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return np.nan
    dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def combined_view_delta(translation: float, angle_deg: float) -> float:
    t = 0.0 if not math.isfinite(translation) else float(translation)
    a = 0.0 if not math.isfinite(angle_deg) else float(angle_deg) / 180.0
    return float(math.sqrt(t * t + a * a))


def q_sign(q: np.ndarray) -> np.ndarray:
    out = np.zeros(q.shape, dtype=np.int8)
    finite = np.isfinite(q)
    out[finite & (q >= 0.5)] = 1
    out[finite & (q < 0.5)] = -1
    return out


def emission_llr_active(q: np.ndarray, strength: np.ndarray, eta: float) -> np.ndarray:
    eta = min(max(float(eta), EPS), 1.0 - EPS)
    return strength * (2.0 * q - 1.0) * math.log(eta / (1.0 - eta))


def median_window(arr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.asarray([], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(arr[rows], axis=1)


def transition_odds_arrays(p00: np.ndarray, p01: np.ndarray, p10: np.ndarray, p11: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p01 / np.maximum(p00, EPS), p10 / np.maximum(p11, EPS)


def threshold_crossings(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    out = np.full(pre.shape, "unknown", dtype=object)
    finite = np.isfinite(pre) & np.isfinite(post)
    out[finite] = "none"
    out[finite & (pre < 0.5) & (post >= 0.5)] = "up"
    out[finite & (pre >= 0.5) & (post < 0.5)] = "down"
    return out


def update_scene(scene: SceneInfo, *, eta: float) -> tuple[DenseSceneState, list[dict[str, Any]]]:
    state = DenseSceneState.make(scene.instance, scene.scene, scene.gaussian_count or 1)
    events: list[dict[str, Any]] = []
    for frame in scene.frames:
        fields = load_frame_fields(frame.path)
        idx = fields["gaussian_index"]
        if idx.size == 0:
            continue
        state.ensure(int(np.max(idx)))
        q = fields["q"]
        strength = fields["evidence_strength"]
        raw_mass = fields["raw_mass"]
        actions = fields["action_name"]
        p_pre = fields["p_active_pre"]
        p_post = fields["p_active_post"]
        p00, p01, p10, p11 = fields["p00"], fields["p01"], fields["p10"], fields["p11"]
        pflip = fields["pflip"]
        visible = fields["visible_observation_count"]
        current_slot = fields["current_slot"]

        previous_observed_frame = state.last_frame[idx].copy()
        observation_gap = np.where(
            previous_observed_frame >= 0,
            frame.index - previous_observed_frame - 1,
            np.nan,
        )
        state.observed_count[idx] += 1
        first = state.first_frame[idx] < 0
        if np.any(first):
            state.first_frame[idx[first]] = frame.index
        state.last_frame[idx] = frame.index

        # Cohorts describe the committed representation lifecycle, whose
        # authoritative active label is whether a slot is currently open.
        active_now = current_slot >= 0
        state.currently_active[idx] = active_now
        is_open = actions == "OPEN"
        is_close = actions == "CLOSE"
        is_keep = actions == "KEEP"
        is_reopen_name = actions == "REOPEN"
        inferred_reopen = is_open & (state.close_count[idx] > 0)
        is_event = is_open | is_close | is_reopen_name
        state.ever_active[idx] |= active_now | is_open | is_reopen_name
        state.open_count[idx[is_open | is_reopen_name]] += 1
        state.reopen_count[idx[inferred_reopen | is_reopen_name]] += 1
        state.close_count[idx[is_close]] += 1
        state.keep_count[idx[is_keep]] += 1
        state.transition_count[idx[is_open | is_close | is_reopen_name]] += 1

        finite_q = np.isfinite(q)
        if np.any(finite_q):
            qidx = idx[finite_q]
            qv = q[finite_q]
            state.q_sum[qidx] += qv
            state.q_sq_sum[qidx] += qv * qv
            prev = state.prev_q[qidx]
            has_prev = np.isfinite(prev)
            if np.any(has_prev):
                pidx = qidx[has_prev]
                delta = qv[has_prev] - prev[has_prev]
                state.q_abs_delta_sum[pidx] += np.abs(delta)
                state.q_delta_count[pidx] += 1
            sign = q_sign(qv)
            prev_sign = state.prev_q_sign[qidx]
            flip = (prev_sign != 0) & (sign != 0) & (prev_sign != sign)
            if np.any(flip):
                state.q_sign_flip_count[qidx[flip]] += 1
            state.prev_q[qidx] = qv
            state.prev_q_sign[qidx] = sign

        if np.any(is_event):
            erows = idx[is_event]
            eactions = actions[is_event]
            prior_mass = median_window(state.keep_mass, erows)
            prior_strength = median_window(state.keep_strength, erows)
            prior_view = median_window(state.keep_view_delta, erows)
            prior_q = median_window(state.keep_q, erows)
            ev_mass = raw_mass[is_event]
            ev_strength = strength[is_event]
            ev_q = q[is_event]
            odds01, odds10 = transition_odds_arrays(p00[is_event], p01[is_event], p10[is_event], p11[is_event])
            selected_odds = np.where((eactions == "OPEN") | (eactions == "REOPEN"), odds01, odds10)
            ell_active = emission_llr_active(ev_q, ev_strength, eta)
            crossing = threshold_crossings(p_pre[is_event], p_post[is_event])
            mass_ratio = ev_mass / np.maximum(prior_mass, EPS)
            strength_ratio = ev_strength / np.maximum(prior_strength, EPS)
            view_ratio = np.full(erows.shape, np.nan, dtype=np.float64)
            finite_prior_view = np.isfinite(prior_view)
            view_ratio[finite_prior_view] = frame.view_delta / np.maximum(prior_view[finite_prior_view], EPS)
            q_delta = ev_q - prior_q
            q_flip = (q_sign(ev_q) != 0) & (q_sign(prior_q) != 0) & (q_sign(ev_q) != q_sign(prior_q))
            event_gap = observation_gap[is_event]
            event_is_reopen = inferred_reopen[is_event] | is_reopen_name[is_event]
            for j, gi in enumerate(erows.tolist()):
                events.append(
                    {
                        "instance": scene.instance,
                        "scene": scene.scene,
                        "frame_index": frame.index,
                        "frame_name": frame.name,
                        "gaussian_index": int(gi),
                        "action": str(eactions[j]),
                        "is_reopen": bool(event_is_reopen[j]),
                        "cohort": "",  # filled after scene finalization
                        "raw_mass": csv_float(ev_mass[j]),
                        "evidence_strength": csv_float(ev_strength[j]),
                        "prior_keep_mass_median": csv_float(prior_mass[j]),
                        "prior_keep_strength_median": csv_float(prior_strength[j]),
                        "mass_vs_prior_keep_ratio": csv_float(mass_ratio[j]),
                        "strength_vs_prior_keep_ratio": csv_float(strength_ratio[j]),
                        "view_delta": csv_float(frame.view_delta),
                        "translation_delta_from_previous": csv_float(frame.translation_delta),
                        "angular_delta_from_previous_degrees": csv_float(frame.angular_delta_deg),
                        "prior_keep_view_delta_median": csv_float(prior_view[j]),
                        "view_delta_vs_prior_keep_ratio": csv_float(view_ratio[j]),
                        "q": csv_float(ev_q[j]),
                        "prior_keep_q_median": csv_float(prior_q[j]),
                        "q_delta_vs_prior_keep": csv_float(q_delta[j]),
                        "q_sign_flip_vs_prior_keep": bool(q_flip[j]),
                        "p_active_pre": csv_float(p_pre[is_event][j]),
                        "p_active_post": csv_float(p_post[is_event][j]),
                        "p00": csv_float(p00[is_event][j]),
                        "p01": csv_float(p01[is_event][j]),
                        "p10": csv_float(p10[is_event][j]),
                        "p11": csv_float(p11[is_event][j]),
                        "pflip": csv_float(pflip[is_event][j]),
                        "transition_odds_01": csv_float(odds01[j]),
                        "transition_odds_10": csv_float(odds10[j]),
                        "selected_transition_odds": csv_float(selected_odds[j]),
                        "emission_llr_active": csv_float(ell_active[j]),
                        "emission_llr_close": csv_float(-ell_active[j]),
                        "threshold_crossing": str(crossing[j]),
                        "controller_threshold_margin": csv_float(
                            p_post[is_event][j] - 0.6
                            if str(eactions[j]) in {"OPEN", "REOPEN"}
                            else 0.4 - p_post[is_event][j]
                        ),
                        "visible_observation_count": csv_float(visible[is_event][j]),
                        "observation_gap_since_previous": csv_float(event_gap[j]),
                    }
                )

        if np.any(is_keep):
            kidx = idx[is_keep]
            pos = state.keep_pos[kidx] % KEEP_WINDOW
            state.keep_mass[kidx, pos] = raw_mass[is_keep]
            state.keep_strength[kidx, pos] = strength[is_keep]
            state.keep_view_delta[kidx, pos] = frame.view_delta
            state.keep_q[kidx, pos] = q[is_keep]
            state.keep_pos[kidx] += 1
            state.keep_seen[kidx] += 1

    codes = state.cohort_codes()
    names = np.asarray(
        [
            "reopen/repeated",
            "close-only",
            "stable-open",
            "stable-inactive",
            "insufficient-observation",
        ],
        dtype=object,
    )
    for row in events:
        code = int(codes[int(row["gaussian_index"])])
        row["cohort"] = str(names[code]) if code >= 0 else "unobserved"
    return state, events


def csv_float(value: Any) -> float | str:
    try:
        out = float(value)
    except Exception:
        return ""
    return out if math.isfinite(out) else ""


def collect_cohort_stats(states: Sequence[DenseSceneState]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = [
        "reopen/repeated",
        "close-only",
        "stable-open",
        "stable-inactive",
        "insufficient-observation",
    ]
    for label_i, label in enumerate(labels):
        count = observed = opens = closes = reopens = keeps = transitions = signflips = 0
        q_means: list[np.ndarray] = []
        q_vars: list[np.ndarray] = []
        q_abs: list[np.ndarray] = []
        for st in states:
            codes = st.cohort_codes()
            m = codes == label_i
            count += int(np.sum(m))
            observed += int(np.sum(st.observed_count[m]))
            opens += int(np.sum(st.open_count[m]))
            closes += int(np.sum(st.close_count[m]))
            reopens += int(np.sum(st.reopen_count[m]))
            keeps += int(np.sum(st.keep_count[m]))
            transitions += int(np.sum(np.maximum(0, st.transition_count[m] - 1)))
            signflips += int(np.sum(st.q_sign_flip_count[m]))
            q_means.append(st.q_mean()[m])
            q_vars.append(st.q_variance()[m])
            q_abs.append(st.q_abs_delta_mean()[m])
        rows.append(
            {
                "cohort": label,
                "gaussian_count": count,
                "observed_row_count": observed,
                "open_count": opens,
                "close_count": closes,
                "reopen_count": reopens,
                "keep_count": keeps,
                "repeated_transition_count": transitions,
                "q_mean_mean": nanmean_concat(q_means),
                "q_variance_mean": nanmean_concat(q_vars),
                "q_abs_delta_mean": nanmean_concat(q_abs),
                "q_sign_flip_total": signflips,
            }
        )
    return rows


def nanmean_concat(chunks: Sequence[np.ndarray]) -> float | str:
    nonempty = [c.reshape(-1) for c in chunks if c.size]
    arr = np.concatenate(nonempty) if nonempty else np.asarray([])
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else ""


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: Sequence[Mapping[str, Any]], key: str, action: str | None = None) -> np.ndarray:
    out: list[float] = []
    for row in rows:
        if action is not None and row.get("action") != action:
            continue
        value = finite_float(row.get(key), np.nan)
        if math.isfinite(value):
            out.append(value)
    return np.asarray(out, dtype=np.float64)


def point_biserial(values: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.all(y == y[0]):
        return None
    out = float(np.corrcoef(x, y.astype(float))[0, 1])
    return out if math.isfinite(out) else None


def build_summary(root: Path, output_dir: Path, scenes: Sequence[SceneInfo], states: Sequence[DenseSceneState], events: Sequence[Mapping[str, Any]], cohort_rows: Sequence[Mapping[str, Any]], eta: float) -> dict[str, Any]:
    close_mass = finite_values(events, "mass_vs_prior_keep_ratio", action="CLOSE")
    close_strength = finite_values(events, "strength_vs_prior_keep_ratio", action="CLOSE")
    close_gaps = finite_values(events, "observation_gap_since_previous", action="CLOSE")
    reopen_gaps = np.asarray(
        [
            finite_float(row.get("observation_gap_since_previous"), np.nan)
            for row in events
            if bool(row.get("is_reopen"))
        ],
        dtype=np.float64,
    )
    view_ratio = finite_values(events, "view_delta_vs_prior_keep_ratio")
    selected_odds = finite_values(events, "selected_transition_odds")
    sign_flip = np.asarray([bool(r.get("q_sign_flip_vs_prior_keep")) for r in events], dtype=bool)
    reopen_event = np.asarray([r.get("cohort") == "reopen/repeated" or r.get("action") == "REOPEN" for r in events], dtype=bool)
    q_abs_chunks: list[np.ndarray] = []
    q_var_chunks: list[np.ndarray] = []
    reopen_chunks: list[np.ndarray] = []
    for st in states:
        obs = st.observed_mask()
        q_abs_chunks.append(st.q_abs_delta_mean()[obs])
        q_var_chunks.append(st.q_variance()[obs])
        reopen_chunks.append((st.cohort_codes()[obs] == 0))
    q_abs_all = np.concatenate(q_abs_chunks) if q_abs_chunks else np.asarray([])
    q_var_all = np.concatenate(q_var_chunks) if q_var_chunks else np.asarray([])
    reopen_all = np.concatenate(reopen_chunks) if reopen_chunks else np.asarray([], dtype=bool)
    threshold_counts: dict[str, int] = {}
    for row in events:
        key = str(row.get("threshold_crossing", "unknown"))
        threshold_counts[key] = threshold_counts.get(key, 0) + 1
    h1 = summarize(close_mass).get("q50")
    h2 = summarize(view_ratio).get("q50")
    h3 = summarize(selected_odds).get("q75")
    h4 = point_biserial(q_abs_all, reopen_all)
    baseline_mismatches = 0
    base_equal = True
    for scene in scenes:
        scene_summary = scene.manifest.get("scene_summary", {})
        if not isinstance(scene_summary, Mapping):
            continue
        baseline = scene_summary.get("baseline_lifecycle_event_structure")
        if isinstance(baseline, Mapping) and baseline.get("exists", True):
            baseline_mismatches += int(not bool(baseline.get("matches")))
        base_equal = base_equal and bool(scene_summary.get("base_bitwise_equal", False))
    lifecycle_counts = {
        "open": int(sum(np.sum(st.open_count) for st in states)),
        "close": int(sum(np.sum(st.close_count) for st in states)),
        "reopen": int(sum(np.sum(st.reopen_count) for st in states)),
        "keep": int(sum(np.sum(st.keep_count) for st in states)),
        "same_scene_repeated_transition_events": int(
            sum(np.sum(np.maximum(0, st.transition_count - 1)) for st in states)
        ),
    }
    return {
        "contract": "paslcd_d1_view_consistency_offline_no_gt_multiscene_analysis",
        "input_root": str(root),
        "output_dir": str(output_dir),
        "eta": float(eta),
        "scene_count": len(scenes),
        "frame_count": int(sum(len(s.frames) for s in scenes)),
        "observed_gaussian_namespace_count": int(sum(np.sum(st.observed_mask()) for st in states)),
        "event_count": len(events),
        "lifecycle_counts": lifecycle_counts,
        "gt_used_for_decision": False,
        "optimizer_used": False,
        "representation_training_used": False,
        "baseline_structure_mismatch_count": int(baseline_mismatches),
        "base_bitwise_equal": bool(base_equal),
        "scene_manifests": [str(s.manifest_path) for s in scenes],
        "cohorts": cohort_rows,
        "h1_close_vs_prior_keep_mass": {
            "raw_mass_ratio_close_to_median_preceding_keep": summarize(close_mass),
            "strength_ratio_close_to_median_preceding_keep": summarize(close_strength),
            "close_observation_gap_since_previous": summarize(close_gaps),
            "close_after_gap_fraction": (
                float(np.mean(close_gaps >= 1)) if close_gaps.size else None
            ),
            "reopen_after_gap_fraction": (
                float(np.mean(reopen_gaps[np.isfinite(reopen_gaps)] >= 1))
                if np.any(np.isfinite(reopen_gaps))
                else None
            ),
        },
        "h2_view_delta_vs_transition": {
            "event_view_delta_ratio_to_same_gaussian_keep": summarize(view_ratio),
            "event_view_delta": summarize(finite_values(events, "view_delta")),
            "translation_delta_from_previous": summarize(finite_values(events, "translation_delta_from_previous")),
            "angular_delta_from_previous_degrees": summarize(finite_values(events, "angular_delta_from_previous_degrees")),
        },
        "h3_threshold_crossing_transition_odds_emission_llr": {
            "threshold_crossing_counts": threshold_counts,
            "selected_transition_odds": summarize(selected_odds),
            "transition_odds_01": summarize(finite_values(events, "transition_odds_01")),
            "transition_odds_10": summarize(finite_values(events, "transition_odds_10")),
            "emission_llr_active": summarize(finite_values(events, "emission_llr_active")),
            "emission_llr_close": summarize(finite_values(events, "emission_llr_close")),
            "controller_threshold_margin": summarize(
                finite_values(events, "controller_threshold_margin")
            ),
            "default_max_transition_odds_reference": (0.01 / 0.99) * 9.0,
        },
        "h4_q_volatility_sign_flip_association": {
            "q_delta_vs_prior_keep": summarize(finite_values(events, "q_delta_vs_prior_keep")),
            "event_q_sign_flip_rate": float(np.mean(sign_flip)) if sign_flip.size else None,
            "event_q_sign_flip_reopen_point_biserial": point_biserial(sign_flip.astype(float), reopen_event) if sign_flip.size else None,
            "gaussian_q_variance_reopen_point_biserial": point_biserial(q_var_all, reopen_all),
            "gaussian_q_abs_delta_reopen_point_biserial": h4,
        },
        "recommended_next_ablation": recommend_next_ablation(h1, h2, h3, h4, float(np.mean(sign_flip)) if sign_flip.size else None),
        "output_files": list(OUTPUT_FILES),
    }


def recommend_next_ablation(close_mass_ratio_median: Any, view_ratio_median: Any, odds_q75: Any, q_volatility_corr: Any, sign_flip_rate: Any) -> dict[str, Any]:
    scores = {"visibility/view_consistency": 0.0, "emission_calibration": 0.0, "transition_prior": 0.0}
    reasons = {k: [] for k in scores}
    if close_mass_ratio_median is not None and math.isfinite(float(close_mass_ratio_median)) and abs(math.log(max(float(close_mass_ratio_median), EPS))) > math.log(1.5):
        scores["visibility/view_consistency"] += 1.0
        reasons["visibility/view_consistency"].append("CLOSE raw_mass differs from same-Gaussian preceding KEEP median")
    if view_ratio_median is not None and math.isfinite(float(view_ratio_median)) and float(view_ratio_median) > 1.25:
        scores["visibility/view_consistency"] += 1.0
        reasons["visibility/view_consistency"].append("events occur at larger camera deltas than KEEP median")
    if q_volatility_corr is not None and abs(float(q_volatility_corr)) >= 0.15:
        scores["emission_calibration"] += abs(float(q_volatility_corr)) * 3.0
        reasons["emission_calibration"].append("q volatility is associated with reopen/repeated cohort")
    if sign_flip_rate is not None and float(sign_flip_rate) >= 0.10:
        scores["emission_calibration"] += 1.0
        reasons["emission_calibration"].append("event rows frequently flip q sign relative to KEEP median")
    if odds_q75 is not None and math.isfinite(float(odds_q75)) and float(odds_q75) >= (0.01 / 0.99) * 9.0:
        scores["transition_prior"] += 1.0
        reasons["transition_prior"].append("event transition odds approach/exceed default max odds reference")
    ranking = sorted(scores, key=lambda k: (-scores[k], k))
    return {
        "decision": "D2 always: run the next ablation with this no-GT multiscene diagnostic contract; prioritize the highest flagged factor.",
        "ranking": [{"ablation_focus": k, "score": scores[k], "reasons": reasons[k]} for k in ranking],
        "flags": {
            "visibility_or_view": scores["visibility/view_consistency"] > 0,
            "emission": scores["emission_calibration"] > 0,
            "transition": scores["transition_prior"] > 0,
        },
    }


def scatter(path: Path, title: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, *, hline: float | None = None, vline: float | None = None) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.any(mask):
        if np.sum(mask) > 200_000:
            rng = np.random.default_rng(0)
            keep = rng.choice(np.flatnonzero(mask), size=200_000, replace=False)
            mask2 = np.zeros_like(mask)
            mask2[keep] = True
            mask = mask2
        ax.scatter(x[mask], y[mask], s=5, alpha=0.25, edgecolors="none")
    else:
        ax.text(0.5, 0.5, "no finite samples", ha="center", va="center", transform=ax.transAxes)
    if hline is not None:
        ax.axhline(hline, color="tab:red", linestyle="--", linewidth=1)
    if vline is not None:
        ax.axvline(vline, color="tab:red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_plots(output_dir: Path, scenes: Sequence[SceneInfo], states: Sequence[DenseSceneState], events: Sequence[Mapping[str, Any]], eta: float, seed: int) -> None:
    close = [r for r in events if r.get("action") == "CLOSE"]
    scatter(output_dir / "mass_at_close_vs_keep.png", "CLOSE raw_mass vs preceding KEEP median", finite_values(close, "prior_keep_mass_median"), finite_values(close, "raw_mass"), "median preceding KEEP raw_mass", "CLOSE raw_mass")
    scatter(output_dir / "view_delta_vs_transition.png", "Event view delta vs transition odds", finite_values(events, "view_delta"), finite_values(events, "selected_transition_odds"), "view delta", "selected transition odds", hline=(0.01 / 0.99) * 9.0)
    q_abs: list[float] = []
    repeated: list[float] = []
    for st in states:
        obs = st.observed_mask()
        q_abs.extend(st.q_abs_delta_mean()[obs].tolist())
        repeated.extend((st.cohort_codes()[obs] == 0).astype(float).tolist())
    scatter(output_dir / "q_volatility_vs_reopen.png", "q volatility vs reopen/repeated cohort", np.asarray(q_abs), np.asarray(repeated), "mean |Δq|", "reopen/repeated (1=yes)")
    scatter(output_dir / "transition_odds_at_events.png", "P01/P00 vs P10/P11 at events", finite_values(events, "transition_odds_01"), finite_values(events, "transition_odds_10"), "P01/P00", "P10/P11", hline=(0.01 / 0.99) * 9.0, vline=(0.01 / 0.99) * 9.0)
    representative_trajectory_plot(output_dir / "representative_gaussian_trajectories.png", scenes, states, events, eta=eta, seed=seed)


def representative_trajectory_plot(path: Path, scenes: Sequence[SceneInfo], states: Sequence[DenseSceneState], events: Sequence[Mapping[str, Any]], *, eta: float, seed: int) -> None:
    plt = _load_pyplot()
    del eta
    selected = select_representatives(states, seed=seed, limit=8)
    traj = {key: {"x": [], "q": [], "p": [], "mass": []} for key in selected}
    scene_map = {(s.instance, s.scene): s for s in scenes}
    for key in selected:
        inst, scene, gi = key
        sinfo = scene_map.get((inst, scene))
        if sinfo is None:
            continue
        for frame in sinfo.frames:
            fields = load_frame_fields(frame.path)
            pos = np.flatnonzero(fields["gaussian_index"] == gi)
            if pos.size:
                j = int(pos[0])
                traj[key]["x"].append(frame.index)
                traj[key]["q"].append(float(fields["q"][j]))
                traj[key]["p"].append(float(fields["p_active_post"][j]))
                traj[key]["mass"].append(float(fields["raw_mass"][j]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=150, sharex=True)
    for key, tr in traj.items():
        if not tr["x"]:
            continue
        label = f"{key[0]}/{key[1]}/g{key[2]}"
        axes[0].plot(tr["x"], tr["q"], linewidth=1, label=label)
        axes[1].plot(tr["x"], tr["p"], linewidth=1)
        axes[2].plot(tr["x"], tr["mass"], linewidth=1)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("q")
    axes[1].set_ylabel("P(active)")
    axes[2].set_ylabel("raw_mass")
    axes[2].set_xlabel("frame index")
    axes[0].set_title("Representative Gaussian trajectories (seed 0)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def select_representatives(states: Sequence[DenseSceneState], *, seed: int, limit: int) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    chosen: list[tuple[str, str, int]] = []
    for cohort_code in (0, 1, 2, 3):
        candidates: list[tuple[int, str, str, int]] = []
        for st in states:
            codes = st.cohort_codes()
            idxs = np.flatnonzero(codes == cohort_code)
            for gi in idxs[: min(idxs.size, 1000)]:
                candidates.append((int(st.observed_count[gi]), st.instance, st.scene, int(gi)))
        if candidates:
            _, inst, scene, gi = sorted(candidates, reverse=True)[0]
            chosen.append((inst, scene, gi))
    all_obs: list[tuple[str, str, int]] = []
    for st in states:
        for gi in np.flatnonzero(st.observed_mask())[:5000]:
            all_obs.append((st.instance, st.scene, int(gi)))
    rng.shuffle(all_obs)
    for key in all_obs:
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="D1 output root or one scene directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to <input_root>/view_consistency_analysis")
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--eta", type=float, default=0.9, help="Emission reliability for ell_active=w*(2q-1)*log(eta/(1-eta))")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.5 < float(args.eta) < 1.0:
        raise ValueError("--eta must be in (0.5, 1)")
    root = args.input_root.resolve()
    out = (args.output_dir if args.output_dir is not None else root / "view_consistency_analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    scenes = discover_scene_infos(root, args.manifest_name)
    states: list[DenseSceneState] = []
    events: list[dict[str, Any]] = []
    for scene in scenes:
        state, scene_events = update_scene(scene, eta=float(args.eta))
        states.append(state)
        events.extend(scene_events)
    cohorts = collect_cohort_stats(states)
    write_csv(out / "cohort_stats.csv", cohorts)
    write_csv(out / "event_windows.csv", events)
    summary = build_summary(root, out, scenes, states, events, cohorts, float(args.eta))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plots(out, scenes, states, events, eta=float(args.eta), seed=int(args.seed))
    print(json.dumps({"output_dir": str(out), "scenes": len(scenes), "frames": summary["frame_count"], "events": len(events), "output_files": list(OUTPUT_FILES)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
