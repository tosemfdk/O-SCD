"""D1 detector-only view-consistency diagnostic for PASLCD direct binary lifespans.

The runner performs exactly one causal detector/controller pass per scene using
cached PASLCD O-SCD/SAM cues and fixed cameras. It deliberately does not load GT
masks, run representation optimization, or change detector/controller semantics.
Outputs are sparse observed-row diagnostics under ``outputs/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.prepare_paslcd_fixed_pose_cues import (
    DEFAULT_CUE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INSTANCES,
    DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT,
    DEFAULT_OSCD_OUTPUT_ROOT,
    DEFAULT_SCENES,
    SceneSpec,
    discover_scenes,
    parse_csv,
    sha256_file,
)
from experiments.run_online_binary_state_lifespan_thaw import (
    BASE_PLY_REL,
    LifecycleEvent,
    RunConfig,
    controller_events,
    event_diagnostics,
    filter_config,
    make_controller,
    posterior_run_diagnostics,
    quantile_summary,
    run_config_from_args,
    same_scene_repeated_transition_diagnostics,
    serializable_arguments,
)
from temporal.binary_state_filter import BinaryStateFilter
from temporal.binary_state_lifespan_controller import BinaryLifespanAction

DEFAULT_OUTPUT_ROOT = Path("outputs/paslcd_d1_view_consistency_diagnostic")
DEFAULT_BASELINE_LIFECYCLE_ROOT = Path("outputs/paslcd_direct_binary_state_benchmark_res4/u16/B0_binary_dc")
EVENT_STRUCTURE_FIELDS = (
    "gaussian_index",
    "decision_timestamp",
    "old_binary_label",
    "new_binary_label",
    "action",
    "old_slot",
    "new_current_slot",
)
FRAME_NPZ_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SparseObservedRows:
    frame_index: int
    view_id: str
    observed_gaussian_indices: list[int]
    transition_llr: list[float]
    p_transition: list[float]
    camera_center: list[float]
    camera_forward: list[float]


@dataclass(frozen=True)
class DiagnosticEvent:
    gaussian_index: int
    frame_index: int
    view_id: str
    action: str
    old_binary_label: int
    new_binary_label: int
    old_slot: int
    new_current_slot: int
    transition_llr: float
    p_transition: float


def transition_llr_without_prior(*, p_transition: float, transition_prior: float) -> float:
    p = float(p_transition)
    prior = float(transition_prior)
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError("transition probability must be finite in the open interval (0, 1)")
    if not (math.isfinite(prior) and 0.0 < prior < 1.0):
        raise ValueError("transition prior must be finite in the open interval (0, 1)")
    return float(math.log(p / (1.0 - p)) - math.log(prior / (1.0 - prior)))


def max_transition_llr_without_prior(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("at least one transition LLR is required")
    return max(vals)


def camera_center_from_w2c(w2c: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(w2c, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("w2c must be a 4x4 matrix")
    c2w = np.linalg.inv(matrix)
    return c2w[:3, 3].astype(float).tolist()


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("vector must have finite nonzero norm")
    return arr / norm


def camera_forward_from_w2c(w2c: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(w2c, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("w2c must be a 4x4 matrix")
    c2w = np.linalg.inv(matrix)
    return _normalize_vector(c2w[:3, 2]).astype(float).tolist()


def angular_delta_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    va = _normalize_vector(a)
    vb = _normalize_vector(b)
    return float(np.degrees(np.arccos(float(np.clip(np.dot(va, vb), -1.0, 1.0)))))


def write_sparse_observed_rows_jsonl(path: Path, rows: Sequence[SparseObservedRows]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row)
            payload["schema"] = "paslcd_view_consistency_sparse_observed_rows_v1"
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def read_sparse_observed_rows_jsonl(path: Path) -> list[SparseObservedRows]:
    out: list[SparseObservedRows] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            payload = json.loads(line)
            schema = payload.pop("schema", None)
            if schema != "paslcd_view_consistency_sparse_observed_rows_v1":
                raise ValueError(f"{path}:{line_number} has unsupported schema {schema!r}")
            row = SparseObservedRows(**payload)
            lengths = {
                len(row.observed_gaussian_indices),
                len(row.transition_llr),
                len(row.p_transition),
            }
            if len(lengths) != 1:
                raise ValueError(f"{path}:{line_number} has mismatched sparse-row lengths")
            if len(set(row.observed_gaussian_indices)) != len(row.observed_gaussian_indices):
                raise ValueError(f"{path}:{line_number} contains duplicate Gaussian indices")
            if any(not math.isfinite(float(value)) for value in row.transition_llr):
                raise ValueError(f"{path}:{line_number} contains non-finite transition LLR")
            if any(not 0.0 <= float(value) <= 1.0 for value in row.p_transition):
                raise ValueError(f"{path}:{line_number} contains invalid transition probability")
            out.append(row)
    return out


def merge_sparse_observed_rows(*, num_gaussians: int, sparse: SparseObservedRows) -> list[dict[str, Any]]:
    if int(num_gaussians) < 0:
        raise ValueError("num_gaussians must be nonnegative")
    lengths = {
        len(sparse.observed_gaussian_indices),
        len(sparse.transition_llr),
        len(sparse.p_transition),
    }
    if len(lengths) != 1:
        raise ValueError("sparse observed-row fields must have equal lengths")
    indices = [int(value) for value in sparse.observed_gaussian_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("sparse observed-row indices must be unique")
    if any(value < 0 or value >= int(num_gaussians) for value in indices):
        raise IndexError("sparse observed-row index is out of bounds")
    dense = [
        {"gaussian_index": i, "observed": False, "transition_llr": None, "p_transition": None}
        for i in range(int(num_gaussians))
    ]
    for idx, llr, p_transition in zip(
        sparse.observed_gaussian_indices, sparse.transition_llr, sparse.p_transition
    ):
        dense[int(idx)].update(
            {"observed": True, "transition_llr": float(llr), "p_transition": float(p_transition)}
        )
    return dense


def event_structure_key(event: DiagnosticEvent) -> tuple[Any, ...]:
    return (
        int(event.gaussian_index),
        int(event.frame_index),
        str(event.view_id),
        str(event.action),
        int(event.old_binary_label),
        int(event.new_binary_label),
        int(event.old_slot),
        int(event.new_current_slot),
    )


def event_structure_sha256(events: Sequence[DiagnosticEvent]) -> str:
    digest = hashlib.sha256()
    for key in sorted(event_structure_key(event) for event in events):
        digest.update(json.dumps(key, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_transition_cohorts(events: Sequence[DiagnosticEvent]) -> dict[str, Any]:
    ordered = sorted(events, key=event_structure_key)
    transition_events = [event for event in ordered if event.action in {"OPEN", "CLOSE"}]
    max_by_gaussian: dict[int, float] = {}
    for event in transition_events:
        idx = int(event.gaussian_index)
        max_by_gaussian[idx] = max(max_by_gaussian.get(idx, float("-inf")), float(event.transition_llr))
    return {
        "event_count": len(events),
        "transition_event_count": len(transition_events),
        "unique_gaussian_count": len({int(event.gaussian_index) for event in ordered}),
        "max_transition_llr_by_gaussian": dict(sorted(max_by_gaussian.items())),
        "event_structure_sha256": event_structure_sha256(ordered),
    }


def run_detector_only_view_consistency_diagnostic(
    *,
    output_dir: Path,
    detector,
    gt_loader=None,
    optimizer_step=None,
) -> dict[str, Any]:
    del gt_loader, optimizer_step
    rows = list(detector())
    write_sparse_observed_rows_jsonl(output_dir / "observed_rows.jsonl", rows)
    return {
        "gt_used_for_training": False,
        "gt_loaded": False,
        "optimizer_steps": 0,
        "observed_row_count": int(sum(len(row.observed_gaussian_indices) for row in rows)),
        "frame_count": len(rows),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def finite_probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise argparse.ArgumentTypeError("value must be finite and in (0, 1)")
    return parsed


def require_outputs_path(path: Path) -> Path:
    resolved = path.resolve()
    outputs = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs)
    except ValueError as exc:
        raise ValueError(f"diagnostic output must be under {outputs}: {path}") from exc
    return path


def lifecycle_event_structure_sha256(path: Path, *, max_timestamp: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if max_timestamp is not None and int(event.get("decision_timestamp", -1)) >= int(max_timestamp):
                continue
            structure = [event.get(field) for field in EVENT_STRUCTURE_FIELDS]
            digest.update(json.dumps(structure, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _tensor_checksum(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def base_tensor_snapshots(base: Any) -> dict[str, torch.Tensor]:
    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    return {
        name: getattr(base, name).detach().cpu().clone()
        for name in names
        if isinstance(getattr(base, name, None), torch.Tensor)
    }


def tensor_snapshot_checksums(snapshots: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {name: _tensor_checksum(value) for name, value in sorted(snapshots.items())}


def compare_tensor_snapshots(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(before) != set(after):
        return {"bitwise_equal": False, "max_abs": float("inf"), "per_tensor_max_abs": {}}
    bitwise = True
    max_abs = 0.0
    per_tensor: dict[str, float] = {}
    for name in sorted(before):
        lhs = before[name]
        rhs = after[name].to(dtype=lhs.dtype)
        same = torch.equal(lhs, rhs)
        bitwise = bitwise and bool(same)
        diff = float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0
        per_tensor[name] = diff
        max_abs = max(max_abs, diff)
    return {"bitwise_equal": bool(bitwise), "max_abs": float(max_abs), "per_tensor_max_abs": per_tensor}


def action_name(value: int) -> str:
    return BinaryLifespanAction(int(value)).name


def json_array(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True), dtype=np.str_)


def camera_metadata(camera_json: Mapping[str, Any], *, w2c: np.ndarray) -> dict[str, Any]:
    width = int(camera_json["width"])
    height = int(camera_json["height"])
    fx = float(camera_json["fx"])
    fy = float(camera_json["fy"])
    k = np.asarray(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    c2w = np.linalg.inv(w2c)
    center = c2w[:3, 3].astype(np.float64)
    forward = c2w[:3, 2].astype(np.float64)
    norm = float(np.linalg.norm(forward))
    if norm > 0.0:
        forward = forward / norm
    return {
        "fx": fx,
        "fy": fy,
        "width": width,
        "height": height,
        "intrinsics": k,
        "w2c": w2c.astype(np.float64),
        "c2w": c2w.astype(np.float64),
        "center": center,
        "forward": forward,
        "camera_json": dict(camera_json),
    }




def list_images_no_gt(folder: Path) -> list[str]:
    names = [
        path.name
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
    names.sort()
    if not names:
        raise FileNotFoundError(f"No images found in {folder}")
    return names


def build_causal_records_no_gt(source_path: Path, *, max_frames: int | None = None) -> tuple[list[Any], list[str]]:
    image_dir = source_path / "inference_scene" / "images"
    names = list_images_no_gt(image_dir)
    if max_frames is not None:
        names = names[:max_frames]
    records: list[Any] = []
    segment_ids: dict[str, int] = {}
    for index, name in enumerate(names):
        stem = Path(name).stem
        segment_name = stem.rsplit("_frame_", 1)[0] if "_frame_" in stem else source_path.name
        if segment_name not in segment_ids:
            segment_ids[segment_name] = len(segment_ids)
        records.append(
            SimpleNamespace(
                global_index=index,
                segment_id=segment_ids[segment_name],
                segment_name=segment_name,
                name=name,
                image_path=str(image_dir / name),
                mask_path="",
            )
        )
    return records, names


def write_frame_npz(
    path: Path,
    *,
    spec: SceneSpec,
    frame_record: Any,
    camera: Mapping[str, Any],
    w2c: np.ndarray,
    rows: torch.Tensor,
    evidence: Any,
    pre_p_active: torch.Tensor,
    filter_update: Any,
    decision: Any,
    active_count_before: int,
    active_count_after: int,
    translation_delta_from_previous: float | None,
    angular_delta_from_previous_degrees: float | None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cam = camera_metadata(camera, w2c=w2c)
    cpu_rows = rows.detach().cpu().to(torch.int64).numpy()

    def cpu_float(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().to(torch.float32).numpy()

    action = decision.action.detach().cpu().to(torch.int16).numpy()
    old_slot = decision.old_slot.detach().cpu().to(torch.int64).numpy()
    current_slot = decision.current_slot.detach().cpu().to(torch.int64).numpy()
    visible_observations = decision.visible_observations.detach().cpu().to(torch.int64).numpy()
    action_names = np.asarray([action_name(int(v)) for v in action], dtype=np.str_)
    frame_meta = {
        "schema_version": FRAME_NPZ_SCHEMA_VERSION,
        "dataset": "PASLCD",
        "instance": spec.instance,
        "scene": spec.scene,
        "frame_name": str(frame_record.name),
        "frame_stem": Path(str(frame_record.name)).stem,
        "timestamp": int(frame_record.global_index),
        "segment_id": int(frame_record.segment_id),
        "segment_name": str(getattr(frame_record, "segment_name", spec.scene)),
        "source_image_path": str(frame_record.image_path),
        "fixed_cameras_json": str(spec.cameras_json),
        "cue_cache_root": str(spec.output_dir),
        "active_count_before": int(active_count_before),
        "active_count_after": int(active_count_after),
        "observed_row_count": int(cpu_rows.shape[0]),
        "sparse_absence_semantics": "rows absent from this shard were unobserved and therefore held with no filter/controller update",
        "camera_center": cam["center"].tolist(),
        "camera_forward": cam["forward"].tolist(),
        "translation_delta_from_previous": translation_delta_from_previous,
        "angular_delta_from_previous_degrees": angular_delta_from_previous_degrees,
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(FRAME_NPZ_SCHEMA_VERSION, dtype=np.int64),
        timestamp=np.asarray(int(frame_record.global_index), dtype=np.int64),
        frame_name=np.asarray(str(frame_record.name), dtype=np.str_),
        gaussian_index=cpu_rows,
        e_plus=cpu_float(evidence.e_plus[rows]),
        e_minus=cpu_float(evidence.e_minus[rows]),
        raw_mass=cpu_float(evidence.total_mass[rows]),
        delta_a=cpu_float(evidence.delta_a[rows]),
        delta_b=cpu_float(evidence.delta_b[rows]),
        q=cpu_float(filter_update.q),
        evidence_strength=cpu_float(filter_update.evidence_strength),
        p_active_pre=cpu_float(pre_p_active),
        p_active_post=cpu_float(filter_update.p_active),
        p00=cpu_float(filter_update.p_00),
        p01=cpu_float(filter_update.p_01),
        p10=cpu_float(filter_update.p_10),
        p11=cpu_float(filter_update.p_11),
        pflip=cpu_float(filter_update.p_flip),
        action=action,
        action_name=action_names,
        old_binary_label=decision.old_binary_label.detach().cpu().to(torch.int8).numpy(),
        new_binary_label=decision.new_binary_label.detach().cpu().to(torch.int8).numpy(),
        old_slot=old_slot,
        current_slot=current_slot,
        visible_observation_count=visible_observations,
        active_count_before=np.asarray(int(active_count_before), dtype=np.int64),
        active_count_after=np.asarray(int(active_count_after), dtype=np.int64),
        camera_intrinsics=cam["intrinsics"],
        camera_w2c=cam["w2c"],
        camera_c2w=cam["c2w"],
        camera_center=cam["center"],
        camera_forward=cam["forward"],
        translation_delta_from_previous=np.asarray(np.nan if translation_delta_from_previous is None else translation_delta_from_previous, dtype=np.float64),
        angular_delta_from_previous_degrees=np.asarray(np.nan if angular_delta_from_previous_degrees is None else angular_delta_from_previous_degrees, dtype=np.float64),
        camera_fx=np.asarray(cam["fx"], dtype=np.float64),
        camera_fy=np.asarray(cam["fy"], dtype=np.float64),
        camera_width=np.asarray(cam["width"], dtype=np.int64),
        camera_height=np.asarray(cam["height"], dtype=np.int64),
        camera_json=json_array(cam["camera_json"]),
        frame_metadata=json_array(frame_meta),
    )
    counts = {a.name: int((action == int(a)).sum()) for a in BinaryLifespanAction}
    return {
        **frame_meta,
        "npz": path.name,
        "open_count": counts.get("OPEN", 0),
        "keep_count": counts.get("KEEP", 0),
        "close_count": counts.get("CLOSE", 0),
        "none_count": counts.get("NONE", 0),
        "uncertain_count": counts.get("UNCERTAIN", counts.get("HOLD", 0)),
        "p_active_stats": quantile_summary(filter_update.p_active),
        "p_flip_stats": quantile_summary(filter_update.p_flip),
        "p_01_stats": quantile_summary(filter_update.p_01),
        "p_10_stats": quantile_summary(filter_update.p_10),
        "q_stats": quantile_summary(filter_update.q),
        "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
        "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
        "raw_alpha_t_mass": float(evidence.total_mass.sum().item()),
    }


def run_scene(spec: SceneSpec, args: argparse.Namespace, config: RunConfig) -> dict[str, Any]:
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        camera_json_to_w2c,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import file_checksum, seed_everything
    from gaussian_renderer import render_change  # noqa: F401 - validates renderer availability for alpha-T evidence
    from scene import GaussianModel
    from temporal import TemporalGeometryChangeModel
    from temporal.change_evidence import accumulate_change_evidence

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    seed_everything(config.seed)
    scene_started = time.time()
    records, names = build_causal_records_no_gt(spec.source_path, max_frames=args.max_frames)
    timestamps = [int(record.global_index) for record in records]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise RuntimeError("views are not in strict global timestamp order")
    base_ply = (spec.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(spec.output_dir, base_ply, args.resolution)
    camera_checksum = file_checksum(spec.cameras_json)
    cached_camera_checksum = cue_metadata.get("fixed_cameras_sha256")
    if cached_camera_checksum is not None and cached_camera_checksum != camera_checksum:
        raise ValueError(
            "cue cache/fixed camera checksum mismatch: "
            f"{cached_camera_checksum} != {camera_checksum}"
        )
    cameras = load_fixed_camera_index(spec.cameras_json)
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    base_before = base_tensor_snapshots(base)
    base_before_checksums = tensor_snapshot_checksums(base_before)
    gaussian_count = int(base.get_xyz.shape[0])
    model = TemporalGeometryChangeModel.from_gaussians(base, max_states=config.max_states)
    model.reset_all_lifespans_closed()
    device = base.get_xyz.device
    dtype = base.get_xyz.dtype
    tracker = BinaryStateFilter(
        gaussian_count,
        filter_config(config),
        device=device,
        dtype=dtype,
    )
    controller = make_controller(model, config)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=dtype, device=device)
    scene_dir = args.output_root / spec.instance / spec.scene
    frame_dir = scene_dir / "frames"
    scene_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    events: list[LifecycleEvent] = []
    intrinsics: np.ndarray | None = None
    torch.cuda.reset_peak_memory_stats()
    previous_center: np.ndarray | None = None
    previous_forward: np.ndarray | None = None
    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        views, _poses, current_intrinsics = build_fixed_cue_views(
            [record], cameras, spec.output_dir, args.resolution
        )
        view = views[0]
        if intrinsics is None:
            intrinsics = current_intrinsics
        elif not np.array_equal(intrinsics, current_intrinsics):
            raise ValueError("fixed views do not share one intrinsic matrix")
        evidence = accumulate_change_evidence(
            view,
            base,
            pipe,
            background,
            view.candidate_map,
            cue_mode=config.bayes_cue_mode,
            cue_threshold=config.bayes_cue_threshold,
            cue_scale=config.bayes_cue_scale,
            count_mode=config.evidence_count_mode,
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
        )
        observed = (
            (evidence.total_mass > 0)
            & (evidence.total_mass >= float(config.min_evidence_mass))
            & ((evidence.delta_a + evidence.delta_b) > 0)
        )
        rows = torch.nonzero(observed, as_tuple=False).flatten()
        active_before = int((model.current_state_index >= 0).sum().item())
        pre_p = tracker.p_active[rows].clone()
        update = tracker.update(
            evidence.delta_a[rows],
            evidence.delta_b[rows],
            total_mass=evidence.total_mass[rows],
            indices=rows,
            timestamp=timestamp,
        )
        decision = controller.update(update, timestamp=timestamp, optimizer=None)
        frame_events = controller_events(decision)
        events.extend(frame_events)
        active_after = int((model.current_state_index >= 0).sum().item())
        stem = Path(str(record.name)).stem
        cam = cameras[stem]
        w2c = camera_json_to_w2c(cam)
        cam_meta = camera_metadata(cam, w2c=w2c)
        center = cam_meta["center"]
        forward = cam_meta["forward"]
        if previous_center is None or previous_forward is None:
            translation_delta = None
            angular_delta_degrees = None
        else:
            translation_delta = float(np.linalg.norm(center - previous_center))
            dot = float(np.clip(np.dot(forward, previous_forward), -1.0, 1.0))
            angular_delta_degrees = float(np.degrees(np.arccos(dot)))
        previous_center = center.copy()
        previous_forward = forward.copy()
        npz_path = frame_dir / f"{timestamp:06d}_{stem}.npz"
        row = write_frame_npz(
            npz_path,
            spec=spec,
            frame_record=record,
            camera=cam,
            w2c=w2c,
            rows=rows,
            evidence=evidence,
            pre_p_active=pre_p,
            filter_update=update,
            decision=decision,
            active_count_before=active_before,
            active_count_after=active_after,
            translation_delta_from_previous=translation_delta,
            angular_delta_from_previous_degrees=angular_delta_degrees,
        )
        row["frame_runtime_seconds"] = float(time.time() - frame_started)
        row["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        row["event_count"] = len(frame_events)
        frame_rows.append(row)
    base_after = base_tensor_snapshots(base)
    base_after_checksums = tensor_snapshot_checksums(base_after)
    base_audit = compare_tensor_snapshots(base_before, base_after)
    if not base_audit["bitwise_equal"]:
        raise RuntimeError(f"immutable base tensor drift detected: {base_audit}")
    evdiag = event_diagnostics(events)
    same_scene_diag = same_scene_repeated_transition_diagnostics(events, ())
    lifecycle_path = scene_dir / "lifecycle_events.jsonl"
    with lifecycle_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    lifecycle_sha = lifecycle_event_structure_sha256(lifecycle_path)
    baseline_path = baseline_event_path(args, spec)
    baseline_check: dict[str, Any] | None = None
    if baseline_path is not None:
        baseline_check = {"path": str(baseline_path), "exists": baseline_path.exists()}
        if baseline_path.exists():
            baseline_sha = lifecycle_event_structure_sha256(baseline_path, max_timestamp=args.max_frames)
            baseline_check.update(
                {
                    "structure_sha256": baseline_sha,
                    "matches": baseline_sha == lifecycle_sha,
                    "max_timestamp": args.max_frames,
                }
            )
            if not args.allow_baseline_mismatch and not baseline_check["matches"]:
                raise RuntimeError(
                    f"baseline lifecycle event structure mismatch for {spec.instance}/{spec.scene}: "
                    f"{lifecycle_sha} != {baseline_sha}"
                )
    summary = {
        "schema_version": 1,
        "script": "experiments/run_paslcd_view_consistency_diagnostic.py",
        "diagnostic_id": "D1_detector_only_view_consistency",
        "contract": "detector_only_sparse_observed_rows_no_gt_no_optimizer",
        "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
        "transition_equation": "normalize [(1-b)(1-p01)L0, (1-b)p01L1, bp10L0, b(1-p10)L1]; p_active=P01+P11; p_flip=P01+P10",
        "runtime_seconds": float(time.time() - scene_started),
        "dataset": "PASLCD",
        "instance": spec.instance,
        "scene": spec.scene,
        "source_path": str(spec.source_path),
        "base_ply": str(base_ply),
        "base_ply_sha256": sha256_file(base_ply),
        "fixed_cameras_json": str(spec.cameras_json),
        "fixed_cameras_sha256": camera_checksum,
        "fixed_camera_source": spec.camera_source,
        "cue_cache_root": str(spec.output_dir),
        "cue_cache_metadata": cue_metadata,
        "run_config": asdict(config),
        "run_arguments": serializable_arguments(args),
        "one_detector_pass_per_scene": True,
        "sparse_absence_semantics": "rows absent from a frame NPZ shard were unobserved and therefore held with no filter/controller update",
        "base_bitwise_equal": bool(base_audit["bitwise_equal"]),
        "base_max_drift": float(base_audit["max_abs"]),
        "base_checksum_before": base_before_checksums,
        "base_checksum_after": base_after_checksums,
        "base_checksum_equal": base_before_checksums == base_after_checksums,
        "gt_used": False,
        "gt_used_for_training": False,
        "gt_loaded_after_inference_only": False,
        "manual_boundaries_used_for_inference": False,
        "optimizer_used": False,
        "representation_training_used": False,
        "fixed_topology": True,
        "frames": len(frame_rows),
        "processed_frame_count": len(names),
        "gaussian_count": gaussian_count,
        "event_count": len(events),
        "observed_row_count": int(sum(row["observed_row_count"] for row in frame_rows)),
        **evdiag,
        **same_scene_diag,
        "keep_count": int(sum(row["keep_count"] for row in frame_rows)),
        "uncertain_count": int(sum(row["uncertain_count"] for row in frame_rows)),
        "none_count": int(sum(row["none_count"] for row in frame_rows)),
        "final_active_gs": int((model.current_state_index >= 0).sum().item()),
        "posterior_diagnostics": posterior_run_diagnostics(frame_rows),
        "camera_intrinsics": None if intrinsics is None else intrinsics.tolist(),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "binary_filter_resident_state_bytes": int(
            tracker.p_active.numel() * tracker.p_active.element_size()
            + tracker.visible_observations.numel() * tracker.visible_observations.element_size()
            + tracker.last_timestamp.numel() * tracker.last_timestamp.element_size()
        ),
        "frame_npz_schema_version": FRAME_NPZ_SCHEMA_VERSION,
        "frame_npz_count": len(frame_rows),
        "expected_frame_npz": [str(Path("frames") / Path(row["npz"]).name) for row in frame_rows],
        "lifecycle_event_structure_sha256": lifecycle_sha,
        "baseline_lifecycle_event_structure": baseline_check,
        "output_files": ["summary.json", "manifest.json", "frame_metrics.csv", "lifecycle_events.jsonl", "frames/*.npz"],
    }
    manifest = {
        "schema_version": 1,
        "diagnostic_id": summary["diagnostic_id"],
        "summary_json": "summary.json",
        "frame_count": len(frame_rows),
        "frames": _manifest_frames_with_relative_npz(frame_rows),
        "scene_summary": {k: v for k, v in summary.items() if k not in {"cue_cache_metadata", "run_arguments"}},
    }
    write_scene_tables(scene_dir, frame_rows)
    (scene_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def baseline_event_path(args: argparse.Namespace, spec: SceneSpec) -> Path | None:
    if args.baseline_lifecycle_events is not None:
        return args.baseline_lifecycle_events
    if args.baseline_lifecycle_root is None:
        args.baseline_lifecycle_root = DEFAULT_BASELINE_LIFECYCLE_ROOT
    return args.baseline_lifecycle_root / spec.instance / spec.scene / "lifecycle_events.jsonl"


def baseline_match_status(args: argparse.Namespace, spec: SceneSpec, summary: Mapping[str, Any]) -> dict[str, Any] | None:
    path = baseline_event_path(args, spec)
    if path is None or not path.exists():
        return None
    baseline_sha = lifecycle_event_structure_sha256(path, max_timestamp=args.max_frames)
    current_sha = str(summary.get("lifecycle_event_structure_sha256", ""))
    return {"path": str(path), "structure_sha256": baseline_sha, "matches": baseline_sha == current_sha}


def _manifest_frames_with_relative_npz(frame_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for row in frame_rows:
        copied = dict(row)
        copied["npz"] = str(Path("frames") / Path(str(row["npz"])).name)
        frames.append(copied)
    return frames


def scene_outputs_complete(scene_dir: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    required = ("summary.json", "manifest.json", "frame_metrics.csv", "lifecycle_events.jsonl")
    if not all((scene_dir / name).is_file() for name in required):
        return False
    frames = manifest.get("frames", [])
    if len(frames) != int(summary.get("frames", -1)):
        return False
    for frame in frames:
        rel = Path(str(frame.get("npz", "")))
        if rel.is_absolute() or ".." in rel.parts or not (scene_dir / rel).is_file():
            return False
    return True


def load_resumable_scene_summary(
    args: argparse.Namespace,
    spec: SceneSpec,
    config: RunConfig,
) -> dict[str, Any] | None:
    scene_dir = args.output_root / spec.instance / spec.scene
    summary_path = scene_dir / "summary.json"
    manifest_path = scene_dir / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not scene_outputs_complete(scene_dir, summary, manifest):
        return None
    expected_frames = len(
        list_images_no_gt(spec.source_path / "inference_scene" / "images")
    )
    if args.max_frames is not None:
        expected_frames = min(expected_frames, int(args.max_frames))
    if int(summary.get("processed_frame_count", -1)) != expected_frames:
        return None
    if json.dumps(summary.get("run_config"), sort_keys=True) != json.dumps(
        asdict(config), sort_keys=True
    ):
        return None
    expected_paths = {
        "source_path": spec.source_path,
        "fixed_cameras_json": spec.cameras_json,
        "cue_cache_root": spec.output_dir,
    }
    for key, expected in expected_paths.items():
        try:
            cached = Path(str(summary.get(key, ""))).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if cached != expected.resolve():
            return None
    base_ply = (spec.source_path / BASE_PLY_REL).resolve()
    if summary.get("base_ply_sha256") != sha256_file(base_ply):
        return None
    if summary.get("fixed_cameras_sha256") != sha256_file(spec.cameras_json):
        return None
    baseline = summary.get("baseline_lifecycle_event_structure")
    if baseline is not None and not bool(baseline.get("matches")) and not args.allow_baseline_mismatch:
        return None
    current_baseline = baseline_match_status(args, spec, summary)
    if current_baseline is not None:
        if not current_baseline["matches"] and not args.allow_baseline_mismatch:
            return None
        summary["baseline_lifecycle_event_structure"] = current_baseline
    return dict(summary)


def write_scene_tables(scene_dir: Path, frame_rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        flat_rows.append(
            {
                "timestamp": row["timestamp"],
                "frame_name": row["frame_name"],
                "npz": str(Path("frames") / Path(row["npz"]).name),
                "observed_row_count": row["observed_row_count"],
                "open_count": row["open_count"],
                "keep_count": row["keep_count"],
                "close_count": row["close_count"],
                "none_count": row["none_count"],
                "uncertain_count": row["uncertain_count"],
                "active_count_before": row["active_count_before"],
                "active_count_after": row["active_count_after"],
                "positive_pseudocount_mass": row["positive_pseudocount_mass"],
                "negative_pseudocount_mass": row["negative_pseudocount_mass"],
                "raw_alpha_t_mass": row["raw_alpha_t_mass"],
                "frame_runtime_seconds": row["frame_runtime_seconds"],
                "cuda_peak_memory_bytes": row["cuda_peak_memory_bytes"],
            }
        )
    with (scene_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]) if flat_rows else ["timestamp"])
        writer.writeheader()
        writer.writerows(flat_rows)


def aggregate_report(summaries: Sequence[Mapping[str, Any]], args: argparse.Namespace, started: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script": "experiments/run_paslcd_view_consistency_diagnostic.py",
        "diagnostic_id": "D1_detector_only_view_consistency",
        "output_root": str(args.output_root),
        "dataset_root": str(args.dataset_root),
        "cue_root": str(args.cue_root),
        "scene_count": len(summaries),
        "frame_count": int(sum(int(s["frames"]) for s in summaries)),
        "event_count": int(sum(int(s["event_count"]) for s in summaries)),
        "open_count": int(sum(int(s["open_count"]) for s in summaries)),
        "close_count": int(sum(int(s["close_count"]) for s in summaries)),
        "reopen_count": int(sum(int(s["reopen_count"]) for s in summaries)),
        "keep_count": int(sum(int(s["keep_count"]) for s in summaries)),
        "uncertain_count": int(sum(int(s["uncertain_count"]) for s in summaries)),
        "observed_row_count": int(sum(int(s.get("observed_row_count", 0)) for s in summaries)),
        "same_scene_repeated_transition_event_count": int(sum(int(s.get("same_scene_repeated_transition_event_count", 0)) for s in summaries)),
        "same_scene_repeated_gaussian_count": int(sum(int(s.get("same_scene_repeated_gaussian_count", 0)) for s in summaries)),
        "base_bitwise_equal": bool(all(bool(s.get("base_bitwise_equal", False)) for s in summaries)),
        "base_max_drift": max((float(s.get("base_max_drift", 0.0)) for s in summaries), default=0.0),
        "runtime_seconds": float(time.time() - started),
        "gt_used": False,
        "optimizer_used": False,
        "representation_training_used": False,
        "one_detector_pass_per_scene": True,
        "baseline_structure_mismatches": int(
            sum(
                s.get("baseline_lifecycle_event_structure") is not None
                and not bool(s["baseline_lifecycle_event_structure"].get("matches"))
                for s in summaries
            )
        ),
        "scenes": [
            {
                "instance": s["instance"],
                "scene": s["scene"],
                "frames": s["frames"],
                "event_count": s["event_count"],
                "open_count": s["open_count"],
                "close_count": s["close_count"],
                "reopen_count": s["reopen_count"],
                "observed_row_count": s.get("observed_row_count", 0),
                "same_scene_repeated_transition_event_count": s.get("same_scene_repeated_transition_event_count", 0),
                "same_scene_repeated_gaussian_count": s.get("same_scene_repeated_gaussian_count", 0),
                "base_bitwise_equal": s.get("base_bitwise_equal"),
                "base_max_drift": s.get("base_max_drift"),
                "lifecycle_event_structure_sha256": s["lifecycle_event_structure_sha256"],
                "summary_json": str(args.output_root / s["instance"] / s["scene"] / "summary.json"),
                "manifest_json": str(args.output_root / s["instance"] / s["scene"] / "manifest.json"),
            }
            for s in summaries
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--oscd-output-root", type=Path, default=DEFAULT_OSCD_OUTPUT_ROOT)
    parser.add_argument("--oscd-fallback-output-root", type=Path, default=DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT)
    parser.add_argument("--cue-root", type=Path, default=DEFAULT_CUE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--instances", type=str, default=None)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--bayes-cue-mode", choices=("binary", "soft"), default="binary")
    parser.add_argument("--bayes-cue-threshold", type=finite_probability, default=0.5)
    parser.add_argument("--bayes-cue-scale", type=float, default=1.0)
    parser.add_argument("--evidence-count-mode", choices=("raw", "capped"), default="capped")
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=float, default=1e-6)
    parser.add_argument("--state-emission-reliability", type=float, default=0.9)
    parser.add_argument("--inactive-to-active-prior", type=float, default=0.01)
    parser.add_argument("--active-to-inactive-prior", type=float, default=0.01)
    parser.add_argument("--initial-active-probability", type=float, default=0.5)
    parser.add_argument("--filter-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--open-probability", type=float, default=0.6)
    parser.add_argument("--close-probability", type=float, default=0.4)
    parser.add_argument("--max-states", type=positive_int, default=8)
    parser.add_argument("--updates-per-frame", type=positive_int, default=1, help="Accepted for RunConfig compatibility; D1 does not optimize and uses one detector update per frame.")
    parser.add_argument("--thaw-parameters", default="dc", help="Accepted for RunConfig compatibility; ignored because D1 is detector-only.")
    parser.add_argument("--detector-only", action="store_true", default=True)
    parser.add_argument("--evaluation-threshold", type=finite_probability, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-lifecycle-events", type=Path, default=None)
    parser.add_argument("--baseline-lifecycle-root", type=Path, default=DEFAULT_BASELINE_LIFECYCLE_ROOT)
    parser.add_argument("--allow-baseline-mismatch", action="store_true", help="Do not fail when the baseline lifecycle structure differs; intended only for partial smoke diagnostics.")
    parser.add_argument("--resume", action="store_true", help="Skip completed scenes after validating manifest shards and baseline match status.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root = require_outputs_path(args.output_root)
    # Force the no-training D1 contract while preserving the direct binary filter config surface.
    args.detector_only = True
    args.updates_per_frame = 1
    config = run_config_from_args(args)
    specs = discover_scenes(
        args.dataset_root,
        args.oscd_output_root,
        args.cue_root,
        instances=parse_csv(args.instances, DEFAULT_INSTANCES),
        scenes=parse_csv(args.scenes, DEFAULT_SCENES),
        oscd_fallback_output_root=args.oscd_fallback_output_root,
        resolution=float(args.resolution),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        if args.resume:
            cached = load_resumable_scene_summary(args, spec, config)
            if cached is not None:
                print(f"RESUME D1 {spec.instance}/{spec.scene}", flush=True)
                summaries.append(cached)
                continue
        print(f"RUN D1 {spec.instance}/{spec.scene}", flush=True)
        summaries.append(run_scene(spec, args, config))
    report = aggregate_report(summaries, args, started)
    (args.output_root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
