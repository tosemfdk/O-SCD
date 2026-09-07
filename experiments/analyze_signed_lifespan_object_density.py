"""Posthoc object-mask analysis for signed lifespan score densification.

This tool intentionally runs *after* a causal runner finishes.  It reads runner
source-event logs plus object masks and never feeds object/GT masks back into a
causal experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

DEFAULT_NEW_OBJECT = Path(
    "data/Instance_1/scene_change3/object_masks/inference_base__object_004"
)
DEFAULT_REMOVED_OBJECT = Path(
    "data/Instance_1/scene_change3/object_masks/reference_render_base__object_010"
)

FRAME_ALIASES = ("frame_name", "frame", "image_name", "view_name", "name")
TIMESTAMP_ALIASES = ("timestamp", "frame_index", "frame_idx", "t")
X_ALIASES = ("x", "screen_x", "projected_x", "pixel_x", "u", "mean2d_x")
Y_ALIASES = ("y", "screen_y", "projected_y", "pixel_y", "v", "mean2d_y")
WIDTH_ALIASES = ("image_width", "width", "W", "w")
HEIGHT_ALIASES = ("image_height", "height", "H", "h")
ACTION_ALIASES = ("action", "event", "density_action", "topology_action")
SCORE_ALIASES = ("density_score", "score", "signed_lifespan_score", "signed_score")
SUPPORT_ALIASES = ("signed_support", "support", "signed_cue_support")
GENERATION_ALIASES = ("generation", "gen", "source_generation")
POSITIVE_ACTION_TOKENS = ("clone", "split", "grow", "densify", "birth", "positive")
NEGATIVE_ACTION_TOKENS = ("prune", "suppress", "attenuate", "negative", "remove")


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    path: Path
    expected_sign: int


@dataclass(frozen=True)
class MaskSeries:
    label: str
    path: Path
    expected_sign: int
    by_stem: Mapping[str, np.ndarray]
    by_index: Sequence[tuple[str, np.ndarray]]
    height: int
    width: int


def _first(row: Mapping[str, Any], aliases: Sequence[str], default: Any = None) -> Any:
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(obj)
    return rows


def load_event_rows(path: Path) -> list[dict[str, Any]]:
    """Load CSV, JSONL, or a JSON list of source-level density events."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("events", "rows", "sources"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"JSON input must be a list or contain events/rows/sources: {path}")
        if not all(isinstance(row, dict) for row in data):
            raise ValueError(f"JSON event list must contain objects: {path}")
        return list(data)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _image_files(mask_dir: Path) -> list[Path]:
    if mask_dir.is_file():
        return [mask_dir]
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(mask_dir.glob(pattern))
    return sorted(files)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return array > 0


def load_mask_series(spec: ObjectSpec) -> MaskSeries:
    files = _image_files(spec.path)
    if not files:
        raise FileNotFoundError(f"no mask images found for {spec.label}: {spec.path}")
    items = [(path.stem, _load_mask(path)) for path in files]
    shapes = {mask.shape for _, mask in items}
    if len(shapes) != 1:
        raise ValueError(f"mask sizes differ for {spec.label}: {sorted(shapes)}")
    height, width = items[0][1].shape
    by_stem = {stem: mask for stem, mask in items}
    return MaskSeries(
        label=spec.label,
        path=spec.path,
        expected_sign=spec.expected_sign,
        by_stem=by_stem,
        by_index=items,
        height=height,
        width=width,
    )


def load_prediction_masks(prediction_dir: Path) -> Mapping[str, np.ndarray]:
    """Load optional posthoc prediction masks keyed by filename stem."""
    files = _image_files(prediction_dir)
    if not files:
        raise FileNotFoundError(f"no prediction mask images found: {prediction_dir}")
    return {path.stem: _load_mask(path) for path in files}


def _prediction_for_object_frame(
    predictions: Mapping[str, np.ndarray],
    frame_name: str,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    pred = predictions.get(frame_name)
    if pred is None:
        for candidate, mask in predictions.items():
            if candidate in frame_name or frame_name in candidate:
                pred = mask
                break
    if pred is None:
        return None
    if pred.shape == target_shape:
        return pred
    image = Image.fromarray((pred.astype(np.uint8) * 255), mode="L")
    resized = image.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def analyze_prediction_masks(
    objects: Sequence[MaskSeries],
    prediction_dir: Path,
) -> dict[str, Any]:
    """Posthoc object support covered by runner prediction masks."""
    predictions = load_prediction_masks(prediction_dir)
    object_summaries: dict[str, dict[str, Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    for obj in objects:
        total_object_pixels = 0
        total_predicted_inside = 0
        total_prediction_pixels = 0
        matched_frames = 0
        missing_frames = 0
        for frame_name, object_mask in obj.by_index:
            pred = _prediction_for_object_frame(predictions, frame_name, object_mask.shape)
            object_pixels = int(object_mask.sum())
            if pred is None:
                if object_pixels > 0:
                    missing_frames += 1
                continue
            matched_frames += 1
            predicted_inside = int((pred & object_mask).sum())
            prediction_pixels = int(pred.sum())
            total_object_pixels += object_pixels
            total_predicted_inside += predicted_inside
            total_prediction_pixels += prediction_pixels
            if object_pixels > 0 or prediction_pixels > 0:
                frame_rows.append(
                    {
                        "object": obj.label,
                        "frame_name": frame_name,
                        "object_pixels": object_pixels,
                        "predicted_inside_object_pixels": predicted_inside,
                        "prediction_pixels": prediction_pixels,
                        "predicted_object_fraction": predicted_inside / object_pixels if object_pixels else None,
                        "object_fraction_of_prediction": predicted_inside / prediction_pixels if prediction_pixels else None,
                    }
                )
        object_summaries[obj.label] = {
            "expected_sign": obj.expected_sign,
            "matched_prediction_frames": matched_frames,
            "missing_active_object_prediction_frames": missing_frames,
            "object_pixels": total_object_pixels,
            "predicted_inside_object_pixels": total_predicted_inside,
            "prediction_pixels": total_prediction_pixels,
            "predicted_object_fraction": (
                total_predicted_inside / total_object_pixels if total_object_pixels else None
            ),
            "object_fraction_of_prediction": (
                total_predicted_inside / total_prediction_pixels if total_prediction_pixels else None
            ),
        }
    return {
        "prediction_mask_dir": str(prediction_dir),
        "objects": object_summaries,
        "per_frame": frame_rows,
    }


def _mask_for_event(series: MaskSeries, row: Mapping[str, Any]) -> tuple[str, np.ndarray] | None:
    frame_name = _first(row, FRAME_ALIASES)
    if frame_name is not None:
        stem = Path(str(frame_name)).stem
        if stem in series.by_stem:
            return stem, series.by_stem[stem]
        # Accept logs that include suffixes/prefixes around the actual mask stem.
        for candidate, mask in series.by_stem.items():
            if candidate in stem or stem in candidate:
                return candidate, mask
    timestamp = _first(row, TIMESTAMP_ALIASES)
    if timestamp is not None:
        idx = _as_int(timestamp, -1)
        if 0 <= idx < len(series.by_index):
            return series.by_index[idx]
    return None


def _event_xy(row: Mapping[str, Any], mask_width: int, mask_height: int) -> tuple[int, int] | None:
    x_raw = _first(row, X_ALIASES)
    y_raw = _first(row, Y_ALIASES)
    if x_raw in (None, "") or y_raw in (None, ""):
        return None
    x = _as_float(x_raw, np.nan)
    y = _as_float(y_raw, np.nan)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    width = _as_float(_first(row, WIDTH_ALIASES), float(mask_width))
    height = _as_float(_first(row, HEIGHT_ALIASES), float(mask_height))
    if width > 0 and height > 0 and (abs(width - mask_width) > 1e-3 or abs(height - mask_height) > 1e-3):
        x *= mask_width / width
        y *= mask_height / height
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    if xi < 0 or yi < 0 or xi >= mask_width or yi >= mask_height:
        return None
    return xi, yi


def _action_class(action: str) -> str:
    lower = action.lower()
    if any(token in lower for token in POSITIVE_ACTION_TOKENS):
        return "positive_growth"
    if any(token in lower for token in NEGATIVE_ACTION_TOKENS):
        return "negative_suppress"
    return "other"


def _score(row: Mapping[str, Any]) -> float:
    value = _first(row, SCORE_ALIASES)
    if value is None:
        value = _first(row, SUPPORT_ALIASES)
    return _as_float(value, 0.0)


def _empty_stats() -> dict[str, Any]:
    return {
        "inside_event_count": 0,
        "positive_score_count": 0,
        "negative_score_count": 0,
        "zero_score_count": 0,
        "correct_sign_count": 0,
        "wrong_sign_count": 0,
        "positive_growth_count": 0,
        "negative_suppress_count": 0,
        "other_action_count": 0,
        "child_generation_count": 0,
        "score_sum": 0.0,
        "abs_score_sum": 0.0,
        "positive_score_sum": 0.0,
        "negative_score_sum": 0.0,
    }


def _update_stats(stats: dict[str, Any], row: Mapping[str, Any], score: float, action_class: str) -> None:
    stats["inside_event_count"] += 1
    stats["score_sum"] += score
    stats["abs_score_sum"] += abs(score)
    if score > 0:
        stats["positive_score_count"] += 1
        stats["positive_score_sum"] += score
    elif score < 0:
        stats["negative_score_count"] += 1
        stats["negative_score_sum"] += score
    else:
        stats["zero_score_count"] += 1
    if action_class == "positive_growth":
        stats["positive_growth_count"] += 1
    elif action_class == "negative_suppress":
        stats["negative_suppress_count"] += 1
    else:
        stats["other_action_count"] += 1
    if _as_int(_first(row, GENERATION_ALIASES), 0) > 0:
        stats["child_generation_count"] += 1


def _finalize_stats(stats: dict[str, Any], expected_sign: int) -> dict[str, Any]:
    count = int(stats["inside_event_count"])
    correct = int(stats["correct_sign_count"])
    wrong = int(stats["wrong_sign_count"])
    out = dict(stats)
    out["expected_sign"] = expected_sign
    out["mean_score"] = stats["score_sum"] / count if count else 0.0
    out["mean_abs_score"] = stats["abs_score_sum"] / count if count else 0.0
    out["sign_correct_fraction"] = correct / (correct + wrong) if (correct + wrong) else None
    out["positive_growth_fraction"] = stats["positive_growth_count"] / count if count else 0.0
    out["negative_suppress_fraction"] = stats["negative_suppress_count"] / count if count else 0.0
    return out


def analyze_events(
    rows: Sequence[Mapping[str, Any]],
    objects: Sequence[MaskSeries],
) -> dict[str, Any]:
    totals = {obj.label: _empty_stats() for obj in objects}
    per_frame: dict[tuple[str, str], dict[str, Any]] = {}
    matched_events = 0
    missing_xy = 0
    unmatched_frame = 0

    for row in rows:
        row_matched_any = False
        for obj in objects:
            frame_mask = _mask_for_event(obj, row)
            if frame_mask is None:
                continue
            frame_name, mask = frame_mask
            xy = _event_xy(row, obj.width, obj.height)
            if xy is None:
                missing_xy += 1
                continue
            x, y = xy
            if not mask[y, x]:
                continue
            score = _score(row)
            action = str(_first(row, ACTION_ALIASES, ""))
            action_class = _action_class(action)
            signed = 1 if score > 0 else (-1 if score < 0 else 0)
            key = (obj.label, frame_name)
            frame_stats = per_frame.setdefault(
                key,
                {"object": obj.label, "frame_name": frame_name, **_empty_stats()},
            )
            for stats in (totals[obj.label], frame_stats):
                _update_stats(stats, row, score, action_class)
                if signed != 0:
                    if signed == obj.expected_sign:
                        stats["correct_sign_count"] += 1
                    else:
                        stats["wrong_sign_count"] += 1
            row_matched_any = True
        if row_matched_any:
            matched_events += 1
        else:
            # Count only events with an identifiable frame as spatial misses.  This
            # distinguishes empty masks from schema/frame-name mistakes.
            if any(_mask_for_event(obj, row) is not None for obj in objects):
                pass
            else:
                unmatched_frame += 1

    object_summaries = {
        label: _finalize_stats(stats, next(obj.expected_sign for obj in objects if obj.label == label))
        for label, stats in totals.items()
    }
    frame_rows = []
    for (label, frame_name), stats in sorted(per_frame.items()):
        expected_sign = next(obj.expected_sign for obj in objects if obj.label == label)
        frame_rows.append(_finalize_stats(stats, expected_sign) | {"object": label, "frame_name": frame_name})
    return {
        "schema_version": 1,
        "causal_gt_contract": "object masks are loaded only by this posthoc analyzer",
        "event_count": len(rows),
        "matched_inside_event_rows": matched_events,
        "missing_xy_rows": missing_xy,
        "unmatched_frame_rows": unmatched_frame,
        "objects": object_summaries,
        "per_frame": frame_rows,
    }


def write_per_frame_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = list(summary.get("per_frame", []))
    fieldnames = [
        "object",
        "frame_name",
        "expected_sign",
        "inside_event_count",
        "correct_sign_count",
        "wrong_sign_count",
        "sign_correct_fraction",
        "positive_growth_count",
        "negative_suppress_count",
        "other_action_count",
        "positive_score_count",
        "negative_score_count",
        "mean_score",
        "mean_abs_score",
        "score_sum",
        "positive_score_sum",
        "negative_score_sum",
        "child_generation_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_object_spec(text: str) -> ObjectSpec:
    """Parse LABEL=PATH[:SIGN], where SIGN is new/add/+ or removed/remove/-."""
    if "=" not in text:
        raise argparse.ArgumentTypeError("object spec must be LABEL=PATH[:SIGN]")
    label, rest = text.split("=", 1)
    path_text, sep, sign_text = rest.rpartition(":")
    if not sep or sign_text.lower() not in {"new", "add", "+", "positive", "removed", "remove", "-", "negative"}:
        path_text = rest
        sign_text = "+" if "new" in label.lower() or "add" in label.lower() else "-"
    sign = 1 if sign_text.lower() in {"new", "add", "+", "positive"} else -1
    return ObjectSpec(label=label, path=Path(path_text), expected_sign=sign)


def default_object_specs() -> list[ObjectSpec]:
    return [
        ObjectSpec("sc3_new_object_004", DEFAULT_NEW_OBJECT, 1),
        ObjectSpec("sc3_removed_object_010", DEFAULT_REMOVED_OBJECT, -1),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="density-source CSV/JSONL/JSON from causal runner")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--prediction-mask-dir",
        type=Path,
        default=None,
        help="Optional posthoc runner thresholded_render directory for object-support prediction stats.",
    )
    parser.add_argument(
        "--object",
        dest="objects",
        action="append",
        type=parse_object_spec,
        help="LABEL=MASK_DIR[:new|removed|+|-]. Repeatable. Defaults to SC3 object_004 NEW and object_010 REMOVED.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    specs = args.objects if args.objects else default_object_specs()
    rows = load_event_rows(args.events)
    masks = [load_mask_series(spec) for spec in specs]
    summary = analyze_events(rows, masks)
    if args.prediction_mask_dir is not None:
        summary["prediction_masks"] = analyze_prediction_masks(masks, args.prediction_mask_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_per_frame_csv(args.output_csv, summary)


if __name__ == "__main__":
    main()
