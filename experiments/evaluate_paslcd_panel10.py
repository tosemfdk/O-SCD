"""Evaluate the unchanged uncapped Panel10 viewer on independent PASLCD scenes.

Preparation supplies scene-local, causal SAM/DA3/Stage-2 inputs. Each scene uses
one new viewer/replay; no lifespan, optimizer, or calibration crosses scenes.
GT and baseline masks are only used for post-step metrics, never for training.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from experiments.prepare_paslcd_fixed_pose_cues import (
    DEFAULT_DATASET_ROOT, DEFAULT_INSTANCES, DEFAULT_OSCD_OUTPUT_ROOT,
    DEFAULT_SCENES, discover_scenes, parse_csv,
)
from experiments.evaluate_bayesian_da3_dc_hypotheses import binary_metrics, aggregate_binary
from experiments.view_bayesian_detector_steps import BayesianDetectorViewer, bayes_factor, load_union_mask, parse_args as viewer_args

DEFAULT_INPUTS = Path('outputs/paslcd_panel10_inputs_20260907')
DEFAULT_OUTPUT = Path('outputs/paslcd_panel10_uncapped_u120_20260907')
SOURCE_FILES = (
    'experiments/evaluate_paslcd_panel10.py',
    'experiments/prepare_paslcd_panel10_inputs.py',
    'experiments/view_bayesian_detector_steps.py',
    'experiments/panel10_split_training.py',
    'experiments/panel10_seed_topology.py',
    'temporal/lifespan_gate_beta.py',
    'temporal/single_candidate_beta.py',
    'temporal/change_evidence.py',
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scene_viewer_args(spec, inputs, output, *, updates=120, max_frames=None, port=8091,
                      first_open_bayes_factor_threshold=None, detector_pixel_cue='unchanged',
                      detector_gaussian_cue='unchanged'):
    options = [
        '--source-path', str(spec.source_path), '--gt-format', 'binary',
        '--fixed-cameras-json', str(spec.cameras_json), '--cue-cache-root', str(spec.output_dir),
        '--host', '127.0.0.1', '--port', str(port), '--capture-dir', str(output / 'captures'),
        '--cue-fusion', 'l1_power_product', '--product-exponent', '0.3',
        '--cue-mode', 'soft', '--cue-scale', '2', '--cue-remap', 'learned_sigmoid',
        '--cue-boundary-json', str(inputs / 'learned_boundaries_causal.json'),
        '--da3-seed-checkpoint', str(inputs / 'da3_seed_replay.pt'),
        '--sam-sign-trace-root', str(inputs), '--training-partition', 'panel10_new',
        '--da3-detector-cue-source', 'shared', '--da3-max-rows', '0',
        '--no-train-never-open-geometry', '--representation-updates', str(updates),
        '--current-view-probability', '0.33', '--representation-cue-amplitude', '2',
        '--dc-replay-mode', 'sampled',
        '--detector-pixel-cue', detector_pixel_cue,
        '--detector-gaussian-cue', detector_gaussian_cue,
    ]
    if first_open_bayes_factor_threshold is not None:
        options += ['--first-open-bayes-factor-threshold', str(first_open_bayes_factor_threshold)]
    if max_frames is not None:
        options += ['--max-frames', str(max_frames)]
    return viewer_args(options)


def frozen_snapshot(optimizer, frozen):
    result = []
    for group in optimizer.param_groups:
        p = group['params'][0]
        result.append((group['name'], p.detach()[frozen].clone(), {
            k: v.detach()[frozen].clone() for k, v in optimizer.state[p].items()
            if torch.is_tensor(v) and v.ndim and len(v) == len(frozen)
        }))
    return result


def verify_frozen(optimizer, entries, frozen):
    for group, (name, before, state) in zip(optimizer.param_groups, entries):
        assert group['name'] == name
        p = group['params'][0]
        assert torch.equal(p.detach()[:len(frozen)][frozen], before), f'parameter drift: {name}'
        for k, v in state.items():
            assert torch.equal(optimizer.state[p][k][:len(frozen)][frozen], v), f'Adam drift: {name}/{k}'


def verify_scene_masks(output, source, rows):
    """Independently recompute all saved prediction/GT counts after a scene."""
    for row in rows:
        path = output / 'captures' / f'{int(row["timestamp"]):06d}_prediction_mask.png'
        with Image.open(path) as im:
            raw = np.asarray(im)
        assert raw.ndim == 3 and raw.shape[2] == 3 and np.isin(raw, [0, 255]).all()
        assert np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2])
        h, w = raw.shape[:2]
        gt = load_union_mask((source / 'gt_mask' / f'{Path(row["frame_name"]).stem}.png',), width=w, height=h)
        recomputed = binary_metrics(raw[..., 0] > 0, gt)
        for key in ('tp', 'tn', 'fp', 'fn', 'iou', 'f1'):
            assert abs(float(row[f'gt_{key}']) - recomputed[key]) < 1e-12, (row['frame_name'], key)
    return {'frames_verified': len(rows), 'saved_mask_counts_exact': True}


def run_scene(spec, args):
    output = args.output_root / spec.instance / spec.scene
    inputs = args.inputs_root / spec.instance / spec.scene
    output.mkdir(parents=True, exist_ok=True)
    config = scene_viewer_args(spec, inputs, output, updates=args.updates,
                               max_frames=args.max_frames, port=args.port,
                               first_open_bayes_factor_threshold=args.first_open_bayes_factor_threshold,
                               detector_pixel_cue=args.detector_pixel_cue,
                               detector_gaussian_cue=args.detector_gaussian_cue)
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCE_FILES}
    input_paths = [inputs / n for n in ('input_preparation_metadata.json',
                  'learned_boundaries_causal.json', 'causal_pca_posterior_arrays.npz', 'da3_seed_replay.pt')]
    input_paths += sorted((inputs / 'da3metric_cache').glob('frame_*.npz'))
    write_json(output / 'input_sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
    write_json(output / 'configuration.json', vars(config))
    write_json(output / 'source_sha256.json', hashes)
    started = time.perf_counter()
    viewer = BayesianDetectorViewer(config)
    replay = viewer.replay
    expected = 25 if args.max_frames is None else min(25, args.max_frames)
    assert replay.total_frames == expected and replay.seed_tracker is None
    reference = {n: getattr(replay.base, n).detach().clone() for n in
                 ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')}
    rows, audit, evidence_times = [], [], []
    detector_stats = {}
    evidence_impl = replay._accumulate_change_evidence
    train_impl = replay._train_representation
    update_impl = replay.tracker.update

    def update(*a, **kw):
        result = update_impl(*a, **kw)
        first = kw['first_open']
        committed = result.candidate_committed.bool()
        stats = detector_stats[kw['timestamp']]
        stats['first_open_commits'] += int((committed & first).sum())
        stats['reopen_commits'] += int((committed & ~first & ~kw['current_active']).sum())
        stats['close_commits'] += int((committed & kw['current_active']).sum())
        stats['first_open_observation_sum'] += int(result.visible_observations[committed & first].sum())
        return result

    def evidence(view, probe, *a, **kw):
        assert probe.get_xyz.shape[0] == replay.count + int((~replay.seed_retired).sum())
        assert all(t <= replay.current_index for t in replay.accepted_da3_birth_global)
        evidence_times.append(replay.current_index)
        result = evidence_impl(view, probe, *a, **kw)
        assert float((result.delta_a + result.delta_b).max()) <= 1.000001
        q = replay._normalized_cue_target(view)[0]
        if args.detector_pixel_cue == 'binary_q05':
            assert torch.equal(result.cue.squeeze(0), (q > 0.5).to(q.dtype))
        else:
            assert torch.equal(result.cue.squeeze(0), q)
        if args.detector_gaussian_cue == 'binary_q05':
            assert result.count_mode == 'capped_binary'
            weight = (result.total_mass / config.evidence_mass_saturation).clamp(0., 1.)
            weight = torch.where(result.total_mass >= config.min_evidence_mass, weight, torch.zeros_like(weight))
            positive = result.e_plus > result.e_minus
            assert torch.equal(result.delta_a, torch.where(positive, weight, 0.))
            assert torch.equal(result.delta_b, torch.where(positive, 0., weight))
            assert not bool(((result.delta_a > 0) & (result.delta_b > 0)).any())
        detector_stats[replay.current_index] = dict(first_open_commits=0, reopen_commits=0,
            close_commits=0, first_open_observation_sum=0,
            detector_positive_mass=float(result.e_plus.sum()),
            detector_negative_mass=float(result.e_minus.sum()),
            detector_positive_only_rows=int(((result.delta_a > 0) & (result.delta_b == 0)).sum()),
            detector_negative_only_rows=int(((result.delta_b > 0) & (result.delta_a == 0)).sum()),
            detector_mixed_rows=int(((result.delta_a > 0) & (result.delta_b > 0)).sum()))
        return result

    def train(timestamp):
        assert evidence_times.count(timestamp) == 1
        base_frozen = ~replay.lifecycle.active_mask(timestamp)
        seed_frozen = ~replay.seed_lifecycle.active_mask(timestamp)
        base_before = frozen_snapshot(replay.base_optimizer, base_frozen)
        seed_before = frozen_snapshot(replay.seed_optimizer, seed_frozen)
        base_dc = replay.change_dc.detach().clone()
        seed_dc = replay.seed_model.new_dc.detach().clone()
        probe_before = {n: getattr(replay.seed_detector_probe, n).detach().clone()
                        for n in ('_xyz', '_scaling', '_rotation', '_opacity')}
        result = train_impl(timestamp)
        verify_frozen(replay.base_optimizer, base_before, base_frozen)
        verify_frozen(replay.seed_optimizer, seed_before, seed_frozen)
        for n, before in reference.items():
            assert torch.equal(getattr(replay.base, n), before), f'reference drift: {n}'
        for n, before in probe_before.items():
            assert torch.equal(getattr(replay.seed_detector_probe, n)[:len(before)], before), f'probe drift: {n}'
        assert evidence_times.count(timestamp) == 1
        assert result['future_view_accesses'] == result['lifespan_render_violations'] == 0
        item = replay.representation_replay[-1]
        assert torch.equal(item.new_target + (item.cue_target - item.new_target), item.cue_target)
        audit.append(dict(timestamp=timestamp, joint_evidence_passes=1, reference_drift=0,
                          fixed_probe_drift=0, frozen_parameter_drift=0, frozen_adam_drift=0,
                          future_view_accesses=0, lifespan_render_violations=0,
                          base_dc_changed_rows=int((replay.change_dc.detach() != base_dc).flatten(1).any(1).sum()),
                          seed_dc_changed_rows=int((replay.seed_model.new_dc.detach()[:len(seed_dc)] != seed_dc).flatten(1).any(1).sum())))
        return result

    replay._accumulate_change_evidence = evidence
    replay._train_representation = train
    replay.tracker.update = update
    video_dir = output / 'video_frames'
    video_dir.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    try:
        with viewer.lock:
            viewer.gui_next.disabled = viewer.gui_previous.disabled = viewer.gui_reset.disabled = True
            for t in range(expected):
                step = replay.step()
                viewer._refresh_images()
                # Evaluation only after detector and all120 representation updates.
                pred = viewer.predicted_change_mask_image[..., 0] > 0
                gt = replay.current_view.gt_change_mask
                metrics = binary_metrics(pred, gt)
                row = dict(instance=spec.instance, scene=spec.scene,
                           pixels=int(gt.size), gt_positive_pixels=int(gt.sum()),
                           **{f'gt_{k}': v for k, v in metrics.items()}, **asdict(step))
                row['seed_archived'] = replay.seed_model.num_gaussians
                row['seed_retired'] = int(replay.seed_retired.sum())
                row['tracker_capacity'] = replay.tracker.num_gaussians
                row.update(detector_stats[t])
                for name, directory in [('online', 'change_mask'), ('refined', 'change_mask_refined')]:
                    baseline = args.oscd_output_root / spec.instance / spec.scene / 'renders' / directory / f'{Path(step.frame_name).stem}.png'
                    mask = load_union_mask((baseline,), width=pred.shape[1], height=pred.shape[0])
                    row.update({f'{name}_{k}': v for k, v in binary_metrics(mask, gt).items()})
                rows.append(row)
                write_csv(output / 'frame_metrics.csv', rows)
                write_json(output / 'audit.json', audit)
                with Image.open(output / 'captures' / f'{t:06d}_dashboard.png') as dashboard:
                    frame = Image.new('RGB', (dashboard.width, dashboard.height + 40), 'black')
                    frame.paste(dashboard, (0, 40))
                    ImageDraw.Draw(frame).text((12, 7),
                        f'PASLCD {spec.instance}/{spec.scene} | Frame {t+1:02d}/{expected} | '
                        f'first BF {args.first_open_bayes_factor_threshold or 30:g} / {args.detector_pixel_cue} | '
                        f'Gaussian {args.detector_gaussian_cue} | '
                        f'IoU {metrics["iou"]:.4f} | F1 {metrics["f1"]:.4f}',
                        font=font, fill='white')
                    frame.save(video_dir / f'{t:06d}.png')
                print('FRAME', spec.instance, spec.scene, t, json.dumps(metrics), 'archive', replay.seed_model.num_gaussians, flush=True)
        verification = verify_scene_masks(output, spec.source_path, rows)
        summary = dict(instance=spec.instance, scene=spec.scene, frames=len(rows),
                       metrics=aggregate_binary(rows, 'gt'),
                       baseline_online=aggregate_binary(rows, 'online'),
                       baseline_refined=aggregate_binary(rows, 'refined'),
                       final_seed_archive=replay.seed_model.num_gaussians,
                       final_seed_open=int(replay.seed_lifecycle.active_mask(expected - 1).sum()),
                       final_seed_retired=int(replay.seed_retired.sum()),
                       total_root_births=sum(r['da3_seed_accepted_now'] for r in rows),
                       total_density_children=sum(r['da3_density_children'] for r in rows),
                       verification=verification, runtime_seconds=time.perf_counter() - started,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated())
        write_json(output / 'summary.json', summary)
        print('SCENE_COMPLETE', json.dumps(summary), flush=True)
    finally:
        viewer.server.stop()
    return summary


def aggregate_runs(specs, args):
    rows, scenes = [], []
    videos = args.output_root / 'video_frames'
    videos.mkdir(exist_ok=True)
    for spec in specs:
        output = args.output_root / spec.instance / spec.scene
        summary = json.loads((output / 'summary.json').read_text())
        scene_rows = list(csv.DictReader((output / 'frame_metrics.csv').open()))
        verify_scene_masks(output, spec.source_path, scene_rows)
        scenes.append(summary)
        for row in scene_rows:
            source = output / 'video_frames' / f'{int(row["timestamp"]):06d}.png'
            target = videos / f'{len(rows):06d}.png'
            if target.exists():
                target.unlink()
            os.link(source, target)
            rows.append(row)
    # aggregate_binary expects numeric counts/metrics, not CSV strings.
    numeric = [{k: (v if k in {'instance', 'scene', 'frame_name'} or v in ('', None) else float(v))
                for k, v in row.items()} for row in rows]
    write_csv(args.output_root / 'frame_metrics.csv', rows)
    result = dict(scene_count=len(scenes), frames=len(rows),
                  metrics=aggregate_binary(numeric, 'gt'),
                  baseline_online=aggregate_binary(numeric, 'online'),
                  baseline_refined=aggregate_binary(numeric, 'refined'),
                  scenes=scenes, total_seed_archive=sum(s['final_seed_archive'] for s in scenes),
                  all_saved_masks_verified=True,
                  comparison_note='Current u120 vs original online u16; refined additionally uses future views. Single seed0.')
    write_json(args.output_root / 'summary.json', result)
    mp4 = args.output_root / 'viewer_dashboard_all_frames.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'warning', '-framerate', '10', '-i', str(videos / '%06d.png'),
                    '-frames:v', str(len(rows)), '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', str(mp4)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_read_frames,width,height,duration,r_frame_rate,codec_name', '-of', 'json', str(mp4)]))
    assert int(probe['streams'][0]['nb_read_frames']) == len(rows)
    decode = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(mp4), '-f', 'null', '-'], stderr=subprocess.STDOUT)
    assert not decode.strip(), decode
    write_json(args.output_root / 'video_verification.json', probe)
    print('BENCHMARK_COMPLETE', json.dumps(result['metrics']), str(mp4), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--oscd-output-root', type=Path, default=DEFAULT_OSCD_OUTPUT_ROOT)
    parser.add_argument('--cue-root', type=Path, default=Path('outputs/paslcd_fixed_pose_cues_res4_v1'))
    parser.add_argument('--inputs-root', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--instances', default=','.join(DEFAULT_INSTANCES))
    parser.add_argument('--scenes', default=','.join(DEFAULT_SCENES))
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--first-open-bayes-factor-threshold', type=bayes_factor, default=None)
    parser.add_argument('--detector-pixel-cue', choices=('unchanged', 'binary_q05'), default='unchanged')
    parser.add_argument('--detector-gaussian-cue', choices=('unchanged', 'binary_q05'), default='unchanged')
    parser.add_argument('--scene-worker', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.detector_gaussian_cue != 'unchanged' and args.detector_pixel_cue != 'unchanged':
        parser.error('Gaussian binarization requires soft pixels, without pixel binarization')
    if args.updates < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('updates/max-frames must be positive')
    return args


def main():
    args = parse_args()
    specs = discover_scenes(args.dataset_root, args.oscd_output_root, args.cue_root,
                            instances=parse_csv(args.instances, DEFAULT_INSTANCES),
                            scenes=parse_csv(args.scenes, DEFAULT_SCENES), resolution=4.0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.scene_worker:
        if len(specs) != 1:
            raise ValueError('scene worker requires exactly one scene')
        run_scene(specs[0], args)
        return
    if not args.aggregate_only:
        for spec in specs:
            output = args.output_root / spec.instance / spec.scene
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-m', 'experiments.evaluate_paslcd_panel10',
                       '--dataset-root', str(args.dataset_root), '--oscd-output-root', str(args.oscd_output_root),
                       '--cue-root', str(args.cue_root), '--inputs-root', str(args.inputs_root),
                       '--output-root', str(args.output_root), '--instances', spec.instance, '--scenes', spec.scene,
                       '--updates', str(args.updates), '--port', str(args.port), '--scene-worker']
            command += ['--detector-pixel-cue', args.detector_pixel_cue]
            command += ['--detector-gaussian-cue', args.detector_gaussian_cue]
            if args.first_open_bayes_factor_threshold is not None:
                command += ['--first-open-bayes-factor-threshold', str(args.first_open_bayes_factor_threshold)]
            if args.max_frames is not None:
                command += ['--max-frames', str(args.max_frames)]
            write_json(output / 'command.json', command)
            print('START', spec.instance, spec.scene, flush=True)
            with (output / 'run.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    aggregate_runs(specs, args)


if __name__ == '__main__':
    main()
