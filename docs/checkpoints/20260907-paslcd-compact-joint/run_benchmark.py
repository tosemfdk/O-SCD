"""Bounded PASLCD25-frame paired comparison; final snapshot audit outside training timer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments import evaluate_paslcd_panel10 as evaluation
from experiments import panel10_split_training as split
from experiments import view_bayesian_detector_steps as viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', choices=['off', 'snapshot'], required=True)
    parser.add_argument('--scene', choices=['Cantina','Garden'], required=True)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--render-mode', choices=['split', 'joint_channels'], default='split')
    parser.add_argument('--port', type=int, default=8092)
    parser.add_argument('--profile-components', action=argparse.BooleanOptionalAction, default=True,
                        help='Disable nested synchronization for fair whole-training wall measurements.')
    args = parser.parse_args()
    assert args.updates == 120 and not args.profile_components
    assert (args.cache,args.render_mode) in [('off','split'),('snapshot','joint_channels')]
    source_root = Path(viewer.__file__).resolve().parents[1]
    evaluation.SOURCE_FILES = tuple(str(source_root/p) for p in evaluation.SOURCE_FILES)
    def tensor_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if args.output_root.exists():
        raise FileExistsError('Each benchmark must use a fresh output directory')
    args.output_root.mkdir(parents=True)
    totals = defaultdict(float)
    calls = defaultdict(int)
    per_frame = []
    final_replay = None
    in_training = False
    mask_audits = 0

    def audit_life(life, timestamp):
        nonlocal mask_audits
        cache = life.state_snapshots
        if cache is None:
            return
        for t in range(timestamp + 1):
            actual = [life.active_mask(t), life.never_open_mask(t), life.materialized_mask(t)]
            life.state_snapshots = None
            try:
                expected = [life.active_mask(t), life.never_open_mask(t), life.materialized_mask(t)]
            finally:
                life.state_snapshots = cache
            assert all(torch.equal(a, b) for a, b in zip(actual, expected)), (timestamp, t)
            mask_audits += 1

    original_config = evaluation.scene_viewer_args
    @functools.wraps(original_config)
    def config(*a, **k):
        result = original_config(*a, **k)
        result.lifespan_state_cache = args.cache
        result.panel10_render_mode = args.render_mode
        result.da3_birth_coverage = '3d_plus_2d'
        result.da3_coverage_2d_sigma = 2.0
        assert result.first_open_bayes_factor_threshold == 10 and result.bayes_factor_threshold == 30
        assert result.detector_gaussian_cue == 'binary_q05' and result.cue_mode == 'soft'
        return result
    evaluation.scene_viewer_args = config

    original_train = viewer.BayesianDetectorReplay._train_representation
    @functools.wraps(original_train)
    def train(self, timestamp):
        nonlocal in_training, final_replay
        final_replay = self
        torch.cuda.synchronize()
        started = time.perf_counter()
        in_training = True
        try:
            result = original_train(self, timestamp)
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            in_training = False
        totals['training_seconds'] += elapsed
        calls['training'] += 1
        item = self.representation_replay[-1]
        per_frame.append(dict(timestamp=timestamp, training_seconds=elapsed,
                              sampled_indices=result['sampled_indices'],
                              raw_soft_q_sha256=tensor_hash(item.cue_target),
                              new_target_sha256=tensor_hash(item.new_target)))
        return result
    viewer.BayesianDetectorReplay._train_representation = train

    def timed_training_call(name, function):
        @functools.wraps(function)
        def wrapper(*a, **k):
            if not in_training:
                return function(*a, **k)
            calls[name] += 1
            if not args.profile_components:
                return function(*a, **k)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = function(*a, **k)
            torch.cuda.synchronize()
            totals[name + '_seconds'] += time.perf_counter() - started
            return result
        return wrapper
    split._partition_masks = timed_training_call('partition_masks', split._partition_masks)
    split.build_partition_view = timed_training_call('assemble', split.build_partition_view)
    split._render = timed_training_call('render', split._render)

    original_seal = viewer.DetectorReplayLifespan.seal_snapshot
    @functools.wraps(original_seal)
    def seal(self, timestamp):
        torch.cuda.synchronize()
        started = time.perf_counter()
        original_seal(self, timestamp)
        torch.cuda.synchronize()
        totals['seal_seconds'] += time.perf_counter() - started
        calls['seal'] += 1
    viewer.DetectorReplayLifespan.seal_snapshot = seal

    run_args = evaluation.parse_args([
        '--instances', 'Instance_1', '--scenes', args.scene, '--updates', str(args.updates),
        '--port', str(args.port), '--scene-worker', '--output-root', str(args.output_root),
        '--inputs-root', 'outputs/paslcd_panel10_inputs_20260907',
        '--first-open-bayes-factor-threshold', '10', '--detector-gaussian-cue', 'binary_q05',
    ])
    spec = evaluation.discover_scenes(run_args.dataset_root, run_args.oscd_output_root,
        run_args.cue_root, instances=('Instance_1',), scenes=(args.scene,), resolution=4.)[0]
    torch.cuda.synchronize()
    started = time.perf_counter()
    summary = evaluation.run_scene(spec, run_args)
    torch.cuda.synchronize()
    totals['runner_seconds'] = time.perf_counter() - started
    audit_started = time.perf_counter()
    if args.audit:
        audit_life(final_replay.lifecycle,24)
        audit_life(final_replay.seed_lifecycle,24)
    torch.cuda.synchronize()
    totals['final_snapshot_audit_seconds'] = time.perf_counter()-audit_started
    passes = 1 if args.render_mode == 'joint_channels' else 2
    assert calls['training'] == 25 and calls['assemble'] == 25 * args.updates * passes
    assert calls['render'] == calls['assemble']
    totals['training_including_seal_seconds'] = totals['training_seconds'] + totals['seal_seconds']
    # Nested mask time is part of assembly, and assembly is part of training.
    totals['assemble_excluding_masks_seconds'] = totals['assemble_seconds'] - totals['partition_masks_seconds']
    caches = {name: getattr(final_replay, name).state_snapshots
              for name in ['lifecycle', 'seed_lifecycle']}
    if args.cache == 'snapshot':
        assert all(cache is not None and len(cache.frames) == 25 for cache in caches.values())
        assert all(cache.interval_fallbacks == 0 for cache in caches.values())
    sources = [Path(__file__), Path(viewer.__file__), Path(split.__file__),
               source_root/'temporal/lifespan_state_snapshots.py', source_root/'temporal/seed_projected_occupancy.py']
    result = dict(scene=args.scene,cache=args.cache, render_mode=args.render_mode,birth_coverage='3d_plus_2d',
        profile_components=args.profile_components,
        updates=args.updates, frames=25, audit=args.audit,
        mask_audits=mask_audits, seconds=dict(totals), calls=dict(calls), per_frame=per_frame,
        caches={name: cache.statistics() if cache is not None else None for name, cache in caches.items()},
        summary=summary, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        note='CUDA-synchronized function wall time; training includes nested assembly/masks. '
             'Compression sealing counted separately and included in the fair training total. '
             'Cue/depth inputs cached; runner includes setup/audits/captures/savedmask verification, excludes finalsnapshot audit. '
             'External process wall includes imports and finalsnapshot audit. Concurrent persistent viewer may add variation.')
    (args.output_root / 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SNAPSHOT_BENCHMARK_COMPLETE', json.dumps(dict(cache=args.cache, seconds=dict(totals),
          metrics=summary['metrics'], audits=mask_audits)), flush=True)


if __name__ == '__main__':
    main()
