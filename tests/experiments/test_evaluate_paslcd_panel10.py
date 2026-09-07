from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.evaluate_paslcd_panel10 import scene_viewer_args, verify_scene_masks
from experiments.evaluate_bayesian_da3_dc_hypotheses import binary_metrics


def test_prepare_and_evaluate_default_input_roots_match():
    from experiments.evaluate_paslcd_panel10 import DEFAULT_INPUTS
    from experiments.prepare_paslcd_panel10_inputs import DEFAULT_OUTPUT_ROOT
    assert DEFAULT_INPUTS == DEFAULT_OUTPUT_ROOT


def test_scene_options_keep_uncapped_panel10_contract_and_use_scene_inputs(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for name in ('learned_boundaries_causal.json', 'da3_seed_replay.pt'):
        (inputs / name).touch()
    spec = SimpleNamespace(source_path=tmp_path/'dataset'/'Cantina', cameras_json=tmp_path/'cameras.json',
                           output_dir=tmp_path/'cues')
    args = scene_viewer_args(spec, inputs, tmp_path/'output')
    assert args.source_path == spec.source_path
    assert args.fixed_cameras_json == spec.cameras_json
    assert args.cue_cache_root == spec.output_dir
    assert args.cue_boundary_json == inputs/'learned_boundaries_causal.json'
    assert args.sam_sign_trace_root == inputs
    assert args.gt_format == 'binary'
    assert args.da3_max_rows == 0 and args.representation_updates == 120
    assert args.training_partition == 'panel10_new' and args.da3_detector_cue_source == 'shared'
    assert args.bayes_factor_threshold == 30 and args.current_view_probability == .33
    assert args.first_open_bayes_factor_threshold is None and args.detector_pixel_cue == 'unchanged'
    assert args.detector_gaussian_cue == 'unchanged'
    assert args.da3_birth_max_per_frame == 1024 and args.da3_density_max_children == 128
    assert not args.train_never_open_geometry
    assert args.host == '127.0.0.1' and args.port == 8091
    ablation = scene_viewer_args(spec, inputs, tmp_path/'ablation',
                                first_open_bayes_factor_threshold=10., detector_pixel_cue='binary_q05')
    assert ablation.first_open_bayes_factor_threshold == 10. and ablation.bayes_factor_threshold == 30.
    assert ablation.detector_pixel_cue == 'binary_q05'
    assert ablation.cue_mode == 'soft' and ablation.cue_scale == 2.
    gaussian = scene_viewer_args(spec, inputs, tmp_path/'gaussian',
                                first_open_bayes_factor_threshold=10., detector_gaussian_cue='binary_q05')
    assert gaussian.detector_gaussian_cue == 'binary_q05'
    assert gaussian.detector_pixel_cue == 'unchanged' and gaussian.cue_mode == 'soft'
    assert gaussian.first_open_bayes_factor_threshold == 10. and gaussian.bayes_factor_threshold == 30.


def test_saved_mask_verification_uses_binary_gt_and_catches_tampered_metric(tmp_path):
    source = tmp_path/'source'
    out = tmp_path/'out'
    (source/'gt_mask').mkdir(parents=True)
    (out/'captures').mkdir(parents=True)
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    Image.fromarray(gt.astype('uint8')*255).save(source/'gt_mask'/'a.png')
    Image.fromarray(np.repeat(pred[..., None], 3, axis=2).astype('uint8')*255).save(out/'captures'/'000000_prediction_mask.png')
    row = dict(timestamp=0, frame_name='a.jpg', **{f'gt_{k}': v for k,v in binary_metrics(pred,gt).items()})
    assert verify_scene_masks(out, source, [row])['saved_mask_counts_exact']
    row['gt_tp'] = 99
    with pytest.raises(AssertionError):
        verify_scene_masks(out, source, [row])
