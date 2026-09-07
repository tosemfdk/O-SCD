from pathlib import Path

import torch

from experiments.summarize_bayesian_da3_dc_hypotheses import (
    compare_controlled_state,
)


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "base_current_state_index": torch.tensor([0]),
        "base_num_states": torch.tensor([1]),
        "base_open_timestamp": torch.tensor([2]),
        "seed_xyz": torch.tensor([[value, 0.0, 0.0]]),
        "seed_opacity": torch.tensor([[0.0]]),
        "seed_scaling": torch.tensor([[0.0, 0.0, 0.0]]),
        "seed_rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "seed_start": torch.tensor([3.0]),
        "seed_end": torch.tensor([float("inf")]),
        "seed_geometry_update_counts": torch.tensor([4]),
        "accepted_da3_source_rows": torch.tensor([9]),
        "accepted_da3_birth_global": torch.tensor([1]),
    }


def test_controlled_state_comparison_separates_discrete_and_float_geometry(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save(_state(1.0), reference)
    torch.save(_state(1.25), candidate)

    comparison = compare_controlled_state(reference, candidate)

    assert comparison["all_discrete_equal"]
    assert comparison["geometry_max_absolute_difference"]["seed_xyz"] == 0.25
    assert comparison["accepted_source_rows_jaccard"] == 1.0
    assert comparison["common_geometry_max_absolute_difference"]["seed_xyz"] == 0.25
