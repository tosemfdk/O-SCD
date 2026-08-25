import torch
from torch import nn

from temporal.masked_optimizer import MaskedRowAdam


def test_closed_row_parameter_and_moments_do_not_drift_then_resume():
    parameter = nn.Parameter(torch.zeros(2, 1))
    optimizer = MaskedRowAdam(
        {"dc": parameter},
        thaw_names=("dc",),
        lrs={"dc": 0.1},
    )

    # Build value and momentum for row 0 while its lifespan is OPEN.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.tensor([[1.0], [0.0]])
        optimizer.step(torch.tensor([True, False]))
    closed_value = parameter[0].detach().clone()
    closed_state = {
        name: value[0].detach().clone()
        for name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }

    # Optimize only row 1.  Retained Adam momentum must not move row 0.
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.tensor([[9.0], [1.0]])
        optimizer.step(torch.tensor([False, True]))
    assert torch.equal(parameter[0], closed_value)
    for name, expected in closed_state.items():
        assert torch.equal(optimizer.state[parameter][name][0], expected)

    # REOPEN deliberately resumes both the value and its optimizer history.
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.tensor([[1.0], [0.0]])
    optimizer.step(torch.tensor([True, False]))
    assert not torch.equal(parameter[0], closed_value)
    assert optimizer.state[parameter]["step"][0] == closed_state["step"] + 1


def test_masked_row_adam_supports_all_direct_gaussian_tensor_shapes():
    parameters = {
        "dc": nn.Parameter(torch.zeros(3, 1, 3)),
        "xyz": nn.Parameter(torch.zeros(3, 3)),
        "features_rest": nn.Parameter(torch.zeros(3, 15, 3)),
        "opacity": nn.Parameter(torch.zeros(3, 1)),
        "scaling": nn.Parameter(torch.zeros(3, 3)),
        "rotation": nn.Parameter(torch.zeros(3, 4)),
    }
    optimizer = MaskedRowAdam(parameters, lrs={name: 0.01 for name in parameters})
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step(torch.tensor([False, True, False]))
    for parameter in parameters.values():
        assert torch.count_nonzero(parameter[0]) == 0
        assert torch.count_nonzero(parameter[1]) > 0
        assert torch.count_nonzero(parameter[2]) == 0


def test_masked_row_adam_accepts_distinct_parameter_row_masks():
    parameters = {
        "dc": nn.Parameter(torch.zeros(3, 1)),
        "xyz": nn.Parameter(torch.zeros(3, 1)),
        "opacity": nn.Parameter(torch.zeros(3, 1)),
    }
    optimizer = MaskedRowAdam(
        parameters,
        thaw_names=tuple(parameters),
        lrs={name: 0.01 for name in parameters},
    )
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step(
        {
            "dc": torch.tensor([True, True, False]),
            "xyz": torch.tensor([True, False, False]),
            "opacity": torch.tensor([True, True, False]),
        }
    )

    assert torch.count_nonzero(parameters["dc"][:2]) == 2
    assert torch.count_nonzero(parameters["opacity"][:2]) == 2
    assert torch.count_nonzero(parameters["xyz"][0]) == 1
    assert torch.count_nonzero(parameters["xyz"][1:]) == 0
    assert optimizer.state[parameters["xyz"]]["step"].tolist() == [1, 0, 0]
