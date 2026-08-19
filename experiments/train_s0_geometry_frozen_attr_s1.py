"""Run S1 with S0 influence rows frozen by persistent model metadata."""

from experiments.train_s0_influence_freeze_s1 import (
    MODEL_BUFFER_OUTPUT,
    parse_args,
    train,
)


def main() -> None:
    args = parse_args(
        default_output=MODEL_BUFFER_OUTPUT,
        default_freeze_implementation="model_buffer_gradient_mask",
    )
    train(args)


if __name__ == "__main__":
    main()
