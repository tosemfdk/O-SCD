"""Continue from shared-geometry S0 with every geometry row fixed and S1 DC only."""

from experiments.train_s0_influence_freeze_s1 import (
    ALL_GEOMETRY_OUTPUT,
    parse_args,
    train,
)


def main() -> None:
    args = parse_args(
        default_output=ALL_GEOMETRY_OUTPUT,
        default_freeze_implementation="model_buffer_gradient_mask",
        default_freeze_scope="all_geometry",
    )
    train(args)


if __name__ == "__main__":
    main()
