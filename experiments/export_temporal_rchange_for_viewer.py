"""CLI for exporting one DC-only temporal R_change state as a viewer PLY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal.viewer_export import (  # noqa: E402
    ensure_dc_state_viewer_export,
    export_dc_state_for_viewer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="DC-only temporal_rchange_checkpoint.pt")
    parser.add_argument("--state-id", type=int, default=0, help="Temporal state slot to inspect")
    parser.add_argument("--output-ply", type=Path, default=None, help="Static viewer PLY output path")
    parser.add_argument("--force", action="store_true", help="Rebuild even when a matching export exists")
    args = parser.parse_args()

    if args.force:
        output_ply = args.output_ply
        if output_ply is None:
            checkpoint = args.checkpoint.expanduser().resolve()
            output_ply = checkpoint.parent / "viewer" / f"state_{args.state_id:03d}_change.ply"
        result = export_dc_state_for_viewer(
            args.checkpoint,
            output_ply,
            state_id=args.state_id,
        )
        result["reused"] = False
    else:
        result = ensure_dc_state_viewer_export(
            args.checkpoint,
            state_id=args.state_id,
            output_ply=args.output_ply,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
