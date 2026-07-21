# Animate a scene's inference frames (the original query-view RGBs, in dataset
# order) into a downscaled looping GIF. Frames come from the cached
# frame_context.pt, so no dataset reload.
#
#   python experiments/make_inf_gif.py --scenes Garden Porch --width 560
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = os.path.join(REPO, "outputs", "change_nbv", "diagnostics",
                   "rchange_importance", "Instance_1")
OUT = os.path.join(REPO, "experiments", "figures", "inf_gifs")


def make_gif(scene, width, ms):
    ctx = torch.load(os.path.join(CTX, scene, "frame_context.pt"),
                     weights_only=False)
    orig = ctx["original"].float().numpy() / 255.0     # (V, 3, H, W)
    frames = []
    for i in range(orig.shape[0]):
        a = (np.clip(orig[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(a)
        h = int(round(width * im.height / im.width))
        im = im.resize((width, h), Image.LANCZOS)
        # a small quantized palette keeps the file light and loops clean
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=96))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{scene.lower()}_inference.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, optimize=True, disposal=2)
    mb = os.path.getsize(out) / 1e6
    print(f"{scene}: {len(frames)} frames @ {width}px -> "
          f"{os.path.relpath(out, REPO)}  ({mb:.2f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["Garden", "Porch"])
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--ms", type=int, default=350, help="ms per frame")
    args = ap.parse_args()
    for s in args.scenes:
        make_gif(s, args.width, args.ms)


if __name__ == "__main__":
    main()
