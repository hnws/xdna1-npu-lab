"""Build calibration and evaluation sets from your own camera snapshots.

Frigate keeps a clean snapshot per event in <storage>/clips/*-clean.webp (or
use any folder of JPG/PNG/WebP frames). Each snapshot yields a letterboxed
full frame and a random square crop (Frigate feeds the detector square
regions around motion), resized to 320x320 RGB uint8.

  python make_calib.py /path/to/frigate/storage/clips --per-camera 40 --calib 100 --eval 50
  -> data/calib.npy (2 * --calib images), data/eval.npy (2 * --eval images)

Calibrating on your own cameras matches your scenes better than COCO.
"""
import argparse
import collections
import glob
import os
import random

import numpy as np
from PIL import Image

S = 320

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--pattern", default="*-clean.webp", help="glob inside folder (Frigate event snapshots by default)")
ap.add_argument("--per-camera", type=int, default=40, help="max snapshots per camera (file-name prefix before '-')")
ap.add_argument("--calib", type=int, default=100, help="snapshots for calibration")
ap.add_argument("--eval", type=int, default=50, help="held-out snapshots for evaluation")
args = ap.parse_args()

rng = random.Random(0)
by_cam = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
    by_cam[os.path.basename(f).split("-")[0]].append(f)
picked = []
for fs in by_cam.values():  # spread across cameras and time
    rng.shuffle(fs)
    picked += fs[:args.per_camera]
rng.shuffle(picked)
need = args.calib + args.eval
assert len(picked) >= need, f"found {len(picked)} snapshots, need {need}"
print({k: len(v) for k, v in by_cam.items()})


def letterbox(im):
    w, h = im.size
    r = S / max(w, h)
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(im.resize((round(w * r), round(h * r))), (0, 0))
    return canvas


def crop(im):
    w, h = im.size
    s = rng.randint(min(w, h) // 3, min(w, h))
    x, y = rng.randint(0, w - s), rng.randint(0, h - s)
    return im.crop((x, y, x + s, y + s)).resize((S, S))


calib, evals = [], []
for i, f in enumerate(picked[:need]):
    im = Image.open(f).convert("RGB")
    (calib if i < args.calib else evals).extend([np.asarray(letterbox(im)), np.asarray(crop(im))])
os.makedirs("data", exist_ok=True)
np.save("data/calib.npy", np.stack(calib))
np.save("data/eval.npy", np.stack(evals))
print(f"data/calib.npy: {len(calib)} images, data/eval.npy: {len(evals)} images")
