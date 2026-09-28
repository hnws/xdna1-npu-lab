"""How closely does a quantized YOLO-NAS model match the FP32 one on held-out
Frigate snapshots? The FP32 detections are the reference (it is the model
Frigate runs today); an int8 detection matches when it has the same class and
IoU >= 0.5.

  .venv/bin/python yolonas_int8/evaluate.py models/yolo_nas_l_320_xint8.onnx [...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
import nas_raw  # noqa: E402

HERE = Path(__file__).parent
REF = HERE.parent / "models/yolo_nas_l_320_raw.onnx"
THRESH = 0.5  # Frigate's default min_score is 0.5
NAMES = {0: "person", 2: "car", 15: "cat", 16: "dog"}


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def run(path, images):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out, t0 = [], time.perf_counter()
    for im in images:
        out.append(nas_raw.decode(s.run(None, {"images": nas_raw.preprocess(im)}), conf=0.3))
    return out, (time.perf_counter() - t0) / len(images) * 1000


def compare(ref, test):
    tp = fn = fp = 0
    dscore, per_cls = [], {}
    for r, t in zip(ref, test):
        r, t = r[r[:, 4] >= THRESH], t[t[:, 4] >= THRESH]
        used = np.zeros(len(t), bool)
        for d in r:
            c = int(d[5])
            pc = per_cls.setdefault(c, [0, 0])
            pc[1] += 1
            cand = np.where((t[:, 5] == c) & ~used)[0]
            if len(cand):
                o = iou(d, t[cand])
                j = o.argmax()
                if o[j] >= 0.5:
                    used[cand[j]] = True
                    tp += 1
                    pc[0] += 1
                    dscore.append(t[cand[j], 4] - d[4])
                    continue
            fn += 1
        fp += int((~used).sum())
    return tp, fn, fp, dscore, per_cls


def main():
    images = np.load(HERE / "data/eval.npy")
    ref, ms = run(REF, images)
    print(f"{len(images)} held-out images; FP32 reference: {sum(len(r[r[:, 4] >= THRESH]) for r in ref)} "
          f"detections >= {THRESH} ({ms:.0f} ms/img on CPU)")
    for path in sys.argv[1:]:
        test, ms = run(path, images)
        tp, fn, fp, ds, pc = compare(ref, test)
        print(f"\n{Path(path).name}  ({ms:.0f} ms/img on CPU)")
        print(f"  kept {tp}/{tp + fn} FP32 detections ({tp / max(tp + fn, 1):.1%}), "
              f"{fp} extra, score shift mean {np.mean(ds):+.3f} (|shift| {np.mean(np.abs(ds)):.3f})")
        print("  per class: " + ", ".join(f"{NAMES.get(c, c)} {k}/{n}"
                                          for c, (k, n) in sorted(pc.items(), key=lambda kv: -kv[1][1])[:6]))


if __name__ == "__main__":
    main()
