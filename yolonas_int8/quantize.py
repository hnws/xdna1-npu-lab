"""Quantize the raw-head YOLO-NAS-L to XINT8 for the XDNA 1 NPU (Vitis AI EP).

Same recipe as Frigate PR #24470 for YOLO-NAS-S: AMD Quark XINT8 (power-of-two
scales, which XDNA 1 requires), CLE off, optional AdaRound -- but calibrated
on this site's own Frigate snapshots (data/calib.npy) instead of COCO.

Run in the npu-quark container:
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD/..":/w -w /w/yolonas_int8 \
      npu-quark python3 quantize.py [--adaround]
"""
import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../models/yolo_nas_l_320_raw.onnx")
ap.add_argument("--adaround", action="store_true")
ap.add_argument("--iters", type=int, default=300)
args = ap.parse_args()
dst = "../models/yolo_nas_l_320_xint8%s.onnx" % ("_adaround" if args.adaround else "")

calib = np.load("data/calib.npy")  # (N, 320, 320, 3) uint8 RGB


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(calib)

    def get_next(self):
        x = next(self.it, None)
        if x is None:
            return None
        return {"images": np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32) / 255.0}


qc = get_default_config("XINT8")
qc.include_cle = False
if args.adaround:
    qc.include_fast_ft = True
    ft = dict(qc.extra_options.get("FastFinetune", {}))
    ft.update(OptimAlgorithm="adaround", DataSize=len(calib), NumIterations=args.iters, BatchSize=2)
    qc.extra_options["FastFinetune"] = ft
print(f"XINT8, CLE off, AdaRound={args.adaround}, {len(calib)} calibration images", flush=True)
t = time.time()
ModelQuantizer(Config(global_quant_config=qc)).quantize_model(args.src, dst, Reader())
print(f"quantized in {(time.time() - t) / 60:.1f} min -> {dst}", flush=True)

# Same sanity checks as the PR: every scale a power of two, model runs.
g = onnx.load(dst)
init = {i.name: i for i in g.graph.initializer}
sc = np.concatenate([np.atleast_1d(onnx.numpy_helper.to_array(init[n.input[1]])).ravel()
                     for n in g.graph.node
                     if n.op_type in ("QuantizeLinear", "DequantizeLinear") and n.input[1] in init])
l2 = np.log2(sc.astype(np.float64))
print("QuantizeLinear nodes:", sum(n.op_type == "QuantizeLinear" for n in g.graph.node))
assert np.allclose(l2, np.round(l2)), "scales are not all powers of two"
o = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(
    None, {"images": np.zeros((1, 3, 320, 320), np.float32)})
print("outputs:", [x.shape for x in o])
