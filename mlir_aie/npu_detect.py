#!/usr/bin/env python3
"""Frigate-style object detection on the Ryzen AI NPU.

Runs the exact detector model your Frigate uses (YOLO-NAS-L, 320x320, from
/config/model_cache/yolo_nas_l_320.onnx) with Frigate's preprocessing, and
puts all 119 convolutions on the NPU:

  * weights are uploaded once and stay resident in NPU buffers,
  * activation/output buffers are pre-allocated per layer,
  * every layer runs through npu_gemm.UGemm, so the whole network uses a few
    resident NPU programs instead of reloading one per layer shape.

The CPU does im2col, bias, activations, concat/upsample (PyTorch), and the
model's own NMS post-processing (onnxruntime) -- the post-processing is split
off the ONNX graph unchanged.

Usage:
    .venv/bin/python npu_detect.py images/camera_snapshot.jpg        # detect + draw boxes
    .venv/bin/python npu_detect.py images/camera_snapshot.jpg --bench 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx.utils import extract_model
from onnx2torch import convert
from PIL import Image, ImageDraw

import aie.iron as iron
from aie.iron.device import from_name

from npu_gemm import UGemm

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models/yolo_nas_l_320.onnx"  # Frigate's exported YOLO-NAS-L
SIZE = 320
# Boundary between the network and its NMS post-processing in this ONNX file.
NET_OUTPUTS = ["/model/heads/Mul_output_0", "/model/heads/Sigmoid_output_0"]
FINAL_OUTPUT = "graph2_flat_predictions"
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
        "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
        "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat "
        "baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl "
        "banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave "
        "oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()

# NPU tile configs (m, k, n) and their rough sustained GFLOPS, for picking the
# config per layer. Each config is one resident NPU program (driver limit: 6).
CONFIGS = {(32, 64, 32): 1000.0, (16, 64, 32): 700.0, (32, 64, 16): 730.0, (16, 64, 16): 300.0}
CALL_OVERHEAD_S = 0.3e-3

bf16 = iron.str_to_dtype("bf16")


def round_up(x, a):
    return -(-x // a) * a


def split_model():
    """Split the Frigate ONNX into the network and its NMS post-processing."""
    net, post = MODEL.with_suffix(".net.onnx"), MODEL.with_suffix(".post.onnx")
    if not net.exists():
        extract_model(str(MODEL), str(net), ["input"], NET_OUTPUTS)
    if not post.exists():
        extract_model(str(MODEL), str(post), NET_OUTPUTS, [FINAL_OUTPUT])
    return net, post


def preprocess(img):
    """Frigate's input for this model: 320x320, BGR, NCHW, uint8."""
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE)))[:, :, ::-1]
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt


TRANSPOSE_BYTES_PER_S = 4e9  # rough CPU strided-copy rate, for the cost model
MAX_DMA_REPEAT = 64  # AIE DMA buffer-descriptor dimension limit: N // (4*n) must fit


class NpuGemmLayer(torch.nn.Module):
    """A conv (or a kernel==stride transposed conv) as one GEMM on the NPU.

    Preferred orientation is channel-first, OUT[Cout, L] = W[Cout, K] @ P[K, L],
    where P is exactly what F.unfold produces and OUT is already NCHW, so no
    transposes are needed on the CPU. Layers whose L is too wide for the DMA
    descriptors use OUT[L, Cout] = P^T @ W^T instead.
    """

    def __init__(self, conv, in_hw, gemms, timer):
        super().__init__()
        self.timer = timer
        self.transposed = isinstance(conv, torch.nn.ConvTranspose2d)
        H, W = in_hw
        w = conv.weight.detach().float()
        if self.transposed:
            cin, cout, kh, kw = w.shape
            self.kh, self.kw, self.cout = kh, kw, cout
            self.out_hw = (H * kh, W * kw)
            L, K, C = H * W, cin, cout * kh * kw
            wm = w.reshape(cin, C).T  # (C, K), rows ordered (cout, kh, kw)
            bias = conv.bias.detach().float().repeat_interleave(kh * kw) if conv.bias is not None else None
        else:
            cout, cin, kh, kw = w.shape
            self.k, self.s, self.p = tuple(conv.kernel_size), tuple(conv.stride), tuple(conv.padding)
            Ho = (H + 2 * self.p[0] - kh) // self.s[0] + 1
            Wo = (W + 2 * self.p[1] - kw) // self.s[1] + 1
            self.out_hw = (Ho, Wo)
            L, K, C = Ho * Wo, cin * kh * kw, cout
            wm = w.reshape(cout, K)  # (C, K), columns ordered (cin, kh, kw) like F.unfold
            bias = conv.bias.detach().float() if conv.bias is not None else None
        self.L, self.K, self.C = L, K, C
        self.bias = bias

        # Candidates: (estimated seconds, orientation, config). Try the cheapest
        # one that compiles.
        cands = []
        for cfg, gflops in CONFIGS.items():
            am, ak, an = gemms[cfg].align
            for cf in (True, False):
                M, N = (C, L) if cf else (L, C)
                Mp, Kp, Np = round_up(M, am), round_up(K, ak), round_up(N, an)
                if Np // an > MAX_DMA_REPEAT:
                    continue
                est = 2 * Mp * Kp * Np / (gflops * 1e9) + CALL_OVERHEAD_S
                if not cf:  # CPU transposes of the input patches and the output
                    est += (L * K + L * C) * 4 / TRANSPOSE_BYTES_PER_S
                cands.append((est, not cf, cfg, (Mp, Kp, Np)))
        for _, not_cf, cfg, shape in sorted(cands):
            cf = not not_cf
            try:
                gemms[cfg].kernel(*shape)  # compile now, before any NPU run
            except RuntimeError:
                continue
            self.cf, self.cfg, self.gemm, (self.Mp, self.Kp, self.Np) = cf, cfg, gemms[cfg], shape
            break
        else:
            raise RuntimeError(f"no NPU GEMM config fits conv L={L} K={K} C={C}")

        # Resident weights: W as the left operand (channel-first) or W^T as the right one.
        wt = torch.zeros(self.Mp, self.Kp) if self.cf else torch.zeros(self.Kp, self.Np)
        if self.cf:
            wt[:C, :K] = wm
            self.a = self._resident(wt)
            self.b = iron.zeros((self.Kp, self.Np), dtype=bf16, device="npu")  # patches
        else:
            wt[:K, :C] = wm.T
            self.b = self._resident(wt)
            self.a = iron.zeros((self.Mp, self.Kp), dtype=bf16, device="npu")  # patches^T
        self.c = iron.zeros((self.Mp, self.Np), dtype=np.float32, device="npu")
        self.run = None  # bound fast call, created on first forward

    @staticmethod
    def _resident(t):
        return iron.tensor(t.to(torch.bfloat16).view(torch.int16).numpy().view(bf16), dtype=bf16, device="npu")

    def shape(self):
        return (self.Mp, self.Kp, self.Np)

    def _pointwise(self):
        return self.transposed or (self.k == (1, 1) and self.s == (1, 1) and self.p == (0, 0))

    def _patches(self, x):
        """(K, L) im2col matrix, as F.unfold would produce."""
        x = x.to(torch.bfloat16)
        if self._pointwise():
            return x[0].flatten(1)
        return F.unfold(x, self.k, padding=self.p, stride=self.s)[0]

    def _im2col_into(self, x, dst):
        """Write the (K, L) im2col matrix straight into dst[:K, :L] in one pass."""
        x = x[0].to(torch.bfloat16)
        Np = dst.shape[1]
        if self._pointwise():
            dst[:self.K, :self.L].copy_(x.flatten(1))
            return
        (kh, kw), (sh, sw), (ph, pw) = self.k, self.s, self.p
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph))
        cin, Hp, Wp = x.shape
        Ho, Wo = self.out_hw
        src = x.as_strided((cin, kh, kw, Ho, Wo), (Hp * Wp, Wp, 1, sh * Wp, sw))
        out = dst.as_strided((cin, kh, kw, Ho, Wo), (kh * kw * Np, kw * Np, Np, Wo, 1))
        out.copy_(src)

    def forward(self, x):
        t0 = time.perf_counter()
        staged = self.b if self.cf else self.a
        with staged.overwrite() as buf:
            dst = torch.from_numpy(buf.view(np.int16)).view(torch.bfloat16)
            if self.cf:
                self._im2col_into(x, dst)
            else:
                dst[:self.L, :self.K].copy_(self._patches(x).T)
        t1 = time.perf_counter()
        if self.run is None:
            self.run = self.gemm.bind(self.a, self.b, self.c, self.Mp, self.Kp, self.Np)
        self.run()
        t2 = time.perf_counter()
        out = torch.from_numpy(self.c.numpy())
        y = out[:self.C, :self.L] if self.cf else out[:self.L, :self.C].T  # (C, L)
        if self.bias is not None:
            y = y + self.bias[:, None]
        Ho, Wo = self.out_hw
        if self.transposed:
            H, W = Ho // self.kh, Wo // self.kw
            y = y.reshape(self.cout, self.kh, self.kw, H, W).permute(0, 3, 1, 4, 2)
            y = y.reshape(1, self.cout, Ho, Wo)
        else:
            y = y.reshape(1, self.C, Ho, Wo)
        y = y.contiguous()
        t3 = time.perf_counter()
        self.timer.add("stage", t1 - t0)
        self.timer.add("npu", t2 - t1)
        self.timer.add("unstage", t3 - t2)
        return y


class NpuDetector:
    def __init__(self, verbose=True):
        net_path, post_path = split_model()
        self.post = ort.InferenceSession(str(post_path), providers=["CPUExecutionProvider"])
        self.cpu_net = convert(str(net_path)).eval()
        self.net = convert(str(net_path)).eval()
        self.timer = Timer()

        # Dry run on the CPU to learn each conv's input size.
        in_hw = {}
        hooks = [m.register_forward_hook(lambda mod, i, o, n=n: in_hw.__setitem__(n, tuple(i[0].shape[2:])))
                 for n, m in self.net.named_modules()
                 if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
        with torch.no_grad():
            self.net(torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8))
        for h in hooks:
            h.remove()

        iron.set_current_device(from_name("npu", n_cols=4))
        gemms = {cfg: UGemm(*cfg) for cfg in CONFIGS}
        self.layers = []
        cpu_convs = 0
        for name, mod in list(self.net.named_modules()):
            if not isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                continue
            ok = mod.groups == 1 and tuple(mod.dilation) == (1, 1)
            if isinstance(mod, torch.nn.ConvTranspose2d):
                ok = ok and tuple(mod.kernel_size) == tuple(mod.stride) and tuple(mod.padding) == (0, 0) \
                     and tuple(mod.output_padding) == (0, 0)
            elif not isinstance(mod.padding, tuple):
                ok = False  # e.g. padding="same"
            if not ok:
                cpu_convs += 1
                continue
            layer = NpuGemmLayer(mod, in_hw[name], gemms, self.timer)
            parent, _, child = name.rpartition(".")
            (self.net.get_submodule(parent) if parent else self.net)._modules[child] = layer
            self.layers.append(layer)

        # (Each layer compiled its instruction stream while being built, i.e.
        # before the first NPU run -- see UGemm.prepare.)
        self.programs = sum(1 for g in gemms.values() if any(l.gemm is g for l in self.layers))
        if verbose:
            used = {cfg: n for cfg in CONFIGS if (n := sum(l.cfg == cfg for l in self.layers))}
            cf = sum(l.cf for l in self.layers)
            print(f"[{len(self.layers)} conv layers on the NPU ({cf} channel-first), {cpu_convs} on "
                  f"the CPU; {self.programs} resident NPU programs; layers per (m,k,n) tile: {used}]",
                  file=sys.stderr)

    def detect(self, x, net=None):
        """x: (1,3,320,320) uint8 BGR. Returns (N, 7) rows [batch, x1, y1, x2, y2, score, class]."""
        with torch.no_grad():
            boxes, scores = (net or self.net)(torch.from_numpy(x))
        t0 = time.perf_counter()
        out = self.post.run(None, {NET_OUTPUTS[0]: boxes.numpy(), NET_OUTPUTS[1]: scores.numpy()})[0]
        self.timer.add("nms", time.perf_counter() - t0)
        return out


def draw(img, dets, path, min_score):
    img = img.convert("RGB")
    sx, sy = img.width / SIZE, img.height / SIZE
    d = ImageDraw.Draw(img)
    for _, x1, y1, x2, y2, score, cls in dets:
        if score < min_score:
            continue
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        d.rectangle(box, outline=(255, 64, 64), width=3)
        d.text((box[0] + 4, box[1] + 2), f"{COCO[int(cls)]} {score:.2f}", fill=(255, 255, 0))
    img.save(path)


def fmt(dets, min_score):
    return ", ".join(f"{COCO[int(c)]} {s:.2f}" for *_, s, c in dets if s >= min_score) or "nothing"


def throughput(make_fn, threads, seconds):
    """Run make_fn() workers in `threads` threads for `seconds`; return (fps, package watts)."""
    import threading
    sys.path.insert(0, str(HERE.parent / "igpu"))
    from bench_llm import PowerMeter
    fns = [make_fn() for _ in range(threads)]
    for fn in fns:
        fn()  # warm-up
    stop, counts = threading.Event(), [0] * threads

    def work(i):
        while not stop.is_set():
            fns[i]()
            counts[i] += 1
    meter = PowerMeter()
    with meter:
        ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
    return sum(counts) / dt, meter.mean


def idle_watts(seconds=5):
    sys.path.insert(0, str(HERE.parent / "igpu"))
    from bench_llm import PowerMeter
    meter = PowerMeter()
    with meter:
        time.sleep(seconds)
    return meter.mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("--bench", type=int, default=0, help="also time N frames per backend")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--throughput", type=int, default=0, metavar="SECONDS",
                    help="also measure sustained detections/s and package power per backend")
    args = ap.parse_args()

    img = Image.open(args.image)
    x = preprocess(img)

    t0 = time.perf_counter()
    det = NpuDetector()
    # A second detector (own buffers, same resident NPU programs) lets two frames
    # be in flight in --throughput mode. Built before any NPU run; see UGemm.prepare.
    det2 = NpuDetector(verbose=False) if args.throughput else None
    print(f"[ready in {time.perf_counter() - t0:.1f}s]", file=sys.stderr)

    ref_sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    ref = ref_sess.run(None, {"input": x})[0]
    out = det.detect(x)
    print(f"NPU:              {fmt(out, args.min_score)}")
    print(f"onnxruntime CPU:  {fmt(ref, args.min_score)}")
    same = len(out) == len(ref) and np.array_equal(out[:, 6], ref[:, 6])
    if same and len(out):
        print(f"  -> same objects; max box diff {np.abs(out[:, 1:5] - ref[:, 1:5]).max():.2f} px, "
              f"max score diff {np.abs(out[:, 5] - ref[:, 5]).max():.3f}")
    else:
        print("  -> detections differ")
    outp = Path(args.image).with_name(Path(args.image).stem + "_npu.jpg")
    draw(img, out, outp, args.min_score)
    print(f"annotated: {outp}")

    if args.bench:
        n = args.bench
        rows = []

        def bench(label, fn):
            for _ in range(3):
                fn()
            det.timer.t.clear()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            dt = (time.perf_counter() - t0) / n
            parts = {k: v / n * 1000 for k, v in det.timer.t.items()}
            rows.append((label, dt * 1000, parts))

        bench("NPU (this script)", lambda: det.detect(x))
        bench("CPU, onnxruntime (full model)", lambda: ref_sess.run(None, {"input": x}))
        bench("CPU, PyTorch (same graph as NPU path)", lambda: det.detect(x, net=det.cpu_net))
        print(f"\nper-frame latency, {n} frames, 320x320 YOLO-NAS-L (~30 GFLOP/frame):")
        for label, ms, parts in rows:
            extra = ""
            if "npu" in parts:
                other = ms - sum(parts.values())
                extra = ("  [" + ", ".join(f"{k} {v:.1f}" for k, v in parts.items())
                         + f", other CPU ops {other:.1f} ms]")
            print(f"  {label:38} {ms:7.1f} ms  = {1000 / ms:5.1f} fps{extra}")

    if args.throughput:
        secs = args.throughput
        idle = idle_watts()
        dets = iter([det, det2])
        cpu_sess = lambda: (lambda: ref_sess.run(None, {"input": x}))
        rows = [
            ("NPU, 1 frame in flight", throughput(lambda: (lambda: det.detect(x)), 1, secs)),
            ("NPU, 2 frames in flight", throughput(lambda d=None: (lambda d=next(dets): lambda: d.detect(x))(), 2, secs)),
            ("CPU onnxruntime", throughput(cpu_sess, 1, secs)),
        ]
        print(f"\nsustained, {secs}s each; package power idle {idle:.1f} W:")
        for label, (fps, w) in rows:
            print(f"  {label:26} {fps:5.1f} detections/s   {w:5.1f} W   "
                  f"{(w - idle) / fps:5.2f} J/detection above idle")


if __name__ == "__main__":
    main()
