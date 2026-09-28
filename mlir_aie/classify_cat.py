#!/usr/bin/env python3
"""Is there a cat in this picture? -- ResNet-50 on the AMD Ryzen AI NPU.

Every convolution and the final fully-connected layer of an ImageNet-trained
ResNet-50 run as bf16 matrix multiplies on the NPU (XDNA / "RyzenAI-npu1"),
using the whole-array GEMM design from AMD's mlir-aie (IRON) project. The CPU
only does the cheap glue: im2col, bias + ReLU, pooling, and residual adds.

Each distinct GEMM shape is JIT-compiled to an NPU binary on first use and
cached, so the first run takes a minute or two and later runs are fast.

Usage:
    ./npu-run.sh python3 classify_cat.py images/cat.jpg [more.jpg ...] [--check]

--check also runs the stock PyTorch model on the CPU and compares its output
with the NPU's.
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import aie.iron as iron
from aie.iron.device import from_name

from whole_array import whole_array

# ImageNet classes 281-285 are the domestic cats (tabby, tiger cat, Persian,
# Siamese, Egyptian cat); 286-293 are the big cats (cougar ... cheetah).
DOMESTIC_CATS = range(281, 286)
BIG_CATS = range(286, 294)

# GEMM tiling for the NPU: each of the 4x4 AIE cores computes m x n output
# tiles, consuming k-deep slices. M, K, N are padded to fit these tiles.
TILE_M, TILE_K, TILE_N = 64, 64, 32
N_AIE_ROWS, N_AIE_COLS = 4, 4
M_ALIGN = TILE_M * N_AIE_ROWS * 2  # the design moves 2 row-blocks per step
K_ALIGN = TILE_K
N_ALIGN = TILE_N * N_AIE_COLS


def round_up(x, a):
    return (x + a - 1) // a * a


class NpuGemm:
    """C = A @ B on the NPU (bf16 inputs, fp32 accumulate/output)."""

    def __init__(self):
        iron.set_current_device(from_name("npu", n_cols=N_AIE_COLS))
        self.npu_seconds = 0.0
        self.calls = 0

    def __call__(self, a, b):
        M, K = a.shape
        _, N = b.shape
        Mp, Kp, Np = round_up(M, M_ALIGN), round_up(K, K_ALIGN), round_up(N, N_ALIGN)

        a_p = np.zeros((Mp, Kp), dtype=np.float32)
        a_p[:M, :K] = a
        b_p = np.zeros((Kp, Np), dtype=np.float32)
        b_p[:K, :N] = b

        bf16 = iron.str_to_dtype("bf16")
        a_t = iron.tensor(a_p.astype(bf16), dtype=bf16, device="npu")
        b_t = iron.tensor(b_p.astype(bf16), dtype=bf16, device="npu")
        c_t = iron.zeros((Mp, Np), dtype=np.float32, device="npu")

        t0 = time.perf_counter()
        whole_array(
            a_t, b_t, c_t,
            M=Mp, K=Kp, N=Np,
            m=TILE_M, k=TILE_K, n=TILE_N,
            n_aie_cols=N_AIE_COLS,
            dtype_in_str="bf16", dtype_out_str="f32",
        )
        self.npu_seconds += time.perf_counter() - t0
        self.calls += 1
        # .numpy() is a view of the NPU buffer's mapping, which is freed with
        # c_t -- copy the result out before returning.
        return c_t.numpy().reshape(Mp, Np)[:M, :N].copy()


def fold_bn(conv, bn):
    """Fold eval-mode BatchNorm into the preceding conv's weight and bias."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w.detach(), b.detach()


class NpuResNet50:
    def __init__(self, model, gemm):
        self.gemm = gemm
        # Pre-fold every conv+bn pair into GEMM-ready (K, Cout) weights.
        self.layers = {}
        self._add("stem", model.conv1, model.bn1)
        for li in range(1, 5):
            for bi, blk in enumerate(getattr(model, f"layer{li}")):
                p = f"layer{li}.{bi}"
                self._add(f"{p}.conv1", blk.conv1, blk.bn1)
                self._add(f"{p}.conv2", blk.conv2, blk.bn2)
                self._add(f"{p}.conv3", blk.conv3, blk.bn3)
                if blk.downsample is not None:
                    self._add(f"{p}.down", blk.downsample[0], blk.downsample[1])
        self.model = model
        self.fc_w = model.fc.weight.detach().T.numpy()  # (2048, 1000)
        self.fc_b = model.fc.bias.detach().numpy()

    def _add(self, name, conv, bn):
        w, b = fold_bn(conv, bn)
        cout = w.shape[0]
        self.layers[name] = dict(
            w=w.reshape(cout, -1).T.contiguous().numpy(),  # (Cin*kh*kw, Cout)
            b=b.numpy(),
            k=conv.kernel_size, s=conv.stride, p=conv.padding,
        )

    def conv(self, x, name, relu=True):
        """Convolution as im2col (CPU) + GEMM (NPU). x: (1, C, H, W) tensor."""
        L = self.layers[name]
        kh, kw = L["k"]
        H, W = x.shape[2:]
        Ho = (H + 2 * L["p"][0] - kh) // L["s"][0] + 1
        Wo = (W + 2 * L["p"][1] - kw) // L["s"][1] + 1
        if (kh, kw) == (1, 1):
            patches = x[:, :, :: L["s"][0], :: L["s"][1]].flatten(2)
        else:
            patches = F.unfold(x, (kh, kw), padding=L["p"], stride=L["s"])
        a = patches[0].T.numpy()  # (Ho*Wo, Cin*kh*kw)
        y = self.gemm(a, L["w"]) + L["b"]
        if relu:
            y = np.maximum(y, 0)
        return torch.from_numpy(np.ascontiguousarray(y.T)).reshape(1, -1, Ho, Wo)

    def __call__(self, x):
        x = self.conv(x, "stem")
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        for li in range(1, 5):
            for bi, blk in enumerate(getattr(self.model, f"layer{li}")):
                p = f"layer{li}.{bi}"
                idt = self.conv(x, f"{p}.down", relu=False) if blk.downsample is not None else x
                y = self.conv(x, f"{p}.conv1")
                y = self.conv(y, f"{p}.conv2")
                y = self.conv(y, f"{p}.conv3", relu=False)
                x = torch.relu(y + idt)
        feat = x.mean(dim=(2, 3)).numpy()  # (1, 2048)
        return self.gemm(feat, self.fc_w)[0] + self.fc_b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="also run on the CPU with PyTorch and compare")
    args = ap.parse_args()

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).eval()
    labels = weights.meta["categories"]
    preprocess = weights.transforms()

    gemm = NpuGemm()
    net = NpuResNet50(model, gemm)

    for path in args.images:
        x = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)

        gemm.npu_seconds, gemm.calls = 0.0, 0
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = net(x)
        total = time.perf_counter() - t0

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top5 = np.argsort(probs)[::-1][:5]
        p_cat = probs[list(DOMESTIC_CATS)].sum()
        p_big = probs[list(BIG_CATS)].sum()

        print(f"\n=== {path} ===")
        for i in top5:
            print(f"  {probs[i]*100:6.2f}%  {labels[i]}")
        # These weights were trained with label smoothing, so probabilities are
        # spread out; judge by the top-1 class, and report the summed cat share.
        if top5[0] in DOMESTIC_CATS:
            verdict = f"CAT! ({p_cat*100:.1f}% total on domestic-cat classes)"
        elif top5[0] in BIG_CATS:
            verdict = f"a big cat ({p_big*100:.1f}%), not a house cat"
        else:
            verdict = f"no cat ({p_cat*100:.1f}% on domestic-cat classes)"
        print(f"  -> {verdict}")
        print(f"  [{gemm.calls} GEMMs on NPU, {gemm.npu_seconds*1000:.0f} ms in NPU calls, "
              f"{total*1000:.0f} ms total]")

        if args.check:
            with torch.no_grad():
                ref = model(x)[0].numpy()
            ref_p = torch.softmax(torch.from_numpy(ref), 0).numpy()
            print(f"  CPU check: top-1 '{labels[ref.argmax()]}' ({ref_p.max()*100:.2f}%), "
                  f"max |logit diff| = {np.abs(ref - logits).max():.3f}, "
                  f"top-1 {'MATCHES' if ref.argmax() == logits.argmax() else 'DIFFERS'}")


if __name__ == "__main__":
    sys.exit(main())
