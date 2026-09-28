#!/usr/bin/env python3
"""Chat with an LLM running on the Ryzen AI NPU (XDNA 1 / Phoenix).

Qwen2.5-0.5B-Instruct, with every linear layer -- the attention q/k/v/o
projections, the MLP gate/up/down projections, and the vocabulary (LM) head --
executed as bf16 GEMMs on the NPU. The weights are uploaded once and stay in
NPU-mapped buffers. The CPU does the small non-matmul work: RMSNorm, RoPE,
attention over the KV cache, SiLU, residual adds and sampling.

Ollama and the other turnkey NPU runners (FastFlowLM, Lemonade) don't support
this first-generation NPU; this is a from-scratch alternative on AMD's mlir-aie
(IRON) toolchain, in the spirit of github.com/Scottcjn/open-xdna.

Usage:
    .venv/bin/python npu_llm.py "Why is the sky blue?"
    .venv/bin/python npu_llm.py              # interactive chat
    .venv/bin/python npu_llm.py --check "Hi"  # compare with PyTorch on CPU
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import aie.iron as iron
from aie.iron.device import from_name

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlir_aie"))
from whole_array import whole_array  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Every GEMM is (ROWS x K) @ (K x N): up to ROWS tokens at once. Decode uses
# one row; prefill fills up to ROWS rows per pass. The 4x4-core design needs
# ROWS to be a multiple of TILE_M * 4 rows * 2, and N a multiple of 32 * 4.
ROWS, TILE_M, TILE_K, TILE_N, N_AIE_COLS = 128, 16, 64, 32, 4
# Widest N the design handles here; N=9728 timed out on this NPU. The LM head
# is split into chunks of this width.
MAX_N = 4864

bf16 = iron.str_to_dtype("bf16")


class NpuMatmul:
    """Runs x @ W on the NPU for weight matrices that stay resident there.

    Activation and output buffers are shared per shape, since the layers run
    one at a time.
    """

    def __init__(self):
        iron.set_current_device(from_name("npu", n_cols=N_AIE_COLS))
        self._in = {}
        self._out = {}
        self.seconds = 0.0
        self.calls = 0

    def upload(self, w):
        """w: (K, N) float tensor -> resident NPU weight handle."""
        K, N = w.shape
        assert K % TILE_K == 0 and N % (TILE_N * N_AIE_COLS) == 0 and N <= MAX_N, w.shape
        # .T views are column-major; the NPU expects row-major (K, N).
        w = np.ascontiguousarray(w.detach().to(torch.float32).numpy()).astype(bf16)
        return iron.tensor(w, dtype=bf16, device="npu")

    def __call__(self, x, w):
        """x: (T, K) float32 numpy, T <= ROWS; w: handle from upload()."""
        T, K = x.shape
        N = w.shape[1]
        a = self._in.get(K)
        if a is None:
            a = self._in[K] = iron.zeros((ROWS, K), dtype=bf16, device="npu")
        c = self._out.get(N)
        if c is None:
            c = self._out[N] = iron.zeros((ROWS, N), dtype=np.float32, device="npu")
        a[:T] = x.astype(bf16)
        t0 = time.perf_counter()
        whole_array(
            a, w, c,
            M=ROWS, K=K, N=N, m=TILE_M, k=TILE_K, n=TILE_N,
            n_aie_cols=N_AIE_COLS, dtype_in_str="bf16", dtype_out_str="f32",
        )
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return c.numpy()[:T].copy()


class NpuQwen2:
    def __init__(self, hf_model, mm):
        cfg = hf_model.config
        self.mm = mm
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.vocab = cfg.vocab_size
        rope = getattr(cfg, "rope_parameters", None) or {"rope_theta": cfg.rope_theta}
        inv = 1.0 / (rope["rope_theta"] ** (np.arange(0, self.hd, 2) / self.hd))
        self.inv_freq = inv.astype(np.float32)

        m = hf_model.model
        f32 = lambda t: t.detach().to(torch.float32).numpy()
        self.embed = f32(m.embed_tokens.weight)
        self.final_norm = f32(m.norm.weight)
        self.layers = []
        for i, L in enumerate(m.layers):
            at, ml = L.self_attn, L.mlp
            qkv_w = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight])
            qkv_b = torch.cat([at.q_proj.bias, at.k_proj.bias, at.v_proj.bias])
            self.layers.append(dict(
                ln1=f32(L.input_layernorm.weight),
                ln2=f32(L.post_attention_layernorm.weight),
                qkv=mm.upload(qkv_w.T), qkv_b=f32(qkv_b),
                o=mm.upload(at.o_proj.weight.T),
                gate=mm.upload(ml.gate_proj.weight.T),
                up=mm.upload(ml.up_proj.weight.T),
                down=mm.upload(ml.down_proj.weight.T),
            ))
            print(f"\ruploading weights to NPU: layer {i + 1}/{self.n_layers}", end="", file=sys.stderr)
        # Tied LM head: embed^T, zero-padded to a whole number of MAX_N chunks.
        head = torch.from_numpy(self.embed).T
        n_chunks = -(-self.vocab // MAX_N)
        head = torch.nn.functional.pad(head, (0, n_chunks * MAX_N - self.vocab))
        self.head = [mm.upload(head[:, j * MAX_N:(j + 1) * MAX_N]) for j in range(n_chunks)]
        print("\r" + " " * 50 + "\r", end="", file=sys.stderr)
        self.reset()

    def reset(self):
        self.k_cache = [np.zeros((self.n_kv, 0, self.hd), np.float32) for _ in self.layers]
        self.v_cache = [np.zeros((self.n_kv, 0, self.hd), np.float32) for _ in self.layers]

    @property
    def pos(self):
        return self.k_cache[0].shape[1]

    def _rms(self, x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * w

    def _rope(self, x, positions):
        """x: (heads, T, hd); rotate-half RoPE as in Qwen2."""
        ang = positions[:, None].astype(np.float32) * self.inv_freq[None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1)
        sin = np.concatenate([np.sin(ang)] * 2, -1)
        h = self.hd // 2
        rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
        return x * cos + rot * sin

    def forward(self, ids):
        """Feed up to ROWS new tokens; returns logits for the last one."""
        T = len(ids)
        assert T <= ROWS
        p0 = self.pos
        positions = np.arange(p0, p0 + T)
        H, KV, hd = self.n_heads, self.n_kv, self.hd
        # Causal mask over [cache + new tokens].
        mask = np.triu(np.full((T, p0 + T), -np.inf, np.float32), k=p0 + 1)

        x = self.embed[ids]
        for i, L in enumerate(self.layers):
            h = self._rms(x, L["ln1"])
            qkv = self.mm(h, L["qkv"]) + L["qkv_b"]
            q = qkv[:, :H * hd].reshape(T, H, hd).transpose(1, 0, 2)
            k = qkv[:, H * hd:(H + KV) * hd].reshape(T, KV, hd).transpose(1, 0, 2)
            v = qkv[:, (H + KV) * hd:].reshape(T, KV, hd).transpose(1, 0, 2)
            q, k = self._rope(q, positions), self._rope(k, positions)
            K = self.k_cache[i] = np.concatenate([self.k_cache[i], k], 1)
            V = self.v_cache[i] = np.concatenate([self.v_cache[i], v], 1)
            K = np.repeat(K, H // KV, 0)
            V = np.repeat(V, H // KV, 0)
            s = q @ K.transpose(0, 2, 1) / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att = (s @ V).transpose(1, 0, 2).reshape(T, H * hd)
            x = x + self.mm(att, L["o"])

            h = self._rms(x, L["ln2"])
            g = self.mm(h, L["gate"])
            u = self.mm(h, L["up"])
            x = x + self.mm(g / (1 + np.exp(-g)) * u, L["down"])

        h = self._rms(x[-1:], self.final_norm)
        logits = np.concatenate([self.mm(h, w) for w in self.head], 1)[0]
        return logits[:self.vocab]

    def prefill(self, ids):
        logits = None
        for s in range(0, len(ids), ROWS):
            logits = self.forward(ids[s:s + ROWS])
        return logits


def sample(logits, temperature, rng):
    if temperature <= 0:
        return int(logits.argmax())
    p = np.exp((logits - logits.max()) / temperature)
    return int(rng.choice(len(p), p=p / p.sum()))


STOP_TOKENS = (151643, 151645)  # <|endoftext|>, <|im_end|>


def stream_generate(net, tok, messages, max_new, temperature, rng, info=None):
    """Yields the reply as text pieces. Fills `info` with counts and timings."""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):  # newer transformers return a BatchEncoding
        ids = ids["input_ids"]
    info = {} if info is None else info
    net.reset()
    mm = net.mm
    mm.seconds, mm.calls = 0.0, 0

    t0 = time.perf_counter()
    logits = net.prefill(ids)
    t_prefill = time.perf_counter() - t0
    info.update(prompt_tokens=len(ids), prefill_s=t_prefill, tokens=[], finish="length")

    out, printed = info["tokens"], ""
    t1 = time.perf_counter()
    while len(out) < max_new:
        nxt = sample(logits, temperature, rng)
        if nxt == tok.eos_token_id or nxt in STOP_TOKENS:
            info["finish"] = "stop"
            break
        out.append(nxt)
        text = tok.decode(out, skip_special_tokens=True)
        if not text.endswith("\ufffd"):  # wait for complete UTF-8 chars
            yield text[len(printed):]
            printed = text
        logits = net.forward([nxt])
    info.update(decode_s=time.perf_counter() - t1, npu_s=mm.seconds, npu_calls=mm.calls)


def generate(net, tok, messages, max_new, temperature, rng):
    info = {}
    for piece in stream_generate(net, tok, messages, max_new, temperature, rng, info):
        print(piece, end="", flush=True)
    print(flush=True)
    out, t_decode = info["tokens"], info["decode_s"]
    stats = (f"[prompt {info['prompt_tokens']} tok in {info['prefill_s']:.2f}s | "
             f"{len(out)} tok in {t_decode:.2f}s = {len(out) / max(t_decode, 1e-9):.2f} tok/s | "
             f"{info['npu_calls']} NPU GEMMs, "
             f"{info['npu_s'] / (info['prefill_s'] + t_decode) * 100:.0f}% of time in NPU calls]")
    return out, stats


def load(verbose=True):
    """Load the tokenizer, the fp32 PyTorch model and the NPU model."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    t0 = time.perf_counter()
    net = NpuQwen2(hf, NpuMatmul())
    if verbose:
        print(f"[{MODEL_ID} loaded onto the NPU in {time.perf_counter() - t0:.1f}s]", file=sys.stderr)
    return tok, hf, net


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompt", nargs="?", help="one-shot prompt (omit for interactive chat)")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--check", action="store_true",
                    help="also generate greedily with PyTorch on the CPU and compare")
    args = ap.parse_args()

    tok, hf, net = load()
    rng = np.random.default_rng()
    system = {"role": "system", "content": "You are a helpful assistant."}

    if args.prompt:
        msgs = [system, {"role": "user", "content": args.prompt}]
        out, stats = generate(net, tok, msgs, args.max_new, args.temperature, rng)
        print(stats, file=sys.stderr)
        if args.check:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            with torch.no_grad():
                full = hf.generate(ids, max_new_tokens=len(out), do_sample=False)[0]
                ref_logits = hf(full[None]).logits[0].numpy()
            prompt, ref = ids[0].tolist(), full[ids.shape[1]:].tolist()
            # Greedy runs diverge after the first near-tie (bf16 vs fp32), so
            # compare step by step on the CPU's sequence (teacher forcing).
            net.reset()
            logits = net.prefill(prompt)
            agree, max_diff = 0, 0.0
            for i, t in enumerate(ref):
                agree += int(logits.argmax()) == t
                max_diff = max(max_diff, np.abs(logits - ref_logits[len(prompt) - 1 + i]).max())
                logits = net.forward([t])
            same = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), len(ref))
            print(f"[CPU check vs PyTorch fp32: next-token prediction agrees on {agree}/{len(ref)} steps, "
                  f"max |logit diff| {max_diff:.2f}; free-running greedy text identical for "
                  f"{same}/{len(ref)} tokens]", file=sys.stderr)
        return

    history = [system]
    print("Chat with Qwen2.5-0.5B on the NPU. Empty line or Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            user = input("\n> ").strip()
        except EOFError:
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        out, stats = generate(net, tok, history, args.max_new, args.temperature, rng)
        history.append({"role": "assistant", "content": tok.decode(out, skip_special_tokens=True)})
        print(stats, file=sys.stderr)


if __name__ == "__main__":
    main()
