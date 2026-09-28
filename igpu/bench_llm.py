#!/usr/bin/env python3
"""Benchmark LLM speed and power: Ryzen AI NPU vs Radeon 760M vs CPU.

Backends (started and stopped by this script as needed; one already listening
on its port is reused and left running):

  npu     npu_server.py (this repo), :11435 -- Qwen2.5-0.5B bf16 on the NPU
  vulkan  llama.cpp llama-server, Vulkan on the 760M, :11437 -- build it
          first (README.md, "Radeon 760M")
  rocm    Ollama with ROCm on the 760M via ./ollama-760m.sh, :11436
  cpu     the system Ollama, :11434, with num_gpu=0

The llama.cpp/Ollama backends run the GGUF files the system Ollama downloaded.
Every request starts with a unique line, so no backend can reuse a cached
prompt. Speeds are the backends' own prompt/generation counters.

Package power is sampled from the amdgpu "PPT" sensor, which on this APU
covers the whole SoC (CPU + iGPU + NPU). An idle baseline is measured right
before each row; "J/tok" is the energy above idle per generated token (whole
request, prefill included). Background load on the machine (containers etc.)
adds a few watts of noise, so rows drawing less than NOISE_W above idle show
n/a.

Usage:
    .venv/bin/python bench_llm.py
    .venv/bin/python bench_llm.py --only npu,vulkan --tokens 64 --runs 1
"""

import argparse
import glob
import itertools
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "/var/lib/ollama")
LLAMA_SERVER = os.path.join(HERE, "..", "llama.cpp/build-vulkan/bin/llama-server")
VULKAN_DEVICE = os.environ.get("VULKAN_DEVICE", "Vulkan1")  # Vulkan0 = RTX 3060 Ti
NOISE_W = 2.0

PROMPT = ("Explain how a refrigerator works to a curious ten-year-old. "
          "Cover the compressor, the refrigerant, and why the back of the fridge is warm.")

Q05_F16 = "qwen2.5:0.5b-instruct-fp16"
# (device label, backend, model). The NPU runs bf16 Qwen2.5-0.5B, so the like-
# for-like rows use the fp16 GGUF; q4 is what Ollama runs by default.
DEFAULT_MATRIX = [
    ("NPU", "npu", "qwen2.5-npu:0.5b"),
    ("760M Vulkan", "vulkan", Q05_F16),
    ("760M ROCm", "rocm", Q05_F16),
    ("CPU", "cpu", Q05_F16),
    ("760M Vulkan", "vulkan", "qwen2.5:0.5b"),
    ("760M Vulkan", "vulkan", "qwen2.5:1.5b"),
    ("760M Vulkan", "vulkan", "qwen2.5:3b"),
    ("760M Vulkan", "vulkan", "llama3.2:latest"),
    ("760M Vulkan", "vulkan", "qwen2.5:7b"),
    ("760M ROCm", "rocm", "qwen2.5:7b"),
    ("CPU", "cpu", "qwen2.5:7b"),
]

_nonce = itertools.count()


def unique_prompt():
    return f"(request {os.getpid()}-{next(_nonce)})\n{PROMPT}"


# ---- power -----------------------------------------------------------------

class PowerMeter:
    """Samples the APU package power (amdgpu PPT, microwatts) in a thread."""

    def __init__(self, hz=10):
        paths = [p for p in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_input")
                 if open(p.replace("_input", "_label")).read().strip() == "PPT"]
        self.path = paths[0] if paths else None
        self.period = 1 / hz
        self.samples = []
        self._stop = threading.Event()

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self.t0 = time.perf_counter()
        if self.path:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(int(open(self.path).read()) / 1e6)
            time.sleep(self.period)

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self._stop.set()
        if self.path:
            self._t.join()

    @property
    def mean(self):
        return statistics.fmean(self.samples) if self.samples else float("nan")


# ---- HTTP / process helpers --------------------------------------------------

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def is_up(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def wait_up(url, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_up(url):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        time.sleep(0.5)
    raise RuntimeError(f"{url} not up after {timeout}s")


def spawn(cmd, env=None):
    log = open(os.path.join(HERE, ".bench-server.log"), "ab")
    return subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=log,
                            env={**os.environ, **(env or {})}, start_new_session=True)


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


def gguf_path(model):
    """Ollama model name -> its GGUF blob, via the manifest."""
    name, _, tag = model.partition(":")
    manifest = os.path.join(OLLAMA_MODELS, "manifests/registry.ollama.ai/library", name, tag or "latest")
    layers = json.load(open(manifest))["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"].endswith(".model"))
    return os.path.join(OLLAMA_MODELS, "blobs", digest.replace(":", "-"))


# ---- backends ----------------------------------------------------------------
# Each backend: load(model), run(model, n) -> (prompt_n, prompt_tps, gen_n, gen_tps),
# where(model) -> "NPU"/"GPU"/"CPU", unload(model), close().

class OllamaBackend:
    def __init__(self, url, launcher=None, cpu_only=False):
        self.url, self.cpu_only, self.proc = url, cpu_only, None
        if not is_up(url + "/api/version"):
            if launcher is None:
                raise RuntimeError(f"no Ollama at {url}")
            self.proc = spawn([launcher])
            wait_up(url + "/api/version", self.proc)

    def _chat(self, model, n):
        opts = {"temperature": 0, "num_predict": n, "seed": 1}
        if self.cpu_only:
            opts["num_gpu"] = 0
        return post(f"{self.url}/api/chat", {
            "model": model, "stream": False, "keep_alive": "5m", "options": opts,
            "messages": [{"role": "user", "content": unique_prompt()}]})

    def load(self, model):
        self._chat(model, 8)

    def run(self, model, n):
        r = self._chat(model, n)
        return (r["prompt_eval_count"], r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9),
                r["eval_count"], r["eval_count"] / (r["eval_duration"] / 1e9))

    def where(self, model):
        with urllib.request.urlopen(f"{self.url}/api/ps", timeout=10) as r:
            for m in json.loads(r.read())["models"]:
                if m["name"] == model:
                    gpu = m.get("size_vram", 0) / m["size"]
                    return "CPU" if gpu == 0 else f"{gpu:.0%} GPU"
        return "?"

    def unload(self, model):
        try:
            post(f"{self.url}/api/generate", {"model": model, "keep_alive": 0})
        except Exception:
            pass

    def close(self):
        stop(self.proc)


class NpuBackend(OllamaBackend):
    def __init__(self, url="http://127.0.0.1:11435"):
        self.url, self.cpu_only, self.proc = url, False, None
        if not is_up(url + "/api/version"):
            self.proc = spawn([sys.executable, os.path.join(HERE, "..", "llm", "npu_server.py"),
                               "--port", url.rsplit(":", 1)[1]])
            wait_up(url + "/api/version", self.proc)

    def where(self, model):
        return "NPU"

    def unload(self, model):
        pass


class LlamaServerBackend:
    """llama.cpp's llama-server on the 760M via Vulkan, one process per model."""

    def __init__(self, port=11437):
        if not os.path.exists(LLAMA_SERVER):
            raise RuntimeError(f"{LLAMA_SERVER} not built")
        self.port, self.url, self.proc, self.model = port, f"http://127.0.0.1:{port}", None, None

    def load(self, model):
        if self.model != model:
            self.unload(self.model)
            self.proc = spawn([LLAMA_SERVER, "-m", gguf_path(model), "--port", str(self.port),
                               "-dev", VULKAN_DEVICE, "-ngl", "99", "-fa", "on", "-c", "4096",
                               "--no-webui"])
            wait_up(self.url + "/health", self.proc)
            self.model = model
        self.run(model, 8)

    def run(self, model, n):
        r = post(f"{self.url}/v1/chat/completions", {
            "messages": [{"role": "user", "content": unique_prompt()}],
            "max_tokens": n, "temperature": 0, "seed": 1, "cache_prompt": False})
        t = r["timings"]
        return t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]

    def where(self, model):
        return "GPU (VK)"

    def unload(self, model):
        stop(self.proc)
        self.proc, self.model = None, None

    def close(self):
        self.unload(self.model)


def make_backend(name):
    if name == "npu":
        return NpuBackend()
    if name == "vulkan":
        return LlamaServerBackend()
    if name == "rocm":
        return OllamaBackend("http://127.0.0.1:11436", launcher=os.path.join(HERE, "ollama-760m.sh"))
    if name == "cpu":
        return OllamaBackend("http://127.0.0.1:11434", cpu_only=True)
    raise ValueError(name)


# ---- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=128, help="tokens to generate per run")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per row (after a warm-up)")
    ap.add_argument("--only", help="comma list of backends: npu,vulkan,rocm,cpu")
    ap.add_argument("--models", help="comma list of model names to keep")
    args = ap.parse_args()

    rows = DEFAULT_MATRIX
    if args.only:
        rows = [r for r in rows if r[1] in args.only.split(",")]
    if args.models:
        rows = [r for r in rows if r[2] in args.models.split(",")]

    meter = PowerMeter()
    print(f"power sensor: {meter.path or 'none found; power columns will be empty'}\n")
    print("| device | model | runs on | prefill tok/s | decode tok/s | idle W | load W | J/tok |")
    print("|---|---|---|--:|--:|--:|--:|--:|")

    backends = {}
    try:
        for label, bname, model in rows:
            try:
                if bname not in backends:
                    backends[bname] = make_backend(bname)
                be = backends[bname]
                be.load(model)  # also a warm-up generation
                with meter:
                    time.sleep(4)
                idle = meter.mean
                where = be.where(model)
                pre, dec, watts, jpt = [], [], [], []
                for _ in range(args.runs):
                    with meter:
                        pn, ptps, gn, gtps = be.run(model, args.tokens)
                    pre.append(ptps)
                    dec.append(gtps)
                    watts.append(meter.mean)
                    extra = meter.mean - idle
                    jpt.append(extra * meter.seconds / gn if extra >= NOISE_W else float("nan"))
                be.unload(model)
            except Exception as e:
                print(f"| {label} | {model} | FAILED: {e} | | | | | |", flush=True)
                continue
            j = statistics.median(jpt)
            jtxt = "n/a" if j != j else f"{j:.2f}"
            print(f"| {label} | {model} | {where} | {statistics.median(pre):.0f} | "
                  f"{statistics.median(dec):.1f} | {idle:.1f} | {statistics.median(watts):.1f} | {jtxt} |",
                  flush=True)
    finally:
        for be in backends.values():
            be.close()

    print(f"\n{args.tokens} generated tokens per run, prompt ~{len(PROMPT.split()) + 20} tokens, "
          f"median of {args.runs} runs, greedy. J/tok = energy above idle per generated token "
          f"(n/a when under {NOISE_W} W above idle, i.e. within noise).")


if __name__ == "__main__":
    main()
