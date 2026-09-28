# xdna1-npu-lab

This repo gets the first-generation AMD XDNA NPU (Ryzen 7040/8040 "Phoenix" and "Hawk Point", and the Ryzen 8000G desktop APUs) doing real work on Linux. Its main result is **Frigate object detection on the NPU**: YOLO-NAS-L in int8 at 19–24 ms per detection, for about 6 W.

Ollama, FastFlowLM and AMD Lemonade don't support this NPU generation, and AMD's Ryzen AI Linux release officially targets only the newer XDNA 2. Everything here was built and measured on a **Ryzen 5 8600G** (Arch Linux, kernel 6.18, in-tree `amdxdna` driver, NPU firmware 1.5.2.380), with a Radeon 760M iGPU and an RTX 3060 Ti in the same box.

| What | Result | Directory |
|---|---|---|
| Frigate detector: YOLO-NAS-L 320, XINT8 with AdaRound, via AMD's Vitis AI runtime | **18.8 ms, 57 detections/s, +6 W; keeps 95.7% of FP32 detections**; runs in Frigate 0.18 via its `zmq` detector | `yolonas_int8/`, `frigate_npu/` |
| Hand-built CNN inference on mlir-aie (IRON): ResNet-50, then YOLO-NAS-L | correct, but 118 ms per detection: the CPU work between layers dominates | `mlir_aie/` |
| Shape-agnostic bf16 GEMM: one resident NPU program for all shapes | up to 1.58 TFLOPS; avoids the driver's 6-context limit | `mlir_aie/npu_gemm.py` |
| LLM on the NPU: Qwen2.5-0.5B, all linear layers on the NPU, Ollama/OpenAI-compatible API | works, but only 2.5 tokens/s | `llm/` |
| Radeon 760M vs CPU vs NPU for LLMs | the 760M (llama.cpp Vulkan) wins: 124 tok/s on 0.5B Q4, 15.5 tok/s on 7B Q4 | `igpu/` |

The verdicts: use the NPU for always-on vision at a few watts, and the iGPU for LLMs.

## 0. Make the NPU usable

The NPU needs XRT with the `amdxdna` plugin (Arch: `xrt`, `xrt-plugin-amdxdna`). `xrt-smi examine` should list `RyzenAI-npu1`.

XRT locks a 64 MB buffer. With the common 8 MB memlock default, every NPU program fails with `mmap(... len=67108864 ...) failed (err=-11)`. To fix it:

```sh
sudo mkdir -p /etc/security/limits.d
echo "$USER - memlock unlimited" | sudo tee /etc/security/limits.d/99-npu.conf
# log out and in; `ulimit -l` must print "unlimited"
```

Without root, `./npu-run.sh <command>` runs a command with unlimited memlock via Docker. Containers that use the NPU need `--device /dev/accel/accel0 --ulimit memlock=-1:-1`.

## 1. Frigate detection on the NPU (`yolonas_int8/`, `frigate_npu/`)

This is the working path. AMD's Vitis AI runtime from Ryzen AI 1.7.1 compiles the whole int8 model into one NPU pipeline. It isn't officially supported on Phoenix, but it runs. The recipe follows [Frigate PR #24470](https://github.com/blakeblackshear/frigate/pull/24470), which covers YOLO-NAS-S and YOLOv8s. This repo runs **YOLO-NAS-L**, cut from the same ONNX file Frigate already uses, and calibrates it on your own camera snapshots.

Measured on the 8600G (accuracy is compared on 100 held-out snapshots against the FP32 model; there is no hand-labelled ground truth):

| Backend | Per detection | Sustained | Energy above idle | FP32 detections kept |
|---|--:|--:|--:|--:|
| NPU, YOLO-NAS-L XINT8 + AdaRound | 18.8 ms | 57.5/s | 0.11 J (+6 W) | 95.7% (person 59/60) |
| NPU, YOLO-NAS-L XINT8 (no AdaRound) | 19.8 ms | 54.9/s | | 76.5% |
| CPU, YOLO-NAS-L FP32 (onnxruntime) | ~70 ms | 14.9/s | 1.86 J | 100% |
| RTX 3060 Ti, YOLO-NAS-L FP32 (Frigate's onnx detector) | 11.4 ms | | | 100% |

989 of the 996 graph nodes run on the NPU. Inside Frigate (zmq round trip included), it reports 23.7 ms and tops out at about 42 detections/s.

### Build the model

You need Frigate's exported `yolo_nas_l_320.onnx` in `models/` (see Frigate's YOLO-NAS docs; the weights are under Deci's license, so they aren't included here). You also need AMD's `ryzen_ai-1.7.1.tgz` in the repo root, downloaded from AMD's Ryzen AI Linux page with an account and EULA.

```sh
cd yolonas_int8
python make_calib.py /path/to/frigate/storage/clips          # -> data/calib.npy, data/eval.npy (your cameras)
python -c "import nas_raw; nas_raw.make_raw('../models/yolo_nas_l_320.onnx', '../models/yolo_nas_l_320_raw.onnx')"
docker build -t npu-quark -f docker/Dockerfile.quark docker
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$PWD/..":/w -w /w/yolonas_int8 \
    npu-quark python3 quantize.py --adaround                 # ~1 h on the CPU -> models/yolo_nas_l_320_xint8_adaround.onnx
python evaluate.py ../models/yolo_nas_l_320_xint8_adaround.onnx   # accuracy vs FP32 on the CPU
./npu_bench.sh ../models/yolo_nas_l_320_xint8_adaround.onnx       # builds the npu-ryzenai image, runs on the NPU
```

`nas_raw.make_raw` extracts the raw 6-output head (cls and reg per stride 8/16/32). It also flips the stem convolution from BGR to RGB, so the model takes RGB 0–1 float input. With the ported decode, it reproduces Frigate's detections exactly.

### Run it in Frigate (0.18+)

Frigate's built-in `zmq` detector sends each preprocessed frame to an external process. `frigate_npu/` is that process: a sidecar container on Frigate's Docker network. Frigate's image stays stock, so NVIDIA or other hardware can keep doing stream decoding and embeddings.

```sh
cd frigate_npu
docker compose up -d        # needs the npu-ryzenai image from npu_bench.sh and the model in ../models
```

Copy `yolo_nas_l_320_xint8_adaround.onnx` into Frigate's `/config/model_cache` too. Frigate only checks that the file exists; inference happens in the sidecar. Then set:

```yaml
detectors:
  npu:
    type: zmq
    endpoint: tcp://frigate-npu:5555
model:
  path: /config/model_cache/yolo_nas_l_320_xint8_adaround.onnx
  model_type: ssd
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: rgb
  input_dtype: float
  labelmap_path: /labelmap/coco-80.txt
```

The compose file assumes Frigate's network is called `frigate_default`. `test_frigate_side.py` runs Frigate's own `LocalObjectDetector` and zmq plugin against the sidecar, so you can test before touching your config. Right after a Frigate restart, everything in view is re-detected, so expect a minute of skipped frames.

### Settings that decide whether it works on Phoenix

- **`xlnx_enable_py3_round=0`** (a provider option). Without it, the whole graph silently runs on the CPU.
- **`XILINX_XRT=/usr`**. Without it, the session only compiles ("XRT is not installed ... generates model binaries only").
- **`target=X1`** and the `voe-4.0-linux_x86_64/xclbins/phoenix/4x4.xclbin` overlay.
- **Only XINT8 models with power-of-two scales reach the NPU on XDNA 1.** AdaRound matters: without it, detections kept drop from 96% to 78%.
- **The partition report is the only proof of NPU execution** (`XLNX_ONNX_EP_REPORT_FILE`): check its NPU node count. The provider name alone proves nothing.

## 2. Hand-built NPU inference on mlir-aie (`mlir_aie/`)

This uses AMD's open [mlir-aie](https://github.com/Xilinx/mlir-aie) 1.4.3 (IRON) plus llvm-aie, from pip wheels (Python 3.14 works), with the system `pyxrt`.

- **`classify_cat.py`:** ResNet-50 with every convolution on the NPU as bf16 GEMMs. It matches PyTorch within 0.05 logits.
- **`npu_gemm.py` (`UGemm`):** a bf16 GEMM on the 4×4 core array whose loop counts are runtime parameters. The XDNA 1 driver keeps at most **6 hardware contexts**, and stock mlir-aie compiles one program per (M, K, N), so a network with dozens of layer shapes reloads programs on almost every call. With `UGemm`, one program per tile config serves every shape. It reaches 0.78 ms per call when cycling shapes, and up to 1.58 TFLOPS.
- **`npu_detect.py`:** Frigate's YOLO-NAS-L with all 119 convolutions on the NPU: resident weights, a channel-first GEMM (so im2col needs no transposes), and direct XRT dispatch. Its detections are identical to onnxruntime's at 118 ms, of which about 45 ms is NPU work and about 55 ms is CPU work between layers. That's why a whole-network compiler (section 1) wins.

## 3. LLM on the NPU (`llm/`)

`npu_llm.py` runs Qwen2.5-0.5B-Instruct with all linear layers as bf16 GEMMs on the NPU, and the rest on the CPU. Next-token agreement with FP32 PyTorch is 51/60, versus 49/60 for PyTorch's own bf16. It runs at 2.5 tok/s, because each token needs about 150 NPU dispatches. `npu_server.py` serves the model on port 11435 with the Ollama API (`/api/chat`, `/api/generate`, ...) and the OpenAI API (`/v1/chat/completions`); `OLLAMA_HOST=127.0.0.1:11435 ollama run qwen2.5-npu:0.5b` works. It's a demonstration, not a practical LLM backend.

## 4. Radeon 760M for LLMs (`igpu/`)

- **Stock Ollama 0.34 drops integrated GPUs.** The runner logs "no usable GPU found", while `ollama ps` still shows "100% GPU", so it's running on the CPU.
- **`ollama-760m.sh`** forces the 760M through ROCm (`OLLAMA_IGPU_ENABLE=1`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`). It hangs the GPU on some models, and is slower than Vulkan.
- **llama.cpp with `-DGGML_VULKAN=ON`** (RADV; the 760M was `Vulkan1` here) is fast and stable. Measured with `bench_760m.sh`:

| Model | 760M prompt (pp512) | CPU prompt | 760M generation (tg128) | CPU generation |
|---|--:|--:|--:|--:|
| Qwen2.5 0.5B Q4 | 4299 tok/s | 746 tok/s | 129.8 tok/s | 127.5 tok/s |
| Qwen2.5 3B Q4 | 666 tok/s | 155 tok/s | 33.9 tok/s | 28.2 tok/s |
| Qwen2.5 7B Q4 | 306 tok/s | 70 tok/s | 15.6 tok/s | 12.7 tok/s |

The 760M processes prompts 4–6× faster than the CPU but generates only 1.0–1.2× faster, because both share the same DDR5 bandwidth. `bench_llm.py` compares the NPU, 760M (Vulkan and ROCm) and CPU, including package power from the amdgpu PPT sensor.

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `mmap(... len=67108864) failed (err=-11)` | memlock limit | section 0 |
| Every NPU call takes tens of ms | more than 6 distinct NPU programs in rotation | one resident program (`UGemm`) or a whole-network compiler |
| New shape's first run fails `EXEC_CMD ... err=-22` | instruction stream compiled after NPU work already ran in the process | compile every shape up front |
| `aie.dma_bd Size exceeds [1:64]`, or a GEMM timeout | DMA buffer-descriptor dimension limited to 64 repeats | keep `N / (4n) <= 64` |
| `static assertion ... % (4 * t)` | bf16 matmul tile | `n` must be a multiple of 16 |
| `iron` tensor data turns to garbage or segfaults | `.numpy()` is a view of freed buffer memory | `.copy()` results |
| GPU containers fail after an NVIDIA driver update: `open /usr/lib/libnvidia-...so.<old>` | stale `/etc/cdi/nvidia.yaml` | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |

## License

Apache-2.0, see `LICENSE`. For third-party code and what's deliberately not included (AMD Ryzen AI, YOLO-NAS weights), see `NOTICE`.
