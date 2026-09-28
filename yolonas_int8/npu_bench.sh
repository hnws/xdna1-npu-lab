#!/usr/bin/env bash
# Benchmark an XINT8 model on the XDNA 1 NPU with AMD's Ryzen AI 1.7.1 runtime.
#   ./npu_bench.sh [model.onnx]
# First run builds the npu-ryzenai image from ../ryzen_ai-1.7.1.tgz (extracted
# to ryzenai_pkg/), mirroring the runtime part of AMD's install_ryzen_ai.sh.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-../models/yolo_nas_l_320_xint8.onnx}

if ! docker image inspect npu-ryzenai >/dev/null 2>&1; then
  [ -d ryzenai_pkg ] || { mkdir -p ryzenai_pkg && tar -xzf ../ryzen_ai-1.7.1.tgz -C ryzenai_pkg; }
  docker rm -f ryzenai-build >/dev/null 2>&1 || true
  docker run --name ryzenai-build -v "$PWD/ryzenai_pkg:/pkg:ro" \
    -v "$PWD/docker/install_ryzenai.sh:/install.sh:ro" ubuntu:24.04 bash /install.sh
  docker commit ryzenai-build npu-ryzenai && docker rm ryzenai-build
fi

VOE=/opt/ryzenai/lib/python3.12/site-packages/voe-4.0-linux_x86_64
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --device /dev/accel/accel0 --ulimit memlock=-1:-1 \
  -e XILINX_XRT=/usr -e XLNX_VART_FIRMWARE=$VOE/xclbins/phoenix/4x4.xclbin \
  -e VITISAI_EP_JSON_CONFIG=$VOE/vaip_config.json \
  -v "$PWD/..:/w" -w /w/yolonas_int8 \
  npu-ryzenai bash -c "source /opt/ryzenai/bin/activate && python npu_bench.py '$MODEL'"
