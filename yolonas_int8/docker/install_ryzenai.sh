#!/usr/bin/env bash
# Runs inside ubuntu:24.04 with the extracted ryzen_ai-1.7.1.tgz at /pkg.
# Mirrors install_ryzen_ai.sh (runtime parts only) into /opt/ryzenai.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip ca-certificates wget gnupg \
    software-properties-common libgl1 libglib2.0-0 patch
add-apt-repository -y ppa:amd-team/xrt
apt-get update
apt-get install -y --no-install-recommends libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu
python3.12 -m venv /opt/ryzenai --copies
/opt/ryzenai/bin/pip install uv
ARGS="--prerelease=allow --find-links /pkg"
/opt/ryzenai/bin/uv pip install --python /opt/ryzenai/bin/python "ryzen-ai>=1.7.0.dev0,<1.8.0.dev0" \
    "device-essentials-phx>=1.5.0.dev0,<1.8.0.dev0" $ARGS
/opt/ryzenai/bin/uv pip uninstall --python /opt/ryzenai/bin/python onnxruntime onnxruntime-vitisai || true
/opt/ryzenai/bin/uv pip install --python /opt/ryzenai/bin/python onnxruntime-vitisai pillow $ARGS
SP=$(ls -d /opt/ryzenai/lib/python*/site-packages)
patch -u /opt/ryzenai/bin/activate -i "$SP/ryzen_ai/scripts/activate.patch"
tar -xf /pkg/voe-min.tar.gz -C "$SP"
rm -rf /var/lib/apt/lists/*
