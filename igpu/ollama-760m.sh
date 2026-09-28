#!/usr/bin/env bash
# Run a second Ollama that uses the Radeon 760M iGPU, on port 11436, sharing
# the system Ollama's models (read-only). The system Ollama (port 11434)
# currently runs everything on the CPU, because Ollama >= 0.3x drops
# integrated GPUs by default and ROCm 7.2 has no kernels for the 760M (gfx1103).
#   OLLAMA_IGPU_ENABLE=1              don't drop the iGPU
#   HSA_OVERRIDE_GFX_VERSION=11.0.0   run it with gfx1100 kernels
# WARNING: unreliable. Borrowed kernels hang the GPU ("HW Exception ... GPU Hang")
# on some models: llama3.2 and qwen2.5:7b with 11.0.0, qwen2.5:1.5b+ with 11.0.2.
# llama.cpp's Vulkan backend (README.md, "Radeon 760M") is faster and never hung.
# Use: OLLAMA_HOST=127.0.0.1:11436 ollama run qwen2.5:7b
export OLLAMA_IGPU_ENABLE=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11436}
export OLLAMA_MODELS=${OLLAMA_MODELS:-/var/lib/ollama}
export OLLAMA_NOPRUNE=1
export OLLAMA_FLASH_ATTENTION=1
exec ollama serve
