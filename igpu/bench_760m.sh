#!/usr/bin/env bash
# llama.cpp benchmark of the Radeon 760M (Vulkan/RADV) vs the CPU, on the
# GGUF files the system Ollama already downloaded. Same llama.cpp build for
# both, so it compares devices, not software stacks.
#
#   pp512 = prompt processing (prefill), tokens/s for a 512-token prompt
#   tg128 = text generation (decode), tokens/s while generating 128 tokens
#
# Usage: ./bench_760m.sh [ollama-model ...]     (default: the Qwen2.5 ladder)
# Build llama.cpp first: see README.md ("Radeon 760M").
set -euo pipefail
cd "$(dirname "$0")"

BENCH=../llama.cpp/build-vulkan/bin/llama-bench
MODELS_DIR=${OLLAMA_MODELS:-/var/lib/ollama}
DEVICE=${DEVICE:-Vulkan1}  # the 760M; Vulkan0 is the RTX 3060 Ti
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(qwen2.5:0.5b-instruct-fp16 qwen2.5:0.5b qwen2.5:1.5b
                                  qwen2.5:3b llama3.2:latest qwen2.5:7b)

# Ollama model name -> path of its GGUF blob, via the manifest.
blob() {
  local name=${1%%:*} tag=${1#*:}
  [ "$tag" = "$1" ] && tag=latest
  local manifest=$MODELS_DIR/manifests/registry.ollama.ai/library/$name/$tag
  local digest
  digest=$(python3 -c 'import json,sys
m = json.load(open(sys.argv[1]))
print(next(l["digest"] for l in m["layers"] if l["mediaType"].endswith(".model")))' "$manifest")
  echo "$MODELS_DIR/blobs/${digest/:/-}"
}

args=()
for m in "${models[@]}"; do
  args+=(-m "$(blob "$m")")
  echo "$m -> $(blob "$m")" >&2
done

run() {  # $1 = label, rest = llama-bench device args
  local label=$1; shift
  echo -e "\n### $label"
  "$BENCH" "${args[@]}" "$@" -p 512 -n 128 -r 3 -fa 1 -o md 2>/dev/null
}

run "Radeon 760M (Vulkan, $DEVICE)" -dev "$DEVICE" -ngl 99
run "CPU (Ryzen 5 8600G, 6 threads)" -dev none -ngl 0 -t 6
