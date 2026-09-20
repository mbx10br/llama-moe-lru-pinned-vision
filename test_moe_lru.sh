#!/usr/bin/env bash
set -e

# Benchmark runner for MoE Expert LRU Cache + Pinned Vision
# Default paths assume standard llama.cpp build directory
BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
MODEL="${MODEL_PATH:-./models/Qwen3.8-Flash-Next-Q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}"
MMPROJ="${MMPROJ_PATH:-./models/Qwen3.8-Flash-Next-Q2_0/mmproj-Qwen3.8-Flash-Next-BF16.gguf}"
IMAGE="${IMAGE_PATH:-./sample.jpg}"

CACHE_SLOTS=${MOE_CACHE_SLOTS:-144}
CACHE_INSERTS=${MOE_CACHE_INSERTS:-3}
CONTEXT_SIZE=${CONTEXT_SIZE:-138000}

echo "=== Running Qwen 122B with MoE Expert LRU Cache + Pinned Vision ==="
echo "Slots: $CACHE_SLOTS | Inserts/step: $CACHE_INSERTS | Context: $CONTEXT_SIZE"

CMD="MTMD_PINNED_HOST=1 nice -n 19 $BIN \
  -m \"$MODEL\" \
  --mmproj \"$MMPROJ\" \
  -ngl 999 -cmoe -fa on \
  --moe-expert-cache \"$CACHE_SLOTS\" \
  --moe-expert-cache-inserts \"$CACHE_INSERTS\" \
  -c \"$CONTEXT_SIZE\" \
  -ctk q5_1 -ctv q5_1 \
  -t 8 \
  -p 'Explain briefly how MoE partitioning with LRU cache on GPU works.' \
  -n 128 --temp 0.7 -st --simple-io"

if [ -f "$IMAGE" ]; then
  CMD="$CMD --image \"$IMAGE\""
fi

eval "$CMD"
