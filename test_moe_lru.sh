#!/usr/bin/env bash
set -e

# Benchmark runner for MoE Expert LRU Cache + Pinned Vision + Zero-Copy DMA
# Default paths assume standard llama.cpp build directory
BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
MODEL="${MODEL_PATH:-./models/Qwen3.8-Flash-Next-Q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}"
MMPROJ="${MMPROJ_PATH:-./models/Qwen3.8-Flash-Next-Q2_0/mmproj-Qwen3.8-Flash-Next-BF16.gguf}"
IMAGE="${IMAGE_PATH:-./sample.jpg}"

# Profile selection: 'throughput' (default, 144 slots / 138k ctx) or 'context' (110 slots / 256k ctx)
PROFILE="${PROFILE:-throughput}"

if [ "$PROFILE" = "context" ]; then
    CACHE_SLOTS=${MOE_CACHE_SLOTS:-110}
    CONTEXT_SIZE=${CONTEXT_SIZE:-256000}
    PROMPT_TEXT=${PROMPT:-"Explain how running 256k context in a 16GB GPU is achieved via MoE cache and pinned memory."}
    echo "=== Running Profile: ULTRA-CONTEXT (256k tokens, 110 MoE slots) ==="
else
    CACHE_SLOTS=${MOE_CACHE_SLOTS:-144}
    CONTEXT_SIZE=${CONTEXT_SIZE:-138000}
    PROMPT_TEXT=${PROMPT:-"Explain briefly how MoE partitioning with LRU cache on GPU works."}
    echo "=== Running Profile: HIGH-THROUGHPUT (138k tokens, 144 MoE slots @ ~15 t/s) ==="
fi

CACHE_INSERTS=${MOE_CACHE_INSERTS:-3}
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
  -p \"$PROMPT_TEXT\" \
  -n 128 --temp 0.7 -st --simple-io"

if [ -f "$IMAGE" ]; then
  CMD="$CMD --image \"$IMAGE\""
fi

eval "$CMD"
