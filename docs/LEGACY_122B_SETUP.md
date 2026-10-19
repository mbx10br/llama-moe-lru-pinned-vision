# 🛠️ Original 122B build and launch reference

These are the September 2026 commands for the original slot-based MoE cache + pinned-vision patch. They are retained as a reference for the published [unified patch](../moe_lru_cache_and_pinned_vision.patch), with its legacy `--moe-expert-cache` and `MTMD_PINNED_HOST` interfaces.

They do not install the later local sleep-cache or APEX-inspired integration. The original instructions did not freeze an upstream SHA; applying the old patch to today's master may require a rebase. Match the source and patch generation before building. The current architecture, test configurations and measured results are in the [main README](../README.md).

Expert host registration is an optional dedicated-host policy. The shared laptop keeps it disabled so Linux can reclaim and swap pageable expert memory. The historical memlock example below changes system policy; it is not required for the laptop profile.

---


### 1. System Setup (Linux Memlock for Zero-Copy DMA)
To allow CUDA to lock physical host RAM pages for zero-copy DMA without memory staging copies:
```bash
sudo bash -c 'cat <<EOF > /etc/security/limits.d/99-cuda-memlock.conf
* soft memlock unlimited
* hard memlock unlimited
EOF'
```
*(Log out and log back in, or run `ulimit -l unlimited` in your active shell).*

### 2. Apply the Unified Patch
Clone a clean copy of `llama.cpp` and apply [`moe_lru_cache_and_pinned_vision.patch`](../moe_lru_cache_and_pinned_vision.patch):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git apply /path/to/moe_lru_cache_and_pinned_vision.patch
```

### 3. Build with CUDA Support
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-server
```

### 4. Run Inference (Choose Your Profile)

#### 🔥 Profile 1: High-Throughput (Default - 15.16 tokens/s)
Best for general chat, coding, and fast reasoning with 138k context:
```bash
MTMD_PINNED_HOST=1 nice -n 19 ./build/bin/llama-cli \
  -m /path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --image /path/to/sample.jpg \
  -ngl 999 -cmoe -fa on \
  --moe-expert-cache 144 \
  --moe-expert-cache-inserts 3 \
  -c 138000 -ctk q5_1 -ctv q5_1 \
  -t 8 \
  -p "Describe what you see in this image." \
  -n 128 --temp 0.7 -st
```

#### 🌌 Profile 2: Ultra-Context (256k Tokens - 12.20 tokens/s)
Best for whole-repository analysis, massive document ingestion, and book-length context:
```bash
MTMD_PINNED_HOST=1 nice -n 19 ./build/bin/llama-cli \
  -m /path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  -ngl 999 -cmoe -fa on \
  --moe-expert-cache 110 \
  --moe-expert-cache-inserts 3 \
  -c 256000 -ctk q5_1 -ctv q5_1 \
  -t 8 \
  -p "Analyze this 200k token document..." \
  -n 128 --temp 0.7 -st
```

### 5. Run as an OpenAI-Compatible Server
```bash
MTMD_PINNED_HOST=1 nice -n 19 ./build/bin/llama-server \
  -m /path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  -ngl 999 -cmoe -fa on \
  --moe-expert-cache 144 \
  --moe-expert-cache-inserts 3 \
  -c 138000 -ctk q5_1 -ctv q5_1 \
  -t 8 \
  --host 0.0.0.0 --port 8080 \
  -a qwen3.8-122b \
  --jinja -np 1 --temp 0.7 --top-p 0.95
```

[Original published description](https://github.com/mbx10br/llama-moe-lru-pinned-vision/blob/8c54b4f2d9e1817026fd991f6e367f1f72a1c686/README.md) · [Technical walkthrough](../TECHNICAL_WALKTHROUGH.md) · [Back to the project](../README.md)
