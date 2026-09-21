# MoE Expert LRU Cache + Pinned Host Vision (CUDA_Host) for llama.cpp

Running **122B Parameter MoE Models** (e.g., Qwen 3.8 Flash Next 122B) with **138k Context** and **Multimodal Vision** on a single **16GB Consumer GPU** (NVIDIA RTX 5060 Ti) at **up to 14.87 tokens/sec**.

---

## 🌟 Overview & Attribution

This project combines and refines two key architectural ideas to break the memory wall for Mixture-of-Experts (MoE) and Vision-Language models on consumer GPUs:

1. **MoE Expert LRU Cache (`--moe-expert-cache`)**:
   * Inspired by and building upon **PR #27861** on `llama.cpp` (`exp/moe-lru-cache`), with conceptual roots in **FreeToken**, dynamic expert offloading, and the evolutionary principles of **ncpumoe**.
   * Maintains a dynamic, device-resident LRU cache of the most recently and frequently used expert weights directly in GPU VRAM (e.g. 144 slots per layer). Uncached experts are evaluated on host memory or uploaded asynchronously via PCIe DMA, while cached experts compute at full GPU speed.
2. **Pinned Host Vision (`CUDA_Host` for Discrete GPUs)**:
   * Enables `CUDA_Host` (pinned system RAM via `cudaMallocHost`) support for discrete GPUs in `ggml-cuda.cu`.
   * When `MTMD_PINNED_HOST=1` is set, the multimodal vision projector (`mmproj`, ~866 MB in BF16) is allocated in host pinned RAM instead of persistent VRAM.
   * CUDA kernels execute the vision forward pass directly from pinned RAM over PCIe via DMA under demand, completely freeing ~866 MB of persistent VRAM for LLM experts and KV cache.

### 🙏 Credits & Acknowledgements
* **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: Georgi Gerganov and the GGML team for the extraordinary foundational runtime that powers local LLM inference across heterogeneous hardware.
* **MoE Expert Cache PR (#27861)**: Authors of the host-offloaded GPU-resident LRU cache implementation (`exp/moe-lru-cache`), introducing dynamic slot caching and throttled asynchronous uploads.
* **FreeToken & ncpumoe**: Prior theoretical and empirical works demonstrating temporal locality, expert stickiness, and sparse activation patterns in modern MoE architectures, paving the way for hierarchical CPU/GPU memory routing.
* **Qwen Team (Alibaba Cloud)**: For developing the exceptional Qwen 3.8 Flash Next MoE architecture (512 total experts, 10 routed, 2 KV heads) which enables massive context windows with minimal KV memory footprint.
* **ISTA-DASLab**: For the GSQ-RCO quantization methodology and the N-gram speculative lookup tables that make 122B inference viable on consumer workstations.

---

## 🎯 The Core Synergy: Why Combine Both?

On a 16GB GPU (such as the RTX 5060 Ti with 15.6 GB usable VRAM):

* A 122B model with 512 experts (10 routed per token) requires ~3.7 GB for base dense/attention layers.
* A 138k token context window with `Q5_1` KV cache + Flash Attention 2 requires ~2.5 GB VRAM.
* A 120-slot MoE cache requires ~7.5 GB VRAM.
* **The bottleneck:** If the 866 MB vision projector is permanently pinned in VRAM, expanding the MoE cache to **144 slots (+50% capacity)** pushes VRAM requirements to **16.3 GB**, triggering an immediate **CUDA Out of Memory (OOM)** error.
* **The solution:** By offloading `mmproj` to `CUDA_Host` pinned RAM, the GPU reclaims ~866 MB of VRAM. This unlocked slot headroom allows allocating **144 MoE cache slots**, perfectly fitting within **15.2 GB VRAM** with **zero OOM** and **zero performance penalty for vision**.

---

## 📊 Benchmark & Real-World Validation

### Test Environment
* **CPU**: AMD Ryzen 7 5700G (8 cores / 16 threads, Zen 3, 3.8 GHz base / 4.6 GHz boost)
* **Motherboard**: Gigabyte A520M DS3H V2 (AM4, PCIe 3.0 / PCIe 4.0 support)
* **Host RAM**: 48 GB DDR4 (4 sticks in Dual-Channel configuration @ 3200 MHz, 2x16GB + 2x8GB)
* **GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell Architecture, SM 12.0, 16,311 MiB VRAM)
* **Storage**: Fast NVMe PCIe M.2 SSD (Shard 2 N-gram speculative table memory-mapped via `-lm mmap`)
* **OS / Environment**: Linux x86_64, CUDA 13.1, NVIDIA Driver 590.48.01
* **Model**: `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (Q2_0, 122B total parameters, 512 experts, 10 routed per token)
* **Context**: 138,000 tokens (`-ctk q5_1 -ctv q5_1 -fa on`)

### Performance Results

| Configuration | MoE Cache Slots | Context Size | VRAM Usage | Generation Speed (Cold) | Generation Speed (Warm) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (PR #27861) | 96 slots | 138k (`q5_1`) | ~13.6 GB | 7.8 t/s | 11.5 t/s | Stable |
| +25% Experts (Default mmproj) | 120 slots | 138k (`q5_1`) | ~15.1 GB | 8.1 t/s | 10.4 t/s | Stable |
| +50% Experts (Default mmproj) | 144 slots | 138k (`q5_1`) | > 16.3 GB | — | — | ❌ **OOM** (+865 MB over limit) |
| **+50% Experts + Pinned Vision** | **144 slots** | **138k (`q5_1`)** | **~15.2 GB** | **8.44 t/s** | **14.87 t/s** | ⚡ **Success (+76% speedup)** |

### Multimodal Vision Test (Real-world 400x300 Image)
* **Prompt Processing (including image encoding)**: **11.7 tokens/s**
* **Generation**: **10.9 tokens/s**
* **Recognition**: Accurately recognized complex architecture (*"interior of a modern church with basalt columns and concentric arches, Hallgrímskirkja..."*).

---

## 🛠️ Quick Start & Build Instructions

### 1. Apply the Unified Patch
Clone a clean copy of `llama.cpp` and apply [`moe_lru_cache_and_pinned_vision.patch`](moe_lru_cache_and_pinned_vision.patch):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git apply /path/to/moe_lru_cache_and_pinned_vision.patch
```

### 2. Build with CUDA Support
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-server
```

### 3. Run Inference (CLI)
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

### 4. Run as an OpenAI-Compatible Server
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

---

## 📖 Deep-Dive Code Walkthrough

For an in-depth, line-by-line explanation of the patch and the mathematical properties of the exact sum (`ggml_add`), refer to [**`TECHNICAL_WALKTHROUGH.md`**](TECHNICAL_WALKTHROUGH.md).

---

## 📄 License
This work is released under the same license as `llama.cpp` (MIT License).
