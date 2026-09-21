# Technical Walkthrough: MoE LRU Cache + Pinned Host Vision (CUDA_Host)

This document provides a line-by-line technical explanation of the codebase modifications that enable running a 122B parameter Mixture-of-Experts (MoE) model on a 16GB consumer GPU via a 3-tier memory offloading architecture (VRAM + RAM + NVMe).

---

## 🏗️ 1. Architecture Overview

```mermaid
flowchart TD
    subgraph GPU_VRAM ["GPU VRAM (16 GB)"]
        DenseLayers["Dense Layers & Attention Heads"]
        KVCache["KV Cache (138k tokens @ Q5_1 + Flash Attention 2)"]
        MoECache["MoE LRU Cache (144 slots per layer)"]
        DummySlot["Dummy Slot (slot n_slots: permanently zero)"]
    end

    subgraph Host_RAM ["Host System RAM (48 GB DDR4 @ 3200 MHz Dual-Channel, 4 sticks)"]
        AllExperts["512 Experts per layer (Authoritative Weights)"]
        MMPROJ["Vision Projector (mmproj BF16 - 866 MB)<br/><b>Allocated in CUDA_Host (Pinned)</b>"]
        CPUTable["Host Mapping Table (host_table)"]
    end

    subgraph NVMe_Storage ["NVMe Storage"]
        MMapFile["Shard 2: N-Gram Speculative Table (27 GB mmap)"]
    end

    Router["Top-K Router"] -->|Expert IDs| CheckCache{Is in Cache?}
    CheckCache -->|Yes (Hit)| MoECache
    CheckCache -->|No (Miss)| AllExperts

    MoECache -->|GPU Forward Pass| GPU_Out["GPU Cache Output"]
    AllExperts -->|CPU Forward Pass (on Miss)| CPU_Out["CPU Host Output"]
    GPU_Out --> Merge["ggml_add (Exact Mathematical Sum)"]
    CPU_Out --> Merge
    Merge --> FinalOut["Next Layer"]

    MMPROJ -.->|DMA / PCIe on-demand| DenseLayers
```

---

## 🔍 2. Pinned Vision Patch (`CUDA_Host` on Discrete GPUs)

### File 1: `ggml/src/ggml-cuda/ggml-cuda.cu`

In `llama.cpp`, the CUDA backend historically restricted `CUDA_Host` buffers (pinned host RAM allocated via `cudaMallocHost`) strictly to integrated GPUs (such as NVIDIA Tegra / Jetson or APUs):

```diff
@@ -5320,7 +5320,7 @@ static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const g
 static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
     ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
     const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
-    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
+    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || ggml_backend_buft_is_cuda_host(buft);
 }
```

#### Line-by-Line Breakdown:
* `static bool ggml_backend_cuda_device_supports_buft(...)`: Checks whether the CUDA device accepts a given buffer type (`buft`).
* `const bool integrated = ...`: Identifies if the GPU is an integrated SoC.
* `- (integrated && ggml_backend_buft_is_cuda_host(buft))`: The legacy code rejected `CUDA_Host` buffers on discrete GPUs (e.g. RTX 5060 Ti).
* `+ || ggml_backend_buft_is_cuda_host(buft)`: **The fix.** Discrete GPUs can now access and execute operations on pinned host memory over the PCIe bus via DMA.

---

### File 2: `tools/mtmd/clip.cpp`

Allocates the multimodal vision projector (`mmproj`, ~866 MB in BF16) in pinned host RAM when `MTMD_PINNED_HOST=1` is set:

```cpp
// tools/mtmd/clip.cpp (line 3558)
// Default buffer type (usually persistent VRAM on GPU when offload is active)
ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);

// Pinned Host RAM Interception
if (std::getenv("MTMD_PINNED_HOST") != nullptr) {
    // 1. Retrieve the device associated with the GPU backend (e.g. CUDA0)
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx_clip.backend);
    
    // 2. Query the device for its host pinned buffer type (cudaMallocHost)
    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    
    // 3. If supported, switch allocation to CUDA_Host
    if (host_buft) {
        buft = host_buft;
        LOG_INF("%s: CLIP weights allocated in pinned host memory (CUDA_Host)\n", __func__);
    }
}

// Actual tensor allocation in pinned host RAM
ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

---

## ⚡ 3. The MoE Expert LRU Cache Subsystem

### File 3: `src/llama-moecache.h`

Defines per-layer cache structures and companion tensors:

```cpp
#pragma once
#include <cstdint>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1; // Layer index (0 to n_layers - 1)

    int32_t n_slots = 0; // Number of GPU cache slots (e.g. 144)

    // Authoritative weights residing in host system RAM (e.g. 512 experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // GPU companion tensors for cached experts:
    // Shape: [ne0, ne1, n_slots + 1]
    // The extra slot (index n_slots) is permanently filled with zeros (dummy slot)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // Mapping tables: Expert ID -> Cache Slot
    // dev_table: resides in GPU VRAM (read by ggml_get_rows)
    // host_table: resides in Host RAM (read by CPU kernel to skip cached experts)
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;
};

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts);
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);
void llama_moe_cache_step();
```

---

### File 4: `src/llama-moecache.cpp`

#### A. LRU Bookkeeping & Data Structures
```cpp
struct layer_state {
    llama_moe_cache_layer pub;

    std::vector<int32_t>  slot_expert;    // slot -> expert_id (-1 if empty)
    std::vector<int32_t>  expert_slot;    // expert_id -> slot (-1 if uncached)
    std::vector<uint64_t> slot_last_use;  // slot -> Lamport logical clock of last hit
    std::vector<int32_t>  pending;        // cache misses waiting for upload

    std::vector<bool>     slot_in_flight; // upload currently in progress for slot

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};
```

#### B. Asynchronous Background Upload Worker (Zero Decode Stutter)
Weight uploads occur off the main decode thread via an asynchronous worker:

```cpp
mc->worker = std::thread([mc]() {
    for (;;) {
        upload_job j;
        {
            std::unique_lock<std::mutex> lk(mc->wmtx);
            mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
            if (mc->stop) return;
            j = mc->todo.front();
            mc->todo.pop_front();
        }
        
        auto & ls = mc->layers[j.layer_idx];
        // Stream UP, GATE, and DOWN weight matrices to GPU slot via PCIe DMA
        upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
        upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
        upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
        
        {
            std::lock_guard<std::mutex> lk(mc->wmtx);
            j.done = true;
            mc->done.push_back(j);
        }
    }
});
```

#### C. Synchronization & Eviction: `llama_moe_cache_step()`
Invoked strictly **between decode steps** when no compute graph is active:

1. **Publish Completed Uploads**: Updates `dev_table` and `host_table` so subsequent steps route to the newly resident expert in VRAM.
2. **LRU Victim Selection**: Selects the slot with the lowest `slot_last_use` timestamp.
3. **Eviction**: Points the evicted expert's table entries to the dummy zero slot (`n_slots`).
4. **Queue New Uploads**: Submits new upload tasks to the worker thread adhering to `max_inserts` (e.g. 3 per step).

---

## 🧮 4. Exact Mathematical Sum Principle (`ggml_add`)

How does the runtime split computation between GPU and CPU without numerical divergence?

$$\text{Final Output} = \text{Output}_{\text{GPU}} + \text{Output}_{\text{CPU}}$$

1. **On GPU (Cache Chain)**:
   * Cached experts compute matrix multiplication using their assigned slot.
   * Uncached experts map to slot `n_slots` (which contains **all zeros**). Hence, $\text{Output}_{\text{GPU}} = 0$.

2. **On CPU (Host Chain)**:
   * The CPU kernel reads `src[3]` (`host_table`):
     ```c
     if (moe_tbl && moe_tbl[i02] != moe_dummy) {
         // Expert is cached on GPU! Skip computation and zero destination row:
         memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
         continue;
     }
     ```
   * Cached experts produce $\text{Output}_{\text{CPU}} = 0$.
   * Uncached experts compute normally on host RAM.

3. **In `llama-graph.cpp`**:
   ```cpp
   experts = ggml_add(ctx0, experts, down_g);
   ```
   Since exactly one side computes the real value and the other produces 0:
   $$\text{Real Value} + 0 = \text{Real Value}$$
   The result is **exact to the bit with zero precision loss!**
