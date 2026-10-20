# Upstream refresh and cache qualification · 2026-10-20

The current laptop runtime was rebuilt from official llama.cpp revision
`39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` with this project's selective pinned vision,
sleep-cache preservation, APEX-inspired anchors and private dual-prefix restore.
Upstream SLRU and checkpoint invalidation remain intact. The previous native
patch and all historical benchmarks remain available.

[Source bundle and manifests](../patches/llama.cpp/) ·
[Aggregate evidence](../benchmarks/upstream-refresh-20261020.json) ·
[Historical return benchmarks](PREFIX_BENCHMARKS.md)

## Configuration

| Setting | Qualified recipe |
| --- | --- |
| Hardware / build | RTX 3070 Ti Laptop 8 GiB; Linux; CUDA 13.1.1, SM86, Release |
| Model | Ornith 1.5 35B A3B Q4_K_M |
| Context / parallel slots | 131,072 / 1 |
| K / V cache | Q8_0 / Q5_1 |
| Expert VRAM budget | 3,968 MiB; upstream 50/50 SLRU |
| Conversation RAM budget / checkpoints | 3,072 MiB / four checkpoints with the APEX-inspired grid |
| Batch / microbatch | 512 / 128 |
| Host memory | Pageable/mmap expert banks; selectively pinned vision with CUDA compute |
| Speculation / normal idle timeout | MTP and DFlash off / 1,200 seconds |
| Protected startup states | Two independent, validated pre-user SSD bases |

The production profile stayed byte-identical. Replacing the binary/libraries
changed runtime identity, so both protected bases were rebuilt once. The new
files then survived native restart and public requests without being rewritten.
Whole-expert host registration remains unset, preserving Linux reclaim/swap/zram
flexibility on a shared laptop.

## Generation and local L3

Both builds used the same model, context, KV geometry, expert budget and greedy
number-list prompt. Each ran a 128-token warmup followed by 512 generated tokens.

| Build | Generation · tok/s | Full 512-token request · s |
| --- | ---: | ---: |
| Previous local build | 41.75 | 12.704 |
| Refreshed build | **44.47** | **11.924** |

The observed generation increase was **6.51%**
in this single ordered pair. It is a short throughput check, not a statistical
speed claim or a prediction for every workload.

The unchanged local Bonsai/Ornith **L3 task passed 23/23 checks** in one attempt,
seed 42, with creator sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0,
repeat penalty 1 and unrestricted thinking within a 32,768-token output limit.
It completed normally after **12,161 output tokens**
and **507.500 s** of API wall time; native generation was
**25.03 tok/s**. Output counts include reasoning.
Generated code was graded only in the existing isolated container. This local
battery does not establish a universal model-quality ranking.

## Cache, sleep, vision and actual client payloads

Synthetic cache fixtures passed cold, edited-history and repeated-input checks.
Both builds produced the same short output token IDs in those fixtures. A small
image correctly returned `Red`, using the pinned-vision/CUDA path.

The synthetic sleep case retained **three states / 938.851 MiB** in RAM and
reduced process VRAM to **256 MiB**. It woke in the same process, and both
protected families restored again after a fresh process start. These are small
fixtures; 131,072 is configured capacity, not a filled-context stress test.

The two real SDK-captured client prefixes were then rebuilt: **16,686 tokens for
Web** and **22,976 for Telegram**. Seven private cases covered preparation,
alternation, sleep/wake and fresh native restart. The accelerated sleep check
used a temporary **5-second** idle timeout; the permanent setting stayed at
**1,200 seconds**. Short private-stage timings can therefore include an extra
weight reload and must not be described as steady-state hot response time.

Finally, two short requests with the captured client formats traversed the real
public FIFO and supervisor, using the public compatibility alias:

| Client format | Cached / input tokens | Full request wall · s | Starting condition |
| --- | ---: | ---: | --- |
| Web | 16,686 / 16,794 | 51.902 | Weights loaded inside this request |
| Telegram | 22,976 / 23,055 | 3.555 | Same loaded backend, switched prefix |

Both returned the expected short response, preserved the SSD file hashes and
left **2/2 bases ready**. No real tools, production actions or Telegram messages
were executed. These are direct API tests with actual captured client payloads,
not complete live Hermes turns. No streaming TTFT was measured in this refresh.

## Source validation and limits

- CUDA build passed; 41 native component tests / 172 assertions and 15 selection-policy assertions passed.
- The portable llama.cpp controller passed all 12 CPU tests. All 11 patched/new native files reproduce byte for byte from the pinned source; the historical ten-file generation still verifies.
- Independent optional Tabby qualification is documented in its [own AGPL integration](../integrations/tabby/#validation-and-limits): 43 portable CPU tests, 19 separate native-AST cases and eight GPU cases. These are not added to the llama.cpp score.
- Initial test-adapter errors are retained in the evidence history. The canonical L3 prompt/grader and production public validator were not changed to accommodate those errors.
- Ordinary RAM conversations remain bounded and do not survive a full process restart. Protected SSD bases contain only the stable pre-user prefix, not a saved user greeting or full chat history.
- Full-context quality, universal floating-point parity and instantaneous long-conversation concurrency are not established. Earlier strict grid numerical differences remain documented.
