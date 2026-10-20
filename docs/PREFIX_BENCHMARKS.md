# Persistent-prefix return benchmarks · 2026-10-20

Protected pre-user checkpoints survived a complete restart of both inference stacks: **4/4 SSD bases preserved byte for byte, with no warmup or inference request sent during restart**. Actual user calls then reused prior work. Returning from SSD still requires loading the model when its weights are absent.

[Machine-readable measurements](../benchmarks/persistent-prefix-return-20261020.json) include every retained completion in the fixed observation window, sources identified by SHA-256, cancellations and measurement limits. Raw conversations and logs are not published.

## What was measured

The laptop runs a 35B A3B GGUF Q4_K_M model through the local llama.cpp adaptation: 102,400-token context, one slot, Q8_0 K/V, batch 512, microbatch 128, 3,712 MiB expert cache, 3,072 MiB RAM prompt cache and four recurrent checkpoints. Pinned vision remains enabled; these measurements are text only.

The desktop runs a 27B EXL3 SC-3.00-H4 model through TabbyAPI/ExLlamaV3 with a DFlash2 Q4 draft. Its user workload differs from the laptop workload. The tables describe return behavior; they do not rank model quality or establish a matched speed comparison.

Three measurements must stay separate:

- **Controlled TTFT:** client wall time to the first nonempty text/reasoning SSE token.
- **Native first token:** the desktop engine's reported first-token timing.
- **Proxy first byte:** the proxy's `ttft_s` metric, which can include admission overhead and can arrive as stream framing before visible text.

Generation tok/s measures generation, while total request time also includes new input, output length and other overhead. A complete Hermes task can contain multiple requests and tools.

## Before restart: ten controlled requests

Four requests created/replaced a protected prefix; six reused prepared state. These short-response tests were performed **before** the full restart. Warm interface switches specifically reported a RAM source on the laptop.

| Environment / interface | Phase | First nonempty SSE token · s | Request wall · s | Cached / input tokens |
| :--- | :--- | ---: | ---: | ---: |
| 35B laptop · web | prepare and first response | 276.307 | 278.146 | 16,233 / 16,345 |
| 35B laptop · web | warm repeat | 1.041 | 1.785 | 16,325 / 16,345 |
| 35B laptop · telegram | prepare and first response | 256.795 | 260.762 | 22,976 / 23,059 |
| 35B laptop · telegram | warm repeat | 1.493 | 3.457 | 23,039 / 23,059 |
| 35B laptop · web | warm ram switch | 1.213 | 3.048 | 16,341 / 16,345 |
| 35B laptop · telegram | warm ram switch | 0.981 | 1.816 | 23,055 / 23,059 |
| 27B desktop · telegram | prepare and first response | 23.894 | 23.955 | 22,784 / 23,081 |
| 27B desktop · web | prepare and first response | 16.737 | 17.177 | 16,128 / 16,378 |
| 27B desktop · telegram | warm repeat | 0.509 | 0.679 | 23,040 / 23,081 |
| 27B desktop · web | warm repeat | 0.875 | 1.046 | 16,128 / 16,378 |

The first preparation requests include loading and/or prefix preparation in client wall time. Their final native-call cache counters can already include the newly prepared prefix. Therefore these rows must not be presented as a clean preservation-OFF/ON speedup or as ten SSD-return measurements.

## Complete restart: SSD persistence, no artificial warmup

| Stack | What was restarted | Protected bases preserved | State before the first user call |
| :--- | :--- | :--- | :--- |
| 35B laptop | Hermes Web, supervisor and request bridges/proxies; prior native process ended | **2/2**, binary and metadata hashes unchanged | Lazy cold: no native model process or weights loaded |
| 27B desktop | Hermes Telegram microVM, broker, Tabby and text proxy; all process IDs changed | **2/2**, checkpoint bytes/hashes unchanged | Backend startup loaded weights and restored SSD bases to RAM |

The desktop's history and configuration checks passed. No test question, inference request, Telegram message or warmup was sent during restart. Startup modes differ: the desktop paid loading cost before the first user call; the laptop paid it inside that call.

## After restart: real laptop calls

The first user call reused **16,233 / 16,340 input tokens (99.35%)**, evaluating only **107 new tokens** in the final request. Its API wall time was **76.5 s**; the corresponding first Hermes turn took **77.7 s**.

The supervisor recorded two `disk_imported` events, then a valid protected prefix, **55.771 s after the request started**. That interval combines loading, readiness and prefix handling. It does **not** isolate model loading or SSD read time. Native diagnostics are intentionally not retained by this supervisor, so native TTFT and generation tok/s are unavailable.

| Completed API call | Cached / input tokens | New input | Output tokens | API wall · s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16,233 / 16,340 | 107 | 99 | 76.5 |
| 2 | 16,336 / 16,528 | 192 | 403 | 25.3 |
| 3 | 16,524 / 16,769 | 245 | 161 | 12.1 |
| 4 | 16,765 / 18,800 | 2,035 | 367 | 56.4 |
| 5 | 18,796 / 21,305 | 2,509 | 282 | 67.4 |
| 6 | 21,301 / 26,511 | 5,210 | 354 | 89.8 |
| 7 | 26,507 / 27,382 | 875 | 438 | 39.7 |
| 8 | 27,378 / 31,992 | 4,614 | 212 | 74.7 |

These are growing, real requests rather than repeated one-word fixtures. For example, evaluating 5,210 new input tokens and producing 354 output tokens took 89.8 s despite reusing the preceding 21,301 tokens. Reuse avoids replaying all history; it does not make a new suffix or generation free.

The lifecycle window contains 9 completed requests, 1 cancellation and 1 in-progress request. 8 completions have matching retained API token/latency records. Unmatched lifecycle completions and unfinished work are not assigned invented metrics.

## After restart: real desktop calls

The first completed streamed user request reused **23,040 / 23,273 input tokens (99.00%)**, with **233 new tokens**. It reported **0.97 s native first token**, **2.710 s proxy first byte**, **4.133 s proxy service time** and **24.1 generation tok/s**. The model and protected SSD state were already loaded during backend startup.

The table retains all 18 completed streamed requests in the snapshot. The JSON also includes 6 nonstreamed completions, which may include auxiliary work; they are not silently mixed into streamed TTFT results. Row numbers refer to completion order in the full JSON.

| Row | Cached / input tokens | New input | Native first token · s | Proxy first byte · s | Native gen · tok/s | Proxy service · s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23,040 / 23,273 | 233 | 0.97 | 2.71 | 24.1 | 4.133 |
| 3 | 23,040 / 23,425 | 385 | 4.31 | 5.762 | 65.8 | 26.884 |
| 4 | 23,296 / 25,018 | 1,722 | 2.36 | 4.129 | 61.8 | 4.878 |
| 5 | 24,832 / 27,275 | 2,443 | 3.14 | 4.323 | 57.7 | 23.52 |
| 6 | 27,136 / 27,984 | 848 | 1.41 | 2.598 | 57.7 | 11.838 |
| 7 | 27,904 / 29,041 | 1,137 | 1.74 | 2.939 | 61.3 | 8.323 |
| 8 | 28,928 / 30,516 | 1,588 | 2.2 | 3.411 | 56.7 | 8.843 |
| 9 | 30,464 / 32,249 | 1,785 | 2.65 | 3.878 | 49.2 | 15.394 |
| 10 | 32,000 / 33,659 | 1,659 | 2.41 | 3.635 | 48.5 | 58.1 |
| 13 | 34,048 / 34,432 | 384 | 1.03 | 1.654 | 67.3 | 6.511 |
| 14 | 34,304 / 35,225 | 921 | 1.73 | 3.005 | 53.3 | 40.282 |
| 17 | 35,072 / 37,551 | 2,479 | 3.59 | 4.826 | 50.1 | 37.577 |
| 18 | 37,376 / 39,964 | 2,588 | 3.61 | 4.876 | 58.9 | 61.26 |
| 19 | 48,128 / 48,390 | 262 | 0.76 | 2.243 | 93.7 | 3.509 |
| 20 | 48,384 / 71,504 | 23,120 | 36.5 | 32.72 | 52.6 | 61.453 |
| 22 | 71,424 / 74,461 | 3,037 | 5.87 | 7.414 | 42.5 | 34.793 |
| 23 | 74,496 / 74,767 | 271 | 0.84 | 2.479 | 55.9 | 5.173 |
| 24 | 74,752 / 75,331 | 579 | 1.58 | 3.011 | 61.2 | 12.32 |

The long-tail case with **71,504 input tokens** reused **48,384** and evaluated **23,120 new tokens**; its native first-token time was **36.5 s**, and proxy service time **61.453 s**. It remains in the table. Proxy first-byte timing can be earlier than native first-token timing because the clocks observe different events, including stream framing.

Logs show protected-state restoration during startup and later `exact-prefix-match` / `restored-to-ram` events. No `rebuilding` transition was observed after the restart boundary in this window. One later `partially-ready` line is retained in the evidence; these logs do not establish its cause or prove both protected entries remained ready continuously.

## Interpreting the result

The operator reported that continuation felt immediate after model loading. This is subjective feedback, **not** a separately measured post-load latency.

The measured result is narrower: compatible calculations survived a complete process restart on SSD, were restored, and were reused by actual requests. Warm RAM return, SSD restoration, weight loading, new input processing, generation and whole-task time are distinct costs. Changing histories, cache eviction, new suffixes and failed compatibility checks can still require substantial work.

The snapshot is passive and ends with ongoing user activity. No new workload was run to improve the numbers, and no quality conclusion is drawn from this timing sample. Hardware/model details and the earlier LRU, vision, sleep and APEX experiments remain in the [main project documentation](../README.md).
