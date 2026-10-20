# Persistent agent prefixes: how the pieces work together

The native patch keeps model calculations reusable; the client integration
makes the initial calculations stable. Neither changes an agent's execution
permissions. The same pattern applies to two flows of one model, such as Web
and Telegram, while each different model/quantization/runtime owns separate
states and identities.

## From the original flow to the protected base

```text
Original
  instructions + tools -> first user message + plugin bootstrap
                        -> ordinary model cache -> answer

First checkpoint preparation, with a mismatched fixture
  task-specific startup base != normal chat context
                        -> rebuild base -> process bootstrap + question

Current
  real client factory: instructions + tools + static plugin sections
                        -> verified Web or Telegram startup checkpoint
                        -> process only the uncached suffix -> answer
```

The initial mismatch was a preparation error: a Kanban task fixture differed
from a normal chat, and the Superpowers bootstrap was outside the pre-user
boundary. The engine correctly refused incompatible state. Capturing the real
Hermes construction path and putting the unchanged static bootstrap in native
system sections corrected this episode; it does not imply that all unrelated
native cache failures were client preparation problems.

## Three forms of retained work

| Storage | Contents | Survives | Limits |
| --- | --- | --- | --- |
| Active native context | Current KV, recurrent state and checkpoints | Compatible continued requests | Context geometry and model identity |
| Bounded RAM prompt pool | Saved conversation entries and their recovery states | Same-process client switches and compatible native sleep/wake | Eviction, available snapshots and configured RAM payload budget |
| Protected SSD bases | Two stable instruction/tool prefixes before any user message | Process shutdown and persistent storage remaining intact | Exact compatibility and per-base integrity; no complete user history backup |

APEX-inspired interior anchors improve recovery when a conversation changes in
the middle. They use the existing checkpoint count; they are not another RAM
cache layer. Protected SSD bases solve a separate problem: retaining the
frequently repeated start of an agent across a process restart.

Sleep still reloads model weights. A full process restart additionally loses
the ordinary RAM pool. The controller restores each compatible SSD base and
evaluates new content. Long conversations can require substantial new prefill
even when their startup prefix is protected. A full machine reboot was not
performed in the serving-stack restart benchmark.

## Matching and updates

1. Render/tokenize the actual tool-aware request with the native template and
   preserve its real provider/template options.
2. Determine a system/developer prefix that is also a token prefix of the full
   request, ending before its first user message. Exclude only the documented
   client variable tail; do not cache a greeting to fill a missing boundary.
3. Require exact prefix tokens, verified file digest and compatible runtime/model
   identity. If there is no exact base, fill an empty slot or select the closest
   occupied base for reconstruction. A genuine tie uses LRU and a stable ID.
4. The native selector compares usable active, RAM and SSD states using actual
   hybrid/recurrent checkpoint eligibility. Equal scores prefer active, then RAM,
   then disk. Text overlap alone does not prove recurrent state can be rewound.
5. Preserve the exact advanced active state in the same bounded RAM pool before
   replacing it or computing a changed base. If preservation cannot be completed,
   defer the base update and continue the ordinary inference path.
6. Commit only the selected pre-user artifact and metadata; keep the other base
   intact. Validate each independently after restart. Never persist a real user
   conversation as one of these protected bases.

The llama.cpp controller skips text-only preview for image/audio histories.
Those histories need real native multimodal chunks; a preview could fetch an
image twice and still choose the wrong cached state. Ordinary inference receives
the original request unchanged.

## Published components and pinned versions

| Component | Export and compatibility |
| --- | --- |
| llama.cpp | [Versioned native patches and hash manifests](../patches/llama.cpp/): current qualified base `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`; original `620d2225cc436ab0b001e3200e590e3b286ef9ab` preserved |
| llama.cpp host controller | [Portable modules and model-free tests](../integrations/llama_cpp/); caller owns serialization, identities and private native transport |
| Hermes / Superpowers | [Separate integration](../integrations/hermes/); exact upstream revisions and legacy hook differences documented there |
| Tabby / EXL3 | [Optional AGPL-3.0 integration](../integrations/tabby/); separate engine, dependency versions and source contract |

These exports do not contain production service files, private profiles, remote
access rules, credentials, prompts, saved token vectors or model weights. The
optional Tabby integration is separately licensed; it is not part of the MIT
llama.cpp patch. Source packaging does not update the active local runtimes.

## Tested laptop configuration

The original source patch was built for Linux/CUDA 13.1/SM86 and exercised with Ornith
1.5 35B A3B Q4_K_M on an RTX 3070 Ti Laptop GPU. The tested process used one
slot, 102,400 context tokens, Q8_0 K/V, batch 512, microbatch 128, a 3,712 MiB
expert cache, a 3,072 MiB RAM prompt pool, four checkpoints and 1,200-second
idle sleep. Expert banks remained pageable/mmap; immutable vision weights were
pinned; MTP/DFlash and optional whole-expert host registration were off.

The later 39da130f refresh was built with CUDA 13.1.1/SM86 and qualified with
the same model at **131,072 context tokens, Q8_0 K / Q5_1 V, 3,968 MiB experts**,
the same 3,072 MiB RAM pool and four checkpoints. Batch/microbatch remain 512/128,
one slot and speculation off. Vision remains selectively pinned; expert banks
remain pageable. The original numerical tables describe their original recipes;
changing the KV/context geometry requires preparing compatible SSD bases again.

An illustrative native launch, after applying the pinned patch:

```sh
mkdir -m 700 -p /private/model-prefix-cache
GGML_CUDA_NO_PINNED=1 ORNITH_MMPROJ_HOST_WEIGHTS=1 ORNITH_CHECKPOINT_GRID=1 \
  ./build/bin/llama-server \
  -m /models/model.gguf --mmproj /models/projector.gguf \
  --alias local-model --host 127.0.0.1 --port 8102 \
  -ngl 999 -cmoe --moe-cache-mib 3712 --load-mode mmap --fit off \
  -c 102400 -np 1 -t 12 -tb 12 -b 512 -ub 128 -fa on \
  -ctk q8_0 -ctv q8_0 --jinja \
  --cache-ram 3072 --ctx-checkpoints 4 --sleep-idle-seconds 1200 \
  --slot-save-path /private/model-prefix-cache/
```

Replace model/projector paths, GPU geometry and resource limits appropriately.
Leave `GGML_CUDA_REGISTER_HOST` unset to keep expert registration off; setting it
to the string `0` enables the presence-based path in this patch generation.
The native launch alone does not install the prefix controller or Hermes patches.
Follow each component's integration contract before exposing a public client API.

The protected committed SSD payload limit is 2 GiB per base / 4 GiB combined.
The two final laptop bases used approximately 1.01 GiB combined, separate from
the RAM cache payload budget. The 27B backend has different geometry, cache
budgets and quantization; do not compare their bytes per token as if identical.

## Client preparation and validation

The Web and Telegram factories have different homes, working directories and
tool availability. An isolated test SessionDB can accidentally change where
Hermes resolves its home. Bind the real profile explicitly, keep temporary
state/output separate, and verify the real home is the one used to assemble
SOUL, AGENTS and memories. SDK mock transports can capture the actual outgoing
request without sending inference or running tools.

Compare two newly constructed sessions with different questions. They must have
the same stable system/tool prefix; the bootstrap must occur once in the system
and zero times in user messages. The actual tested Web flow had 18 visible tools
and a protected base of 16,233 laptop-native tokens. Telegram had 23 visible tools
and 22,976 laptop-native tokens. Schema/prompt overhead and tokenization differ
on the 27B engine; these counts are observations, not hard-coded requirements.

New sessions adopt the revised static prompt. Existing sessions retain their
frozen prompts and use the compatibility hook where needed. This avoids silently
rewriting conversation history. Memory/tool/plugin updates can invalidate a base;
the next compatible request rebuilds the selected base rather than accepting a
similar but incorrect saved state.

## Visibility and alerts

Expose only aggregate/per-base readiness, token counts, bytes and bounded reason
codes. Report a newly missing, damaged or invalidated base once per changed
condition and clear the notice on recovery. A normal Web/Telegram selection
switch is not an invalidation. Do not infer a token-hit percentage merely from
process RSS or the existence of a saved file. A base can be ready on SSD while
the model itself remains unloaded.

## Measurements and validation limits

See [the return benchmarks](PREFIX_BENCHMARKS.md) and their linked numeric evidence.
Cold loading, native prefill, native first token, proxy first byte and whole-agent
turn duration are separate measurements. Missing measurements remain missing;
they are not reconstructed from perceived speed.

The experimental checkpoint grid has known strict numerical edge differences,
documented in the original cache-return evidence. Short controlled acceptance
does not establish full-context quality, universal floating-point parity or
instant concurrent switching of arbitrarily long conversations.
