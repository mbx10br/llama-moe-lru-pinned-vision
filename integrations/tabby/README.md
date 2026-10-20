# TabbyAPI / EXL3: reusable context and two protected prefix bases

This is the portable source export of the tested Tabby integration. It adds
bounded native context persistence and **two independent pre-user checkpoints on
SSD**, outside the evictable context cache. The normal native request path still
decides which longer conversation state to reuse.

```text
Exact rendered system + tools, before any user message
               │
          compare BOTH token vectors
          ┌────┴─────────────────────┐
       exact                    neither exact
       reuse              rebuild the closest entry
          │                preserve the other entry
          └──────────────┬───────────┘
                 private SSD bases
                 base-1    base-2
                      │
        import both on model load / missing RAM copy
                      │
      native bounded tmpfs tier → CPU cache → GPU context
```

It contains **three source modules, a five-file Tabby hook patch, and 43 CPU
tests**. It does not include model weights, checkpoints, client prompts, client
tool catalogs, credentials, bots, network rules, service units or a GPU scheduler.
No additional TCP administrative endpoint is added.

## Exact compatibility and provenance

| Component | Tested source |
|---|---|
| TabbyAPI | [`be74bf0a00bcb3a518e6feb7606f150c189be637`](https://github.com/theroyallab/tabbyAPI/commit/be74bf0a00bcb3a518e6feb7606f150c189be637) |
| EXL3 | [Public `v1.6.3`](https://github.com/turboderp-org/exllamav3/tree/v1.6.3), installed as 1.6.3 |
| Native cache API | `DiskPageCache`, CPU page cache, paired recurrent state and `flush_cache_to_disk(wait=True)` from that public release |
| CPU export tests | Standard-library-only tests; no model, CUDA or EXL3 import required |
| Host-dependent control | Linux private Unix socket with `SO_PEERCRED`; a dedicated bounded tmpfs owned by the backend user |

All 12 checked generator/recurrent source files in the tested runtime were
byte-identical to the public EXL3 tag. A private EXL3 fork is **not** needed for
this export. Other releases and other architectures have not been qualified.
The protected-prefill path specifically requires a paired recurrent checkpoint;
it is not presented as a tested general cache for every Transformer.

[PROVENANCE.json](PROVENANCE.json) records baseline and tested hashes. The hook
patch reproduces the five deployed files byte for byte. The three exported
modules differ from their tested versions only by their license/date comment
headers. No running backend was modified to create this export.

## Exact-anchor repair (Base 1, 2026-10-20)

A static prefix can end between the native 4,096-token checkpoint intervals.
After a longer conversation branch, the ordinary native flush may prune that
interior recurrent checkpoint even though the exact KV page chain remains.
Previously, protected publication then failed with
`paired-recurrent-state-missing`; the existing SSD generation remained selected.
The defect was reproduced without memory pressure.

The protected transaction now keeps its prefill job and quiescence ownership
until publication. It performs the ordinary flush first, then validates the
exact live target/draft stash and force-writes **one missing anchor**, drains the
native writer, validates its full paired layout, and commits the SSD bundle.
There is no second global flush/prune between the anchor write and publication.
Normal native spacing, pruning and cache budgets remain unchanged.

Publication uses a fresh immutable generation even for identical inputs. Files
and directories are synced before the atomic selection pointer changes; a
pre-commit copy/writer/pointer failure retains the previous selected generation.
Checks cover runtime fingerprint, absolute positions, recurrent sides/layers,
checkpoint sizes and truncated tensor payloads. A dense draft without recurrent
layers is supported by validating the sides actually present in the runtime.

**Capacity limit:** the extra post-flush anchor write is refused if pending I/O
or insufficient native-tier space prevents safe admission. It does not request
additional eviction to make room. This is a safe failure, not a general repair
for EXL3 eviction progress; ordinary global flush/restore still use native
behavior. Real asynchronous writer races, tensor-parallel ranks and filesystem
power-loss behavior have not been qualified.

## What each piece does

| Piece | Purpose |
|---|---|
| `overlay/common/cache_persistence.py` | Connects the native CPU/tmpfs tier, validates bounded private storage, and offers a local flush/quiesce/status handshake. Graceful generator close occurs before GPU tensors are unloaded. |
| `overlay/common/permanent_prefix.py` | Captures an exact pre-user prefix, publishes immutable page chains plus their paired state, restores compatible artifacts, and migrates the earlier single-base layout without recomputation. |
| `overlay/common/prefix_selection.py` | Compares both entries. Exact tokens plus compatible runtime allow reuse; similarity only chooses which entry to replace. |
| [Hook patch](../../patches/tabby/0001-exl3-persistent-context-hooks.patch) | Connects storage configuration, model load/unload, the private control channel and native chat-template rendering. |

Native GPU pages, the pinned CPU KV tier and the bounded tmpfs tier remain native
EXL3 caches. The protected SSD files are additional non-LRU originals. A process
restart can reuse tmpfs contents; a host reboot loses tmpfs but can import the
protected SSD bases. Replacing A never requires deleting B. Neither base pins a
permanent additional GPU allocation or increases the configured native budgets.

For two occupied entries, replacement selection uses 70% global five-token
shingle similarity and 30% normalized common-prefix length, with source identity
as an update hint. True ties are marked ambiguous and resolved by least-recent
use, then stable ID. A similar prompt is **never** sufficient to restore KV.

## Apply to the referenced source

Run the compatibility check before changing a checkout. The optional native
path check reads source files without importing EXL3 or touching a GPU:

```bash
python3 integrations/tabby/check_compatibility.py /path/to/tabbyAPI \
  --exl-root /path/to/venv/lib/python3.10/site-packages/exllamav3
```

Then, during a maintenance window chosen by your existing serving system:

```bash
# Run from this repository's root. No installer or service restart is implicit.
git -C /path/to/tabbyAPI apply --check "$PWD/patches/tabby/0001-exl3-persistent-context-hooks.patch"
git -C /path/to/tabbyAPI apply "$PWD/patches/tabby/0001-exl3-persistent-context-hooks.patch"
install -m 644 integrations/tabby/overlay/common/*.py /path/to/tabbyAPI/common/
```

The patch intentionally omits the deployment's extra HTTP cache-stats route.
Use the private control channel or your existing metadata integration. Existing
API authentication and request admission remain the application's responsibility.

The exported lifecycle gate retains historical messages referring to a GPU
handoff to “media.” Its generic contract is a drained lifecycle handoff; no media
service or host policy is included. A caller must drain its existing admission
queue, obtain an `ok/persisted/quiesced` flush response, and only then unload or
stop the model. Failed flushes must not be treated as permission to stop it.
An abandoned flush lease resumes automatically after 60 seconds.

## Storage and opt-in settings

Provide a **dedicated tmpfs mount**, backend-owned with mode `0700`, a separate
private directory for the Unix socket, and an SSD directory outside the tmpfs.
This export does not provision mounts or services. The following is a memory
configuration fragment, not a complete model profile:

```yaml
memory:
  sysmem_recurrent_cache: 4096
  sysmem_kv_cache: 6144
  disk_cache_dir: /run/tabby-context-cache/cache
  disk_cache_size: 7936
  disk_cache_checkpoint_interval: 4096
  cache_control_socket: /run/tabby-cache-control/control.sock
```

Here `/run/tabby-context-cache` is the dedicated mountpoint. The tested guard
retains an 8 GiB tmpfs ceiling and at least 128 MiB of filesystem/index headroom.
All numeric settings above are MiB or tokens as named by Tabby. These are the
tested limits, not a recommendation for every machine; lowering the memory
budgets may evict RAM copies sooner while retaining the protected SSD originals.

The following environment variables opt into the protected feature:

```text
HERMES_PERMANENT_PREFIX_DIR=/var/lib/tabby-prefix-bases
HERMES_PREFIX_STATUS_EXPORT=/run/tabby-prefix-metadata/status.json
```

The historical names are retained to preserve the tested interface; the feature
does not require a Telegram bot or a specific client. Omit the optional status
export when unused. The backend creates private `0700` artifact directories and
`0600` files; status export contains counts/reason codes, not token vectors.
Its parent directory must already exist with suitable ownership.

Use a different protected root for each served model when retaining two bases
per model. This wrapper manages exactly two entries in its configured root; it
does not provision a multi-model directory hierarchy automatically.

```text
protected SSD root/
  catalog.json             entry IDs, selection and recency
  status.json              sanitized metadata
  base-1/current.json → immutable generation/pages + paired recurrent state
  base-2/current.json → immutable generation/pages + paired recurrent state
```

Each entry is limited to 2 GiB, for 4 GiB of committed protected data. Atomic
replacement can temporarily stage a third bundle. The token ceiling is 32,768,
aligned to complete 256-token native pages. The artifact contains token IDs and
native state, so it remains private even though the status export is sanitized.

## Pre-user capture, invalidation and client compatibility

The chat hook renders consecutive leading system messages with the **same native
template and tools** as the real request, then takes a strict token intersection
with that actual request. A synthetic user boundary satisfies templates which
reject a system-only conversation, but it is cut out before tokenization/prefill.
The prefill helper calls no generation/sampling path. Its unprocessed final
sentinel and the real user's message never enter the published checkpoint.

Capture currently opts in tool-aware requests; tiny title/summary requests do not
replace a tool-aware base. The renderer and token intersection were tested with
the Qwen template. A different template must be validated before deployment.

One retained client compatibility rule removes only the final
`\n\nConversation started:` footer from a **deep copy** of leading content. The
original request is unchanged. This keeps a known client's date/session hints
outside the static base; memory/instructions before that marker remain exact.
For clients using that text for another purpose, review this explicit delimiter
contract rather than assuming arbitrary text is canonicalized.

Native cache fingerprint, target/draft cache settings, native Python source
hashes, exact template/defaults and tokenizer file hashes guard each entry.
Changed input or runtime causes a visible expired/rebuild state and updates only
the selected entry. State-transition logging is deduplicated; no application
message or notification service is included. If RAM pages were evicted, a
compatible SSD entry is imported individually when idle. Busy engines defer
preparation instead of cancelling live jobs. Integrity failure in one entry does
not invalidate the other.

Legacy single-base migration renames its generation into `base-1`; it does not
prefill or copy a large checkpoint. Rolling back to the old single-base Python
implementation after migration also requires reversing that layout change;
reverting code alone will not discover the nested generations.

## Validation and limits

```bash
python3 -I -m unittest discover -s integrations/tabby/tests -v
```

The **43 CPU tests** cover reboot-style RAM loss, A → B → A without prefill on
exact use, front edits replacing only the closest base, corruption isolation,
missing middle-page reimport, inode-preserving migration, exact token/paired-state
validation, pre-user neutrality, busy refusal, publication before releasing job
references, bounded tmpfs storage and private socket ownership/lease races. The 14 added
portable regressions cover exact-anchor admission and ordering, side/position/
layer/runtime/payload rejection, safe near-capacity refusal, and immutable
publication across copy, sync and pointer failures. Existing fixture files now
represent the paired native header rather than a generic state placeholder.
These tests use standard-library fixtures; no native EXL3 source is bundled.

Historical native GPU checks used a 27B EXL3 SC-3.00-H4 model with DFlash2 Q4, K5/V4 target
cache, two slots and a 130K configured context. Two independent SDK-assembled
client prefixes were validated with different fresh user requests and alternated
through the ordinary existing admission route. Their hot first-token latencies
were **0.509 s** and **0.875 s**, reusing **23,040/23,081** and **16,128/16,378**
tokens. Both SSD artifacts included the intended static bootstrap and excluded
the user boundary. Initial new-base prefill/publication took 22.795 s / 15.839 s.

The Base 1 repair was subsequently qualified in a separate deployment with
**K8/V4, one slot, 130,000 context tokens and DFlash2 Q4**, without changing the
model profile or native memory budgets. All **19 native-AST CPU cases** and
**eight GPU request cases** passed. That CPU harness exercised native state
layout, serialization, spacing and pruning bodies against tiny synthetic CPU
tensors. It is separate evidence, not a dependency of this portable package.

After a backend restart, both protected bases were restored before new requests:

| Request | Cached / input tokens | Full request wall time |
|---|---:|---:|
| Client B, first return | 17,664 / 17,925 | 2.046 s |
| Client A, first return | 22,784 / 23,062 | 1.198 s |
| Client B, next return | 17,920 / 17,925 | 0.241 s |

These latest times cover complete short requests after backend health returned;
they are **not streaming TTFT and exclude model startup time**. The two committed
bases contain 22,784 and 17,664 tokens, occupying 728,176,027 and 591,815,727 bytes.
The eight cases also covered initial preparation, a longer branch, a static-tail
change and canonical return. They executed no real client tools or Telegram
messages.

These are short-request cache checks, not a general throughput benchmark, a
full-130K recovery test, or a guarantee of sampling/quality equivalence for every
model. All 43 portable CPU tests and the source compatibility checker were rerun
for this export. The native-AST and GPU observations are separate deployment
evidence; creating this package performed no inference, restart or deployment.
See [VALIDATION.json](VALIDATION.json) for the compact, sanitized evidence.

## License and attribution

The source modules, hook patch and integration tests are distributed under
**GNU AGPL version 3** (`AGPL-3.0-only`), with the complete text in [LICENSE](LICENSE).
They extend [TabbyAPI](https://github.com/theroyallab/tabbyAPI), whose referenced
[license](https://github.com/theroyallab/tabbyAPI/blob/be74bf0a00bcb3a518e6feb7606f150c189be637/LICENSE)
is AGPLv3. This subdirectory does not relicense other repository components.

The paired paging/recurrent serialization and native CPU/disk cache machinery
belong to [EXL3](https://github.com/turboderp-org/exllamav3/tree/v1.6.3), credited
to Turboderp and contributors and distributed separately under its
[MIT license](https://github.com/turboderp-org/exllamav3/blob/v1.6.3/LICENSE).
No EXL3 native source or model weights are bundled here. Local changes add the
protected two-base store, exact pre-user capture and lifecycle integration;
they do not claim authorship of the native cache implementation.
