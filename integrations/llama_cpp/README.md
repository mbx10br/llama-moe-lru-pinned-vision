# Two protected startup prefixes for a private llama.cpp server

These are the tested Python modules, exported without changes to the controller
logic. They do not launch a model, create a public API, implement another queue
or replace the serving supervisor. Python's standard library is sufficient.

`prefix_selection.py` chooses which of two managed bases to reuse or update.
`prefix_checkpoint.py` renders/tokenizes through the real native server, saves
only the pre-user prefix, verifies SSD integrity and calls the native selector.
The tests use a fake native transport and disposable directories; they never
call a model or use a personal prompt.

```sh
python3 -B -m unittest discover -s integrations/llama_cpp -p 'test_*.py' -v
```

## Supervisor integration contract

1. Apply/build the pinned native patch in `patches/llama.cpp/`. Keep the native
   administrative server private, behind the existing single-flight supervisor.
2. Construct `PrefixCheckpoint(profile, post, event)`. `post(path, body)` is a
   private JSON POST callback to that server; `event` accepts bounded lifecycle
   metadata. The example profile below illustrates its relevant fields.
3. Before importing states, validate `/props` supports **both**
   `supports_prefix_restore_if_better` and `supports_prefix_preserve_active`.
   An old server can silently ignore new fields, so this guard is required.
4. After a real native child is ready, call `controller.on_load(native_pid)`
   under the same request gate. It imports independently validated SSD bases.
5. Before forwarding a real chat request, call `controller.prepare(payload)`
   under that gate. Forward the original payload unchanged afterward; continue
   ordinary inference if a cache preview/preparation cannot safely proceed.
6. Publish only sanitized `controller.status()` metadata. Never publish saved
   token vectors, checkpoint files, prompts, paths or raw profile identity.

```python
profile = {
    "prefix_dual_cache": True,
    "prefix_checkpoint_dir": "/private/model-prefix-cache",
    "context": 102400,
    "model_sha256": "<verified-model-content-digest>",
    "projector_sha256": "<verified-projector-content-digest>",
    "runtime_sha256": "<verified-runtime-build-digest>",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "context_checkpoints": 4,
}
```

The controller fingerprints the complete supplied profile, except explicitly
irrelevant serving fields. Its caller must include and verify actual model,
projector, tokenizer/template/runtime and cache identities; merely putting a
model's display name or geometry in a profile is insufficient. A separate
runtime/profile must own each checkpoint directory. Directory ownership and
file integrity are checked; cache failures fall back rather than authorize
incompatible state reuse.

The native server also needs `--slot-save-path` pointing at the same private
directory, `--cache-ram 3072`, and the tested single-slot context geometry.
Sleep preservation uses `--sleep-idle-seconds 1200`. Experimental checkpoint
anchors require `ORNITH_CHECKPOINT_GRID=1` and the compatible restrictions in
the native patch. The full tested example is in `docs/PERSISTENT_PREFIXES.md`.

## Boundary and retention

The stored base contains real tool-aware system/developer content **before the
first user**. The native rendering of that leading context must be a token
prefix of the actual complete request. Auxiliary title/summary prompts without
tools do not replace the protected bases. The known Hermes variable tail after
`Conversation started:` is excluded from the saved base and remains in the real
request. This delimiter is client-specific; other clients need a documented
stable boundary, not an arbitrary truncation of their system instructions.

Two bases are managed, with a 2 GiB payload limit each. Similarity chooses an
update victim only. Exact token and compatible runtime identity determine
reusability. Catalog updates are atomic; unreadable references retain uncertain
files instead of deleting another base. An atomic replacement can briefly use
additional disk space beyond the committed 4 GiB total.

The normal RAM pool may preserve advanced conversations. Only pre-user bases
are persisted by this module. Sleep/wake may reuse that RAM; shutdown clears it.
A changed prompt can rebuild one SSD base while preserving the other. Media
requests skip text-only preview and use the native multimodal path.

No host configuration, keys, saved checkpoints or private policy is bundled.
Project integration code follows the MIT foundation of the native runtime.
