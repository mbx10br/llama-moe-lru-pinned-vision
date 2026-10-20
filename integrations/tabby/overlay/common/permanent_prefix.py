# SPDX-License-Identifier: AGPL-3.0-only
# Local cache integration, exported 2026-10-20; see ../../LICENSE.
"""Protected, pre-user EXL3 seed. Administrative socket only; never a chat API.

The SSD copy is outside EXL3's evictable tmpfs root. Only complete content-hashed
pages and their exact paired recurrent checkpoint are restored. No generated
token enters this artifact, and no live conversation is replaced by the seed.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import time
import weakref

PAGE_SIZE = 256
MAX_TOKENS = 32768
MAX_BUNDLE_BYTES = 2 * 1024**3
MAX_BASES = 2
MAX_STORE_BYTES = MAX_BASES * MAX_BUNDLE_BYTES
FORMAT = 1


class PrefixError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def aligned_tokens(tokens):
    if (not isinstance(tokens, list) or not PAGE_SIZE <= len(tokens) <= MAX_TOKENS
            or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens)):
        raise PrefixError("invalid-prefix-tokens")
    return tokens[:len(tokens) // PAGE_SIZE * PAGE_SIZE]


def runtime_identity(container):
    """Include native cache identity, template/tokenization and executable source.

    EXL3's fingerprint covers model weights, quantization, target/draft cache and
    recurrent format. Full source hashes additionally invalidate on engine edits
    that an upstream cache-format version may not yet describe.
    """
    import inspect
    import exllamav3
    engine = container.generator.generator
    cached = getattr(container, "_prefix_runtime_identity", None)
    if cached is not None and cached[0]() is engine:
        return cached[1]
    disk = engine.disk_page_cache
    if disk is None:
        raise PrefixError("disk-tier-unavailable")
    package = Path(inspect.getfile(exllamav3)).parent
    sources = {str(p.relative_to(package)): file_digest(p) for p in sorted(package.rglob("*.py"))}
    template = getattr(container.prompt_template, "raw_template", "") or ""
    identity = {
        "disk_fingerprint": disk.fingerprint,
        "exl_sources": digest(sources),
        "template": hashlib.sha256(template.encode()).hexdigest(),
        "tokenizer_files": {name: file_digest(Path(container.model_dir) / name)
                            for name in ("tokenizer.json", "tokenizer_config.json")
                            if (Path(container.model_dir) / name).is_file()},
        "template_defaults": getattr(container, "template_vars_default", {}),
        "template_force": getattr(container, "template_vars_force", {}),
        "cache_mode": container.cache_mode,
        "draft_cache_mode": container.draft_cache_mode,
        "special_encoding": True,
    }
    # Do not retain a retired generator/cache (and its GPU allocations) after
    # a model switch/unload just to memoize its source/config digest.
    container._prefix_runtime_identity = (weakref.ref(engine), identity)
    return identity


class SinglePrefixStore:
    def __init__(self, root, notify=None, status_export=True):
        self.root = Path(root)
        self.notify = notify or (lambda message: None)
        self.state = {"state": "missing", "reason": "not-prepared", "tokens": 0}
        self._last_notice = None
        self._export_error = False
        self._manifest = None
        self.status_export = status_export
        if not self.root.is_absolute() or self.root.is_symlink():
            raise PrefixError("unsafe-prefix-directory")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise PrefixError("prefix-directory-must-be-private")

    def report(self, state, reason, tokens=0, **details):
        self.state = {"state": state, "reason": reason, "tokens": tokens, **details}
        notice = (state, reason)
        if notice != self._last_notice:
            self.notify(f"Prefix base: {state} ({reason}); {tokens} tokens")
            self._last_notice = notice
        # Sanitized status, no prompt/token IDs, accessible to the existing host
        # observer without adding a TCP service or sending Telegram messages.
        snapshot = {**self.state, "updated_at": time.time()}
        self.atomic_json(self.root / "status.json", snapshot)
        export = os.environ.get("HERMES_PREFIX_STATUS_EXPORT") if self.status_export else None
        if export:
            try:
                self.atomic_json(Path(export), snapshot, mode=0o640)
                self._export_error = False
            except OSError:
                # Status export is optional and never breaks inference or the
                # protected artifact if its runtime directory is unavailable.
                if not self._export_error:
                    self.notify("Prefix base status export unavailable")
                self._export_error = True
        return dict(self.state)

    @staticmethod
    def atomic_json(path, value, mode=0o600):
        fd, temporary = tempfile.mkstemp(prefix=".prefix-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read_manifest(self):
        pointer = self.root / "current.json"
        if not pointer.exists():
            return None, None
        if pointer.is_symlink() or pointer.stat().st_size > 4096:
            raise PrefixError("invalid-prefix-pointer")
        generation = json.loads(pointer.read_text())["generation"]
        if len(generation) != 64 or any(c not in "0123456789abcdef" for c in generation):
            raise PrefixError("invalid-prefix-generation")
        folder = self.root / generation
        if folder.is_symlink():
            raise PrefixError("unsafe-prefix-generation")
        manifest = folder / "manifest.json"
        if manifest.is_symlink() or manifest.stat().st_size > 2 * 1024**2:
            raise PrefixError("invalid-prefix-manifest")
        value = json.loads(manifest.read_text())
        tokens = aligned_tokens(value["tokens"])
        if (value.get("format") != FORMAT or len(tokens) != len(value["tokens"])
                or value.get("token_digest") != digest(tokens)
                or value.get("generation") != generation):
            raise PrefixError("invalid-prefix-manifest")
        return folder, value

    def publish(self, disk, tokens, source_identity, runtime, page_hashes):
        tokens = aligned_tokens(tokens)
        if len(page_hashes) != len(tokens) // PAGE_SIZE:
            raise PrefixError("prefix-page-count-mismatch")
        files = []
        previous = None
        for index, key in enumerate(page_hashes):
            source = Path(disk._page_path(key))
            header = disk._read_page_header(str(source))
            if (header is None or header["phash"] != key or header["prev_hash"] != previous
                    or header.get("page_size") != PAGE_SIZE
                    or header["tokens_len"] not in (PAGE_SIZE * 4, PAGE_SIZE * 8)):
                raise PrefixError("incomplete-prefix-chain")
            with source.open("rb") as stream:
                # Native reader's data_offset is the start of token bytes,
                # not the following KV image (which starts tokens_len later).
                stream.seek(header["data_offset"])
                raw_tokens = stream.read(header["tokens_len"])
            width = header["tokens_len"] // PAGE_SIZE
            decoded = list(struct.unpack("<" + ("i" if width == 4 else "q") * PAGE_SIZE, raw_tokens))
            if decoded != tokens[index * PAGE_SIZE:(index + 1) * PAGE_SIZE]:
                raise PrefixError("prefix-page-tokens-mismatch")
            files.append((source, f"pages/{key.hex()}.kv"))
            previous = key
        state_path = Path(disk._state_path(previous))
        state_header = disk._read_state_header(str(state_path))
        if (state_header is None or not disk.has_state(previous)
                or state_header.get("position") != len(tokens)
                or state_header.get("phash") != previous.hex()):
            raise PrefixError("paired-recurrent-state-missing")
        files.append((state_path, f"states/{previous.hex()}.rs"))
        size = sum(path.stat().st_size for path, _ in files)
        if size > MAX_BUNDLE_BYTES:
            raise PrefixError("prefix-bundle-too-large")
        generation = digest({"tokens": tokens, "source": source_identity, "runtime": runtime})
        destination = self.root / generation
        temporary = Path(tempfile.mkdtemp(prefix=".preparing-", dir=self.root))
        try:
            entries = []
            for path, relative in files:
                target = temporary / relative
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                target.chmod(0o600)
                entries.append({"path": relative, "size": target.stat().st_size, "sha256": file_digest(target)})
            manifest = {"format": FORMAT, "generation": generation, "tokens": tokens,
                        "token_digest": digest(tokens), "source_identity": source_identity,
                        "runtime": runtime, "disk_fingerprint": disk.fingerprint,
                        "anchor": previous.hex(), "files": entries, "bytes": size,
                        "created_at": time.time()}
            self.atomic_json(temporary / "manifest.json", manifest)
            if destination.exists():
                # The replacement is already complete. A corrupted managed
                # copy can be regenerated; it is never selected again as-is.
                if destination.is_symlink():
                    raise PrefixError("unsafe-prefix-generation")
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            self.atomic_json(self.root / "current.json", {"generation": generation})
            self._manifest = manifest
            self.report("ready", "prepared-before-user", len(tokens), bytes=size)
            # Keep only the selected managed artifact. This is a cache, not an
            # automatic user-data backup; failed preparation never alters it.
            for folder in self.root.iterdir():
                if (folder.is_dir() and not folder.is_symlink() and len(folder.name) == 64
                        and folder.name != generation
                        and all(c in "0123456789abcdef" for c in folder.name)):
                    shutil.rmtree(folder)
            return self.state
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def restore(self, disk, runtime):
        try:
            folder, manifest = self.read_manifest()
            if manifest is None:
                self._manifest = None
                return self.report("missing", "not-prepared")
            if manifest["runtime"] != runtime or manifest["disk_fingerprint"] != disk.fingerprint:
                self._manifest = None
                return self.report("expired", "runtime-model-template-changed", len(manifest["tokens"]))
            validated = []
            total = 0
            for entry in manifest["files"]:
                relative = Path(entry["path"])
                if (len(relative.parts) != 2 or relative.parts[0] not in ("pages", "states")
                        or relative.name != str(relative.name) or ".." in relative.parts):
                    raise PrefixError("invalid-prefix-path")
                source = folder / relative
                if source.is_symlink() or source.stat().st_size != entry["size"]:
                    raise PrefixError("prefix-artifact-corrupt")
                if file_digest(source) != entry["sha256"]:
                    raise PrefixError("prefix-artifact-corrupt")
                total += entry["size"]
                if total > MAX_BUNDLE_BYTES:
                    raise PrefixError("prefix-bundle-too-large")
                key = bytes.fromhex(relative.stem)
                if len(key) not in (16, 20, 32):
                    raise PrefixError("invalid-page-key")
                if relative.parts[0] == "pages":
                    header = disk._read_page_header(str(source))
                    target = Path(disk._page_path(key))
                else:
                    header = disk._read_state_header(str(source))
                    target = Path(disk._state_path(key))
                if header is None:
                    raise PrefixError("native-state-incompatible")
                if relative.parts[0] == "states" and (
                        header.get("position") != len(manifest["tokens"])
                        or header.get("phash") != manifest["anchor"]):
                    raise PrefixError("native-state-incompatible")
                validated.append((source, target))
            if total > disk.max_size:
                raise PrefixError("prefix-exceeds-ram-tier-budget")
            # Startup only: no writer, reader promotion or job may interleave
            # with index reconciliation. Caller enforces this ownership rule.
            if disk.pending_pages or disk.pending_states or getattr(disk, "_promotions", 0):
                raise PrefixError("cache-io-busy")
            disk.flush()
            required = sum(source.stat().st_size for source, target in validated if not target.exists())
            if getattr(disk, "total_size", 0) + required > disk.max_size:
                # Let the normal tier choose expendable entries before adding
                # the RAM copy; the SSD artifact is never in its eviction root.
                with disk._lock:
                    disk._make_room(total)
                disk.flush()
            for source, target in validated:
                if target.exists() and file_digest(target) == file_digest(source):
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".seed-", dir=target.parent)
                os.close(fd)
                try:
                    shutil.copyfile(source, temporary)
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            with disk._lock:
                disk._load_index()
                if hasattr(disk, "_order"):
                    disk._order.clear()
            if not disk.has_state(bytes.fromhex(manifest["anchor"])):
                raise PrefixError("restored-checkpoint-missing")
            self._manifest = manifest
            return self.report("ready", "restored-to-ram", len(manifest["tokens"]), bytes=total)
        except (OSError, ValueError, KeyError, PrefixError) as error:
            self._manifest = None
            reason = str(error) if isinstance(error, PrefixError) else type(error).__name__
            return self.report("error", reason)

    def observe(self, tokens):
        """Native hashes always enforce exact matching, even for other clients.

        This diagnostic never discards a newer conversation or labels another
        client's different prompt as corruption of the protected base.
        """
        manifest = self._manifest
        if manifest is None:
            return None
        expected = manifest["tokens"]
        matches = len(tokens) >= len(expected) and tokens[:len(expected)] == expected
        return {"matching": matches, "tokens": len(expected), "source": "protected-base"}


class PrefixStore:
    """Two independent pre-user artifacts; names do not decide cache reuse.

    Each child's native files/parser are unchanged from the validated single
    store. The aggregate catalog carries only stable IDs and last-use metadata.
    It never turns a similarity decision into permission to restore state: that
    still requires exact tokens and exact native runtime compatibility.
    """
    IDS = ("base-1", "base-2")

    def __init__(self, root, notify=None):
        self.root = Path(root)
        self.notify = notify or (lambda message: None)
        self._reporter = SinglePrefixStore(self.root, self.notify)
        self.children = {name: SinglePrefixStore(self.root / name, status_export=False)
                         for name in self.IDS}
        self._selected = self.IDS[0]
        self._last_used = {name: 0.0 for name in self.IDS}
        self._last_selection = None
        self._migration_error = None
        self._migrate_legacy()
        self._load_catalog()

    @property
    def state(self):
        return self._reporter.state

    @property
    def _manifest(self):
        return self.children[self._selected]._manifest

    @staticmethod
    def atomic_json(path, value, mode=0o600):
        return SinglePrefixStore.atomic_json(path, value, mode)

    def _migrate_legacy(self):
        pointer = self.root / "current.json"
        if not pointer.exists():
            return
        try:
            if pointer.is_symlink() or pointer.stat().st_size > 4096:
                raise PrefixError("invalid-legacy-pointer")
            generation = json.loads(pointer.read_text())["generation"]
            if (not isinstance(generation, str) or len(generation) != 64
                    or any(c not in "0123456789abcdef" for c in generation)):
                raise PrefixError("invalid-legacy-generation")
            child = self.children[self.IDS[0]]
            source, destination = self.root / generation, child.root / generation
            if source.is_symlink() or destination.is_symlink():
                raise PrefixError("unsafe-legacy-artifact")
            # Rename, not copy or prefill. Recover a crash between this rename
            # and pointer commit by recognizing the same generation in child 1.
            if source.exists() and not destination.exists():
                os.replace(source, destination)
            if not destination.is_dir():
                raise PrefixError("legacy-artifact-missing")
            child.atomic_json(child.root / "current.json", {"generation": generation})
            child.read_manifest()  # validate before retiring the old pointer
            self._last_used[self.IDS[0]] = time.time()
            self._save_catalog()
            pointer.unlink()
        except (OSError, ValueError, KeyError, TypeError, PrefixError) as error:
            self._migration_error = str(error) if isinstance(error, PrefixError) else type(error).__name__
            self.children[self.IDS[0]].state = {"state": "error", "reason": "legacy-" + self._migration_error,
                                               "tokens": 0}

    def _load_catalog(self):
        path = self.root / "catalog.json"
        if not path.exists():
            return
        try:
            if path.is_symlink() or path.stat().st_size > 16384:
                raise PrefixError("invalid-dual-catalog")
            value = json.loads(path.read_text())
            if value.get("version") != 2 or len(value.get("bases", [])) > MAX_BASES:
                raise PrefixError("invalid-dual-catalog")
            for entry in value["bases"]:
                if entry["id"] not in self.IDS or type(entry.get("last_used")) not in (int, float):
                    raise PrefixError("invalid-dual-catalog")
                self._last_used[entry["id"]] = float(entry["last_used"])
            if value.get("selected_base") in self.IDS:
                self._selected = value["selected_base"]
        except (OSError, ValueError, KeyError, TypeError, PrefixError):
            # Native artifacts remain authoritative and individually readable;
            # damage to recency metadata cannot invalidate either checkpoint.
            self.notify("Prefix catalog recency unavailable; using deterministic slot order")

    def _save_catalog(self):
        self.atomic_json(self.root / "catalog.json", {
            "version": 2, "selected_base": self._selected,
            "storage_limit_bytes": MAX_STORE_BYTES,
            "bases": [{"id": name, "last_used": self._last_used[name]} for name in self.IDS
                      if (self.children[name].root / "current.json").exists()],
        })

    def read_manifest(self, base_id=None):
        name = base_id or self._selected
        return self.children[name].read_manifest()

    def _records(self):
        records = []
        for name, child in self.children.items():
            occupied = (child.root / "current.json").exists()
            manifest = child._manifest
            if occupied and manifest is None:
                try:
                    _, manifest = child.read_manifest()
                except (OSError, ValueError, KeyError, TypeError, PrefixError):
                    pass
            records.append({"id": name, "occupied": occupied,
                            "tokens": manifest["tokens"] if manifest else [],
                            "runtime": manifest.get("runtime") if manifest else None,
                            "source_identity": manifest.get("source_identity") if manifest else None,
                            "valid": child._manifest is not None,
                            "last_used": self._last_used[name]})
        return records

    def select(self, tokens, runtime, source_identity=None):
        # A small identical helper is shared with the laptop implementation.
        from common.prefix_selection import choose_base
        decision = choose_base(tokens, self._records(), runtime=runtime,
                               source_identity=source_identity, capacity=MAX_BASES)
        self._last_selection = decision
        return decision

    def manifest_for(self, base_id):
        return self.children[base_id]._manifest

    def touch(self, base_id):
        self._selected = base_id
        self._last_used[base_id] = time.time()
        self._save_catalog()

    def report(self, state, reason, tokens=0, base_id=None, **details):
        if base_id is not None:
            self._selected = base_id
            self.children[base_id].state = {"state": state, "reason": reason,
                                             "tokens": tokens, **details}
        bases = []
        for name, child in self.children.items():
            occupied = (child.root / "current.json").exists()
            if not occupied and child.state.get("state") == "missing":
                continue
            snapshot = child.state
            manifest = child._manifest
            if occupied and manifest is None:
                try:
                    _, manifest = child.read_manifest()
                except (OSError, ValueError, KeyError, TypeError, PrefixError):
                    pass
            tokens_count = int(snapshot.get("tokens", 0))
            bytes_count = int(snapshot.get("bytes", 0))
            if manifest:
                tokens_count = tokens_count or len(manifest["tokens"])
                saved_size = manifest.get("bytes", 0)
                if type(saved_size) is int and saved_size >= 0:
                    bytes_count = bytes_count or saved_size
            # Raw token IDs, hashes, source identity and prompt text are never
            # exposed through the metadata export or the administrative status.
            bases.append({"id": name, "state": snapshot.get("state", "missing"),
                          "reason": snapshot.get("reason", "not-prepared"),
                          "tokens": tokens_count,
                          "bytes": bytes_count,
                          "selected": name == self._selected})
        ready = sum(base["state"] == "ready" for base in bases)
        if ready and state in ("missing", "error", "expired"):
            state, reason = "ready", "partially-ready"
        selected = next((base for base in bases if base["id"] == self._selected), {})
        return self._reporter.report(state, reason, selected.get("tokens", tokens),
            bytes=selected.get("bytes", details.get("bytes", 0)), bases=bases,
            selected_base=self._selected, count_ready=ready, count_total=len(bases),
            storage_used_bytes=sum(base["bytes"] for base in bases),
            storage_limit_bytes=MAX_STORE_BYTES, base_limit_bytes=MAX_BUNDLE_BYTES,
            selection=self._last_selection)

    def publish(self, disk, tokens, source_identity, runtime, page_hashes, base_id=None):
        selection = self.select(tokens, runtime, source_identity) if base_id is None else None
        name = base_id or selection["id"]
        child = self.children[name]
        child.publish(disk, tokens, source_identity, runtime, page_hashes)
        self.touch(name)
        return self.report("ready", "prepared-before-user", base_id=name,
                           tokens=len(tokens), bytes=child.state.get("bytes", 0))

    def restore(self, disk, runtime, base_id=None):
        names = (base_id,) if base_id is not None else self.IDS
        for name in names:
            child = self.children[name]
            try:
                child.restore(disk, runtime)
                if child._manifest is not None and child.state.get("state") == "ready":
                    # Imported index entries otherwise have last_access=0 and
                    # could be the first LRU victims while importing base 2.
                    # Normal recency keeps both boot seeds available without
                    # pinning their RAM copies or changing the tier budget.
                    if hasattr(disk, "_touch"):
                        for entry in child._manifest["files"]:
                            path = Path(entry["path"])
                            key = bytes.fromhex(path.stem)
                            kind = "page" if path.parts[0] == "pages" else "state"
                            table = disk.pages if kind == "page" else disk.states
                            if key in table:
                                disk._touch(table[key], kind, key)
                        disk.flush()
            except Exception as error:
                # An unreadable artifact can never prevent the other family's
                # native state from being checked and imported independently.
                child._manifest = None
                child.state = {"state": "error", "reason": type(error).__name__, "tokens": 0}
        for name in self.IDS:
            child = self.children[name]
            if child._manifest is not None and not ram_copy_complete(disk, child._manifest):
                child.state = {"state": "pending", "reason": "ram-tier-capacity-conflict",
                               "tokens": len(child._manifest["tokens"]),
                               "bytes": child._manifest.get("bytes", 0)}
        ready = [name for name in self.IDS if self.children[name].state.get("state") == "ready"]
        if ready:
            self._selected = base_id if base_id in ready else (
                self._selected if self._selected in ready else ready[0])
            self._save_catalog()
            all_occupied = sum((child.root / "current.json").exists() for child in self.children.values())
            reason = "restored-to-ram" if len(ready) == all_occupied else "partially-ready"
            return self.report("ready", reason)
        present = [child for child in self.children.values() if (child.root / "current.json").exists()]
        if not present:
            return self.report("missing", "not-prepared")
        state = "expired" if all(child.state.get("state") == "expired" for child in present) else "error"
        return self.report(state, "no-compatible-base")

    def observe(self, tokens):
        for name in self.IDS:
            result = self.children[name].observe(tokens)
            if result and result["matching"]:
                return {**result, "id": name}
        selected = self.children[self._selected].observe(tokens)
        return {**selected, "id": self._selected} if selected else None


def strict_common_prefix(full, leading):
    """The shorter render contains no user text; never extrapolate its tail."""
    limit = min(len(full), len(leading), MAX_TOKENS)
    position = 0
    while position < limit and full[position] == leading[position]:
        position += 1
    return full[:position // PAGE_SIZE * PAGE_SIZE]


def stable_leading_messages(messages):
    """Exclude Hermes' variable date/session/environment footer, without edits
    to the request. Memory/skills/core content before the footer stays exact.
    """
    stable = copy.deepcopy(messages)
    marker = "\n\nConversation started:"
    for message in stable:
        content = message.get("content")
        if isinstance(content, str):
            position = content.rfind(marker)
            if position >= 0:
                message["content"] = content[:position]
    return stable


def ram_copy_complete(disk, manifest):
    """A recurrent anchor alone cannot make an evicted KV chain usable."""
    if not disk.has_state(bytes.fromhex(manifest["anchor"])):
        return False
    return all(disk.has_page(bytes.fromhex(Path(entry["path"]).stem))
               for entry in manifest["files"] if entry["path"].startswith("pages/"))


def auto_prepare(container, full_prompt, leading_prompt, source_identity, add_bos):
    """Called after native Jinja rendering, before enqueueing the real request.

    Native generation still receives its original prompt. Only the complete
    leading system/developer render is eligible; the strict token LCP excludes
    template suffix differences and every user/assistant/tool message.
    """
    store = getattr(container, "permanent_prefix_store", None)
    if store is None:
        return None
    engine = container.generator.generator
    tokenizer = container.tokenizer
    full = tokenizer.encode(full_prompt, add_bos=add_bos,
                            encode_special_tokens=True).tolist()[0]
    leading = tokenizer.encode(leading_prompt, add_bos=add_bos,
                               encode_special_tokens=True).tolist()[0]
    tokens = strict_common_prefix(full, leading)
    if len(tokens) < PAGE_SIZE:
        return store.report("pending", "no-stable-pre-user-prefix")
    runtime = runtime_identity(container)
    selection = store.select(tokens, runtime, source_identity)
    base_id = selection["id"]
    previous = store.manifest_for(base_id)
    if selection["mode"] == "exact":
        # Native LRU may evict its tmpfs copy later. The SSD source remains
        # untouched; the next idle request can re-import it without computing.
        if not ram_copy_complete(engine.disk_page_cache, previous):
            if engine.active_jobs or engine.pending_jobs or container.generator.jobs:
                return store.report("pending", "ram-copy-evicted-backend-busy", len(tokens), base_id=base_id)
            return store.restore(engine.disk_page_cache, runtime, base_id=base_id)
        store.touch(base_id)
        return store.report("ready", "exact-prefix-match", len(tokens), base_id=base_id,
                            bytes=previous.get("bytes", 0))
    if engine.active_jobs or engine.pending_jobs or container.generator.jobs:
        return store.report("pending", "changed-prefix-awaiting-idle", len(tokens), base_id=base_id)
    previous_quiescing = container.cache_quiescing
    if previous_quiescing:
        return store.report("pending", "maintenance-lease-active", len(tokens), base_id=base_id)
    container.cache_quiescing = True
    try:
        store.report("rebuilding", "prefix-or-source-changed", len(tokens), base_id=base_id)
        # Keep the temporary job's page and recurrent references owned until
        # the immutable SSD generation has been committed. Queue-drain cleanup
        # may prune/evict native entries as soon as cancel() releases them.
        return prefill_only(engine, tokens, on_ready=lambda hashes: store.publish(
            engine.disk_page_cache, tokens, source_identity, runtime, hashes, base_id=base_id))
    except Exception as error:
        reason = (str(error) if isinstance(error, PrefixError)
                  else type(error).__name__ + ("-" + str(error.errno) if isinstance(error, OSError) else ""))
        return store.report("error", reason, len(tokens), base_id=base_id)
    finally:
        container.cache_quiescing = previous_quiescing


def prefill_only(engine, tokens, job_factory=None, tensor_factory=None, deadline_seconds=600, on_ready=None):
    # Native iterate() normally supplies this context around page allocation,
    # prefill, checkpoint copies and deallocation. Its inference tensors cannot
    # safely be mutated outside that context, even if no gradients are requested.
    if job_factory is None or tensor_factory is None:
        import torch
        if job_factory is None:
            from exllamav3.generator.job import Job
            job_factory = Job
        with torch.inference_mode():
            return _prefill_only(engine, tokens, job_factory,
                                 tensor_factory or (lambda ids: torch.tensor([ids], dtype=torch.long)),
                                 deadline_seconds, on_ready)
    return _prefill_only(engine, tokens, job_factory, tensor_factory, deadline_seconds, on_ready)


def _prefill_only(engine, tokens, job_factory, tensor_factory, deadline_seconds, on_ready):
    """Process exactly an aligned static prefix, without any generation path.

    Job.prefill deliberately excludes its final input token. A sentinel occupies
    that unprocessed position: it is neither forwarded nor sampled nor archived.
    """
    if engine.active_jobs or engine.pending_jobs:
        raise PrefixError("text-jobs-busy")
    tokens = aligned_tokens(tokens)
    job = job_factory(input_ids=tensor_factory(tokens + [0]), max_new_tokens=1,
                      token_healing=False)
    started = time.monotonic()
    engine.enqueue(job)
    try:
        results = []
        engine.iterate_start_jobs(results)
        if job not in engine.active_jobs:
            failure = next((row.get("error") for row in results if row.get("stage") == "error"), None)
            raise PrefixError("prefix-job-not-started" + ("-" + type(failure).__name__ if failure else "")) from failure
        while not job.is_prefill_done():
            if time.monotonic() - started > deadline_seconds:
                raise PrefixError("prefix-prefill-deadline")
            before = job.sequences[0].kv_position
            job.prefill(results)
            results.clear()
            if job.sequences[0].kv_position <= before and not job.is_prefill_done():
                raise PrefixError("prefix-prefill-no-progress")
        if job.new_tokens != 0 or job.sequences[0].kv_position != len(tokens):
            raise PrefixError("prefix-not-exact")
        if engine.recurrent_cache is None:
            raise PrefixError("paired-recurrent-cache-required")
        job.maybe_stash_recurrent(engine.recurrent_cache, PAGE_SIZE)
        hashes = [page.phash for page in job.sequences[0].allocated_pages[:len(tokens) // PAGE_SIZE]]
        engine.flush_cache_to_disk(wait=True)
        if not engine.disk_page_cache.has_state(hashes[-1]):
            raise PrefixError("paired-recurrent-state-missing")
        return on_ready(hashes) if on_ready is not None else hashes
    finally:
        # The normal cancellation path releases only our temporary job. Earlier
        # conversation pages/checkpoints remain managed by their native LRU.
        engine.cancel(job)
