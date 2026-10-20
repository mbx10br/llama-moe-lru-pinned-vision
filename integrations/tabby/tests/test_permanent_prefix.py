# SPDX-License-Identifier: AGPL-3.0-only
"""CPU invariants for the protected artifact and exact pre-user prefill path."""
import importlib.util
import json
from pathlib import Path
import struct
import shutil
import weakref
import tempfile
import threading
import types
import unittest
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))
spec = importlib.util.spec_from_file_location("permanent_prefix", ROOT / "overlay/common/permanent_prefix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeDisk:
    def __init__(self, root):
        self.dir = str(root)
        self.fingerprint = "model-1-cache-5-4-draft-q4"
        self.max_size = 3 * 1024**3
        self.pending_pages = set()
        self.pending_states = {}
        self._promotions = 0
        self._lock = threading.RLock()
        self.pages = {}
        self.states = {}
        self.recurrent_sides = {"target": object(), "draft": object()}
        self.total_size = 0
        self.writes = []
        Path(root).mkdir()

    def _page_path(self, key):
        return str(Path(self.dir) / "pages" / key[:1].hex() / (key.hex() + ".kv"))

    def _state_path(self, key):
        return str(Path(self.dir) / "states" / (key.hex() + ".rs"))

    def _header(self, path):
        try:
            with open(path, "rb") as stream:
                size = struct.unpack("<I", stream.read(4))[0]
                header = json.loads(stream.read(size))
            if header["fingerprint"] != self.fingerprint:
                return None
            return header
        except OSError:
            return None

    def _read_page_header(self, path):
        header = self._header(path)
        if header:
            header["phash"] = bytes.fromhex(header["phash"])
            header["prev_hash"] = bytes.fromhex(header["prev_hash"]) if header["prev_hash"] else None
        return header

    def _read_state_header(self, path):
        return self._header(path)

    def _side_keymap(self, side):
        return {side + "-native-layer": side + "-layer"}

    def write_state(self, key, stash, force=False):
        # Explicit native-layout fixture; no EXL3 implementation is bundled.
        self.writes.append((key, force))
        sides = {}
        image_offset = 0
        for side in self.recurrent_sides:
            sides[side] = {"ints": {"position": stash[side]["position"],
                                     "checkpoint_size": stash[side]["checkpoint_size"]},
                           "layers": {side + "-layer": {"tensors": [{"offset": 0, "nbytes": 8}]}},
                           "image_offset": image_offset, "image_size": 8}
            image_offset += 8
        header = {"phash": key.hex(), "fingerprint": self.fingerprint,
                  "position": stash["position"], "top": {"position": stash["position"],
                  "checkpoint_size": stash["checkpoint_size"]}, "sides": sides, "data_offset": 0}
        self.save_state_header(key, header, b"S" * image_offset)
        self.states[key] = {}
        return True

    def save_state_header(self, key, header, payload=b"S" * 16):
        for _ in range(10):
            encoded = json.dumps(header).encode()
            header["data_offset"] = 4 + len(encoded)
        encoded = json.dumps(header).encode()
        path = Path(self._state_path(key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<I", len(encoded)) + encoded + payload)

    def has_state(self, key):
        return key in self.states

    def has_page(self, key):
        return key in self.pages

    def flush(self):
        pass

    def _load_index(self):
        self.pages = {bytes.fromhex(p.stem): {} for p in Path(self.dir).rglob("*.kv")}
        self.states = {bytes.fromhex(p.stem): {} for p in Path(self.dir).rglob("*.rs")}

    def create(self, tokens):
        previous = None
        hashes = []
        for position in range(0, len(tokens), 256):
            key = bytes.fromhex(module.digest([position, previous.hex() if previous else None,
                                               tokens[position:position + 256]])[:32])
            header = {"phash": key.hex(), "prev_hash": previous.hex() if previous else None,
                      "fingerprint": self.fingerprint, "page_size": 256,
                      "tokens_len": 256 * 8, "data_offset": 0}
            # Find the fixed point of the header's own encoded length.
            for _ in range(5):
                encoded = json.dumps(header).encode()
                header["data_offset"] = 4 + len(encoded)
            encoded = json.dumps(header).encode()
            path = Path(self._page_path(key))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(struct.pack("<I", len(encoded)) + encoded
                             + struct.pack("<" + "q" * 256, *tokens[position:position + 256]) + b"KV")
            hashes.append(key)
            previous = key
        self.write_state(previous, paired_stash(len(tokens)), force=True)
        self._load_index()
        return hashes


def paired_stash(position, sides=("target", "draft")):
    return {"position": position, "checkpoint_size": 8 * len(sides),
            **{side: {"position": position, "checkpoint_size": 8} for side in sides}}


def paired_state(position, sides=("target", "draft")):
    return types.SimpleNamespace(position=position, checkpoint_size=8 * len(sides),
        **{side: types.SimpleNamespace(position=position, checkpoint_size=8) for side in sides})


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.notices = []
        self.store = module.PrefixStore(self.root / "protected", self.notices.append)
        self.disk = FakeDisk(self.root / "ram")
        self.tokens = list(range(1024))
        self.runtime = {"model": "qwen", "template": "v1", "build": "v1"}

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self):
        hashes = self.disk.create(self.tokens)
        self.store.publish(self.disk, self.tokens, "source-1", self.runtime, hashes)

    def test_ssd_seed_restores_after_entire_ram_root_is_lost_without_user_binding(self):
        self.publish()
        other = FakeDisk(self.root / "ram-after-reboot")
        restarted = module.PrefixStore(self.root / "protected")
        status = restarted.restore(other, self.runtime)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(len(other.pages), 4)
        self.assertEqual(len(other.states), 1)
        self.assertTrue(restarted.observe(self.tokens + [3333])["matching"])
        self.assertTrue(restarted.observe(self.tokens + [7777, 8888])["matching"])
        self.assertFalse(restarted.observe([9999] + self.tokens)["matching"])

    def test_changed_runtime_refuses_import_and_notifies_once(self):
        self.publish()
        other = FakeDisk(self.root / "ram-2")
        changed = {**self.runtime, "template": "changed"}
        self.store.restore(other, changed)
        self.store.restore(other, changed)
        self.assertEqual(self.store.state["state"], "expired")
        self.assertEqual(other.pages, {})
        self.assertEqual(len(self.notices), 2)  # one ready, one expiry

    def test_recurrent_anchor_does_not_mask_an_evicted_middle_page(self):
        self.publish()
        _, manifest = self.store.read_manifest()
        self.assertTrue(module.ram_copy_complete(self.disk, manifest))
        self.disk.pages.pop(next(iter(self.disk.pages)))
        self.assertFalse(module.ram_copy_complete(self.disk, manifest))

    def test_corrupt_state_never_partially_imports_pages(self):
        self.publish()
        folder, manifest = self.store.read_manifest()
        state = folder / manifest["files"][-1]["path"]
        state.write_bytes(state.read_bytes()[:-1] + b"X")
        other = FakeDisk(self.root / "ram-2")
        self.assertEqual(self.store.restore(other, self.runtime)["state"], "error")
        self.assertEqual(other.pages, {})

    def test_wrong_state_position_or_wrong_token_content_cannot_be_published(self):
        hashes = self.disk.create(self.tokens)
        bad = self.tokens.copy()
        bad[700] = 7777
        with self.assertRaisesRegex(module.PrefixError, "tokens-mismatch"):
            self.store.publish(self.disk, bad, "x", self.runtime, hashes)
        state = Path(self.disk._state_path(hashes[-1]))
        state.unlink()
        with self.assertRaisesRegex(module.PrefixError, "recurrent-state-missing"):
            self.store.publish(self.disk, self.tokens, "x", self.runtime, hashes)
        self.assertFalse((self.store.root / "current.json").exists())

    def test_new_prefix_versions_replace_only_managed_cache_and_leave_other_files(self):
        self.publish()
        (self.store.root / "do-not-touch.txt").write_text("user file")
        self.tokens = list(range(1000, 2024))
        self.publish()
        versions = [p for p in self.store.root.iterdir() if p.is_dir()]
        self.assertEqual(len(versions), 2)
        for child in self.store.children.values():
            self.assertEqual(len([p for p in child.root.iterdir() if p.is_dir()]), 1)
        self.assertEqual((self.store.root / "do-not-touch.txt").read_text(), "user file")


class FakeJob:
    def __init__(self, input_ids, **kwargs):
        self.ids = input_ids
        self.new_tokens = 0
        self.sequences = [types.SimpleNamespace(kv_position=0, allocated_pages=[])]

    def is_prefill_done(self):
        return self.sequences[0].kv_position == len(self.ids) - 1

    def prefill(self, results):
        self.processed.extend(self.ids[self.sequences[0].kv_position:self.sequences[0].kv_position + 256])
        self.sequences[0].kv_position += 256

    def maybe_stash_recurrent(self, cache, interval):
        self.stash_position = self.sequences[0].kv_position
        self.recurrent_state = paired_state(self.stash_position)
        self.stash = paired_stash(self.stash_position)


class FakeEngine:
    def __init__(self):
        self.active_jobs = []
        self.pending_jobs = []
        temporary = tempfile.mkdtemp(prefix="prefix-prefill-test-")
        weakref.finalize(self, shutil.rmtree, temporary)
        self.disk_page_cache = FakeDisk(Path(temporary) / "ram")
        self.recurrent_cache = types.SimpleNamespace(get_stashed=lambda key: self.job.stash)

    def enqueue(self, job):
        job.processed = []
        self.pending_jobs.append(job)
        self.job = job

    def iterate_start_jobs(self, results):
        self.active_jobs.extend(self.pending_jobs)
        self.pending_jobs.clear()
        for job in self.active_jobs:
            job.sequences[0].allocated_pages = [types.SimpleNamespace(phash=bytes([n]) * 16)
                                               for n in range(len(job.ids) // 256)]

    def flush_cache_to_disk(self, **kwargs):
        hashes = self.disk_page_cache.create(self.job.processed)
        for page, key in zip(self.job.sequences[0].allocated_pages, hashes):
            page.phash = key

    def cancel(self, job):
        self.active_jobs.remove(job)

    def iterate(self):
        raise AssertionError("Generation must never run")


class DualArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = module.PrefixStore(self.root / "ssd")
        self.disk = FakeDisk(self.root / "ram")
        self.runtime = {"model": "qwen", "template": "v1", "build": "v1"}
        self.a = list(range(1000, 2024))
        self.b = list(range(5000, 6024))

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, tokens, source):
        hashes = self.disk.create(tokens)
        self.store.publish(self.disk, tokens, source, self.runtime, hashes)
        return self.store.state["selected_base"]

    def test_a_b_a_restart_imports_both_and_exact_uses_never_prefill(self):
        self.assertEqual(self.publish(self.a, "profile-a"), "base-1")
        self.assertEqual(self.publish(self.b, "profile-b"), "base-2")
        disk = FakeDisk(self.root / "ram-after-reboot")
        store = module.PrefixStore(self.root / "ssd")
        state = store.restore(disk, self.runtime)
        self.assertEqual((state["count_ready"], state["count_total"]), (2, 2))
        self.assertEqual((len(disk.pages), len(disk.states)), (8, 2))
        self.assertEqual(state["storage_limit_bytes"], 4 * 1024**3)
        mapping = {"a": self.a, "a-user1": self.a + [101010],
                   "b": self.b, "b-user": self.b + [202020],
                   "a-user2": self.a + [303030]}
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw:
                       types.SimpleNamespace(tolist=lambda: [mapping[text]]))
        engine = FakeEngine()
        engine.disk_page_cache = disk
        container = types.SimpleNamespace(permanent_prefix_store=store, tokenizer=tokenizer,
                    generator=types.SimpleNamespace(generator=engine, jobs={}), cache_quiescing=False)
        with mock.patch.object(module, "runtime_identity", return_value=self.runtime), \
             mock.patch.object(module, "prefill_only", side_effect=AssertionError("spurious prefill")):
            for full, leading, wanted in [("a-user1", "a", "base-1"),
                                          ("b-user", "b", "base-2"),
                                          ("a-user2", "a", "base-1")]:
                result = module.auto_prepare(container, full, leading, "metadata-may-change", False)
                self.assertEqual(result["selected_base"], wanted)
                self.assertEqual(result["count_ready"], 2)

    def test_front_edit_updates_closest_a_and_keeps_b_bytes_identical(self):
        self.publish(self.a, "profile-a")
        self.publish(self.b, "profile-b")
        folder_b, manifest_b = self.store.read_manifest("base-2")
        before = {str(p.relative_to(folder_b)): p.read_bytes() for p in folder_b.rglob("*") if p.is_file()}
        changed = self.a.copy()
        changed[:5] = [88881, 88882, 88883, 88884, 88885]
        selected = self.store.select(changed, self.runtime, "changed-source-a")
        self.assertEqual((selected["id"], selected["mode"]), ("base-1", "replace"))
        self.assertEqual(self.publish(changed, "changed-source-a"), "base-1")
        folder_after, same_b = self.store.read_manifest("base-2")
        self.assertEqual(folder_after, folder_b)
        self.assertEqual(manifest_b, same_b)
        after = {str(p.relative_to(folder_after)): p.read_bytes() for p in folder_after.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.store.select(self.b, self.runtime)["mode"], "exact")

    def test_one_corrupt_checkpoint_does_not_prevent_other_base_restoration(self):
        self.publish(self.a, "profile-a")
        self.publish(self.b, "profile-b")
        folder, manifest = self.store.read_manifest("base-1")
        state_path = folder / manifest["files"][-1]["path"]
        state_path.write_bytes(state_path.read_bytes()[:-1] + b"X")
        restarted = module.PrefixStore(self.root / "ssd")
        disk = FakeDisk(self.root / "ram-after-reboot")
        result = restarted.restore(disk, self.runtime)
        self.assertEqual(result["state"], "ready")
        self.assertEqual((result["count_ready"], result["count_total"]), (1, 2))
        by_id = {entry["id"]: entry for entry in result["bases"]}
        self.assertEqual(by_id["base-1"]["state"], "error")
        self.assertEqual(by_id["base-2"]["state"], "ready")
        self.assertEqual((len(disk.pages), len(disk.states)), (4, 1))
        self.assertEqual(restarted.select(self.b, self.runtime)["mode"], "exact")
        self.assertEqual(restarted.select(self.a, self.runtime, "profile-a")["id"], "base-1")

    def test_legacy_single_slot_migrates_by_rename_without_recalculation(self):
        root = self.root / "legacy-ssd"
        legacy = module.SinglePrefixStore(root)
        hashes = self.disk.create(self.a)
        legacy.publish(self.disk, self.a, "profile-a", self.runtime, hashes)
        folder, manifest = legacy.read_manifest()
        inode = folder.stat().st_ino
        original = (folder / "manifest.json").read_bytes()
        migrated = module.PrefixStore(root)
        moved, same = migrated.read_manifest("base-1")
        self.assertEqual(moved.stat().st_ino, inode)
        self.assertEqual((moved / "manifest.json").read_bytes(), original)
        self.assertEqual(same, manifest)
        self.assertFalse((root / "current.json").exists())
        ram = FakeDisk(self.root / "legacy-ram-restored")
        migrated.restore(ram, self.runtime)
        self.assertEqual(migrated.select(self.a, self.runtime)["mode"], "exact")

    def test_missing_one_ram_page_reimports_only_its_base(self):
        self.publish(self.a, "profile-a")
        self.publish(self.b, "profile-b")
        _, ma = self.store.read_manifest("base-1")
        _, mb = self.store.read_manifest("base-2")
        key_a = bytes.fromhex(Path(ma["files"][0]["path"]).stem)
        Path(self.disk._page_path(key_a)).unlink()
        self.disk.pages.pop(key_a)
        before_b = {entry["path"]: Path(self.disk._state_path(bytes.fromhex(mb["anchor"]))).read_bytes()
                    for entry in mb["files"] if entry["path"].startswith("states/")}
        self.store.restore(self.disk, self.runtime, base_id="base-1")
        self.assertTrue(module.ram_copy_complete(self.disk, ma))
        self.assertTrue(module.ram_copy_complete(self.disk, mb))
        after_b = {entry["path"]: Path(self.disk._state_path(bytes.fromhex(mb["anchor"]))).read_bytes()
                   for entry in mb["files"] if entry["path"].startswith("states/")}
        self.assertEqual(before_b, after_b)


class PrefillTests(unittest.TestCase):
    def test_entire_aligned_base_and_no_sentinel_or_user_is_processed(self):
        engine = FakeEngine()
        tokens = list(range(1, 1025))
        hashes = module.prefill_only(engine, tokens, FakeJob, lambda x: x)
        self.assertEqual(engine.job.processed, tokens)
        self.assertEqual(engine.job.ids[-1], 0)
        self.assertEqual(engine.job.stash_position, 1024)
        self.assertEqual(engine.job.new_tokens, 0)
        self.assertEqual(len(hashes), 4)
        self.assertEqual(engine.active_jobs, [])

    def test_busy_engine_is_never_modified(self):
        engine = FakeEngine()
        existing = object()
        engine.active_jobs.append(existing)
        with self.assertRaisesRegex(module.PrefixError, "text-jobs-busy"):
            module.prefill_only(engine, list(range(512)), FakeJob, lambda x: x)
        self.assertEqual(engine.active_jobs, [existing])

    def test_artifact_publication_keeps_job_owned_until_commit_and_releases_on_error(self):
        engine = FakeEngine()
        def publish(hashes):
            self.assertEqual(engine.active_jobs, [engine.job])
            raise RuntimeError("controlled SSD failure")
        with self.assertRaisesRegex(RuntimeError, "controlled SSD failure"):
            module.prefill_only(engine, list(range(512)), FakeJob, lambda x: x, on_ready=publish)
        self.assertEqual(engine.active_jobs, [])

    def test_lcp_stops_before_first_variable_user_token_or_template_suffix(self):
        stable = list(range(800))
        leading = stable + [99999, 99998]  # standalone template ending
        first = stable + [12, 15, 17]  # first user
        second = stable + [99, 98]  # different first user
        self.assertEqual(module.strict_common_prefix(first, leading), stable[:768])
        self.assertEqual(module.strict_common_prefix(second, leading), stable[:768])

    def test_runtime_footer_does_not_change_stable_source_or_modify_request(self):
        first = [{"role": "system", "content": "Core memory and skills\n\nConversation started: 2026-10-19\nSession: A\nCWD: /one"}]
        second = [{"role": "system", "content": "Core memory and skills\n\nConversation started: 2026-10-20\nSession: B\nCWD: /two"}]
        self.assertEqual(module.stable_leading_messages(first), module.stable_leading_messages(second))
        self.assertEqual(module.stable_leading_messages(first)[0]["content"], "Core memory and skills")
        self.assertIn("Session: A", first[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
