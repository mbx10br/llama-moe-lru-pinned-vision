# SPDX-License-Identifier: AGPL-3.0-only
"""Portable Base 1 regressions using explicit native-layout fixtures.

These tests call the exported helpers and real filesystem publication directly.
They do not reproduce native pruning/serialization: the separately reported
19-test deployment qualification used unmodified EXL3 v1.6.3 AST bodies for that.
"""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from test_permanent_prefix import FakeDisk, FakeEngine, FakeJob, module, paired_stash, paired_state


class ExactAnchorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.disk = FakeDisk(self.root / "ram")
        self.tokens = list(range(1024))
        self.hashes = self.disk.create(self.tokens)
        self.anchor = self.hashes[-1]
        self.stash = paired_stash(len(self.tokens))
        self.job = types.SimpleNamespace(recurrent_state=paired_state(len(self.tokens)))
        self.engine = types.SimpleNamespace(disk_page_cache=self.disk,
            recurrent_cache=types.SimpleNamespace(get_stashed=lambda key: self.stash))

    def persist(self):
        return module.persist_exact_anchor(self.engine, self.job, self.anchor, len(self.tokens))

    def forget_anchor(self):
        self.disk.states.pop(self.anchor)
        Path(self.disk._state_path(self.anchor)).unlink()
        self.disk.writes.clear()

    def header(self):
        return self.disk._read_state_header(self.disk._state_path(self.anchor))

    def test_missing_exact_anchor_forced_once_and_writer_drained_before_validation(self):
        self.forget_anchor()
        events = []
        with mock.patch.object(self.disk, "flush", side_effect=lambda: events.append("flush")), \
             mock.patch.object(module, "validate_native_checkpoint", wraps=module.validate_native_checkpoint) as validate:
            self.persist()
            self.assertEqual(events, ["flush"])
            self.assertEqual(self.disk.writes, [(self.anchor, True)])
            validate.assert_called_once_with(self.disk, self.anchor, len(self.tokens))
        self.persist()
        self.assertEqual(len(self.disk.writes), 1)

    def test_near_capacity_rejects_without_entering_write_or_eviction(self):
        self.forget_anchor()
        self.disk.total_size = self.disk.max_size - self.stash["checkpoint_size"] - 4095
        with mock.patch.object(self.disk, "write_state", side_effect=AssertionError("write must not run")):
            with self.assertRaisesRegex(module.PrefixError, "ram-tier-space-required"):
                self.persist()
        self.assertFalse(self.disk.has_state(self.anchor))

    def test_pending_io_or_promotions_refuse_anchor_write(self):
        self.forget_anchor()
        for attribute, value in (("pending_pages", {b"busy"}), ("pending_states", {b"busy": {}}), ("_promotions", 1)):
            with self.subTest(attribute=attribute), mock.patch.object(self.disk, attribute, value):
                with self.assertRaisesRegex(module.PrefixError, "cache-io-busy"):
                    self.persist()
        self.assertEqual(self.disk.writes, [])

    def test_wrong_stash_position_size_or_side_rejected_before_write(self):
        for change in (lambda s: s.update(position=512),
                       lambda s: s.update(checkpoint_size=17),
                       lambda s: s.pop("draft"),
                       lambda s: s["draft"].update(position=512),
                       lambda s: s["target"].update(checkpoint_size=7)):
            with self.subTest(change=change):
                self.stash = paired_stash(1024)
                change(self.stash)
                with self.assertRaisesRegex(module.PrefixError, "anchor-stash-incompatible"):
                    self.persist()

    def test_live_state_position_or_side_layout_mismatch_rejected(self):
        self.job.recurrent_state.position = 512
        with self.assertRaisesRegex(module.PrefixError, "anchor-stash-incompatible"):
            self.persist()
        self.job.recurrent_state = paired_state(1024, ("target",))
        with self.assertRaisesRegex(module.PrefixError, "anchor-stash-incompatible"):
            self.persist()

    def test_dense_draft_without_recurrent_side_is_valid(self):
        self.disk.recurrent_sides = {"target": object()}
        self.stash = paired_stash(1024, ("target",))
        self.job.recurrent_state = paired_state(1024, ("target",))
        self.forget_anchor()
        self.persist()
        self.assertEqual(set(self.header()["sides"]), {"target"})

    def test_write_rejection_or_failed_serialization_never_passes(self):
        self.forget_anchor()
        with mock.patch.object(self.disk, "write_state", return_value=False):
            with self.assertRaisesRegex(module.PrefixError, "anchor-write-rejected"):
                self.persist()
        with mock.patch.object(self.disk, "write_state", return_value=True):
            with self.assertRaisesRegex(module.PrefixError, "state-incompatible"):
                self.persist()

    def test_header_rejects_wrong_runtime_anchor_position_side_or_layer(self):
        original = self.header()
        mutations = (
            lambda h: h.update(fingerprint="other-runtime"),
            lambda h: h.update(phash="00" * 16),
            lambda h: h.update(position=512),
            lambda h: h["top"].update(position=512),
            lambda h: h["top"].update(checkpoint_size=17),
            lambda h: h["sides"].pop("draft"),
            lambda h: h["sides"]["draft"]["ints"].update(position=512),
            lambda h: h["sides"]["draft"]["layers"].update({"unknown-layer": {}}),
            lambda h: h["sides"]["target"].update(image_offset=-1),
            lambda h: h["sides"]["target"]["layers"]["target-layer"]["tensors"][0].update(nbytes=9),
        )
        for change in mutations:
            with self.subTest(change=change):
                header = copy.deepcopy(original)
                change(header)
                self.disk.save_state_header(self.anchor, header)
                with self.assertRaisesRegex(module.PrefixError, "state-incompatible"):
                    self.persist()

    def test_truncated_payload_refused_even_with_index_entry(self):
        path = Path(self.disk._state_path(self.anchor))
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(module.PrefixError, "state-incompatible"):
            self.persist()

    def test_prefill_orders_global_flush_then_anchor_write_then_commit_while_job_owned(self):
        engine = FakeEngine()
        original_flush = engine.flush_cache_to_disk
        events = []
        def prune(**kwargs):
            original_flush(**kwargs)
            key = engine.job.sequences[0].allocated_pages[-1].phash
            engine.disk_page_cache.states.pop(key)
            Path(engine.disk_page_cache._state_path(key)).unlink()
            engine.disk_page_cache.writes.clear()
            events.append("global-flush-prune")
        def commit(hashes):
            self.assertEqual(engine.active_jobs, [engine.job])
            self.assertEqual(engine.disk_page_cache.writes, [(hashes[-1], True)])
            module.validate_native_checkpoint(engine.disk_page_cache, hashes[-1], 1024)
            events.append("commit")
        with mock.patch.object(engine, "flush_cache_to_disk", side_effect=prune), \
             mock.patch.object(engine.disk_page_cache, "flush", side_effect=lambda: events.append("writer-flush")):
            module.prefill_only(engine, self.tokens, FakeJob, lambda ids: ids, on_ready=commit)
        self.assertEqual(events, ["global-flush-prune", "writer-flush", "commit"])
        self.assertEqual(engine.active_jobs, [])
        self.assertEqual(engine.job.processed, self.tokens)


class ImmutablePublicationTests(unittest.TestCase):
    def setUp(self):
        ExactAnchorTests.setUp(self)
        self.store = module.SinglePrefixStore(self.root / "protected")
        self.runtime = {"model": "fixture-runtime"}

    def publish(self):
        return self.store.publish(self.disk, self.tokens, "fixture-source", self.runtime, self.hashes)

    def snapshot(self):
        folder, manifest = self.store.read_manifest()
        return folder, manifest, (self.store.root / "current.json").read_bytes()

    def assert_unchanged(self, snapshot):
        folder, manifest, pointer = snapshot
        self.assertEqual((self.store.root / "current.json").read_bytes(), pointer)
        for entry in manifest["files"]:
            self.assertEqual(module.file_digest(folder / entry["path"]), entry["sha256"])

    def test_same_semantics_publish_fresh_immutable_generation(self):
        self.publish()
        old = self.snapshot()
        original = self.store.atomic_json
        def check_old_before_pointer(path, value, **kwargs):
            if path.name == "current.json":
                self.assert_unchanged(old)
            return original(path, value, **kwargs)
        with mock.patch.object(self.store, "atomic_json", side_effect=check_old_before_pointer):
            self.publish()
        self.assertNotEqual(self.store.read_manifest()[0], old[0])
        # The old artifact is only GC'd after the new pointer is committed.
        self.assertEqual(self.store.read_manifest()[1]["token_digest"], old[1]["token_digest"])

    def test_copy_error_preserves_selected_generation(self):
        self.publish()
        old = self.snapshot()
        with mock.patch.object(module.shutil, "copyfile", side_effect=OSError("controlled copy failure")):
            with self.assertRaises(OSError):
                self.publish()
        self.assert_unchanged(old)

    def test_pointer_error_preserves_same_semantics_selected_generation(self):
        self.publish()
        old = self.snapshot()
        original = self.store.atomic_json
        def fail_pointer(path, value, **kwargs):
            if path.name == "current.json":
                raise OSError("controlled pointer failure")
            return original(path, value, **kwargs)
        with mock.patch.object(self.store, "atomic_json", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                self.publish()
        self.assert_unchanged(old)

    def test_fsync_error_before_pointer_preserves_selected_generation(self):
        self.publish()
        old = self.snapshot()
        with mock.patch.object(module.os, "fsync", side_effect=OSError("controlled sync failure")):
            with self.assertRaises(OSError):
                self.publish()
        self.assert_unchanged(old)
