# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only checks of the scheduler's cache handoff and private control channel."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


control = load("candidate_cache_control", ROOT / "overlay/common/cache_persistence.py")


def fixture():
    disk = SimpleNamespace(
        write_disabled=False, metrics={"write_errors": 0, "skipped_low_space": 0},
        pending_pages=set(), pending_states={},
    )
    engine = SimpleNamespace(
        disk_page_cache=disk, active_jobs=[], pending_jobs=[],
        flush_cache_to_disk=lambda **kw: {"pages": 1, "states": 1},
        get_cache_stats=lambda: {"disk_pages": 1, "disk_states": 1,
                                 "disk_used_bytes": 1024, "disk_max_bytes": 2048,
                                 "private_prompt": "must never leave the process"},
    )
    return SimpleNamespace(loaded=True, load_lock=asyncio.Lock(), cache_quiescing=False,
                           generator=SimpleNamespace(generator=engine, error=None, jobs={}))


class FlushTests(unittest.TestCase):
    def test_success_quiesces_and_returns_only_counts(self):
        container = fixture()
        result = control.CacheControl.flush(container)
        self.assertTrue(container.cache_quiescing)
        self.assertTrue(result["persisted"])
        self.assertNotIn("private_prompt", result)

    def test_busy_jobs_are_not_cancelled_and_quiesce_is_reverted(self):
        for field in ("active_jobs", "pending_jobs"):
            with self.subTest(field=field):
                container = fixture()
                setattr(container.generator.generator, field, [object()])
                with self.assertRaises(RuntimeError):
                    control.CacheControl.flush(container)
                self.assertFalse(container.cache_quiescing)
                self.assertEqual(len(getattr(container.generator.generator, field)), 1)

    def test_latched_engine_or_missing_disk_never_acks(self):
        for fault in ("latched", "missing"):
            container = fixture()
            if fault == "latched":
                container.generator.error = RuntimeError("bad numerical state")
            else:
                container.generator.generator.disk_page_cache = None
            with self.assertRaises(RuntimeError):
                control.CacheControl.flush(container)
            self.assertFalse(container.cache_quiescing)

    def test_write_errors_pending_or_low_space_never_ack(self):
        for fault in ("write", "pending", "space", "more", "disabled"):
            with self.subTest(fault=fault):
                container = fixture()
                engine = container.generator.generator
                def fail(**kw):
                    if fault == "write":
                        engine.disk_page_cache.metrics["write_errors"] += 1
                    elif fault == "pending":
                        engine.disk_page_cache.pending_pages.add(b"private hash")
                    elif fault == "space":
                        engine.disk_page_cache.metrics["skipped_low_space"] += 1
                    elif fault == "disabled":
                        engine.disk_page_cache.write_disabled = True
                    return {"more": fault == "more"}
                engine.flush_cache_to_disk = fail
                with self.assertRaises(RuntimeError):
                    control.CacheControl.flush(container)
                self.assertFalse(container.cache_quiescing)

    def test_tmpfs_guard_rejects_root_disk_and_unbounded_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            memory = SimpleNamespace(disk_cache_dir=directory + "/cache", sysmem_kv_cache=6144,
                                     cache_control_socket="/run/private.sock", disk_cache_size=7936,
                                     disk_cache_checkpoint_interval=4096)
            with self.assertRaisesRegex(ValueError, "tmpfs"):
                control.persistence_kwargs(memory)

    def test_tmpfs_budget_is_bounded_and_keeps_metadata_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            memory = SimpleNamespace(disk_cache_dir=directory + "/cache", sysmem_kv_cache=6144,
                                     cache_control_socket="/run/private.sock", disk_cache_size=7936,
                                     disk_cache_checkpoint_interval=4096)
            mountinfo = f"123 1 0:1 / {directory} rw - tmpfs tmpfs rw\n"
            filesystem = SimpleNamespace(f_blocks=8 * 1024**3 // 4096, f_frsize=4096)
            with patch.object(control.Path, "read_text", return_value=mountinfo), \
                 patch.object(control.os, "statvfs", return_value=filesystem):
                self.assertEqual(control.persistence_kwargs(memory)["disk_cache_size"], 7936 * 1024**2)
                memory.disk_cache_size = 8192
                with self.assertRaisesRegex(ValueError, "metadata"):
                    control.persistence_kwargs(memory)
                memory.disk_cache_size = 7936
                filesystem.f_blocks = 9 * 1024**3 // 4096
                with self.assertRaisesRegex(ValueError, "hard limit"):
                    control.persistence_kwargs(memory)


class SocketTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def peer(action, drain=None):
        writer = MagicMock()
        writer.get_extra_info.return_value.getsockopt.return_value = struct.pack("3i", 123, os.getuid(), os.getgid())
        async def complete():
            pass
        writer.drain.side_effect = drain or complete
        writer.wait_closed.side_effect = complete
        reader = MagicMock()
        async def read():
            return json.dumps({"action": action}).encode() + b"\n"
        reader.readline.side_effect = read
        return reader, writer

    async def test_lost_status_ack_cannot_resume_a_live_flush_lease(self):
        container = fixture()
        server = control.CacheControl(lambda: container)
        reader, writer = self.peer("flush-and-quiesce")
        await server.handle(reader, writer)
        lease = server.release_handle
        async def broken():
            raise BrokenPipeError("status client disconnected")
        reader, writer = self.peer("status", broken)
        await server.handle(reader, writer)
        self.assertTrue(container.cache_quiescing)
        self.assertIs(server.release_handle, lease)
        self.assertFalse(lease.cancelled())
        await server.close()
        self.assertTrue(lease.cancelled())
        self.assertIsNone(server.release_handle)

    async def test_stale_flush_ack_and_expiry_cannot_release_newer_lease(self):
        container = fixture()
        server = control.CacheControl(lambda: container)
        ack_started = asyncio.Event()
        lose_old_ack = asyncio.Event()
        async def late_broken():
            ack_started.set()
            await lose_old_ack.wait()
            raise BrokenPipeError("old flush client disconnected late")
        reader, writer = self.peer("flush-and-quiesce", late_broken)
        task_a = asyncio.create_task(server.handle(reader, writer))
        await ack_started.wait()
        lease_a = server.release_handle
        reader, writer = self.peer("flush-and-quiesce")
        await server.handle(reader, writer)
        lease_b = server.release_handle
        self.assertIsNot(lease_a, lease_b)
        self.assertTrue(lease_a.cancelled())
        lose_old_ack.set()
        await task_a
        self.assertTrue(container.cache_quiescing)
        self.assertIs(server.release_handle, lease_b)
        self.assertFalse(lease_b.cancelled())
        # Even an already-dispatched callback from A cannot resume B.
        server.resume_lease(lease_a)
        self.assertTrue(container.cache_quiescing)
        self.assertIs(server.release_handle, lease_b)
        # The current lease's own expiry still restores service and cleans up.
        server.resume_lease(lease_b)
        self.assertFalse(container.cache_quiescing)
        self.assertIsNone(server.release_handle)
        self.assertTrue(lease_b.cancelled())
        await server.close()

    async def test_lost_flush_ack_does_not_leave_backend_quiesced(self):
        container = fixture()
        server = control.CacheControl(lambda: container)
        writer = MagicMock()
        writer.get_extra_info.return_value.getsockopt.return_value = struct.pack("3i", 123, os.getuid(), os.getgid())
        async def broken():
            raise BrokenPipeError("root helper disconnected")
        writer.drain.side_effect = broken
        writer.wait_closed.side_effect = broken
        reader = MagicMock()
        async def read():
            return b'{"action":"flush-and-quiesce"}\n'
        reader.readline.side_effect = read
        await server.handle(reader, writer)
        self.assertFalse(container.cache_quiescing)
        self.assertIsNone(server.release_handle)

    async def test_untrusted_uid_cannot_flush(self):
        container = fixture()
        server = control.CacheControl(lambda: container)
        writer = MagicMock()
        writer.get_extra_info.return_value.getsockopt.return_value = struct.pack("3i", 123, 65432, 65432)
        async def complete():
            pass
        writer.drain.side_effect = complete
        writer.wait_closed.side_effect = complete
        reader = MagicMock()
        await server.handle(reader, writer)
        reader.readline.assert_not_called()
        self.assertFalse(container.cache_quiescing)
        response = json.loads(writer.write.call_args.args[0])
        self.assertEqual(response["error"], "PermissionError")

    async def test_private_unix_socket_round_trip_and_no_tcp_listener(self):
        container = fixture()
        with tempfile.TemporaryDirectory() as directory:
            os.chmod(directory, 0o700)
            path = Path(directory) / "control.sock"
            server = control.CacheControl(lambda: container)
            await server.start(path)
            try:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(server.server.sockets[0].family, socket.AF_UNIX)
                async def request(action):
                    reader, writer = await asyncio.open_unix_connection(path)
                    writer.write(json.dumps({"action": action}).encode() + b"\n")
                    await writer.drain()
                    result = json.loads(await reader.readline())
                    writer.close()
                    await writer.wait_closed()
                    return result
                self.assertTrue((await request("flush-and-quiesce"))["persisted"])
                self.assertTrue(container.cache_quiescing)
                self.assertFalse((await request("resume"))["quiesced"])
                self.assertFalse(container.cache_quiescing)
                status = await request("status")
                self.assertEqual(status["recurrent_bytes"], 0)
                self.assertIsInstance(status["disk_metrics"], dict)
                self.assertNotIn("private_prompt", status)
                self.assertFalse((await request("unsupported"))["ok"])
            finally:
                await server.close()
            self.assertFalse(path.exists())



if __name__ == "__main__":unittest.main()
