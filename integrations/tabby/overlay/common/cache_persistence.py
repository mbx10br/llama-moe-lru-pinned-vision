# SPDX-License-Identifier: AGPL-3.0-only
# Local cache integration, exported 2026-10-20; see ../../LICENSE.
"""Private Unix control for a bounded, process-independent EXL3 cache.

No TCP endpoint, prompt text, token IDs or cache keys are exposed. The GPU
scheduler must drain its text leases before requesting flush-and-quiesce.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import stat
import struct


def persistence_kwargs(memory):
    """Validate private RAM storage before giving its path to the engine."""
    directory = memory.disk_cache_dir
    if not directory:
        return {}
    if not memory.sysmem_kv_cache or not memory.cache_control_socket:
        raise ValueError("Persistent cache requires the CPU KV tier and private control socket")
    destination = Path(directory)
    if not destination.is_absolute():
        raise ValueError("Cache path must be absolute")
    root = destination.parent
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Cache mount must be a private 0700 directory")
    if info.st_uid != os.getuid():
        raise ValueError("Cache mount must belong to the backend user")
    if destination.is_symlink():
        raise ValueError("Cache directory cannot be a symlink")
    # Verify the exact mountpoint, not just a parent filesystem. A missing
    # systemd mount must never silently turn this cache into writes on /run or /.
    dedicated_tmpfs = False
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        record, separator, filesystem = line.partition(" - ")
        fields = record.split()
        if separator and len(fields) > 4:
            mountpoint = fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
            if mountpoint == str(root.resolve()) and filesystem.split()[0] == "tmpfs":
                dedicated_tmpfs = True
                break
    if not dedicated_tmpfs:
        raise ValueError("Persistent cache must be on the dedicated tmpfs")
    capacity = os.statvfs(root)
    budget = memory.disk_cache_size * 1024**2
    if capacity.f_blocks * capacity.f_frsize > 8 * 1024**3:
        raise ValueError("Cache tmpfs must have a hard limit of at most 8 GiB")
    if budget + 128 * 1024**2 > capacity.f_blocks * capacity.f_frsize:
        raise ValueError("Cache budget must leave room for filesystem/index metadata")
    if memory.disk_cache_checkpoint_interval % 256:
        raise ValueError("Persistent checkpoint spacing must be a multiple of 256 tokens")
    return {
        "disk_cache_dir": directory,
        "disk_cache_size": budget,
        "disk_cache_flush_on_idle": True,
        "disk_cache_checkpoint_interval": memory.disk_cache_checkpoint_interval,
    }


class CacheHandoffError(RuntimeError):
    """Fixed, non-sensitive reason code safe for administrative diagnostics."""


class CacheControl:
    def __init__(self, container_getter):
        self.container_getter = container_getter
        self.server = None
        self.path = None
        self.lock = asyncio.Lock()
        self.release_handle = None

    def resume(self):
        if self.release_handle is not None:
            self.release_handle.cancel()
            self.release_handle = None
        container = self.container_getter()
        if container is not None:
            container.cache_quiescing = False

    def resume_lease(self, handle):
        """An old ACK/callback cannot release a newer administrative lease."""
        if handle is not None and self.release_handle is handle:
            self.resume()

    @staticmethod
    def flush(container):
        if container is None or not container.loaded or container.load_lock.locked():
            raise CacheHandoffError("backend-not-ready")
        wrapper = container.generator
        engine = getattr(wrapper, "generator", None)
        disk = getattr(engine, "disk_page_cache", None)
        if engine is None or disk is None or getattr(wrapper, "error", None) is not None:
            raise CacheHandoffError("persistent-cache-unavailable")
        previous = getattr(container, "cache_quiescing", False)
        container.cache_quiescing = True
        try:
            if wrapper.jobs or engine.active_jobs or engine.pending_jobs:
                raise CacheHandoffError("text-jobs-busy")
            if disk.write_disabled or disk.metrics.get("write_errors", 0):
                raise CacheHandoffError("cache-writer-fault")
            skipped = disk.metrics.get("skipped_low_space", 0)
            # Synchronous on the generator's event-loop thread, like iterate().
            # No enqueue can interleave with these GPU copies or the final check.
            result = engine.flush_cache_to_disk(wait=True)
            if (disk.write_disabled or disk.metrics.get("write_errors", 0)
                    or disk.metrics.get("skipped_low_space", 0) != skipped
                    or disk.pending_pages or disk.pending_states or result.get("more")):
                raise CacheHandoffError("cache-flush-incomplete")
            stats = engine.get_cache_stats()
            return {
                "ok": True, "pid": os.getpid(), "quiesced": True,
                "persisted": True,
                "disk_pages": stats.get("disk_pages", 0),
                "disk_states": stats.get("disk_states", 0),
                "disk_used_bytes": stats.get("disk_used_bytes", 0),
                "disk_max_bytes": stats.get("disk_max_bytes", 0),
            }
        except BaseException:
            container.cache_quiescing = previous
            raise

    async def start(self, path):
        self.path = Path(path)
        parent = self.path.parent
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.getuid()):
            raise ValueError("Control directory must belong to backend user with mode 0700")
        # RuntimeDirectory belongs to this service. Refuse replacing anything
        # except a stale socket owned by the same user.
        if self.path.exists() or self.path.is_symlink():
            info = self.path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("Unexpected file at cache control socket path")
            self.path.unlink()
        self.server = await asyncio.start_unix_server(self.handle, path=str(self.path), limit=2048)
        os.chmod(self.path, 0o600)

    async def handle(self, reader, writer):
        response_release_handle = None
        try:
            peer = writer.get_extra_info("socket").getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            _, uid, _ = struct.unpack("3i", peer)
            if uid not in (0, os.getuid()):
                raise PermissionError("Administrative peer required")
            raw = await asyncio.wait_for(reader.readline(), timeout=5)
            request = json.loads(raw)
            async with self.lock:
                container = self.container_getter()
                if request == {"action": "flush-and-quiesce"}:
                    response = self.flush(container)
                    # An abandoned administrative client must not freeze a
                    # healthy text backend forever. Normal handoff sends
                    # SIGTERM immediately after ACK; the listener then closes.
                    if self.release_handle is not None:
                        self.release_handle.cancel()
                    handle = asyncio.get_running_loop().call_later(60, lambda: self.resume_lease(handle))
                    self.release_handle = handle
                    response_release_handle = handle
                elif request == {"action": "resume"}:
                    self.resume()
                    response = {"ok": True, "pid": os.getpid(), "quiesced": False}
                elif request == {"action": "status"}:
                    engine = getattr(getattr(container, "generator", None), "generator", None)
                    disk = getattr(engine, "disk_page_cache", None)
                    recurrent = getattr(engine, "recurrent_cache", None)
                    tier = getattr(engine, "cpu_page_cache", None)
                    response = {
                        "ok": True, "pid": os.getpid(),
                        "quiesced": getattr(container, "cache_quiescing", False),
                        "recurrent_bytes": getattr(recurrent, "current_size", 0),
                        "recurrent_checkpoints": len(recurrent) if recurrent is not None else 0,
                        "cpu_reserved_bytes": tier.max_slots * tier.slot_size if tier is not None else 0,
                        "cpu_occupied_bytes": len(tier.entries) * tier.slot_size if tier is not None else 0,
                        "disk_metrics": {key: value for key, value in getattr(disk, "metrics", {}).items()
                                         if isinstance(value, (int, float))},
                        "permanent_prefix": dict(getattr(getattr(container, "permanent_prefix_store", None),
                                                          "state", {"state": "disabled"})),
                    }
                else:
                    raise ValueError("Unsupported cache control action")
        except Exception as error:
            # Fixed exception class, never serialize prompts, hashes or engine internals.
            response = {"ok": False, "pid": os.getpid(), "error": str(error) if isinstance(error, CacheHandoffError) else type(error).__name__}
        try:
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except ConnectionError:
            # For example the root helper timed out before flush completed.
            # STATUS carries quiesced=True too, but never owns the lease.
            self.resume_lease(response_release_handle)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def close(self):
        if self.release_handle is not None:
            self.release_handle.cancel()
            self.release_handle = None
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        if self.path is not None:
            self.path.unlink(missing_ok=True)
