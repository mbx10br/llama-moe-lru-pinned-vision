#!/usr/bin/env python3
"""Capture the real first TUI/Web request with a fake, tool-free model response.

Run in a fresh process in the SAME image, environment and working directory as
the intended TUI. This is not a provider: the configured model/URL stay intact.
Network, tool dispatch and non-temporary SQLite opens are denied. Subprocesses
are limited to exact, inspected read-only environment/git metadata probes.
Only the final HTTP JSON body is saved, never authentication headers/config.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sqlite3.dbapi2
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch
from urllib.parse import unquote, urlsplit
import uuid


class CaptureDenied(RuntimeError):
    pass


def private_json(path: Path, value) -> None:
    """Never create a world-readable intermediate containing prompt memories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    os.chmod(path, 0o600)


class FakeModel:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        import httpx
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            raise CaptureDenied("Only the primary chat-completions POST is permitted")
        body = json.loads(request.content)
        if not isinstance(body.get("messages"), list):
            raise CaptureDenied("Missing chat messages")
        if self.requests:
            raise CaptureDenied("A second model call would invalidate the first-turn capture")
        self.requests.append(body)
        model = str(body.get("model", "capture"))
        common = {"id": "chatcmpl-offline-capture", "created": int(time.time()), "model": model}
        if body.get("stream"):
            chunks = [
                {**common, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]},
                {**common, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            wire = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={**common, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def readonly_probe(argv) -> bool:
    """Only exact, inspected Hermes environment/git metadata probes; never a shell."""
    if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, (str, os.PathLike)) for a in argv):
        return False
    command = [os.fsdecode(a) for a in argv]
    executable = Path(command[0]).name
    args = command[1:]
    if executable == "git":
        if args[:1] == ["-C"] and len(args) >= 3:
            args = args[2:]
        return tuple(args) in {
            ("config", "--system", "-z", "--get-all", "safe.directory"),
            ("config", "--global", "-z", "--get-all", "safe.directory"),
            ("rev-parse", "HEAD"), ("rev-parse", "--show-toplevel"),
            ("rev-parse", "--git-dir"), ("rev-parse", "--git-common-dir"),
            ("rev-parse", "--short=8", "HEAD"), ("rev-parse", "--short=8", "origin/main"),
            ("remote", "get-url", "origin"), ("describe", "--tags", "--abbrev=0"),
            ("status", "--porcelain=2", "--branch"), ("log", "-3", "--pretty=%h %s"),
        }
    if executable in ("python", "python3"):
        return args == ["-m", "pip", "--version"] or args in [
            ["-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            ["-c", "import os; print('yes' if os.path.exists(os.path.join(os.path.dirname(os.__file__), 'EXTERNALLY-MANAGED')) else 'no')"],
        ]
    return (executable, tuple(args)) in {("pip", ("--version",)), ("npx", ("--version",)), ("node", ("--version",)), ("npm", ("--version",)), ("uname", ("-p",)), ("uname", ("-m",))}


@contextmanager
def safety_fence(temporary_root: Path):
    """Allow SQLite only in the disposable capture directory; forbid all I/O execution."""
    real_connect = sqlite3.connect
    real_popen = subprocess.Popen
    violations = []
    temporary_root = temporary_root.resolve()

    def denied(*args, **kwargs):
        violations.append("blocked_optional_network:" + ">".join(frame.name for frame in traceback.extract_stack(limit=8)[:-1]))
        raise CaptureDenied("Network and subprocess execution are disabled during capture")

    def popen(args, *positional, **kwargs):
        if not kwargs.get("shell") and readonly_probe(args):
            # Optional index refresh cannot write even during git status.
            kwargs["env"] = {**(kwargs.get("env") or os.environ), "GIT_OPTIONAL_LOCKS": "0"}
            return real_popen(args, *positional, **kwargs)
        executable = Path(os.fsdecode(args[0])).name if isinstance(args, (list, tuple)) and args else "shell"
        violations.append("forbidden_subprocess:" + executable + ":" + ">".join(frame.name for frame in traceback.extract_stack(limit=9)[:-1]))
        raise CaptureDenied("Only inspected read-only metadata probes may execute")

    def deny_execution(*args, **kwargs):
        violations.append("forbidden_subprocess")
        raise CaptureDenied("Shell execution is prohibited during capture")

    def connect(database, *args, **kwargs):
        value = os.fsdecode(database)
        if value != ":memory:":
            if value.startswith("file:"):
                value = unquote(urlsplit(value).path)
            if not Path(value).resolve().is_relative_to(temporary_root):
                violations.append("non_temporary_database")
                raise CaptureDenied("A non-temporary SQLite database was requested")
        return real_connect(database, *args, **kwargs)

    with ExitStack() as stack:
        for name in ("connect", "connect_ex"):
            stack.enter_context(patch.object(socket.socket, name, denied))
        stack.enter_context(patch.object(socket, "create_connection", denied))
        stack.enter_context(patch.object(socket, "getaddrinfo", denied))
        stack.enter_context(patch.object(subprocess, "Popen", popen))
        stack.enter_context(patch.object(os, "system", deny_execution))
        stack.enter_context(patch.object(sqlite3, "connect", connect))
        stack.enter_context(patch.object(sqlite3.dbapi2, "connect", connect))
        yield violations


def register_cached_mcp(config: dict, allow_expired_snapshot: bool = False, snapshot_not_before: float | None = None, expected_total_tools: int | None = None) -> dict:
    """Use Hermes's native lazy path: exact filters, schemas, trust and provenance."""
    from tools.mcp_schema_cache import config_fingerprint, get_cached_entry, _load_all, _cache_path
    from tools.mcp_tool_config import _load_mcp_config
    from tools.mcp_tool_registration import _register_from_cache_sync
    result = {}
    snapshot_entries = _load_all()
    configured = _load_mcp_config()
    if allow_expired_snapshot:
        if snapshot_not_before is None or expected_total_tools is None:
            raise CaptureDenied("TTL0 inspection requires an explicit restart boundary and independently confirmed tool count")
        configured_names = [n for n, v in configured.items() if isinstance(v, dict) and v.get("enabled") is not False]
        count = sum(len(snapshot_entries.get(n, {}).get("tools", [])) for n in configured_names)
        if count != expected_total_tools or _cache_path().stat().st_mtime < snapshot_not_before:
            raise CaptureDenied("MCP snapshot does not match the expected live restart/count evidence")
    for name, server in configured.items():
        if not isinstance(server, dict) or server.get("enabled") is False:
            continue
        fingerprint = config_fingerprint(server)
        entry = get_cached_entry(name, fingerprint)
        expired = entry is None
        if entry is None and allow_expired_snapshot:
            snapshot = snapshot_entries.get(name)
            if isinstance(snapshot, dict) and snapshot.get("fingerprint") == fingerprint and snapshot.get("ttl_ms") == 0 and float(snapshot.get("written_at", 0)) >= snapshot_not_before:
                entry = snapshot
        if entry is None:
            raise CaptureDenied(f"No valid installed MCP schema cache for {name!r}; refresh it normally first")
        names = _register_from_cache_sync(name, server, entry)
        result[name] = {"fingerprint": fingerprint, "names": names, "source": "validated_live_restart_snapshot" if expired else "native_lazy_schema_cache", "expired_snapshot_explicitly_allowed": expired, "ttl_ms": entry.get("ttl_ms"), "written_at": entry.get("written_at"), "restart_boundary": snapshot_not_before if expired else None, "runtime_cache_ttl_unchanged": True}
    return result


def core_hashes(repo: Path) -> dict:
    names = ("run_agent.py", "agent/agent_init.py", "agent/system_prompt.py", "agent/turn_context.py", "agent/turn_api_request.py", "tui_gateway/server.py", "hermes_cli/plugins.py", "tools/tool_search.py", "tools/tool_search_catalog.py")
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names if (repo / name).is_file()}


@contextmanager
def actual_profile_scope():
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    home = get_hermes_home().resolve()
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def capture(repo: Path, output: Path, metadata: Path, user_message: str, required_markers: list[str], allow_expired_mcp_snapshot: bool = False, mcp_snapshot_not_before: float | None = None, expected_mcp_tools: int | None = None) -> dict:
    import httpx
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    # Clear dispatcher identity BEFORE importing any registry or configuration code.
    removed_kanban = sorted(name for name in os.environ if name.startswith("HERMES_KANBAN_"))
    for name in removed_kanban:
        os.environ.pop(name, None)
    fake = FakeModel()
    with tempfile.TemporaryDirectory(prefix="hermes-payload-capture-") as temporary:
        temporary_root = Path(temporary)
        with safety_fence(temporary_root) as violations, ExitStack() as patches:
            # AIAgent resolves its OWN home from the DB parent unless the public
            # profile ContextVar is explicitly bound. A temporary DB alone loses
            # SOUL/skills, even when the HERMES_HOME environment variable is real.
            real_home = patches.enter_context(actual_profile_scope())
            import hermes_state
            from hermes_state import SessionDB
            import tools.async_delegation as async_delegation
            # The global process registry restores its async ledger during import.
            # Redirect it BEFORE importing run_agent/server, not just the agent DB.
            temporary_db = temporary_root / "state.db"
            patches.enter_context(patch.object(hermes_state, "_default_db_path", lambda: temporary_db))
            patches.enter_context(patch.object(async_delegation, "_db_path", lambda: temporary_db))
            secure_db_files = hermes_state._secure_state_db_files

            def secure_temporary_db(path, *args, **kwargs):
                if not Path(path).resolve().is_relative_to(temporary_root):
                    violations.append("non_temporary_database_hardening")
                    raise CaptureDenied("Non-temporary DB hardening/create was requested")
                return secure_db_files(path, *args, **kwargs)

            patches.enter_context(patch.object(hermes_state, "_secure_state_db_files", secure_temporary_db))
            import hermes_cli.banner as banner
            # This optional UI update badge is unrelated to prompt construction.
            patches.enter_context(patch.object(banner, "prefetch_update_check", lambda: None))
            from run_agent import AIAgent
            from tui_gateway import server
            import agent.turn_tool_round as tool_round
            import hermes_cli.mcp_startup as mcp_startup

            def no_tool(*args, **kwargs):
                violations.append("tool_dispatch")
                raise CaptureDenied("Tool execution is prohibited during payload capture")

            patches.enter_context(patch.object(tool_round, "run_tool_round", no_tool))
            patches.enter_context(patch.object(mcp_startup, "start_background_mcp_discovery", no_tool))
            patches.enter_context(patch.object(AIAgent, "_build_keepalive_http_client", staticmethod(lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(fake)))))
            cfg = server._load_cfg()
            mcp = register_cached_mcp(cfg, allow_expired_mcp_snapshot, mcp_snapshot_not_before, expected_mcp_tools)
            sid = "capture-" + uuid.uuid4().hex
            db = SessionDB(db_path=temporary_db)
            agent = None
            try:
                agent = server._make_agent(sid, sid, session_id=sid, session_db=db, platform_override="tui", context_cwd_is_launch_artifact=False)
                from agent.system_prompt import _agent_home
                if _agent_home(agent).resolve() != real_home:
                    raise CaptureDenied("The real agent profile home was replaced by the disposable DB home")
                if getattr(agent, "api_mode", "chat_completions") not in (None, "chat_completions"):
                    raise CaptureDenied("This capture supports only the actual OpenAI chat-completions route")
                agent.skip_background_review = True
                # Suppress fake-turn debug/artifact exports without changing prompt or
                # persistence flags (pre_llm hooks still run and DB persistence is real).
                agent.logs_dir = temporary_root / "sessions"
                agent.logs_dir.mkdir()
                agent._save_session_log = lambda *args, **kwargs: None
                agent._dump_api_request_debug = lambda *args, **kwargs: None
                agent._save_trajectory = lambda *args, **kwargs: None
                # Existing cached schemas were registered before the constructor; do not launch discovery.
                agent._skip_mcp_refresh = True
                if getattr(agent, "_persist_disabled", False):
                    raise CaptureDenied("Persistence-disabled agents omit real pre_llm context")
                result = agent.run_conversation(user_message=user_message, conversation_history=None, task_id=sid)
                if len(fake.requests) != 1 or result.get("final_response", "").strip() != "OK":
                    raise CaptureDenied("The real turn did not finish with exactly one mocked OK response")
                payload = fake.requests[0]
                system = "\n\n".join(str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "system")
                user = "\n\n".join(str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "user")
                date_pos = system.rfind("Conversation started:")
                for marker in required_markers:
                    if marker not in system or date_pos < 0 or system.index(marker) >= date_pos:
                        raise CaptureDenied("A required stable plugin marker is missing before Conversation started")
                    if marker in user:
                        raise CaptureDenied("A stable plugin section was duplicated in the user message")
                names = [tool.get("function", tool).get("name", "") for tool in payload.get("tools", [])]
                if "You have been assigned ONE task from the shared board" in system:
                    raise CaptureDenied("A dispatcher task protocol leaked into normal TUI system context")
                # A blocked attempt is recorded even if Hermes caught the exception; refuse false success.
                if any(not entry.startswith("blocked_optional_network:") for entry in violations):
                    raise CaptureDenied("Capture encountered forbidden I/O; no payload was published: " + repr(violations))
                from hermes_constants import get_hermes_home
                plugin = get_hermes_home() / "plugins" / "superpowers" / "__init__.py"
                from agent import prompt_builder
                soul = prompt_builder.load_soul_md(getattr(getattr(agent, "context_compressor", None), "context_length", None), home_override=real_home)
                if soul and soul not in system:
                    raise CaptureDenied("The actual profile SOUL identity is missing from the final system")
                loaded_memory = {}
                if getattr(agent, "_memory_store", None):
                    for kind, enabled in (("memory", agent._memory_enabled), ("user", agent._user_profile_enabled)):
                        block = agent._memory_store.format_for_system_prompt(kind) if enabled else None
                        loaded_memory[kind] = {"enabled": enabled, "rendered_chars": len(block or ""), "present_in_system": bool(block and block in system)}
                        if block and block not in system:
                            raise CaptureDenied("An enabled actual profile memory block is absent from the final system")
                profile_files = {}
                for relative in ("SOUL.md", "memories/MEMORY.md", "memories/USER.md"):
                    file = real_home / relative
                    if file.is_file():
                        raw = file.read_bytes()
                        profile_files[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                info = {
                    "capture_version": 1, "created_at": int(time.time()), "factory": "tui_gateway.server._make_agent",
                    "platform": str(agent.platform), "model": str(agent.model), "provider": str(agent.provider),
                    "api_mode": str(agent.api_mode), "configured_route_preserved": True,
                    "context_length": getattr(getattr(agent, "context_compressor", None), "context_length", None),
                    "network_requests": 0, "model_calls_mocked": 1, "tool_executions": 0,
                    "blocked_optional_network_probes": sum(v.startswith("blocked_optional_network:") for v in violations),
                    "readonly_environment_probes_allowed": True,
                    "session_db": "disposable_temporary", "live_database_opened": False,
                    "cwd": str(Path.cwd()), "hermes_home": str(get_hermes_home()),
                    "agent_home_explicitly_bound": str(_agent_home(agent)), "actual_profile_files": profile_files,
                    "soul_rendered_chars": len(soul or ""), "soul_present_in_system": bool(soul and soul in system),
                    "loaded_memory_blocks": loaded_memory,
                    "removed_dispatcher_env_names": removed_kanban, "tool_names": names,
                    "mcp": mcp, "system_chars": len(system), "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                    "stable_plugin_markers_checked": required_markers, "core_hashes": core_hashes(repo),
                    "superpowers_plugin_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest() if plugin.is_file() else None,
                    "payload_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                }
                private_json(output, payload)
                private_json(metadata, info)
                return info
            finally:
                if agent is not None:
                    agent.close()
                db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--user-message", default="Hello. Reply only OK; do not execute tools.")
    parser.add_argument("--required-system-marker", action="append", default=[])
    parser.add_argument("--allow-expired-mcp-snapshot", action="store_true", help="Explicit offline inspection only: accept TTL-expired MCP metadata if its connection fingerprint still matches. Records this limitation; never refreshes/executes MCP.")
    parser.add_argument("--mcp-snapshot-not-before", type=float)
    parser.add_argument("--expected-mcp-tools", type=int)
    args = parser.parse_args()
    # Runtime logging can contain private context: silence it, and never print raw errors/config.
    import logging
    logging.disable(logging.CRITICAL)
    try:
        info = capture(args.repo, args.output, args.metadata, args.user_message, args.required_system_marker, args.allow_expired_mcp_snapshot, args.mcp_snapshot_not_before, args.expected_mcp_tools)
    except Exception as exc:
        detail = str(exc) if isinstance(exc, CaptureDenied) else (getattr(exc, "name", "") if isinstance(exc, ModuleNotFoundError) else "")
        print(f"Capture failed: {type(exc).__name__}. {detail} No credentials or raw provider errors are printed.", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "tools": len(info["tool_names"]), "system_chars": info["system_chars"], "mocked_calls": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
