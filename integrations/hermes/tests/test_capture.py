import importlib.util
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture_request", ROOT / "capture_request.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
HTTP_AVAILABLE = importlib.util.find_spec("httpx") is not None and importlib.util.find_spec("openai") is not None


class CaptureTests(unittest.TestCase):
    def test_private_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "private.json"
            capture.private_json(path, {"synthetic": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                capture.private_json(path, {})

    def test_guards_reject_real_db_network_shell_and_unknown_programs(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            with capture.safety_fence(Path(root)) as violations:
                con = sqlite3.connect(Path(root) / "state.db")
                con.execute("create table fixture(value)")
                con.close()
                for action in (lambda: sqlite3.connect(Path(other) / "state.db"),
                               lambda: socket.create_connection(("provider.invalid", 443)),
                               lambda: subprocess.Popen(["unapproved-program"]),
                               lambda: os.system("unapproved-command")):
                    with self.assertRaises(capture.CaptureDenied):
                        action()
                self.assertEqual(len(violations), 4)
                self.assertFalse((Path(other) / "state.db").exists())

    def test_probe_allowlist_does_not_accept_scripts_or_mutation(self):
        self.assertTrue(capture.readonly_probe(["npx", "--version"]))
        self.assertTrue(capture.readonly_probe(["git", "-C", "fixture-project", "status", "--porcelain=2", "--branch"]))
        for cmd in (["git", "reset", "--hard"], ["python3", "-c", "unknown_script()"], ["npx", "package"], "git status"):
            self.assertFalse(capture.readonly_probe(cmd))

    @unittest.skipUnless(HTTP_AVAILABLE, "Optional capture dependencies: httpx and openai")
    def test_sdk_stream_and_body_capture_do_not_persist_auth_headers(self):
        import httpx
        from openai import OpenAI
        fake = capture.FakeModel()
        with OpenAI(api_key="synthetic-never-record", base_url="https://provider.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(fake))) as client:
            stream = client.chat.completions.create(model="fixture-model", messages=[{"role": "user", "content": "control"}], stream=True)
            text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
        self.assertEqual(text, "OK")
        self.assertEqual(fake.requests[0]["model"], "fixture-model")
        self.assertNotIn("synthetic-never-record", json.dumps(fake.requests))
        with self.assertRaises(capture.CaptureDenied):
            fake(httpx.Request("GET", "https://provider.invalid/v1/models"))


@unittest.skipUnless(os.environ.get("HERMES_SOURCE") and HTTP_AVAILABLE,
                     "Set HERMES_SOURCE to a compatible patched checkout for the optional real-factory CPU test")
class RealFactoryTests(unittest.TestCase):
    def test_two_fresh_captures_preserve_actual_profile_plugin_memory_and_schemas(self):
        repo = Path(os.environ["HERMES_SOURCE"]).resolve()
        with tempfile.TemporaryDirectory(prefix="hermes-public-integration-") as temporary:
            root = Path(temporary)
            home = root / "profile"
            plugin = home / "plugins" / "superpowers"
            references = plugin / "skills" / "using-superpowers" / "references"
            references.mkdir(parents=True)
            import shutil
            shutil.copyfile(ROOT / "superpowers_adapter.py", plugin / "__init__.py")
            (plugin / "plugin.yaml").write_text("name: superpowers\nversion: 6.4.2\ndescription: Synthetic integration test\nprovides_hooks: [pre_llm_call]\n")
            (references.parent / "SKILL.md").write_text("---\nname: using-superpowers\n---\nSYNTHETIC_SKILL_BODY")
            (references / "hermes-tools.md").write_text("SYNTHETIC_TOOL_MAPPING")
            (home / "SOUL.md").write_text("SYNTHETIC_PROFILE_IDENTITY")
            (home / "memories").mkdir()
            (home / "memories" / "MEMORY.md").write_text("SYNTHETIC_PROFILE_MEMORY")
            (home / "memories" / "USER.md").write_text("SYNTHETIC_PROFILE_USER")
            (root / "AGENTS.md").write_text("SYNTHETIC_WORKSPACE_CONTEXT")
            config = {"model": {"default": "fixture-model", "provider": "custom", "base_url": "https://provider.invalid/v1", "context_length": 102400},
                      "providers": {"custom": {"base_url": "https://provider.invalid/v1", "context_length": 102400}},
                      "agent": {"max_turns": 1}, "terminal": {"backend": "local"},
                      "plugins": {"enabled": ["superpowers"]}, "compression": {"enabled": False},
                      "memory": {"memory_enabled": True, "user_profile_enabled": True}}
            (home / "config.yaml").write_text(json.dumps(config))
            env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "TERMINAL_"))}
            env.update(HERMES_HOME=str(home), HERMES_TUI_TOOLSETS="file,memory", TERMINAL_CWD=str(root),
                       OPENAI_API_KEY="synthetic-not-secret", HERMES_KANBAN_TASK_ID="must-be-cleared")
            captures = []
            for index, user in enumerate(("first controlled user", "different controlled user")):
                output, metadata = root / f"capture-{index}.json", root / f"metadata-{index}.json"
                args = [sys.executable, str(ROOT / "capture_request.py"), "--repo", str(repo),
                        "--output", str(output), "--metadata", str(metadata), "--user-message", user,
                        "--required-system-marker", "superpowers:using-superpowers bootstrap for hermes"]
                result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                captures.append(json.loads(output.read_text()))
                info = json.loads(metadata.read_text())
                self.assertEqual(info["agent_home_explicitly_bound"], str(home))
                self.assertTrue(info["soul_present_in_system"])
                self.assertTrue(all(v["present_in_system"] for v in info["loaded_memory_blocks"].values()))
            validation_spec = importlib.util.spec_from_file_location("validate_captures", ROOT / "validate_captures.py")
            validation = importlib.util.module_from_spec(validation_spec)
            validation_spec.loader.exec_module(validation)
            self.assertTrue(validation.validate(*captures, markers=["superpowers:using-superpowers bootstrap for hermes"])["valid"])
            system = "\n".join(m["content"] for m in captures[0]["messages"] if m["role"] == "system")
            for marker in ("SYNTHETIC_PROFILE_IDENTITY", "SYNTHETIC_PROFILE_MEMORY", "SYNTHETIC_PROFILE_USER", "SYNTHETIC_WORKSPACE_CONTEXT", "SYNTHETIC_SKILL_BODY", "SYNTHETIC_TOOL_MAPPING"):
                self.assertIn(marker, system)
            self.assertFalse((home / "state.db").exists())


if __name__ == "__main__":
    unittest.main()
