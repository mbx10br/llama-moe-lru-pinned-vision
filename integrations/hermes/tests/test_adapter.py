"""Synthetic skill text only: no installed skills or user profile is copied."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Context:
    def __init__(self):
        self.skills, self.sections, self.hooks = {}, {}, {}

    def register_skill(self, name, path):
        self.skills[name] = path

    def register_system_prompt_section(self, name, text, **kwargs):
        self.sections[name] = (text, kwargs)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.plugin = Path(self.temporary.name) / "plugin"
        references = self.plugin / "skills" / "using-superpowers" / "references"
        references.mkdir(parents=True)
        (references.parent / "SKILL.md").write_text("---\nname: using-superpowers\n---\nSYNTHETIC_INSTRUCTIONS\n" + "Paragraph for chunk boundaries.\n\n" * 230)
        (references / "hermes-tools.md").write_text("SYNTHETIC_TOOL_MAPPING")
        another = self.plugin / "skills" / "fixture-other"
        another.mkdir()
        (another / "SKILL.md").write_text("SYNTHETIC_OTHER_SKILL")
        target = self.plugin / "__init__.py"
        shutil.copyfile(ROOT / "superpowers_adapter.py", target)
        spec = importlib.util.spec_from_file_location("test_prefix_adapter", target)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_sections_preserve_every_bootstrap_character_and_skill(self):
        ctx = Context()
        self.module.register(ctx)
        bootstrap = self.module._build_bootstrap(self.module._skills_dir())
        self.assertEqual("".join(v[0] for v in ctx.sections.values()), bootstrap)
        self.assertEqual(set(ctx.skills), {"using-superpowers", "fixture-other"})
        self.assertTrue(all(len(text) <= 3600 and cfg == {"position": "after_memory", "max_chars": 4000} for text, cfg in ctx.sections.values()))
        frozen = "\n\n".join(text for text, _ in ctx.sections.values())
        self.assertIsNone(ctx.hooks["pre_llm_call"](is_first_turn=True, system_prompt=frozen))
        self.assertIsNone(ctx.hooks["pre_llm_call"](is_first_turn=False, system_prompt="older prompt"))

    def test_legacy_runtime_or_skipped_section_keeps_complete_fallback(self):
        ctx = Context()
        ctx.register_system_prompt_section = None
        self.module.register(ctx)
        bootstrap = self.module._build_bootstrap(self.module._skills_dir())
        self.assertEqual(ctx.hooks["pre_llm_call"](is_first_turn=True), {"context": bootstrap})
        native = Context()
        self.module.register(native)
        partial = next(iter(native.sections.values()))[0]
        self.assertEqual(native.hooks["pre_llm_call"](is_first_turn=True, system_prompt=partial), {"context": bootstrap})

    def test_old_resumed_prompt_is_not_mistaken_for_native_bootstrap(self):
        ctx = Context()
        self.module.register(ctx)
        expected = self.module._build_bootstrap(self.module._skills_dir())
        self.assertEqual(ctx.hooks["pre_llm_call"](is_first_turn=True, system_prompt="old restored bytes"), {"context": expected})


if __name__ == "__main__":
    unittest.main()
