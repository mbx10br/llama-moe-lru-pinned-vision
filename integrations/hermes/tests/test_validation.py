import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("validate_captures", Path(__file__).resolve().parents[1] / "validate_captures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationTests(unittest.TestCase):
    def request(self, user="first", date="one"):
        return {"model": "fixture-model", "messages": [
            {"role": "system", "content": "Stable CORE and PLUGIN\n\nConversation started: " + date},
            {"role": "user", "content": user}], "tools": [{"type": "function", "function": {"name": "fixture_tool", "parameters": {"type": "object"}}}]}

    def test_different_users_and_dates_do_not_enter_protected_material(self):
        left, right = self.request(), self.request(user="second", date="two")
        before = copy.deepcopy(left)
        self.assertTrue(module.validate(left, right, ["PLUGIN"])["valid"])
        protected = module.protected_material(left)
        self.assertEqual(protected["messages"], [{"role": "system", "content": "Stable CORE and PLUGIN"}])
        self.assertEqual(left, before)

    def test_modified_core_tool_schema_or_duplicate_user_bootstrap_is_rejected(self):
        for field in ("core", "schema", "duplicate"):
            left, right = self.request(), self.request()
            if field == "core":
                right["messages"][0]["content"] += "modified"
                right["messages"][0]["content"] = right["messages"][0]["content"].replace("CORE", "DIFFERENT")
            elif field == "schema":
                right["tools"][0]["function"]["parameters"]["required"] = ["changed"]
            else:
                right["messages"][1]["content"] = "PLUGIN"
            with self.assertRaises(ValueError):
                module.validate(left, right, ["PLUGIN"])

    def test_task_protocol_and_missing_static_context_are_rejected(self):
        task = self.request()
        task["messages"][0]["content"] = "You have been assigned ONE task from the shared board"
        with self.assertRaises(ValueError):
            module.validate(task, task)
        with self.assertRaises(ValueError):
            module.protected_material({"messages": [{"role": "user", "content": "no system"}], "tools": []})


if __name__ == "__main__":
    unittest.main()
