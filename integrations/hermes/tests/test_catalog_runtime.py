"""Capability cards preserve progressive discovery, execution and scoped grants."""

import json


def _definition(name, description):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "FULL_PARAMETER_INSTRUCTIONS"}},
            "required": ["query"]}}}


def test_cards_are_stable_schema_free_and_budgeted():
    from tools.registry import registry
    from tools.tool_search_catalog import build_catalog_listing_with_form

    definitions = [_definition(f"capability_card_{i}", f"Find project item {i}. " + "Details. " * 120)
                   for i in range(4)]
    for definition in definitions:
        registry.register(name=definition["function"]["name"], toolset="mcp-capability-cards",
                          schema=definition["function"], handler=lambda args, **kw: "{}")
    text, form = build_catalog_listing_with_form(definitions, style="capabilities")
    reverse, reverse_form = build_catalog_listing_with_form(list(reversed(definitions)), style="capabilities")
    assert (text, form) == (reverse, reverse_form)
    assert form == "capabilities" and len(text.splitlines()) == 4
    assert all(td["function"]["name"] in text for td in definitions)
    assert "query (string)" in text and "FULL_PARAMETER_INSTRUCTIONS" not in text
    assert "Details. Details. Details." not in text
    tiny, _ = build_catalog_listing_with_form(definitions, style="capabilities", max_tokens=30)
    assert tiny is None or len(tiny) <= 30 * 4


def test_profile_setting_reaches_real_bridge_without_losing_or_expanding_grants(tmp_path, monkeypatch):
    import model_tools
    from tools.registry import registry
    from tools.tool_search import load_config, assemble_tool_defs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "tools:\n  tool_search:\n    enabled: on\n    listing_style: capabilities\n    defer: [capability_e2e_search]\n")
    name = "capability_e2e_search"
    definition = _definition(name, "Find local project documents. Complete instructions remain here.")
    registry.register(name=name, toolset="mcp-capability-e2e", schema=definition["function"],
                      handler=lambda args, **kw: json.dumps({"found": args["query"]}))
    outside = _definition("capability_outside", "Unavailable in this session.")
    registry.register(name="capability_outside", toolset="mcp-capability-outside", schema=outside["function"],
                      handler=lambda args, **kw: "{}")
    cfg = load_config()
    assert cfg.listing_style == "capabilities"
    assembled = assemble_tool_defs([definition], config=cfg)
    assert assembled.listing_form == "capabilities"
    original_prefix = json.dumps(assembled.tool_defs, sort_keys=True)
    scope = {"enabled_toolsets": ["mcp-capability-e2e"]}
    search = json.loads(model_tools.handle_function_call(
        function_name="tool_search", function_args={"queries": [name]}, **scope))
    assert name in search["tools"] and "capability_outside" not in search["tools"]
    described = json.loads(model_tools.handle_function_call(
        function_name="tool_describe", function_args={"names": [name]}, **scope))
    assert described["tools"][name]["parameters"] == definition["function"]["parameters"]
    result = json.loads(model_tools.handle_function_call(
        function_name="tool_call", function_args={"calls": [{"name": name, "arguments": {"query": "plan"}}]}, **scope))
    assert result == {"found": "plan"}
    assert json.dumps(assemble_tool_defs([definition], config=cfg).tool_defs,
                      sort_keys=True) == original_prefix
    denied = json.loads(model_tools.handle_function_call(
        function_name="tool_call", function_args={"calls": [{"name": "capability_outside", "arguments": {"query": "plan"}}]}, **scope))
    assert "error" in denied
