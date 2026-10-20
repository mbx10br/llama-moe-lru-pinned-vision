#!/usr/bin/env python3
"""Validate two private real-pipeline captures without running inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def protected_material(payload: dict) -> dict:
    """Inspection material only; never send this as a replacement live request.

    The native archive must render the ORIGINAL request and verify its token
    prefix. A separately formatted JSON object cannot establish token identity.
    """
    messages = []
    for message in payload.get("messages", []):
        if message.get("role") not in ("system", "developer"):
            break
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Only plain-text leading system/developer context is supported")
        marker = "\n\nConversation started:"
        if marker in content:
            content = content.rsplit(marker, 1)[0]
        messages.append({"role": message["role"], "content": content})
    if not messages or not payload.get("tools"):
        raise ValueError("Capture must contain leading system context and the actual tool schemas")
    result = {"model": payload.get("model"), "messages": messages, "tools": payload["tools"]}
    for key in ("chat_template_kwargs", "tool_choice", "parallel_tool_calls"):
        if key in payload:
            result[key] = payload[key]
    return result


def validate(first: dict, second: dict, markers=()) -> dict:
    left, right = protected_material(first), protected_material(second)
    if left != right:
        raise ValueError("Protected material differs; investigate before creating a reusable base")
    system = "\n\n".join(m["content"] for m in left["messages"])
    if "You have been assigned ONE task from the shared board" in system:
        raise ValueError("Dispatcher-only task instructions leaked into this normal-session capture")
    for marker in markers:
        if marker not in system:
            raise ValueError("A required plugin section is missing from the protected system")
        for payload in (first, second):
            if any(marker in str(m.get("content", "")) for m in payload["messages"] if m.get("role") == "user"):
                raise ValueError("A stable plugin section is duplicated into user context")
    encoded = json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"valid": True, "protected_material_identical": True,
            "protected_system_characters": len(system), "tools": len(left["tools"]),
            "inspection_sha256": hashlib.sha256(encoded).hexdigest(),
            "native_token_identity_still_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--required-system-marker", action="append", default=[])
    args = parser.parse_args()
    result = validate(json.loads(args.first.read_text()), json.loads(args.second.read_text()), args.required_system_marker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
