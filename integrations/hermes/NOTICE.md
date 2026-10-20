# Upstream attribution and scope

This directory contains local integration work for
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) and
[obra/superpowers](https://github.com/obra/superpowers). It does not claim an
upstream merge, endorsement or ownership of either project's implementation.

- **Superpowers:** version6.4.2, source commit
  `8ca22dba9a94f28898bbce59f2537ff4d87c747d`.
  `superpowers_adapter.py` derives from the upstream
  [`.hermes-plugin/__init__.py`](https://github.com/obra/superpowers/blob/8ca22dba9a94f28898bbce59f2537ff4d87c747d/.hermes-plugin/__init__.py).
  Copyright(c)2025 Jesse Vincent; MIT license reproduced in
  [`licenses/superpowers-MIT.txt`](licenses/superpowers-MIT.txt).
  The bootstrap builder and skill-directory/frontmatter helpers remain identical
  in behavior and AST to that source. New logic splits unchanged bootstrap text
  into native sections and tests the actual rendered system before suppressing
  the original hook fallback. No upstream or private skill body is bundled.
- **Hermes Agent:** modular patch baseline
  `81805a97ef977b4e1918c0d9945cf720df9952d9`.
  Copyright(c)2025 Nous Research; MIT license reproduced in
  [`licenses/hermes-agent-MIT.txt`](licenses/hermes-agent-MIT.txt).
  Patch contexts and the optional catalog regression test derive from Hermes;
  the prompt hook change and capability-card rendering are local additions.
  The legacy hunk illustrates the same additive hook field on an older source
  structure; it is not a universal version-independent patch.

Original integration additions may be used under the MIT terms reproduced in
this directory. The existing native prompt sections, lifecycle dispatcher,
tool-search bridge, skill loader, model client and agent factory remain Hermes
responsibilities. Native model state, checkpoints, sleep and model identity
validation remain inference-runtime responsibilities.

The export intentionally excludes local profiles, connector-specific catalog
hints, SOUL/AGENTS/MEMORY/USER contents, authentication/configuration values,
recorded conversations, real request captures and maintenance scripts. Test
fixtures are newly generated synthetic text. See [`provenance.json`](provenance.json)
and the SHA256 file manifest for source and export checks.
