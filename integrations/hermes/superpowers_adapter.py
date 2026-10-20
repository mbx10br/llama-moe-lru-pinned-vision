# SPDX-License-Identifier: MIT
# Adapted from obra/superpowers; provenance and copyrights: NOTICE.md.
import os
import re
from pathlib import Path

BOOTSTRAP_MARKER = "superpowers:using-superpowers bootstrap for hermes"


def _bootstrap_sections(bootstrap: str):
    """Keep every instruction inside Hermes' existing 4K-per-section budget."""
    # Length-framed native sections add their own headings. Split at paragraphs
    # where possible; joining the pieces reproduces the original bootstrap.
    sections=[];remaining=bootstrap
    while remaining:
        end=min(3600,len(remaining))
        if end<len(remaining):
            boundary=remaining.rfind('\n\n',0,end)
            if boundary>1800:end=boundary+2
        sections.append(remaining[:end]);remaining=remaining[end:]
    return tuple(sections)


def _sections_present(system_prompt, sections):
    """Verify the actual frozen prompt, including restore/budget-skip cases."""
    if not isinstance(system_prompt,str):return False
    position=0
    for section in sections:
        start=system_prompt.find(section.strip(),position)
        if start<0:return False
        position=start+len(section.strip())
    return True


def _skills_dir() -> str:
    """Locate the stock skills/ tree for either supported install layout.

    - git-clone install (`hermes plugins install obra/superpowers`): the plugin
      dir is the repo root, so `.hermes-plugin/` and `skills/` are siblings and
      this module resolves `../skills`.
    - flattened install (plugin files copied to the plugin dir root): `skills/`
      sits next to this module.

    Raises loudly when neither matches — a bootstrap that silently skips is how
    a broken install masquerades as a working one.
    """
    here = os.path.dirname(os.path.realpath(__file__))
    candidates = (
        os.path.realpath(os.path.join(here, "..", "skills")),
        os.path.realpath(os.path.join(here, "skills")),
    )
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "using-superpowers", "SKILL.md")):
            return cand
    raise RuntimeError(
        "superpowers plugin: cannot find the skills/ tree "
        f"(looked at {candidates}). Reinstall with "
        "`hermes plugins install obra/superpowers`."
    )


def _strip_frontmatter(content: str) -> str:
    match = re.match(r"^---\n[\s\S]*?\n---\n([\s\S]*)$", content)
    return (match.group(1) if match else content).strip()


def _build_bootstrap(skills_dir: str) -> str:
    with open(
        os.path.join(skills_dir, "using-superpowers", "SKILL.md"),
        encoding="utf-8",
    ) as f:
        body = _strip_frontmatter(f.read())

    tools_path = os.path.join(
        skills_dir, "using-superpowers", "references", "hermes-tools.md"
    )
    with open(tools_path, encoding="utf-8") as f:
        tool_mapping = f.read().strip()

    return (
        f"<EXTREMELY_IMPORTANT>\n"
        f"{BOOTSTRAP_MARKER}\n\n"
        f"You have superpowers.\n\n"
        f"The using-superpowers skill content is included below and is already "
        f"loaded for this Hermes session. Follow it now. "
        f"Do not try to load using-superpowers again.\n\n"
        f"{body}\n\n"
        f"## Loading Superpowers Skills on Hermes\n\n"
        f"Superpowers skills are registered with Hermes' native skill loader: "
        f'invoke one with `skill_view("superpowers:skill-name")` '
        f'(for example `skill_view("superpowers:brainstorming")`). '
        f"If a namespaced lookup returns 'not found', read the skill file "
        f"directly instead:\n"
        f'`read_file("{skills_dir}/skill-name/SKILL.md")`\n\n'
        f"The superpowers skills directory is: `{skills_dir}`\n\n"
        f"{tool_mapping}\n"
        f"</EXTREMELY_IMPORTANT>"
    )


def register(ctx):
    skills_dir = _skills_dir()
    bootstrap = _build_bootstrap(skills_dir)
    sections = _bootstrap_sections(bootstrap)

    # Register every stock skill with Hermes' native loader so skill_view can
    # load them on demand. Standard markdown; no conversion (plugin guide).
    # register_skill requires a pathlib.Path — a str raises AttributeError and
    # hermes silently disables the whole plugin (verified 2026-07-23).
    for name in sorted(os.listdir(skills_dir)):
        skill_md = os.path.join(skills_dir, name, "SKILL.md")
        if os.path.isfile(skill_md):
            ctx.register_skill(name, Path(skill_md))

    # Static plugin sections are frozen before the first user and therefore
    # belong to the protected native prefix. Existing sessions keep their old
    # prompt and api_content; the hook below preserves compatibility with them.
    register_section=getattr(ctx,'register_system_prompt_section',None)
    if callable(register_section):
        for index,section in enumerate(sections,1):
            register_section(f'superpowers.bootstrap.{index:02d}',section,
                position='after_memory',max_chars=4000)

    def pre_llm_call(
        session_id=None,
        user_message=None,
        conversation_history=None,
        is_first_turn=None,
        model=None,
        platform=None,
        system_prompt=None,
        **kwargs,
    ):
        if is_first_turn and not _sections_present(system_prompt,sections):
            return {"context": bootstrap}
        return None

    ctx.register_hook("pre_llm_call", pre_llm_call)
