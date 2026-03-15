"""Skill system for chat — loads markdown skill files on demand."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: Path


def _parse_skill_file(path: Path) -> Skill | None:
    """Parse a markdown skill file with key: value frontmatter."""
    try:
        text = path.read_text()
    except OSError as exc:
        logger.warning("Failed to read skill file %s: %s", path, exc)
        return None

    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None

    frontmatter, body = m.group(1), m.group(2).strip()

    fields: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    name = fields.get("name")
    if not name:
        return None

    return Skill(
        name=name,
        description=fields.get("description", ""),
        content=body,
        path=path,
    )


def load_skills_from_dir(path: Path) -> dict[str, Skill]:
    """Load all .md skill files from a directory."""
    if not path.is_dir():
        return {}

    skills: dict[str, Skill] = {}
    for p in sorted(path.glob("*.md")):
        skill = _parse_skill_file(p)
        if skill:
            skills[skill.name] = skill
        else:
            logger.debug("Skipping malformed skill file: %s", p)
    return skills


class SkillManager:
    """Manages loading and accessing skills."""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._skills: dict[str, Skill] = {}

    def load(self) -> None:
        self._skills = load_skills_from_dir(self.skills_dir)
        if self._skills:
            logger.info("Loaded %d skills from %s", len(self._skills), self.skills_dir)

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_skill_index_text(self) -> str:
        """Return a brief index of skills for system prompt injection."""
        if not self._skills:
            return ""
        lines = ["Available skills (use the use_skill tool to load one):"]
        for skill in self._skills.values():
            desc = f" — {skill.description}" if skill.description else ""
            lines.append(f"- {skill.name}{desc}")
        return "\n".join(lines)

    def get_tool_definition(self) -> dict | None:
        """Return OpenAI function-calling format tool definition, or None if no skills."""
        if not self._skills:
            return None
        return {
            "type": "function",
            "function": {
                "name": "use_skill",
                "description": "Load a skill's full instructions to follow for the current task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The skill name to load.",
                            "enum": sorted(self._skills.keys()),
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    def handle_use_skill(self, arguments: dict) -> str:
        """Handle a use_skill tool call. Returns content or error message."""
        name = arguments.get("name", "")
        skill = self._skills.get(name)
        if skill is None:
            return f"Skill '{name}' not found. Available: {', '.join(sorted(self._skills.keys()))}"
        return skill.content
