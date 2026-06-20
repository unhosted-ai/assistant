"""
Skill system — composable tools the agent can call.

A Skill is a small, self-describing unit: name + description + JSON-schema
parameters + a run() that returns a string. The registry turns registered skills
into Ollama tool specs and dispatches tool calls back to the right skill.
"""

from .base import Skill, SkillRegistry, default_registry

__all__ = ["Skill", "SkillRegistry", "default_registry"]
