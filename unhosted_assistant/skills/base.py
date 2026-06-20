"""Skill contract + registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Skill:
    """A self-describing tool the agent can call.

    parameters is a JSON-schema object (the `properties` map + optional
    `required`), e.g. {"type":"object","properties":{"path":{"type":"string"}},
    "required":["path"]}.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]

    def to_tool_spec(self) -> dict[str, Any]:
        """Ollama/OpenAI-style function tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        self._skills[skill.name] = skill
        return skill

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Decorator form: @registry.add("now", "current date/time")."""
        def deco(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(Skill(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                run=fn,
            ))
            return fn
        return deco

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [s.to_tool_spec() for s in self._skills.values()]

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Run a skill by name with kwargs; returns a string result (or a clear
        error string the model can read + recover from)."""
        skill = self._skills.get(name)
        if skill is None:
            return f"[error] unknown skill '{name}'. Available: {', '.join(self.names())}"
        try:
            result = skill.run(**(args or {}))
            return result if isinstance(result, str) else str(result)
        except TypeError as e:
            return f"[error] bad arguments for '{name}': {e}"
        except Exception as e:  # skills must never crash the agent loop
            return f"[error] skill '{name}' failed: {e}"


# Process-wide default registry the built-in skills attach to.
default_registry = SkillRegistry()
