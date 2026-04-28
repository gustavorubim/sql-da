"""Virtual memory and skill files for Deep Agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_da.config import AgentConfig


@dataclass(frozen=True)
class AgentVirtualContext:
    """Files and source paths passed to Deep Agents middleware."""

    files: dict[str, Any]
    memory_sources: tuple[str, ...]
    skill_sources: tuple[str, ...]


def build_agent_virtual_context(agent: AgentConfig) -> AgentVirtualContext:
    """Load configured memory and skill files into Deep Agents virtual paths."""

    try:
        from deepagents.backends.utils import create_file_data
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("deepagents is required to build agent memory and skills") from exc

    files: dict[str, Any] = {}
    memory_sources: list[str] = []
    for path in agent.memory_paths:
        if not path.exists():
            continue
        virtual_path = f"/memory/{path.name}"
        files[virtual_path] = create_file_data(path.read_text(encoding="utf-8"))
        memory_sources.append(virtual_path)

    skill_sources: list[str] = []
    for path in agent.skill_paths:
        if not path.exists():
            continue
        source_path = f"/skills/{path.name}" if path.name != "skills" else "/skills"
        _load_skill_files(path, source_path, files, create_file_data)
        skill_sources.append(source_path)

    return AgentVirtualContext(
        files=files,
        memory_sources=tuple(memory_sources),
        skill_sources=tuple(skill_sources),
    )


def _load_skill_files(
    root: Path,
    virtual_root: str,
    files: dict[str, Any],
    create_file_data: Any,
) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[f"{virtual_root.rstrip('/')}/{relative}"] = create_file_data(
            path.read_text(encoding="utf-8")
        )
