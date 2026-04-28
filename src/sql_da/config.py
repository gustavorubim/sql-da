"""Configuration loading for the SQL deep agent demo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "google_genai:gemini-3-flash-preview"


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""

    uri: str
    path: Path


@dataclass(frozen=True)
class ModelConfig:
    """Model settings for the live Deep Agent path."""

    name: str = DEFAULT_MODEL
    temperature: float = 0.0
    enable_live_agent: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    """Deterministic variance-analysis settings."""

    materiality_abs: float = 5000.0
    materiality_pct: float = 0.05
    max_query_rows: int = 100
    metrics: tuple[str, ...] = (
        "revenue",
        "cogs",
        "gross_margin",
        "payroll_expense",
        "vendor_expense",
        "operating_expense",
        "net_income",
    )


@dataclass(frozen=True)
class RewardConfig:
    """Reward-harness settings."""

    output_path: Path


@dataclass(frozen=True)
class AgentConfig:
    """Deep Agent memory and skill sources."""

    memory_paths: tuple[Path, ...] = ()
    skill_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration."""

    path: Path
    project_root: Path
    database: DatabaseConfig
    lineage_path: Path
    report_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(Path("reports/reward_score.json"))
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ProjectConfig:
        """Load and resolve a YAML configuration file."""

        path = Path(config_path).expanduser().resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        project_root = _infer_project_root(path)
        load_env_file(project_root / ".env")

        database_raw = raw.get("database", {})
        db_path = _resolve_path(project_root, database_raw.get("path", "data/demo_finance.db"))
        db_uri = database_raw.get("uri") or f"sqlite:///{db_path}"
        if (
            db_uri.startswith("sqlite:///")
            and not Path(db_uri.removeprefix("sqlite:///")).is_absolute()
        ):
            db_uri = f"sqlite:///{db_path}"

        model_raw = raw.get("model", {})
        agent_raw = raw.get("agent", {})
        analysis_raw = raw.get("analysis", {})
        reward_raw = raw.get("reward", {})

        model = ModelConfig(
            name=os.getenv("SQL_DA_MODEL", model_raw.get("name", DEFAULT_MODEL)),
            temperature=float(model_raw.get("temperature", 0.0)),
            enable_live_agent=bool(model_raw.get("enable_live_agent", True)),
        )
        analysis = AnalysisConfig(
            materiality_abs=float(analysis_raw.get("materiality_abs", 5000.0)),
            materiality_pct=float(analysis_raw.get("materiality_pct", 0.05)),
            max_query_rows=int(analysis_raw.get("max_query_rows", 100)),
            metrics=tuple(analysis_raw.get("metrics", AnalysisConfig().metrics)),
        )
        reward = RewardConfig(
            output_path=_resolve_path(
                project_root, reward_raw.get("output_path", "reports/reward_score.json")
            )
        )
        agent = AgentConfig(
            memory_paths=tuple(
                _resolve_path(project_root, value)
                for value in agent_raw.get("memory_paths", ["memory/AGENTS.md"])
            ),
            skill_paths=tuple(
                _resolve_path(project_root, value)
                for value in agent_raw.get("skill_paths", ["skills"])
            ),
        )
        return cls(
            path=path,
            project_root=project_root,
            database=DatabaseConfig(uri=db_uri, path=db_path),
            lineage_path=_resolve_path(
                project_root, raw.get("lineage_path", "configs/lineage.yaml")
            ),
            report_dir=_resolve_path(project_root, raw.get("report_dir", "reports")),
            model=model,
            agent=agent,
            analysis=analysis,
            reward=reward,
        )


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment variables."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _infer_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def config_to_dict(config: ProjectConfig) -> dict[str, Any]:
    """Return a serializable subset of config values for report metadata."""

    return {
        "database_uri": config.database.uri,
        "database_path": str(config.database.path),
        "lineage_path": str(config.lineage_path),
        "report_dir": str(config.report_dir),
        "model": config.model.name,
        "agent_memory_paths": [str(path) for path in config.agent.memory_paths],
        "agent_skill_paths": [str(path) for path in config.agent.skill_paths],
        "materiality_abs": config.analysis.materiality_abs,
        "materiality_pct": config.analysis.materiality_pct,
        "max_query_rows": config.analysis.max_query_rows,
        "metrics": list(config.analysis.metrics),
    }
