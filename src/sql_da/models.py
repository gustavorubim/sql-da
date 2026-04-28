"""Small serializable data models used by the analyzer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DriverEvidence:
    """A measured driver that explains part of a metric variance."""

    label: str
    current_value: float
    prior_value: float
    variance: float
    sql: str
    rows: list[dict[str, Any]]
    lineage_nodes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VarianceFinding:
    """A financial variance and its supporting evidence."""

    metric: str
    period: str
    compare_to: str
    current_value: float
    prior_value: float
    absolute_variance: float
    percent_variance: float | None
    is_material: bool
    lineage_path: list[str] = field(default_factory=list)
    drivers: list[DriverEvidence] = field(default_factory=list)
    root_cause: str = ""
    confidence: str = "medium"
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["drivers"] = [driver.to_dict() for driver in self.drivers]
        return data


@dataclass(frozen=True)
class AnalysisResult:
    """Report output paths and finding payloads for a run."""

    run_id: str
    report_path: str
    evidence_path: str
    trace_json_path: str
    trace_csv_path: str
    findings: list[VarianceFinding]
    agent_review: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "report_path": self.report_path,
            "evidence_path": self.evidence_path,
            "trace_json_path": self.trace_json_path,
            "trace_csv_path": self.trace_csv_path,
            "agent_review": self.agent_review,
            "findings": [finding.to_dict() for finding in self.findings],
        }
