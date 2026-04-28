"""Agent research trace recording and artifact export."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """One tabular research trace event."""

    sequence: int
    timestamp: str
    phase: str
    actor: str
    action: str
    tool: str = ""
    input_summary: str = ""
    output_summary: str = ""
    status: str = "ok"
    duration_ms: int | None = None
    artifact: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> dict[str, Any]:
        row = self.to_dict()
        row["metadata"] = json.dumps(self.metadata, sort_keys=True)
        return row


class AgentTrace:
    """Collects and persists the agent's research trail."""

    def __init__(self, reporter: Any | None = None):
        self.events: list[TraceEvent] = []
        self._reporter = reporter

    def record(
        self,
        phase: str,
        actor: str,
        action: str,
        *,
        tool: str = "",
        input_summary: str = "",
        output_summary: str = "",
        status: str = "ok",
        duration_ms: int | None = None,
        artifact: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Append one event and mirror it to the verbose terminal reporter."""

        event = TraceEvent(
            sequence=len(self.events) + 1,
            timestamp=datetime.now(UTC).isoformat(),
            phase=phase,
            actor=actor,
            action=action,
            tool=tool,
            input_summary=compact_text(input_summary),
            output_summary=compact_text(output_summary),
            status=status,
            duration_ms=duration_ms,
            artifact=artifact,
            metadata=metadata or {},
        )
        self.events.append(event)
        if self._reporter is not None:
            self._reporter.trace_event(event.to_dict())
        return event

    @contextmanager
    def span(
        self,
        phase: str,
        actor: str,
        action: str,
        *,
        tool: str = "",
        input_summary: str = "",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record a duration event around a block."""

        started = perf_counter()
        try:
            yield
        except Exception as exc:
            self.record(
                phase,
                actor,
                action,
                tool=tool,
                input_summary=input_summary,
                output_summary=str(exc),
                status="error",
                duration_ms=int((perf_counter() - started) * 1000),
                metadata=metadata,
            )
            raise
        self.record(
            phase,
            actor,
            action,
            tool=tool,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=int((perf_counter() - started) * 1000),
            metadata=metadata,
        )

    def extend_agent_steps(self, steps: list[dict[str, Any]]) -> None:
        """Append normalized live-agent research steps."""

        for step in steps:
            self.record(
                "live_agent",
                str(step.get("actor", "deep-agent")),
                str(step.get("action", "agent message")),
                tool=str(step.get("tool", "")),
                input_summary=str(step.get("input_summary", "")),
                output_summary=str(step.get("output_summary", "")),
                status=str(step.get("status", "ok")),
                metadata=dict(step.get("metadata", {})),
            )

    def write_artifacts(self, run_dir: Path) -> tuple[Path, Path]:
        """Write JSON and CSV trace artifacts."""

        run_dir.mkdir(parents=True, exist_ok=True)
        json_path = run_dir / "agent_trace.json"
        csv_path = run_dir / "agent_trace.csv"
        payload = [event.to_dict() for event in self.events]
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(TraceEvent(0, "", "", "", "").to_csv_row())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                writer.writerow(event.to_csv_row())
        return json_path, csv_path


def compact_text(value: Any, limit: int = 500) -> str:
    """Serialize and trim values for terminal and tabular trace readability."""

    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
