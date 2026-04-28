"""Project-specific read-only tools exposed to the deep agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sql_da.lineage import LineageGraph
from sql_da.sql_safety import ReadOnlySQLRunner
from sql_da.variance import compute_variances, get_metric_drivers


class SQLAnalysisToolkit:
    """Pure-Python tool surface for schema, lineage, variance, and safe SQL."""

    def __init__(
        self,
        db_path: str | Path,
        lineage: LineageGraph,
        max_rows: int,
        metrics: tuple[str, ...],
        materiality_abs: float,
        materiality_pct: float,
    ):
        self.db_path = Path(db_path)
        self.lineage = lineage
        self.runner = ReadOnlySQLRunner(db_path, max_rows=max_rows)
        self.metrics = metrics
        self.materiality_abs = materiality_abs
        self.materiality_pct = materiality_pct

    def list_schema(self) -> dict[str, list[str]]:
        return self.runner.schema()

    def run_sql(self, sql: str) -> list[dict[str, Any]]:
        return self.runner.run(sql)

    def list_lineage_nodes(self) -> list[dict[str, Any]]:
        return self.lineage.list_nodes()

    def get_lineage_node(self, node_id: str) -> dict[str, Any]:
        node = self.lineage.nodes[node_id]
        return {
            "node_id": node.node_id,
            "type": node.node_type,
            "grain": node.grain,
            "description": node.description,
            "columns": list(node.columns),
            "upstream": list(node.upstream),
        }

    def trace_upstream(self, node_id: str) -> list[str]:
        return self.lineage.trace_upstream(node_id)

    def get_variance_summary(self, period: str, compare_to: str) -> list[dict[str, Any]]:
        findings = compute_variances(
            self.db_path,
            period,
            compare_to,
            self.metrics,
            self.materiality_abs,
            self.materiality_pct,
            self.lineage,
        )
        return [finding.to_dict() for finding in findings]

    def get_metric_drivers(self, metric: str, period: str, compare_to: str) -> list[dict[str, Any]]:
        lineage_path = self.lineage.trace_metric(metric)
        return [
            driver.to_dict()
            for driver in get_metric_drivers(self.db_path, metric, period, compare_to, lineage_path)
        ]

    def as_json(self, payload: Any) -> str:
        return json.dumps(payload, indent=2, sort_keys=True)
