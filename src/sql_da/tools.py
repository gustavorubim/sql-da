"""Project-specific read-only tools exposed to the deep agent."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sql_da.data_quality import run_data_quality_checks
from sql_da.lineage import LineageGraph
from sql_da.sql_safety import ReadOnlySQLRunner, SQLSafetyError
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
        try:
            return self.runner.run(sql)
        except SQLSafetyError as exc:
            return [_sql_error("SQLSafetyError", str(exc))]
        except sqlite3.Error as exc:
            return [_sql_error("SQLExecutionError", str(exc))]

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

    def trace_upstream(self, node_id: str, max_depth: int | None = None) -> list[str]:
        return self.lineage.trace_upstream(node_id, max_depth=max_depth)

    def list_relationship_specs(
        self,
        node_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, str]]:
        return self.lineage.list_relationship_specs(node_id or None, limit=limit)

    def list_transformation_specs(
        self,
        node_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.lineage.list_transformation_specs(node_id or None, limit=limit)

    def get_semantic_specs(self) -> dict[str, Any]:
        return self.lineage.semantic_specs()

    def get_semantic_summary(self) -> dict[str, Any]:
        return self.lineage.semantic_summary()

    def search_semantic_specs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.lineage.search_semantic_specs(query, limit=limit)

    def get_node_context(
        self,
        node_id: str,
        include_relationships: bool = True,
        include_transformations: bool = True,
        include_columns: bool = True,
        max_upstream_depth: int = 2,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self.lineage.get_node_context(
            node_id,
            include_relationships=include_relationships,
            include_transformations=include_transformations,
            include_columns=include_columns,
            max_upstream_depth=max_upstream_depth,
            limit=limit,
        )

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
        metric = _metric_key(metric)
        if metric not in self.lineage.metric_mappings:
            return [
                {
                    "error": f"Unknown metric: {metric}",
                    "valid_metrics": sorted(self.lineage.metric_mappings),
                }
            ]
        lineage_path = self.lineage.trace_metric(metric)
        return [
            driver.to_dict()
            for driver in get_metric_drivers(self.db_path, metric, period, compare_to, lineage_path)
        ]

    def get_data_quality_findings(self, period: str) -> list[dict[str, Any]]:
        return [finding.to_dict() for finding in run_data_quality_checks(self.db_path, period)]

    def as_json(self, payload: Any) -> str:
        return json.dumps(payload, indent=2, sort_keys=True)


def _metric_key(metric: str) -> str:
    return "_".join(str(metric).strip().casefold().replace("-", " ").split())


def _sql_error(error_type: str, message: str) -> dict[str, str]:
    return {
        "error": message,
        "error_type": error_type,
        "hint": "Call list_schema and lineage context tools before retrying with valid columns.",
    }
