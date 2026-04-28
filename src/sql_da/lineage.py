"""YAML lineage graph parsing and traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LineageNode:
    """A table, view, or dimension in the lineage graph."""

    node_id: str
    node_type: str
    grain: str
    description: str
    columns: tuple[str, ...]
    upstream: tuple[str, ...] = ()


class LineageGraph:
    """In-memory lineage graph loaded from YAML."""

    def __init__(self, nodes: dict[str, LineageNode], metric_mappings: dict[str, dict[str, str]]):
        self.nodes = nodes
        self.metric_mappings = metric_mappings

    @classmethod
    def from_yaml(cls, path: str | Path) -> LineageGraph:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        nodes = {
            node_id: LineageNode(
                node_id=node_id,
                node_type=str(payload.get("type", "")),
                grain=str(payload.get("grain", "")),
                description=str(payload.get("description", "")),
                columns=tuple(payload.get("columns", [])),
                upstream=tuple(payload.get("upstream", [])),
            )
            for node_id, payload in raw.get("nodes", {}).items()
        }
        return cls(nodes=nodes, metric_mappings=raw.get("metric_mappings", {}))

    def validate(self) -> list[str]:
        errors: list[str] = []
        for node_id, node in self.nodes.items():
            if not node.node_type:
                errors.append(f"{node_id} is missing type")
            if not node.columns:
                errors.append(f"{node_id} has no columns")
            for upstream in node.upstream:
                if upstream not in self.nodes:
                    errors.append(f"{node_id} references missing upstream node {upstream}")
        for metric, mapping in self.metric_mappings.items():
            node_id = mapping.get("node")
            if node_id not in self.nodes:
                errors.append(f"metric {metric} references missing node {node_id}")
            if not mapping.get("value_column"):
                errors.append(f"metric {metric} is missing value_column")
        return errors

    def trace_upstream(self, node_id: str) -> list[str]:
        """Return a depth-first upstream lineage path with the requested node first."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown lineage node: {node_id}")
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(current: str) -> None:
            if current in seen:
                return
            seen.add(current)
            ordered.append(current)
            for upstream in self.nodes[current].upstream:
                visit(upstream)

        visit(node_id)
        return ordered

    def trace_metric(self, metric: str) -> list[str]:
        mapping = self.metric_mappings.get(metric)
        if mapping is None:
            raise KeyError(f"Unknown metric: {metric}")
        return self.trace_upstream(mapping["node"])

    def list_nodes(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.node_id,
                "type": node.node_type,
                "grain": node.grain,
                "description": node.description,
                "upstream": list(node.upstream),
            }
            for node in self.nodes.values()
        ]
