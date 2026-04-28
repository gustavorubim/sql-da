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


@dataclass(frozen=True)
class RelationshipSpec:
    """A configured join or relationship between two lineage nodes."""

    relationship_id: str
    left: str
    right: str
    relationship_type: str = ""
    join_type: str = ""
    description: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "relationship_id": self.relationship_id,
            "left": self.left,
            "right": self.right,
            "relationship_type": self.relationship_type,
            "join_type": self.join_type,
            "description": self.description,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TransformationSpec:
    """A configured column, metric, or table transformation."""

    transformation_id: str
    output: str
    expression: str
    description: str = ""
    grain: str = ""
    operation: str = ""
    inputs: tuple[str, ...] = ()
    source_nodes: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "output": self.output,
            "expression": self.expression,
            "description": self.description,
            "grain": self.grain,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "source_nodes": list(self.source_nodes),
            "group_by": list(self.group_by),
            "filters": list(self.filters),
            "notes": self.notes,
        }


class LineageGraph:
    """In-memory lineage graph loaded from YAML."""

    def __init__(
        self,
        nodes: dict[str, LineageNode],
        metric_mappings: dict[str, dict[str, str]],
        relationships: dict[str, RelationshipSpec] | None = None,
        transformations: dict[str, TransformationSpec] | None = None,
    ):
        self.nodes = nodes
        self.metric_mappings = metric_mappings
        self.relationships = relationships or {}
        self.transformations = transformations or {}

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
        relationships = {
            spec_id: _relationship_spec(spec_id, payload)
            for spec_id, payload in _spec_items(raw.get("relationships", {}), "relationships")
        }
        transformations = {
            spec_id: _transformation_spec(spec_id, payload)
            for spec_id, payload in _spec_items(raw.get("transformations", {}), "transformations")
        }
        return cls(
            nodes=nodes,
            metric_mappings=raw.get("metric_mappings", {}),
            relationships=relationships,
            transformations=transformations,
        )

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
        errors.extend(self._validate_relationships())
        errors.extend(self._validate_transformations())
        return errors

    def trace_upstream(self, node_id: str, max_depth: int | None = None) -> list[str]:
        """Return a depth-first upstream lineage path with the requested node first."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown lineage node: {node_id}")
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(current: str, depth: int) -> None:
            if current in seen:
                return
            seen.add(current)
            ordered.append(current)
            if max_depth is not None and depth >= max(max_depth, 0):
                return
            for upstream in self.nodes[current].upstream:
                visit(upstream, depth + 1)

        visit(node_id, 0)
        return ordered

    def trace_metric(self, metric: str) -> list[str]:
        metric_key = _metric_key(metric)
        mapping = self.metric_mappings.get(metric) or self.metric_mappings.get(metric_key)
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

    def list_relationship_specs(
        self,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Return configured join and relationship specs."""

        relationships = list(self.relationships.values())
        if node_id:
            relationships = [
                relationship
                for relationship in relationships
                if self._resolve_reference_node(relationship.left) == node_id
                or self._resolve_reference_node(relationship.right) == node_id
            ]
        return [relationship.to_dict() for relationship in _limited(relationships, limit)]

    def list_transformation_specs(
        self,
        node_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return configured transformation specs, optionally filtered by node."""

        specs = list(self.transformations.values())
        if node_id:
            specs = [
                spec
                for spec in specs
                if node_id in self._transformation_node_ids(spec)
                or self._resolve_reference_node(spec.output) == node_id
            ]
        return [spec.to_dict() for spec in _limited(specs, limit)]

    def semantic_specs(self) -> dict[str, Any]:
        """Return all configured relationship and transformation specs."""

        return {
            "relationships": self.list_relationship_specs(),
            "transformations": self.list_transformation_specs(),
        }

    def semantic_summary(self) -> dict[str, Any]:
        """Return compact semantic-layer counts for model prompts."""

        return {
            "node_count": len(self.nodes),
            "relationship_count": len(self.relationships),
            "transformation_count": len(self.transformations),
            "metric_count": len(self.metric_mappings),
            "metric_mappings": {
                metric: {
                    "node": mapping.get("node", ""),
                    "value_column": mapping.get("value_column", ""),
                }
                for metric, mapping in self.metric_mappings.items()
            },
            "retrieval_guidance": (
                "Use search_semantic_specs and get_node_context to retrieve only the "
                "lineage slice needed for the current metric, table, or column."
            ),
        }

    def search_semantic_specs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search nodes, relationships, and transformations with bounded results."""

        terms = _query_terms(query)
        results: list[dict[str, Any]] = []
        for node in self.nodes.values():
            payload = {
                "node_id": node.node_id,
                "type": node.node_type,
                "grain": node.grain,
                "description": node.description,
                "columns": list(node.columns),
                "upstream": list(node.upstream),
            }
            score = _match_score(terms, payload)
            if score:
                results.append(
                    {
                        "kind": "node",
                        "id": node.node_id,
                        "score": score,
                        "summary": f"{node.node_type}: {node.description}",
                        "spec": payload,
                    }
                )
        for relationship in self.relationships.values():
            payload = relationship.to_dict()
            score = _match_score(terms, payload)
            if score:
                results.append(
                    {
                        "kind": "relationship",
                        "id": relationship.relationship_id,
                        "score": score,
                        "summary": f"{relationship.left} -> {relationship.right}",
                        "spec": payload,
                    }
                )
        for transformation in self.transformations.values():
            payload = transformation.to_dict()
            score = _match_score(terms, payload)
            if score:
                results.append(
                    {
                        "kind": "transformation",
                        "id": transformation.transformation_id,
                        "score": score,
                        "summary": f"{transformation.output}: {transformation.expression}",
                        "spec": payload,
                    }
                )
        results.sort(key=lambda item: (-int(item["score"]), str(item["kind"]), str(item["id"])))
        return results[: max(limit, 0)]

    def get_node_context(
        self,
        node_id: str,
        *,
        include_relationships: bool = True,
        include_transformations: bool = True,
        include_columns: bool = True,
        max_upstream_depth: int = 2,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return a bounded context packet for one node."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown lineage node: {node_id}")
        node = self.nodes[node_id]
        context: dict[str, Any] = {
            "node": {
                "node_id": node.node_id,
                "type": node.node_type,
                "grain": node.grain,
                "description": node.description,
                "upstream": list(node.upstream),
            },
            "upstream_path": self.trace_upstream(node_id, max_depth=max_upstream_depth),
            "downstream": [
                candidate.node_id
                for candidate in self.nodes.values()
                if node_id in candidate.upstream
            ][: max(limit, 0)],
            "metric_mappings": {
                metric: mapping
                for metric, mapping in self.metric_mappings.items()
                if mapping.get("node") == node_id
            },
        }
        if include_columns:
            context["node"]["columns"] = list(node.columns)
        if include_relationships:
            context["relationships"] = self.list_relationship_specs(node_id, limit=limit)
        if include_transformations:
            context["transformations"] = self.list_transformation_specs(node_id, limit=limit)
        return context

    def _validate_relationships(self) -> list[str]:
        errors: list[str] = []
        for relationship in self.relationships.values():
            if not relationship.left:
                errors.append(f"relationship {relationship.relationship_id} is missing left")
            if not relationship.right:
                errors.append(f"relationship {relationship.relationship_id} is missing right")
            for side_name, reference in (
                ("left", relationship.left),
                ("right", relationship.right),
            ):
                errors.extend(
                    self._validate_reference(
                        reference,
                        f"relationship {relationship.relationship_id} {side_name}",
                    )
                )
        return errors

    def _validate_transformations(self) -> list[str]:
        errors: list[str] = []
        for transformation in self.transformations.values():
            if not transformation.output:
                errors.append(
                    f"transformation {transformation.transformation_id} is missing output"
                )
            else:
                errors.extend(
                    self._validate_reference(
                        transformation.output,
                        f"transformation {transformation.transformation_id} output",
                    )
                )
            if not transformation.expression:
                errors.append(
                    f"transformation {transformation.transformation_id} is missing expression"
                )
            for source_node in transformation.source_nodes:
                if source_node not in self.nodes:
                    errors.append(
                        "transformation "
                        f"{transformation.transformation_id} references missing source node "
                        f"{source_node}"
                    )
            for reference in (*transformation.inputs, *transformation.group_by):
                errors.extend(
                    self._validate_reference(
                        reference,
                        f"transformation {transformation.transformation_id} reference",
                    )
                )
        return errors

    def _validate_reference(self, reference: str, context: str) -> list[str]:
        if not reference:
            return [f"{context} is empty"]
        resolved = self._resolve_reference(reference)
        if resolved is None:
            return [f"{context} {reference} does not match a known node"]
        node_id, column = resolved
        if column and column not in self.nodes[node_id].columns:
            return [f"{context} {reference} references missing column {column} on {node_id}"]
        return []

    def _resolve_reference(self, reference: str) -> tuple[str, str] | None:
        if reference in self.nodes:
            return reference, ""
        for node_id in sorted(self.nodes, key=len, reverse=True):
            prefix = f"{node_id}."
            if reference.startswith(prefix):
                return node_id, reference.removeprefix(prefix)
        return None

    def _resolve_reference_node(self, reference: str) -> str | None:
        resolved = self._resolve_reference(reference)
        return resolved[0] if resolved else None

    def _transformation_node_ids(self, transformation: TransformationSpec) -> set[str]:
        node_ids = set(transformation.source_nodes)
        for reference in (*transformation.inputs, *transformation.group_by):
            node_id = self._resolve_reference_node(reference)
            if node_id:
                node_ids.add(node_id)
        return node_ids


def _spec_items(raw: Any, section: str) -> list[tuple[str, dict[str, Any]]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            (str(spec_id), _ensure_mapping(payload, f"{section}.{spec_id}"))
            for spec_id, payload in raw.items()
        ]
    if isinstance(raw, list):
        items = []
        for index, payload in enumerate(raw, start=1):
            mapping = _ensure_mapping(payload, f"{section}[{index}]")
            spec_id = str(mapping.get("id") or mapping.get("name") or "")
            if not spec_id:
                raise ValueError(f"{section}[{index}] is missing id")
            items.append((spec_id, mapping))
        return items
    raise ValueError(f"{section} must be a mapping or list")


def _ensure_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _relationship_spec(spec_id: str, payload: dict[str, Any]) -> RelationshipSpec:
    return RelationshipSpec(
        relationship_id=spec_id,
        left=str(payload.get("left", "")),
        right=str(payload.get("right", "")),
        relationship_type=str(payload.get("relationship_type", payload.get("cardinality", ""))),
        join_type=str(payload.get("join_type", "")),
        description=str(payload.get("description", "")),
        notes=str(payload.get("notes", "")),
    )


def _transformation_spec(spec_id: str, payload: dict[str, Any]) -> TransformationSpec:
    return TransformationSpec(
        transformation_id=spec_id,
        output=str(payload.get("output", "")),
        expression=str(payload.get("expression", "")),
        description=str(payload.get("description", "")),
        grain=str(payload.get("grain", "")),
        operation=str(payload.get("operation", "")),
        inputs=_string_tuple(payload.get("inputs", ())),
        source_nodes=_string_tuple(payload.get("source_nodes", ())),
        group_by=_string_tuple(payload.get("group_by", ())),
        filters=_string_tuple(payload.get("filters", ())),
        notes=str(payload.get("notes", "")),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        raise ValueError("spec list values must be strings or lists of strings")
    return tuple(str(item) for item in value)


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return values
    return values[: max(limit, 0)]


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(term for term in query.casefold().replace("_", " ").split() if term)


def _match_score(terms: tuple[str, ...], payload: dict[str, Any]) -> int:
    if not terms:
        return 1
    text = " ".join(str(value) for value in _flatten_values(payload)).casefold().replace("_", " ")
    return sum(text.count(term) for term in terms)


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        flattened: list[Any] = []
        for key, item in value.items():
            flattened.extend([key, *_flatten_values(item)])
        return flattened
    if isinstance(value, list | tuple):
        flattened = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    return [value]


def _metric_key(metric: str) -> str:
    return "_".join(str(metric).strip().casefold().replace("-", " ").split())
