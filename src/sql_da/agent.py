"""Optional LangChain Deep Agents integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from sql_da.agent_context import AgentVirtualContext
from sql_da.config import ModelConfig
from sql_da.tools import SQLAnalysisToolkit


@dataclass(frozen=True)
class AgentReview:
    """Result of the optional live-agent review step."""

    enabled: bool
    content: str
    research_steps: list[dict[str, Any]] = field(default_factory=list)


class AgentUnavailable(RuntimeError):
    """Raised when optional Deep Agents dependencies or credentials are unavailable."""


SYSTEM_PROMPT = """You are a finance variance analyst.
Use lineage before writing root-cause claims. Use read-only SQL only.
Every conclusion must cite the metric, relevant lineage nodes, and SQL evidence.
Check data quality evidence before concluding a variance is purely business-driven.
Do not invent tables, columns, customers, vendors, or accounting explanations."""

SUBAGENTS = (
    (
        "lineage-mapper",
        "Trace upstream tables and columns for financial metrics.",
        "Map requested metrics to upstream lineage nodes before analysis. Use sql-lineage skills.",
    ),
    (
        "sql-investigator",
        "Gather read-only SQL evidence for variance drivers.",
        "Write safe SELECT queries, keep result sets narrow, and summarize evidence rows.",
    ),
    (
        "finance-analyst",
        "Convert evidence into finance-facing root-cause explanations.",
        "Explain variance drivers clearly for a finance audience. Use finance-variance skills.",
    ),
    (
        "report-reviewer",
        "Check that every report claim has SQL and lineage evidence.",
        "Reject unsupported claims, unrelated metric drivers, and unresolved limitations.",
    ),
)


def should_run_live_agent(model: ModelConfig) -> bool:
    return model.enable_live_agent and bool(os.getenv("GOOGLE_API_KEY"))


def create_variance_agent(
    model: ModelConfig,
    toolkit: SQLAnalysisToolkit,
    agent_context: AgentVirtualContext | None = None,
) -> Any:  # pragma: no cover
    """Create a Deep Agent with project-specific tools."""

    try:
        from deepagents import create_deep_agent
        from langchain_core.tools import tool
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise AgentUnavailable(str(exc)) from exc

    @tool
    def list_schema() -> str:
        """List database tables, views, and columns."""

        return toolkit.as_json(toolkit.list_schema())

    @tool
    def run_read_only_sql(sql: str) -> str:
        """Run one read-only SELECT or WITH SQL query against the finance database."""

        return toolkit.as_json(toolkit.run_sql(sql))

    @tool
    def list_lineage_nodes() -> str:
        """List all lineage nodes available to the agent."""

        return toolkit.as_json(toolkit.list_lineage_nodes())

    @tool
    def get_lineage_node(node_id: str) -> str:
        """Fetch metadata for one lineage node."""

        return toolkit.as_json(toolkit.get_lineage_node(node_id))

    @tool
    def trace_upstream(node_id: str) -> str:
        """Trace a lineage node upstream to raw source tables."""

        return toolkit.as_json(toolkit.trace_upstream(node_id))

    @tool
    def get_metric_drivers(metric: str, period: str, compare_to: str) -> str:
        """Fetch deterministic driver rows for one metric variance."""

        return toolkit.as_json(toolkit.get_metric_drivers(metric, period, compare_to))

    @tool
    def get_data_quality_findings(period: str) -> str:
        """Fetch deterministic data quality findings for one reporting period."""

        return toolkit.as_json(toolkit.get_data_quality_findings(period))

    tools = [
        list_schema,
        run_read_only_sql,
        list_lineage_nodes,
        get_lineage_node,
        trace_upstream,
        get_metric_drivers,
        get_data_quality_findings,
    ]
    skill_sources = list(agent_context.skill_sources) if agent_context else []
    memory_sources = list(agent_context.memory_sources) if agent_context else []
    return create_deep_agent(
        model=model.name,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=build_subagents(agent_context.skill_sources if agent_context else ()),
        skills=skill_sources or None,
        memory=memory_sources or None,
    )


def build_subagents(skill_sources: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Build custom Deep Agents subagent specs with isolated skill access."""

    return [
        {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "skills": list(skill_sources),
        }
        for name, description, system_prompt in SUBAGENTS
    ]


def run_agent_review(
    model: ModelConfig,
    toolkit: SQLAnalysisToolkit,
    findings: list[dict[str, Any]],
    data_quality_findings: list[dict[str, Any]] | None = None,
    agent_context: AgentVirtualContext | None = None,
) -> AgentReview:
    """Run a live Deep Agent review when credentials are present; otherwise skip."""

    if not should_run_live_agent(model):
        return AgentReview(
            enabled=False, content="Live Gemini review skipped: GOOGLE_API_KEY is not set."
        )
    try:
        agent = create_variance_agent(model, toolkit, agent_context)
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Review these deterministic variance findings. "
                        "Also review the deterministic data quality findings and call out "
                        "whether any material variance may be data-quality driven. "
                        "Use available memory, skills, and subagents where appropriate. "
                        "Identify unsupported claims and suggest concise improvements.\n"
                        f"{json.dumps(_review_payload(findings, data_quality_findings), indent=2)}"
                    ),
                }
            ]
        }
        if agent_context and agent_context.files:
            payload["files"] = agent_context.files
        response = agent.invoke(payload)
    except (
        Exception
    ) as exc:  # pragma: no cover - network/provider failures are environment-specific
        return AgentReview(enabled=True, content=f"Live Gemini review failed: {exc}")
    return AgentReview(
        enabled=True,
        content=_extract_agent_text(response),
        research_steps=_extract_agent_research_steps(response),
    )


def _review_payload(
    findings: list[dict[str, Any]],
    data_quality_findings: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "findings": findings,
        "data_quality_findings": data_quality_findings or [],
    }


def _extract_agent_text(response: Any) -> str:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    if not messages:
        return str(response)
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    return _content_to_text(content)


def _extract_agent_research_steps(response: Any) -> list[dict[str, Any]]:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    steps: list[dict[str, Any]] = []
    for index, message in enumerate(messages, start=1):
        role = _message_role(message)
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get(
                "tool_calls"
            )
        if tool_calls:
            for tool_call in tool_calls:
                steps.append(
                    {
                        "actor": role,
                        "action": "tool call",
                        "tool": _tool_name(tool_call),
                        "input_summary": tool_call,
                        "output_summary": "requested tool execution",
                        "metadata": {"message_index": index},
                    }
                )
        text = _content_to_text(content)
        if text:
            steps.append(
                {
                    "actor": role,
                    "action": "message",
                    "output_summary": text,
                    "metadata": {"message_index": index},
                }
            )
    return steps


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "agent")
    return str(getattr(message, "type", None) or getattr(message, "role", None) or "agent")


def _tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or tool_call.get("function", {}).get("name") or "")
    return str(getattr(tool_call, "name", ""))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)
