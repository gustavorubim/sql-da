"""Optional LangChain Deep Agents integration."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
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
    token_usage: dict[str, int] = field(default_factory=dict)
    cost_estimate: dict[str, Any] = field(default_factory=dict)


class AgentUnavailable(RuntimeError):
    """Raised when optional Deep Agents dependencies or credentials are unavailable."""


SYSTEM_PROMPT = """You are a finance variance analyst.
Use lineage before writing root-cause claims. Use read-only SQL only.
Use configured relationship and transformation specs before inferring joins, summaries, or formulas.
For large lineage graphs, retrieve only the relevant slice with search and node-context tools.
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

ProgressCallback = Callable[[dict[str, Any]], None]

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_SOURCE_LAST_UPDATED = "2026-04-22"


@dataclass(frozen=True)
class ModelTokenPricing:
    """Paid-tier text token prices per 1M tokens."""

    input_usd_per_1m: float
    output_usd_per_1m: float
    cached_input_usd_per_1m: float | None = None


MODEL_TOKEN_PRICING_USD_PER_1M = {
    "gemini-3-flash-preview": ModelTokenPricing(0.50, 3.00, 0.05),
}


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
        from langchain_google_genai import ChatGoogleGenerativeAI
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
    def trace_upstream(node_id: str, max_depth: int = 2) -> str:
        """Trace a lineage node upstream with a bounded maximum depth."""

        return toolkit.as_json(toolkit.trace_upstream(node_id, max_depth=max_depth))

    @tool
    def search_semantic_specs(query: str, limit: int = 10) -> str:
        """Search configured nodes, relationships, and transformations with bounded results."""

        return toolkit.as_json(toolkit.search_semantic_specs(query, limit))

    @tool
    def get_node_context(
        node_id: str,
        include_relationships: bool = True,
        include_transformations: bool = True,
        include_columns: bool = True,
        max_upstream_depth: int = 2,
        limit: int = 20,
    ) -> str:
        """Fetch a bounded context packet for one lineage node."""

        return toolkit.as_json(
            toolkit.get_node_context(
                node_id,
                include_relationships,
                include_transformations,
                include_columns,
                max_upstream_depth,
                limit,
            )
        )

    @tool
    def list_relationship_specs(node_id: str = "", limit: int = 20) -> str:
        """List configured join and relationship specs, optionally filtered by node."""

        return toolkit.as_json(toolkit.list_relationship_specs(node_id, limit))

    @tool
    def list_transformation_specs(node_id: str = "", limit: int = 20) -> str:
        """List configured transformation specs, optionally filtered by lineage node."""

        return toolkit.as_json(toolkit.list_transformation_specs(node_id, limit))

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
        search_semantic_specs,
        get_node_context,
        list_relationship_specs,
        list_transformation_specs,
        get_metric_drivers,
        get_data_quality_findings,
    ]
    skill_sources = list(agent_context.skill_sources) if agent_context else []
    memory_sources = list(agent_context.memory_sources) if agent_context else []
    return create_deep_agent(
        model=ChatGoogleGenerativeAI(**chat_google_genai_kwargs(model)),
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


def chat_google_genai_kwargs(model: ModelConfig) -> dict[str, Any]:
    """Build ChatGoogleGenerativeAI keyword arguments from project config."""

    kwargs = dict(model.args)
    kwargs.setdefault("temperature", model.temperature)
    kwargs["model"] = _normalize_model_name(model.name)
    return kwargs


def run_agent_review(
    model: ModelConfig,
    toolkit: SQLAnalysisToolkit,
    findings: list[dict[str, Any]],
    data_quality_findings: list[dict[str, Any]] | None = None,
    agent_context: AgentVirtualContext | None = None,
    on_step: ProgressCallback | None = None,
) -> AgentReview:
    """Run a live Deep Agent review when credentials are present; otherwise skip."""

    if not should_run_live_agent(model):
        return AgentReview(
            enabled=False,
            content="Live Gemini review skipped: GOOGLE_API_KEY is not set.",
            token_usage=_empty_token_usage(),
            cost_estimate=estimate_llm_cost(model.name, _empty_token_usage(), enabled=False),
        )
    try:
        agent = create_variance_agent(model, toolkit, agent_context)
        review_payload = _review_payload(
            findings,
            data_quality_findings,
            toolkit.get_semantic_summary(),
        )
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
                        f"{json.dumps(review_payload, indent=2)}"
                    ),
                }
            ]
        }
        if agent_context and agent_context.files:
            payload["files"] = agent_context.files
        response = _invoke_agent(agent, payload, on_step)
    except (
        Exception
    ) as exc:  # pragma: no cover - network/provider failures are environment-specific
        empty_usage = _empty_token_usage()
        return AgentReview(
            enabled=True,
            content=f"Live Gemini review failed: {exc}",
            token_usage=empty_usage,
            cost_estimate=estimate_llm_cost(model.name, empty_usage, enabled=True),
        )
    token_usage = extract_agent_token_usage(response)
    return AgentReview(
        enabled=True,
        content=_extract_agent_text(response),
        research_steps=_extract_agent_research_steps(response),
        token_usage=token_usage,
        cost_estimate=estimate_llm_cost(model.name, token_usage, enabled=True),
    )


def _review_payload(
    findings: list[dict[str, Any]],
    data_quality_findings: list[dict[str, Any]] | None,
    semantic_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "findings": findings,
        "data_quality_findings": data_quality_findings or [],
        "semantic_summary": semantic_summary or {},
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
        if role in {"human", "user"}:
            continue
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
        if role == "tool" and text:
            steps.append(
                {
                    "actor": role,
                    "action": "tool result",
                    "tool": _message_name(message),
                    "output_summary": text,
                    "metadata": {"message_index": index},
                }
            )
            continue
        if text:
            steps.append(
                {
                    "actor": role,
                    "action": "agent message",
                    "output_summary": text,
                    "metadata": {"message_index": index},
                }
            )
    return steps


def extract_agent_token_usage(response: Any) -> dict[str, int]:
    """Aggregate token usage reported on agent response messages."""

    usage = _empty_token_usage()
    messages = response.get("messages", []) if isinstance(response, dict) else []
    for message in messages:
        message_usage = _message_usage_metadata(message)
        if not message_usage:
            continue
        normalized = _normalize_token_usage(message_usage)
        for key, value in normalized.items():
            usage[key] += value
    if not usage["total_tokens"]:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def estimate_llm_cost(
    model_name: str,
    token_usage: dict[str, int],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Estimate paid-tier Gemini token cost from usage metadata."""

    normalized_model = _normalize_model_name(model_name)
    estimate: dict[str, Any] = {
        "model": model_name,
        "normalized_model": normalized_model,
        "currency": "USD",
        "pricing_source": PRICING_SOURCE_URL,
        "pricing_source_last_updated": PRICING_SOURCE_LAST_UPDATED,
        "input_tokens": int(token_usage.get("input_tokens", 0)),
        "output_tokens": int(token_usage.get("output_tokens", 0)),
        "total_tokens": int(token_usage.get("total_tokens", 0)),
        "cached_input_tokens": int(token_usage.get("cached_input_tokens", 0)),
        "total_cost_usd": 0.0 if not enabled else None,
        "status": "skipped" if not enabled else "unavailable",
        "note": (
            "Estimate uses Gemini API paid-tier standard text token rates and excludes "
            "free-tier credits, taxes, and non-token tool fees."
        ),
    }
    if not enabled:
        return estimate
    if not any(
        estimate[key]
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")
    ):
        estimate["note"] += " Token usage metadata was not returned by the provider."
        return estimate
    pricing = MODEL_TOKEN_PRICING_USD_PER_1M.get(normalized_model)
    if pricing is None:
        estimate["note"] += " No local pricing table entry exists for this model."
        return estimate

    cached_input_tokens = estimate["cached_input_tokens"]
    standard_input_tokens = max(estimate["input_tokens"] - cached_input_tokens, 0)
    input_cost = standard_input_tokens * pricing.input_usd_per_1m / 1_000_000
    output_cost = estimate["output_tokens"] * pricing.output_usd_per_1m / 1_000_000
    cached_input_cost = (
        cached_input_tokens * pricing.cached_input_usd_per_1m / 1_000_000
        if pricing.cached_input_usd_per_1m is not None
        else 0.0
    )
    estimate.update(
        {
            "status": "estimated",
            "input_usd_per_1m": pricing.input_usd_per_1m,
            "output_usd_per_1m": pricing.output_usd_per_1m,
            "cached_input_usd_per_1m": pricing.cached_input_usd_per_1m,
            "billable_input_tokens": standard_input_tokens,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "cached_input_cost_usd": round(cached_input_cost, 8),
            "total_cost_usd": round(input_cost + output_cost + cached_input_cost, 8),
        }
    )
    return estimate


def _invoke_agent(
    agent: Any,
    payload: dict[str, Any],
    on_step: ProgressCallback | None,
) -> Any:
    stream = getattr(agent, "stream", None)
    if on_step is None or not callable(stream):
        return agent.invoke(payload)
    return _stream_agent(agent, payload, on_step)


def _stream_agent(
    agent: Any,
    payload: dict[str, Any],
    on_step: ProgressCallback,
) -> Any:
    final_response: Any | None = None
    seen_messages: set[str] = set()
    collected_messages: list[Any] = []
    for chunk in _agent_stream_chunks(agent, payload):
        response_candidate = _response_candidate(chunk)
        if response_candidate is not None:
            final_response = response_candidate
        for message in _iter_stream_messages(chunk):
            message_key = _message_fingerprint(message)
            if message_key in seen_messages:
                continue
            seen_messages.add(message_key)
            collected_messages.append(message)
            for step in _steps_for_message(message, len(collected_messages)):
                on_step(step)
    if final_response is not None:
        final_messages = (
            final_response.get("messages", []) if isinstance(final_response, dict) else []
        )
        if len(final_messages) < len(collected_messages):
            return {"messages": collected_messages}
        return final_response
    return {"messages": collected_messages}


def _agent_stream_chunks(agent: Any, payload: dict[str, Any]) -> Any:
    try:
        yield from agent.stream(payload, stream_mode="values")
    except TypeError:
        yield from agent.stream(payload)


def _response_candidate(chunk: Any) -> Any | None:
    if isinstance(chunk, dict) and isinstance(chunk.get("messages"), list):
        return chunk
    if isinstance(chunk, dict):
        for value in chunk.values():
            if isinstance(value, dict) and isinstance(value.get("messages"), list):
                return value
    return None


def _iter_stream_messages(chunk: Any) -> list[Any]:
    if isinstance(chunk, dict) and isinstance(chunk.get("messages"), list):
        return list(chunk["messages"])
    messages: list[Any] = []
    if isinstance(chunk, dict):
        for value in chunk.values():
            if isinstance(value, dict) and isinstance(value.get("messages"), list):
                messages.extend(value["messages"])
    return messages


def _steps_for_message(message: Any, message_index: int) -> list[dict[str, Any]]:
    steps = _extract_agent_research_steps({"messages": [message]})
    return [
        {
            **step,
            "metadata": {**dict(step.get("metadata", {})), "message_index": message_index},
        }
        for step in steps
    ]


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "agent")
    return str(getattr(message, "type", None) or getattr(message, "role", None) or "agent")


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", "") or "")


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


def _message_fingerprint(message: Any) -> str:
    role = _message_role(message)
    name = _message_name(message)
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    message_id = getattr(message, "id", None)
    if message_id is None and isinstance(message, dict):
        message_id = message.get("id")
    if message_id:
        return f"id:{message_id}"
    return json.dumps(
        {
            "role": role,
            "name": name,
            "content": content,
            "tool_calls": _message_tool_calls(message),
        },
        sort_keys=True,
        default=str,
    )


def _message_tool_calls(message: Any) -> Any:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get(
            "tool_calls"
        )
    return tool_calls


def _message_usage_metadata(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    if usage is not None:
        return dict(usage)
    if isinstance(message, dict):
        for key in ("usage_metadata", "usage", "token_usage"):
            value = message.get(key)
            if isinstance(value, dict):
                return value
        response_metadata = message.get("response_metadata")
        if isinstance(response_metadata, dict):
            for key in ("usage_metadata", "usage", "token_usage"):
                value = response_metadata.get(key)
                if isinstance(value, dict):
                    return value
        additional_kwargs = message.get("additional_kwargs")
        if isinstance(additional_kwargs, dict):
            value = additional_kwargs.get("usage_metadata")
            if isinstance(value, dict):
                return value
    return {}


def _normalize_token_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = _first_int(
        usage, ("input_tokens", "prompt_tokens", "input_token_count", "prompt_token_count")
    )
    output_tokens = _first_int(
        usage,
        (
            "output_tokens",
            "completion_tokens",
            "output_token_count",
            "candidates_token_count",
        ),
    )
    total_tokens = _first_int(usage, ("total_tokens", "total_token_count"))
    cached_input_tokens = _first_int(
        usage,
        (
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
        ),
    )
    input_details = usage.get("input_token_details")
    if isinstance(input_details, dict):
        cached_input_tokens = max(
            cached_input_tokens,
            _first_int(input_details, ("cache_read", "cache_read_tokens", "cached_tokens")),
        )
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
    }


def _first_int(values: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


def _normalize_model_name(model_name: str) -> str:
    return model_name.rsplit(":", maxsplit=1)[-1]
