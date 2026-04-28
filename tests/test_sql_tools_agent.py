from pathlib import Path

import pytest

from sql_da.agent import (
    AgentReview,
    ModelConfig,
    _extract_agent_research_steps,
    _extract_agent_text,
    _invoke_agent,
    _iter_stream_messages,
    _message_fingerprint,
    _message_usage_metadata,
    _normalize_token_usage,
    _response_candidate,
    _tool_name,
    build_subagents,
    chat_google_genai_kwargs,
    create_variance_agent,
    estimate_llm_cost,
    extract_agent_token_usage,
    run_agent_review,
    should_run_live_agent,
)
from sql_da.agent_context import build_agent_virtual_context
from sql_da.config import ProjectConfig
from sql_da.lineage import LineageGraph
from sql_da.sql_safety import ReadOnlySQLRunner, SQLSafetyError
from sql_da.synthetic import init_demo_database
from sql_da.tools import SQLAnalysisToolkit

ROOT = Path(__file__).resolve().parents[1]


def _toolkit() -> SQLAnalysisToolkit:
    config = ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")
    init_demo_database(config)
    lineage = LineageGraph.from_yaml(config.lineage_path)
    return SQLAnalysisToolkit(
        config.database.path,
        lineage,
        config.analysis.max_query_rows,
        config.analysis.metrics,
        config.analysis.materiality_abs,
        config.analysis.materiality_pct,
    )


def test_read_only_sql_runner_allows_selects_and_blocks_mutations():
    toolkit = _toolkit()
    runner = ReadOnlySQLRunner(toolkit.db_path, max_rows=1)

    assert runner.run("-- ok\nSELECT * FROM invoices")[0]["invoice_month"] == "2026-01"
    assert len(runner.run("WITH x AS (SELECT 1 AS value) SELECT value FROM x")) == 1
    assert "invoices" in runner.schema()

    for sql in [
        "",
        "DROP TABLE invoices",
        "SELECT 1; SELECT 2",
        "UPDATE invoices SET amount = 0",
        "SELECT replace('a', 'a', 'b')",
    ]:
        with pytest.raises(SQLSafetyError):
            runner.run(sql)


def test_analysis_toolkit_exposes_schema_lineage_and_variances():
    toolkit = _toolkit()

    assert "mart_financials_monthly" in toolkit.list_schema()
    assert toolkit.run_sql(
        "SELECT metric, value FROM mart_financials_monthly WHERE month = '2026-03'"
    )
    assert any(node["node_id"] == "invoices" for node in toolkit.list_lineage_nodes())
    assert toolkit.get_lineage_node("mart_revenue_monthly")["type"] == "mart_view"
    assert "invoices" in toolkit.trace_upstream("mart_revenue_monthly")
    assert any(
        relationship["relationship_id"] == "invoice_product"
        for relationship in toolkit.list_relationship_specs()
    )
    assert any(
        transformation["transformation_id"] == "cloud_cogs_bill_amount"
        for transformation in toolkit.list_transformation_specs("products")
    )
    assert "relationships" in toolkit.get_semantic_specs()
    assert "lineage" not in toolkit.as_json({"lineage": []}).casefold().replace("lineage", "")

    summary = toolkit.get_variance_summary("2026-03", "2026-02")
    drivers = toolkit.get_metric_drivers("vendor_expense", "2026-03", "2026-02")
    data_quality = toolkit.get_data_quality_findings("2026-03")

    assert any(item["metric"] == "revenue" for item in summary)
    assert drivers[0]["label"].startswith("ConsultCo")
    assert any(item["issue_id"] == "duplicate_vendor_bill" for item in data_quality)


def test_agent_review_skips_without_key(monkeypatch):
    toolkit = _toolkit()
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    review = run_agent_review(ModelConfig(enable_live_agent=True), toolkit, [])

    assert review == AgentReview(
        False,
        "Live Gemini review skipped: GOOGLE_API_KEY is not set.",
        token_usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        },
        cost_estimate=review.cost_estimate,
    )
    assert review.cost_estimate["status"] == "skipped"
    assert review.cost_estimate["total_cost_usd"] == 0.0
    assert not should_run_live_agent(ModelConfig(enable_live_agent=True))


def test_agent_context_loads_memory_and_skills():
    config = ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")

    context = build_agent_virtual_context(config.agent)

    assert "/memory/AGENTS.md" in context.files
    assert "/skills/finance-variance/SKILL.md" in context.files
    assert context.memory_sources == ("/memory/AGENTS.md",)
    assert context.skill_sources == ("/skills",)
    subagents = build_subagents(context.skill_sources)
    assert {subagent["name"] for subagent in subagents} == {
        "lineage-mapper",
        "sql-investigator",
        "finance-analyst",
        "report-reviewer",
    }
    assert all(subagent["skills"] == ["/skills"] for subagent in subagents)


def test_gemini_model_kwargs_and_agent_creation(monkeypatch):
    toolkit = _toolkit()
    captured = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured["chat_kwargs"] = kwargs

    def fake_create_deep_agent(**kwargs):
        captured["deep_agent_kwargs"] = kwargs
        return "agent"

    import deepagents
    import langchain_google_genai

    monkeypatch.setattr(
        langchain_google_genai, "ChatGoogleGenerativeAI", FakeChatGoogleGenerativeAI
    )
    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create_deep_agent)

    model = ModelConfig(
        name="google_genai:gemini-3-flash-preview",
        temperature=0.2,
        args={
            "top_p": 0.95,
            "request_timeout": 120,
            "temperature": 0.4,
            "model": "ignored-model-from-args",
        },
    )

    assert chat_google_genai_kwargs(model) == {
        "model": "gemini-3-flash-preview",
        "temperature": 0.4,
        "top_p": 0.95,
        "request_timeout": 120,
    }
    assert create_variance_agent(model, toolkit) == "agent"
    assert captured["chat_kwargs"] == chat_google_genai_kwargs(model)
    assert captured["deep_agent_kwargs"]["model"].__class__ is FakeChatGoogleGenerativeAI
    tool_names = {tool.name for tool in captured["deep_agent_kwargs"]["tools"]}
    assert {"list_relationship_specs", "list_transformation_specs", "get_semantic_specs"} <= (
        tool_names
    )


def test_agent_review_uses_fake_agent_when_key_present(monkeypatch):
    toolkit = _toolkit()
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    context = build_agent_virtual_context(ProjectConfig.from_yaml(ROOT / "configs/demo.yaml").agent)

    class FakeAgent:
        def invoke(self, payload):
            assert "messages" in payload
            assert "/memory/AGENTS.md" in payload["files"]
            assert '"semantic_specs"' in payload["messages"][0]["content"]
            assert "cloud_cogs_bill_amount" in payload["messages"][0]["content"]
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "review ok", "extras": "ignored"}],
                        "tool_calls": [
                            {"name": "get_metric_drivers", "args": {"metric": "revenue"}}
                        ],
                    }
                ]
            }

    monkeypatch.setattr(
        "sql_da.agent.create_variance_agent",
        lambda model, tools, agent_context: FakeAgent(),
    )

    review = run_agent_review(
        ModelConfig(enable_live_agent=True),
        toolkit,
        [{"metric": "revenue"}],
        [{"issue_id": "duplicate_vendor_bill"}],
        context,
    )

    assert should_run_live_agent(ModelConfig(enable_live_agent=True))
    assert review.enabled
    assert review.content == "review ok"
    assert review.research_steps[0]["tool"] == "get_metric_drivers"
    assert _extract_agent_text({"messages": []}) == "{'messages': []}"
    assert _extract_agent_text({"messages": [{"content": "dict content"}]}) == "dict content"


def test_agent_review_streams_live_steps_and_estimates_cost(monkeypatch):
    toolkit = _toolkit()
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    class FakeStreamingAgent:
        def invoke(self, payload):
            raise AssertionError("streaming path should avoid invoke")

        def stream(self, payload, stream_mode=None):
            assert stream_mode == "values"
            tool_call = {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_metric_drivers", "args": {"metric": "revenue"}}],
            }
            tool_result = {
                "role": "tool",
                "name": "get_metric_drivers",
                "content": "driver rows returned",
            }
            final_message = {
                "role": "assistant",
                "content": "review ok",
                "usage_metadata": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                },
            }
            yield {"messages": [{"role": "user", "content": "prompt"}, tool_call]}
            yield {"messages": [{"role": "user", "content": "prompt"}, tool_call, tool_result]}
            yield {
                "messages": [
                    {"role": "user", "content": "prompt"},
                    tool_call,
                    tool_result,
                    final_message,
                ]
            }

    monkeypatch.setattr(
        "sql_da.agent.create_variance_agent",
        lambda model, tools, agent_context: FakeStreamingAgent(),
    )
    streamed_steps = []

    review = run_agent_review(
        ModelConfig(enable_live_agent=True),
        toolkit,
        [{"metric": "revenue"}],
        on_step=streamed_steps.append,
    )

    assert review.enabled
    assert review.content == "review ok"
    assert [step["action"] for step in streamed_steps] == [
        "tool call",
        "tool result",
        "agent message",
    ]
    assert review.token_usage == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
        "cached_input_tokens": 0,
    }
    assert review.cost_estimate["status"] == "estimated"
    assert review.cost_estimate["total_cost_usd"] == 0.0011


def test_token_usage_and_cost_helpers_handle_cached_input():
    response = {
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "usage_metadata": {
                    "prompt_tokens": 500,
                    "completion_tokens": 100,
                    "input_token_details": {"cache_read": 200},
                },
            }
        ]
    }

    usage = extract_agent_token_usage(response)
    estimate = estimate_llm_cost("google_genai:gemini-3-flash-preview", usage, enabled=True)

    assert usage == {
        "input_tokens": 500,
        "output_tokens": 100,
        "total_tokens": 600,
        "cached_input_tokens": 200,
    }
    assert estimate["billable_input_tokens"] == 300
    assert estimate["total_cost_usd"] == 0.00046


def test_agent_message_helpers_handle_runtime_shapes():
    class ObjectToolCall:
        name = "object_tool"

    class ObjectMessage:
        type = "assistant"

        def __init__(self):
            self.content = [{"type": "text", "text": "object message"}, "tail"]
            self.tool_calls = [ObjectToolCall()]
            self.usage_metadata = {"prompt_token_count": 10, "candidates_token_count": 5}

    steps = _extract_agent_research_steps(
        {
            "messages": [
                {"role": "user", "content": "prompt is skipped"},
                ObjectMessage(),
                {"role": "tool", "name": "run_read_only_sql", "content": "rows returned"},
                {
                    "role": "assistant",
                    "content": None,
                    "additional_kwargs": {
                        "tool_calls": [{"function": {"name": "nested_tool"}}],
                    },
                },
            ]
        }
    )

    assert [step["action"] for step in steps] == [
        "tool call",
        "agent message",
        "tool result",
        "tool call",
    ]
    assert steps[0]["tool"] == "object_tool"
    assert steps[2]["tool"] == "run_read_only_sql"
    assert steps[3]["tool"] == "nested_tool"
    assert _tool_name(ObjectToolCall()) == "object_tool"
    assert _message_fingerprint({"id": "msg-1", "content": "ignored"}) == "id:msg-1"
    assert "nested_tool" in _message_fingerprint(
        {
            "role": "assistant",
            "content": None,
            "additional_kwargs": {
                "tool_calls": [{"function": {"name": "nested_tool"}}],
            },
        }
    )
    assert _message_usage_metadata(ObjectMessage()) == {
        "prompt_token_count": 10,
        "candidates_token_count": 5,
    }
    assert _message_usage_metadata({"usage": {"input_tokens": 1}}) == {"input_tokens": 1}
    assert _message_usage_metadata(
        {"response_metadata": {"token_usage": {"total_token_count": 2}}}
    ) == {"total_token_count": 2}
    assert _message_usage_metadata(
        {"additional_kwargs": {"usage_metadata": {"output_tokens": 3}}}
    ) == {"output_tokens": 3}
    assert _message_usage_metadata({"content": "no usage"}) == {}
    assert _normalize_token_usage(
        {"input_tokens": True, "prompt_tokens": "12", "completion_tokens": 3.8}
    ) == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "cached_input_tokens": 0,
    }
    assert (
        estimate_llm_cost("google_genai:unknown", {"input_tokens": 1}, enabled=True)["status"]
        == "unavailable"
    )
    assert (
        estimate_llm_cost("google_genai:gemini-3-flash-preview", {}, enabled=True)["status"]
        == "unavailable"
    )


def test_agent_invocation_stream_fallbacks():
    class InvokeOnlyAgent:
        def invoke(self, payload):
            return {"messages": [{"role": "assistant", "content": payload["content"]}]}

    class TypeErrorStreamingAgent:
        def stream(self, payload, stream_mode=None):
            if stream_mode is not None:
                raise TypeError("stream_mode unsupported")
            yield {"node": {"messages": [{"role": "assistant", "content": "nested"}]}}

    class ShortFinalStreamingAgent:
        def stream(self, payload, stream_mode=None):
            yield {"messages": [{"role": "assistant", "content": "kept"}]}
            yield {"messages": []}

    assert (
        _invoke_agent(InvokeOnlyAgent(), {"content": "invoked"}, None)["messages"][0]["content"]
        == "invoked"
    )

    streamed_steps = []
    nested_response = _invoke_agent(TypeErrorStreamingAgent(), {}, streamed_steps.append)
    assert nested_response["messages"][0]["content"] == "nested"
    assert streamed_steps[0]["output_summary"] == "nested"

    short_response = _invoke_agent(ShortFinalStreamingAgent(), {}, lambda step: None)
    assert short_response["messages"][0]["content"] == "kept"
    assert _response_candidate("not a chunk") is None
    assert _iter_stream_messages("not a chunk") == []
