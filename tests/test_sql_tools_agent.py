from pathlib import Path

import pytest

from sql_da.agent import (
    AgentReview,
    ModelConfig,
    _extract_agent_text,
    build_subagents,
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
    assert "lineage" not in toolkit.as_json({"lineage": []}).casefold().replace("lineage", "")

    summary = toolkit.get_variance_summary("2026-03", "2026-02")
    drivers = toolkit.get_metric_drivers("vendor_expense", "2026-03", "2026-02")

    assert any(item["metric"] == "revenue" for item in summary)
    assert drivers[0]["label"].startswith("ConsultCo")


def test_agent_review_skips_without_key(monkeypatch):
    toolkit = _toolkit()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    review = run_agent_review(ModelConfig(enable_live_agent=True), toolkit, [])

    assert review == AgentReview(False, "Live Gemini review skipped: GOOGLE_API_KEY is not set.")
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


def test_agent_review_uses_fake_agent_when_key_present(monkeypatch):
    toolkit = _toolkit()
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    context = build_agent_virtual_context(ProjectConfig.from_yaml(ROOT / "configs/demo.yaml").agent)

    class FakeAgent:
        def invoke(self, payload):
            assert "messages" in payload
            assert "/memory/AGENTS.md" in payload["files"]
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
        context,
    )

    assert should_run_live_agent(ModelConfig(enable_live_agent=True))
    assert review.enabled
    assert review.content == "review ok"
    assert review.research_steps[0]["tool"] == "get_metric_drivers"
    assert _extract_agent_text({"messages": []}) == "{'messages': []}"
    assert _extract_agent_text({"messages": [{"content": "dict content"}]}) == "dict content"
