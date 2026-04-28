import csv
import json
from pathlib import Path

import pytest

from sql_da.agent import AgentReview
from sql_da.cli import main
from sql_da.config import ProjectConfig
from sql_da.lineage import LineageGraph
from sql_da.reports import evidence_payload, render_markdown
from sql_da.reward import RewardCheck, _keyword_failures, run_quality_gates, run_reward
from sql_da.synthetic import init_demo_database
from sql_da.terminal import TerminalReporter, print_json
from sql_da.variance import compute_variances
from sql_da.workflow import _cost_output_summary, analyze

ROOT = Path(__file__).resolve().parents[1]


def _config() -> ProjectConfig:
    return ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")


def test_workflow_writes_markdown_and_json(monkeypatch):
    config = _config()
    init_demo_database(config)
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    result = analyze(config, "2026-03", "2026-02", run_id="pytest_report")

    report = Path(result.report_path)
    evidence = Path(result.evidence_path)
    trace_json = Path(result.trace_json_path)
    trace_csv = Path(result.trace_csv_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    trace_payload = json.loads(trace_json.read_text(encoding="utf-8"))

    assert report.exists()
    assert evidence.exists()
    assert trace_json.exists()
    assert trace_csv.exists()
    assert result.to_dict()["run_id"] == "pytest_report"
    assert result.to_dict()["trace_csv_path"] == str(trace_csv)
    assert result.to_dict()["llm_cost_estimate"]["status"] == "skipped"
    assert result.to_dict()["llm_cost_estimate"]["total_cost_usd"] == 0.0
    report_text = report.read_text(encoding="utf-8")
    assert "Financial Variance Root-Cause Report" in report_text
    assert "Data Quality Findings" in report_text
    assert payload["findings"][0]["drivers"]
    assert {finding["issue_id"] for finding in payload["data_quality_findings"]} == {
        "invoice_after_contract_end",
        "duplicate_vendor_bill",
        "missing_gl_entry",
    }
    assert any(event["phase"] == "driver_research" for event in trace_payload)
    assert any(event["phase"] == "data_quality" for event in trace_payload)
    assert any(event["action"] == "estimate llm cost" for event in trace_payload)
    with trace_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sequence"] == "1"
    assert {"phase", "actor", "action", "output_summary", "metadata"} <= set(rows[0])
    assert "GOOGLE_API_KEY is not set" in payload["agent_review"]


def test_report_renderer_handles_empty_findings():
    config = _config()
    payload = evidence_payload(config, "empty", "2026-03", "2026-02", [], [], "no review")

    markdown = render_markdown(payload)

    assert "No material variances" in markdown
    assert "No data quality findings" in markdown
    assert "no review" in markdown


def test_report_renderer_handles_finding_without_driver_rows():
    config = _config()
    payload = evidence_payload(config, "manual", "2026-03", "2026-02", [], [], "review")
    payload["findings"] = [
        {
            "metric": "manual_metric",
            "current_value": 10.0,
            "prior_value": 5.0,
            "absolute_variance": 5.0,
            "percent_variance": None,
            "confidence": "low",
            "lineage_path": ["mart_financials_monthly"],
            "root_cause": "Manual test root cause.",
            "drivers": [],
        }
    ]

    markdown = render_markdown(payload)

    assert "No driver rows" in markdown


def test_reward_internal_checks_pass_without_quality_gates(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    config = _config()

    payload = run_reward(config, include_quality_gates=False)

    assert payload["passed"]
    assert payload["score"] == 1.0
    assert config.reward.output_path.exists()
    assert {check["name"] for check in payload["checks"]} >= {
        "demo_db_builds",
        "lineage_validates",
        "known_variance_drivers_detected",
        "known_data_quality_issues_detected",
        "sql_safety_blocks_mutation",
        "report_artifacts_have_evidence",
    }


def test_reward_quality_gates_can_be_scored_with_fake_subprocess(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    calls = []

    def fake_run(command, cwd, check, text, capture_output):
        calls.append((command, cwd, check, text, capture_output))
        return Completed()

    monkeypatch.setattr("sql_da.reward.subprocess.run", fake_run)

    checks = run_quality_gates(ROOT)

    assert len(checks) == 3
    assert all(isinstance(check, RewardCheck) and check.passed for check in checks)
    assert calls[0][0][-3:] == ["ruff", "format", "--check"] or "ruff" in calls[0][0]


def test_reward_quality_gate_branch_and_keyword_failures(monkeypatch):
    monkeypatch.setattr(
        "sql_da.reward.run_internal_checks",
        lambda config, reporter=None: [RewardCheck("internal", True, "ok")],
    )
    monkeypatch.setattr(
        "sql_da.reward.run_quality_gates",
        lambda project_root, reporter=None: [RewardCheck("quality", True, str(project_root))],
    )
    config = _config()

    payload = run_reward(config, include_quality_gates=True)

    assert payload["score"] == 1.0

    db_path = init_demo_database(config)
    findings = compute_variances(
        db_path,
        "2026-03",
        "2026-02",
        ("revenue",),
        1.0,
        0.0,
        LineageGraph.from_yaml(config.lineage_path),
    )

    assert _keyword_failures({}, {"revenue": {"expected_keywords": "missing"}}) == {}
    assert _keyword_failures(
        {"revenue": findings[0]},
        {"revenue": {"expected_keywords": "not-present"}},
    ) == {"revenue": ["not-present"]}


def test_workflow_initializes_missing_db_and_rejects_bad_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    config = _config()
    if config.database.path.exists():
        config.database.path.unlink()

    result = analyze(config, "2026-03", "2026-02", run_id="auto_init")

    assert Path(result.report_path).exists()
    assert Path(result.trace_json_path).exists()

    bad_lineage = tmp_path / "lineage.yaml"
    bad_lineage.write_text(
        "nodes: {}\nmetric_mappings:\n  revenue:\n    node: missing\n", encoding="utf-8"
    )
    bad_config_path = tmp_path / "demo.yaml"
    bad_config_path.write_text(
        f"""
database:
  path: {config.database.path}
lineage_path: {bad_lineage}
report_dir: {tmp_path / "reports"}
""",
        encoding="utf-8",
    )
    bad_config = ProjectConfig.from_yaml(bad_config_path)

    with pytest.raises(ValueError, match="Lineage validation failed"):
        analyze(bad_config, "2026-03", "2026-02")


def test_workflow_records_streamed_live_agent_steps_once(monkeypatch):
    config = _config()
    init_demo_database(config)

    def fake_run_agent_review(
        model,
        toolkit,
        findings,
        data_quality_findings=None,
        agent_context=None,
        on_step=None,
    ):
        step = {
            "actor": "assistant",
            "action": "agent message",
            "output_summary": "streamed review step",
        }
        if on_step:
            on_step(step)
        return AgentReview(
            enabled=True,
            content="review complete",
            research_steps=[step],
            token_usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cached_input_tokens": 0,
            },
            cost_estimate={
                "status": "estimated",
                "total_cost_usd": 0.00011,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        )

    monkeypatch.setattr("sql_da.workflow.run_agent_review", fake_run_agent_review)

    result = analyze(config, "2026-03", "2026-02", run_id="streamed_agent")
    trace_payload = json.loads(Path(result.trace_json_path).read_text(encoding="utf-8"))

    assert result.agent_review == "review complete"
    assert result.llm_cost_estimate["total_cost_usd"] == 0.00011
    assert sum(event["output_summary"] == "streamed review step" for event in trace_payload) == 1
    assert _cost_output_summary({"status": "unavailable"}) == (
        "LLM cost estimate unavailable (unavailable)"
    )


def test_cli_commands(monkeypatch, capsys):
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    assert main(["init-demo-db", "--config", str(ROOT / "configs/demo.yaml")]) == 0
    assert '"created": true' in capsys.readouterr().out

    assert main(["validate-lineage", "--lineage", str(ROOT / "configs/lineage.yaml")]) == 0
    assert '"valid": true' in capsys.readouterr().out

    bad_lineage = ROOT / "reports" / "bad_lineage.yaml"
    bad_lineage.write_text("nodes:\n  bad:\n    columns: []\n", encoding="utf-8")
    assert main(["validate-lineage", "--lineage", str(bad_lineage)]) == 1
    assert '"valid": false' in capsys.readouterr().out

    assert (
        main(
            [
                "analyze",
                "--config",
                str(ROOT / "configs/demo.yaml"),
                "--period",
                "2026-03",
                "--compare-to",
                "2026-02",
                "--run-id",
                "cli_test",
            ]
        )
        == 0
    )
    analyze_output = capsys.readouterr().out
    assert '"run_id": "cli_test"' in analyze_output
    assert '"trace_json_path":' in analyze_output
    assert '"data_quality_findings":' in analyze_output

    assert (
        main(
            [
                "analyze",
                "--config",
                str(ROOT / "configs/demo.yaml"),
                "--period",
                "2026-03",
                "--compare-to",
                "2026-02",
                "--run-id",
                "cli_verbose",
                "--verbose",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"run_id": "cli_verbose"' not in captured.out
    assert "Analysis complete" in captured.err

    assert (
        main(
            [
                "reward",
                "--config",
                str(ROOT / "configs/demo.yaml"),
                "--verbose",
                "--skip-quality-gates",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"passed": true' not in captured.out
    assert "Reward passed" in captured.err

    assert (
        main(
            [
                "reward",
                "--config",
                str(ROOT / "configs/demo.yaml"),
                "--skip-quality-gates",
            ]
        )
        == 0
    )
    assert '"passed": true' in capsys.readouterr().out


def test_rich_reporter_and_json_output(capsys):
    reporter = TerminalReporter(verbose=True)

    reporter.stage("Testing stage")
    with reporter.status("Testing status"):
        pass
    reporter.checks_table({"checks": [{"name": "demo", "passed": True, "detail": "ok"}]})
    reporter.trace_event(
        {
            "phase": "demo",
            "actor": "tester",
            "action": "trace event",
            "status": "ok",
            "output_summary": "done",
        }
    )
    reporter.trace_event(
        {
            "phase": "live_agent",
            "actor": "assistant",
            "action": "tool call",
            "tool": "get_metric_drivers",
            "status": "ok",
            "input_summary": {"metric": "revenue"},
            "output_summary": "",
        }
    )
    reporter.trace_event(
        {
            "phase": "live_agent",
            "actor": "tool",
            "action": "tool result",
            "tool": "get_metric_drivers",
            "status": "ok",
            "output_summary": "x" * 180,
        }
    )
    reporter.trace_event(
        {
            "phase": "live_agent",
            "actor": "assistant",
            "action": "agent message",
            "status": "error",
            "output_summary": "refining narrative",
        }
    )
    reporter.trace_event(
        {
            "phase": "live_agent",
            "actor": "workflow",
            "action": "estimate llm cost",
            "status": "unavailable",
            "metadata": {},
        }
    )
    reporter.trace_table(
        [
            {
                "sequence": 1,
                "phase": "demo",
                "actor": "tester",
                "action": "trace table",
                "output_summary": "done",
            }
        ]
    )
    print_json({"ok": True})

    captured = capsys.readouterr()
    assert '"ok": true' in captured.out
