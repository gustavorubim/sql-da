"""Top-level analysis workflow."""

from __future__ import annotations

from pathlib import Path

from sql_da.agent import run_agent_review
from sql_da.agent_context import build_agent_virtual_context
from sql_da.config import ProjectConfig
from sql_da.lineage import LineageGraph
from sql_da.models import AnalysisResult
from sql_da.reports import write_report_artifacts
from sql_da.synthetic import init_demo_database
from sql_da.terminal import TerminalReporter
from sql_da.tools import SQLAnalysisToolkit
from sql_da.trace import AgentTrace
from sql_da.variance import compute_variances


def analyze(
    config: ProjectConfig,
    period: str,
    compare_to: str,
    run_id: str | None = None,
    reporter: TerminalReporter | None = None,
) -> AnalysisResult:
    """Run deterministic variance analysis and write report artifacts."""

    reporter = reporter or TerminalReporter()
    trace = AgentTrace(reporter=reporter)
    resolved_run_id = run_id or f"{period}_vs_{compare_to}"
    run_dir = config.report_dir / resolved_run_id
    trace.record(
        "orchestration",
        "workflow",
        "start analysis run",
        input_summary={"period": period, "compare_to": compare_to},
        output_summary=f"run_id={resolved_run_id}",
    )
    if not config.database.path.exists():
        with (
            reporter.status("Building synthetic SQLite demo database"),
            trace.span(
                "data_setup",
                "workflow",
                "build synthetic SQLite demo database",
                tool="init_demo_database",
                output_summary=str(config.database.path),
            ),
        ):
            init_demo_database(config)
    else:
        trace.record(
            "data_setup",
            "workflow",
            "reuse existing SQLite demo database",
            output_summary=str(config.database.path),
        )
    with (
        reporter.status("Loading and validating lineage graph"),
        trace.span(
            "lineage",
            "lineage-mapper",
            "load lineage graph",
            tool="LineageGraph.from_yaml",
            input_summary=str(config.lineage_path),
        ),
    ):
        lineage = LineageGraph.from_yaml(config.lineage_path)
    errors = lineage.validate()
    if errors:
        trace.record(
            "lineage",
            "lineage-mapper",
            "validate lineage graph",
            output_summary="; ".join(errors),
            status="error",
        )
        raise ValueError("Lineage validation failed: " + "; ".join(errors))
    trace.record(
        "lineage",
        "lineage-mapper",
        "validate lineage graph",
        output_summary=f"{len(lineage.nodes)} nodes; {len(lineage.metric_mappings)} metrics",
    )

    with (
        reporter.status(f"Computing material variances for {period} vs {compare_to}"),
        trace.span(
            "variance_detection",
            "sql-investigator",
            "compute material variances",
            tool="compute_variances",
            input_summary={"metrics": config.analysis.metrics},
        ),
    ):
        findings = compute_variances(
            config.database.path,
            period,
            compare_to,
            config.analysis.metrics,
            config.analysis.materiality_abs,
            config.analysis.materiality_pct,
            lineage,
        )
    trace.record(
        "variance_detection",
        "sql-investigator",
        "rank material findings",
        output_summary=f"{len(findings)} material findings",
        metadata={"metrics": [finding.metric for finding in findings]},
    )
    _record_finding_research(trace, findings)
    toolkit = SQLAnalysisToolkit(
        config.database.path,
        lineage,
        config.analysis.max_query_rows,
        config.analysis.metrics,
        config.analysis.materiality_abs,
        config.analysis.materiality_pct,
    )
    with (
        reporter.status("Loading Deep Agents memory and skills"),
        trace.span(
            "agent_context",
            "workflow",
            "load memory and skills",
            tool="build_agent_virtual_context",
            input_summary={
                "memory_paths": [str(path) for path in config.agent.memory_paths],
                "skill_paths": [str(path) for path in config.agent.skill_paths],
            },
        ),
    ):
        agent_context = build_agent_virtual_context(config.agent)
    trace.record(
        "agent_context",
        "workflow",
        "prepare virtual Deep Agents files",
        output_summary=(
            f"{len(agent_context.memory_sources)} memory sources; "
            f"{len(agent_context.skill_sources)} skill sources; "
            f"{len(agent_context.files)} virtual files"
        ),
        metadata={
            "memory_sources": list(agent_context.memory_sources),
            "skill_sources": list(agent_context.skill_sources),
        },
    )
    with (
        reporter.status("Running optional Gemini Deep Agent review"),
        trace.span(
            "live_agent",
            "deep-agent",
            "review deterministic findings",
            tool="create_deep_agent.invoke",
            input_summary=f"{len(findings)} findings",
        ),
    ):
        review = run_agent_review(
            config.model,
            toolkit,
            [finding.to_dict() for finding in findings],
            agent_context,
        )
    trace.record(
        "live_agent",
        "deep-agent",
        "summarize review result",
        output_summary=review.content,
        status="ok" if review.enabled else "skipped",
        metadata={"enabled": review.enabled},
    )
    trace.extend_agent_steps(review.research_steps)
    with reporter.status("Writing Markdown and JSON report artifacts"):
        report_path, evidence_path = write_report_artifacts(
            config, resolved_run_id, period, compare_to, findings, review.content
        )
        trace.record(
            "artifact",
            "workflow",
            "write report artifacts",
            output_summary="variance report and evidence JSON written",
            artifact=str(report_path),
            metadata={"evidence_path": str(evidence_path)},
        )
        trace_json_path, trace_csv_path = trace.write_artifacts(run_dir)
        trace.record(
            "artifact",
            "workflow",
            "write trace artifacts",
            output_summary="agent trace JSON and CSV written",
            artifact=str(trace_json_path),
            metadata={"trace_csv_path": str(trace_csv_path)},
        )
        trace_json_path, trace_csv_path = trace.write_artifacts(run_dir)
    reporter.trace_table([event.to_dict() for event in trace.events])
    return AnalysisResult(
        run_id=resolved_run_id,
        report_path=str(Path(report_path)),
        evidence_path=str(Path(evidence_path)),
        trace_json_path=str(Path(trace_json_path)),
        trace_csv_path=str(Path(trace_csv_path)),
        findings=findings,
        agent_review=review.content,
    )


def _record_finding_research(trace: AgentTrace, findings: list) -> None:
    for finding in findings:
        trace.record(
            "lineage",
            "lineage-mapper",
            "trace metric lineage",
            input_summary=finding.metric,
            output_summary=" -> ".join(finding.lineage_path),
            metadata={"metric": finding.metric},
        )
        trace.record(
            "root_cause",
            "finance-analyst",
            "draft deterministic root cause",
            input_summary=finding.metric,
            output_summary=finding.root_cause,
            metadata={
                "metric": finding.metric,
                "absolute_variance": finding.absolute_variance,
                "percent_variance": finding.percent_variance,
            },
        )
        for driver in finding.drivers:
            trace.record(
                "driver_research",
                "sql-investigator",
                "inspect variance driver",
                tool="get_metric_drivers",
                input_summary=f"{finding.metric}: {driver.label}",
                output_summary=(
                    f"current={driver.current_value:,.2f}; "
                    f"prior={driver.prior_value:,.2f}; "
                    f"variance={driver.variance:,.2f}"
                ),
                metadata={
                    "metric": finding.metric,
                    "driver": driver.label,
                    "lineage_nodes": driver.lineage_nodes,
                    "row_count": len(driver.rows),
                },
            )
