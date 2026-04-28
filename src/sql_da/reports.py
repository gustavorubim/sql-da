"""Markdown and JSON report rendering."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sql_da.config import ProjectConfig, config_to_dict
from sql_da.models import DataQualityFinding, VarianceFinding


def write_report_artifacts(
    config: ProjectConfig,
    run_id: str,
    period: str,
    compare_to: str,
    findings: list[VarianceFinding],
    data_quality_findings: list[DataQualityFinding],
    agent_review: str,
) -> tuple[Path, Path]:
    """Write Markdown and JSON evidence artifacts for an analysis run."""

    run_dir = config.report_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "variance_report.md"
    evidence_path = run_dir / "variance_evidence.json"
    evidence = evidence_payload(
        config,
        run_id,
        period,
        compare_to,
        findings,
        data_quality_findings,
        agent_review,
    )
    report_path.write_text(render_markdown(evidence), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return report_path, evidence_path


def evidence_payload(
    config: ProjectConfig,
    run_id: str,
    period: str,
    compare_to: str,
    findings: list[VarianceFinding],
    data_quality_findings: list[DataQualityFinding],
    agent_review: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "period": period,
        "compare_to": compare_to,
        "config": config_to_dict(config),
        "agent_review": agent_review,
        "findings": [finding.to_dict() for finding in findings],
        "data_quality_findings": [finding.to_dict() for finding in data_quality_findings],
    }


def render_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# Financial Variance Root-Cause Report",
        "",
        f"Run: `{evidence['run_id']}`",
        f"Period: `{evidence['period']}` vs `{evidence['compare_to']}`",
        "",
        "## Executive Summary",
        "",
    ]
    findings = evidence["findings"]
    if not findings:
        lines.append("No material variances met the configured thresholds.")
    else:
        lines.append(f"{len(findings)} material variances were identified and investigated.")
    dq_findings = evidence.get("data_quality_findings", [])
    if dq_findings:
        lines.append(
            f"{len(dq_findings)} data quality issue(s) were found that may affect reporting."
        )
    lines.extend(["", "## Findings", ""])
    for finding in findings:
        lines.extend(_finding_markdown(finding))
    lines.extend(["", "## Data Quality Findings", ""])
    if not dq_findings:
        lines.append("No data quality findings were detected for this period.")
    for finding in dq_findings:
        lines.extend(_data_quality_markdown(finding))
    lines.extend(["", "## Agent Review", "", evidence["agent_review"], ""])
    return "\n".join(lines)


def _finding_markdown(finding: dict[str, Any]) -> list[str]:
    pct = finding["percent_variance"]
    pct_text = "n/a" if pct is None else f"{pct:.1%}"
    rows = [
        f"### {finding['metric']}",
        "",
        f"- Current value: {finding['current_value']:,.2f}",
        f"- Prior value: {finding['prior_value']:,.2f}",
        f"- Variance: {finding['absolute_variance']:,.2f} ({pct_text})",
        f"- Confidence: {finding['confidence']}",
        f"- Lineage: {' -> '.join(finding['lineage_path'])}",
        "",
        finding["root_cause"],
        "",
        "| Driver | Current | Prior | Variance |",
        "| --- | ---: | ---: | ---: |",
    ]
    for driver in finding["drivers"]:
        rows.append(
            f"| {driver['label']} | {driver['current_value']:,.2f} | "
            f"{driver['prior_value']:,.2f} | {driver['variance']:,.2f} |"
        )
    if not finding["drivers"]:
        rows.append("| No driver rows | 0.00 | 0.00 | 0.00 |")
    rows.append("")
    return rows


def _data_quality_markdown(finding: dict[str, Any]) -> list[str]:
    rows = [
        f"### {finding['issue_id']}",
        "",
        f"- Severity: {finding['severity']}",
        f"- Table: `{finding['table_name']}`",
        f"- Period: `{finding['period']}`",
        f"- Lineage: {' -> '.join(finding['lineage_nodes'])}",
        f"- Business impact: {finding['business_impact']}",
        f"- Suggested action: {finding['suggested_action']}",
        "",
        finding["description"],
        "",
        "| Evidence Row | Details |",
        "| ---: | --- |",
    ]
    for index, row in enumerate(finding["rows"], start=1):
        detail = ", ".join(f"{key}={value}" for key, value in row.items())
        rows.append(f"| {index} | {detail} |")
    rows.append("")
    return rows
