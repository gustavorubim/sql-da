"""Executable reward harness for implementation verification."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_da.config import ProjectConfig
from sql_da.data_quality import expected_data_quality_rows, run_data_quality_checks
from sql_da.lineage import LineageGraph
from sql_da.sql_safety import ReadOnlySQLRunner, SQLSafetyError
from sql_da.synthetic import init_demo_database, table_counts
from sql_da.terminal import TerminalReporter
from sql_da.variance import compute_variances, expected_driver_rows
from sql_da.workflow import analyze


@dataclass(frozen=True)
class RewardCheck:
    """One reward check result."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def run_reward(
    config: ProjectConfig,
    include_quality_gates: bool = True,
    reporter: TerminalReporter | None = None,
) -> dict[str, Any]:
    """Run deterministic checks, optionally including ruff and pytest subprocess gates."""

    reporter = reporter or TerminalReporter()
    with reporter.status("Running deterministic reward checks"):
        checks = run_internal_checks(config, reporter=reporter)
    if include_quality_gates:
        with reporter.status("Running quality gates"):
            checks.extend(run_quality_gates(config.project_root, reporter=reporter))
    score = round(sum(check.passed for check in checks) / len(checks), 4) if checks else 0.0
    payload = {
        "score": score,
        "passed": all(check.passed for check in checks),
        "checks": [check.to_dict() for check in checks],
    }
    config.reward.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.reward.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def run_internal_checks(
    config: ProjectConfig,
    reporter: TerminalReporter | None = None,
) -> list[RewardCheck]:
    """Run deterministic, credential-free reward checks."""

    reporter = reporter or TerminalReporter()
    checks: list[RewardCheck] = []
    reporter.stage("[1/6] Rebuilding demo database")
    db_path = init_demo_database(config)
    counts = table_counts(db_path)
    checks.append(
        RewardCheck(
            "demo_db_builds",
            all(count > 0 for count in counts.values()),
            f"row_counts={counts}",
        )
    )

    reporter.stage("[2/6] Validating lineage")
    lineage = LineageGraph.from_yaml(config.lineage_path)
    lineage_errors = lineage.validate()
    checks.append(
        RewardCheck("lineage_validates", not lineage_errors, "; ".join(lineage_errors) or "ok")
    )

    reporter.stage("[3/6] Checking seeded variance drivers")
    findings = compute_variances(
        db_path,
        "2026-03",
        "2026-02",
        config.analysis.metrics,
        config.analysis.materiality_abs,
        config.analysis.materiality_pct,
        lineage,
    )
    expected = {row["metric"]: row for row in expected_driver_rows(db_path, "2026-03", "2026-02")}
    detected = {finding.metric: finding for finding in findings}
    missing = sorted(set(expected) - set(detected))
    keyword_failures = _keyword_failures(detected, expected)
    checks.append(
        RewardCheck(
            "known_variance_drivers_detected",
            not missing and not keyword_failures,
            f"missing={missing}; keyword_failures={keyword_failures}",
        )
    )

    reporter.stage("[4/6] Checking seeded data quality issues")
    expected_dq = {row["issue_id"]: row for row in expected_data_quality_rows(db_path, "2026-03")}
    detected_dq = {
        finding.issue_id: finding for finding in run_data_quality_checks(db_path, "2026-03")
    }
    missing_dq = sorted(set(expected_dq) - set(detected_dq))
    dq_keyword_failures = _data_quality_keyword_failures(detected_dq, expected_dq)
    checks.append(
        RewardCheck(
            "known_data_quality_issues_detected",
            not missing_dq and not dq_keyword_failures,
            f"missing={missing_dq}; keyword_failures={dq_keyword_failures}",
        )
    )

    reporter.stage("[5/6] Checking SQL mutation guard")
    runner = ReadOnlySQLRunner(db_path)
    try:
        runner.run("DROP TABLE invoices")
        sql_blocked = False
    except SQLSafetyError:
        sql_blocked = True
    checks.append(RewardCheck("sql_safety_blocks_mutation", sql_blocked, "DROP TABLE was blocked"))

    reporter.stage("[6/6] Checking report artifacts")
    result = analyze(config, "2026-03", "2026-02", run_id="reward_demo", reporter=reporter)
    evidence = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    every_finding_has_evidence = all(
        finding["lineage_path"] and finding["drivers"] and finding["root_cause"]
        for finding in evidence["findings"]
    )
    trace_paths_exist = (
        Path(result.trace_json_path).exists() and Path(result.trace_csv_path).exists()
    )
    report_has_data_quality_evidence = all(
        finding["rows"] and finding["lineage_nodes"]
        for finding in evidence["data_quality_findings"]
    )
    checks.append(
        RewardCheck(
            "report_artifacts_have_evidence",
            (
                every_finding_has_evidence
                and report_has_data_quality_evidence
                and Path(result.report_path).exists()
                and trace_paths_exist
            ),
            (
                f"report={result.report_path}; evidence={result.evidence_path}; "
                f"trace_json={result.trace_json_path}; trace_csv={result.trace_csv_path}"
            ),
        )
    )
    return checks


def run_quality_gates(
    project_root: Path,
    reporter: TerminalReporter | None = None,
) -> list[RewardCheck]:
    """Run ruff and pytest gates from the active Python environment."""

    reporter = reporter or TerminalReporter()
    commands = [
        ("ruff_format_check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("ruff_check", [sys.executable, "-m", "ruff", "check", "."]),
        (
            "pytest_coverage_95",
            [sys.executable, "-m", "pytest", "--cov=src/sql_da", "--cov-fail-under=95"],
        ),
    ]
    checks: list[RewardCheck] = []
    for name, command in commands:
        reporter.stage(f"Running {name}: {' '.join(command)}")
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            text=True,
            capture_output=True,
        )
        detail = (completed.stdout + completed.stderr).strip()
        checks.append(RewardCheck(name, completed.returncode == 0, detail[-1200:] or "ok"))
    return checks


def _keyword_failures(
    detected: dict[str, Any],
    expected: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for metric, row in expected.items():
        finding = detected.get(metric)
        if finding is None:
            continue
        text = " ".join(
            [finding.root_cause, *[driver.label for driver in finding.drivers]]
        ).casefold()
        missing_terms = [
            keyword.strip()
            for keyword in str(row["expected_keywords"]).split(",")
            if keyword.strip().casefold() not in text
        ]
        if missing_terms:
            failures[metric] = missing_terms
    return failures


def _data_quality_keyword_failures(
    detected: dict[str, Any],
    expected: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for issue_id, row in expected.items():
        finding = detected.get(issue_id)
        if finding is None:
            continue
        text = " ".join(
            [
                finding.description,
                finding.check_name,
                finding.table_name,
                finding.business_impact,
                finding.suggested_action,
                *[str(value) for evidence_row in finding.rows for value in evidence_row.values()],
            ]
        ).casefold()
        missing_terms = [
            keyword.strip()
            for keyword in str(row["expected_keywords"]).split(",")
            if keyword.strip().casefold() not in text
        ]
        if missing_terms:
            failures[issue_id] = missing_terms
    return failures
