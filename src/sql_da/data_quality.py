"""Deterministic data quality checks for the demo finance database."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_da.models import DataQualityFinding


@dataclass(frozen=True)
class DataQualityCheck:
    """One SQL-backed data quality rule."""

    issue_id: str
    check_name: str
    severity: str
    table_name: str
    description: str
    business_impact: str
    suggested_action: str
    evidence_sql: str
    lineage_nodes: tuple[str, ...]


INVOICE_AFTER_CONTRACT_END_SQL = """
SELECT
    i.invoice_id,
    i.invoice_month,
    c.customer_name,
    p.product_name,
    cc.contract_id,
    cc.end_month,
    ROUND(i.amount, 2) AS amount
FROM invoices i
JOIN customer_contracts cc USING (contract_id)
JOIN customers c USING (customer_id)
JOIN products p USING (product_id)
WHERE i.invoice_month = :period
  AND cc.end_month IS NOT NULL
  AND i.invoice_month > cc.end_month
ORDER BY i.invoice_id
"""

DUPLICATE_VENDOR_BILL_SQL = """
SELECT
    bill_month,
    vendor_name,
    department_id,
    account_id,
    description,
    ROUND(amount, 2) AS amount,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(bill_id, ',') AS bill_ids,
    ROUND(SUM(amount), 2) AS loaded_amount,
    ROUND(SUM(amount) - amount, 2) AS potential_overstatement
FROM vendor_bills
WHERE bill_month = :period
GROUP BY bill_month, vendor_name, department_id, account_id, description, amount
HAVING COUNT(*) > 1
ORDER BY ABS(potential_overstatement) DESC, vendor_name
"""

MISSING_GL_ENTRY_SQL = """
SELECT
    p.payroll_id,
    p.pay_month,
    d.department_name,
    p.employee_count,
    ROUND(p.gross_pay + p.bonus_pay, 2) AS payroll_amount,
    ROUND(p.bonus_pay, 2) AS bonus_pay
FROM payroll_runs p
JOIN departments d USING (department_id)
LEFT JOIN gl_entries g
  ON g.source_table = 'payroll_runs'
 AND g.source_id = p.payroll_id
WHERE p.pay_month = :period
  AND g.entry_id IS NULL
ORDER BY p.payroll_id
"""

CHECKS = (
    DataQualityCheck(
        issue_id="invoice_after_contract_end",
        check_name="invoice_after_contract_end",
        severity="high",
        table_name="invoices",
        description="Invoices exist after the related contract ended.",
        business_impact="Revenue and downstream COGS can be overstated by stale billing.",
        suggested_action="Review billing cutover controls and reverse invalid post-churn invoices.",
        evidence_sql=INVOICE_AFTER_CONTRACT_END_SQL,
        lineage_nodes=("invoices", "customer_contracts", "customers", "products"),
    ),
    DataQualityCheck(
        issue_id="duplicate_vendor_bill",
        check_name="duplicate_vendor_bill",
        severity="high",
        table_name="vendor_bills",
        description="Vendor bills have exact duplicate business keys in the same month.",
        business_impact="Vendor expense, operating expense, and net income can be misstated.",
        suggested_action="Deduplicate AP loads and add vendor invoice uniqueness checks.",
        evidence_sql=DUPLICATE_VENDOR_BILL_SQL,
        lineage_nodes=("vendor_bills", "accounts", "departments"),
    ),
    DataQualityCheck(
        issue_id="missing_gl_entry",
        check_name="missing_gl_entry",
        severity="medium",
        table_name="gl_entries",
        description="Source payroll runs are missing corresponding GL postings.",
        business_impact=(
            "Subledger-to-GL reconciliation is incomplete even when marts include payroll."
        ),
        suggested_action=(
            "Reconcile source systems to GL postings before publishing financial reports."
        ),
        evidence_sql=MISSING_GL_ENTRY_SQL,
        lineage_nodes=("payroll_runs", "gl_entries", "departments"),
    ),
)


def run_data_quality_checks(db_path: str | Path, period: str) -> list[DataQualityFinding]:
    """Run deterministic SQL-backed data quality checks for a reporting period."""

    findings: list[DataQualityFinding] = []
    for check in CHECKS:
        rows = _query(db_path, check.evidence_sql, {"period": period})
        if not rows:
            continue
        findings.append(
            DataQualityFinding(
                issue_id=check.issue_id,
                check_name=check.check_name,
                severity=check.severity,
                table_name=check.table_name,
                period=period,
                description=_describe_issue(check, rows),
                business_impact=check.business_impact,
                suggested_action=check.suggested_action,
                evidence_sql=check.evidence_sql,
                rows=rows,
                lineage_nodes=list(check.lineage_nodes),
            )
        )
    return findings


def expected_data_quality_rows(db_path: str | Path, period: str) -> list[dict[str, Any]]:
    """Return expected data quality issues seeded in the synthetic database."""

    return _query(
        db_path,
        """
        SELECT issue_id, check_name, severity, table_name, expected_description, expected_keywords
        FROM expected_data_quality_issues
        WHERE period = :period
        ORDER BY issue_id
        """,
        {"period": period},
    )


def _describe_issue(check: DataQualityCheck, rows: list[dict[str, Any]]) -> str:
    if check.issue_id == "invoice_after_contract_end":
        labels = ", ".join(f"{row['customer_name']} / {row['product_name']}" for row in rows[:3])
        return f"{len(rows)} stale invoice row(s) were found after contract end: {labels}."
    if check.issue_id == "duplicate_vendor_bill":
        labels = ", ".join(f"{row['vendor_name']} / {row['description']}" for row in rows[:3])
        return f"{len(rows)} duplicate vendor bill group(s) were found: {labels}."
    if check.issue_id == "missing_gl_entry":
        labels = ", ".join(f"{row['department_name']} payroll {row['payroll_id']}" for row in rows)
        return f"{len(rows)} source row(s) are missing GL entries: {labels}."
    return check.description


def _query(
    db_path: str | Path,
    sql: str,
    params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, dict(params)).fetchall()
    return [dict(row) for row in rows]
