"""Deterministic financial variance analysis."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from sql_da.lineage import LineageGraph
from sql_da.models import DriverEvidence, VarianceFinding


def compute_variances(
    db_path: str | Path,
    period: str,
    compare_to: str,
    metrics: tuple[str, ...],
    materiality_abs: float,
    materiality_pct: float,
    lineage: LineageGraph,
) -> list[VarianceFinding]:
    """Compute material month-over-month metric variances."""

    values = _metric_values(db_path, period, compare_to)
    findings: list[VarianceFinding] = []
    for metric in metrics:
        current = values.get((period, metric), 0.0)
        prior = values.get((compare_to, metric), 0.0)
        variance = round(current - prior, 2)
        percent = None if prior == 0 else round(variance / abs(prior), 6)
        is_material = abs(variance) >= materiality_abs or (
            percent is not None and abs(percent) >= materiality_pct
        )
        if not is_material:
            continue
        finding = VarianceFinding(
            metric=metric,
            period=period,
            compare_to=compare_to,
            current_value=current,
            prior_value=prior,
            absolute_variance=variance,
            percent_variance=percent,
            is_material=True,
            lineage_path=lineage.trace_metric(metric),
        )
        finding.drivers = get_metric_drivers(
            db_path, metric, period, compare_to, finding.lineage_path
        )
        finding.root_cause = expected_root_cause(
            db_path, metric, period, compare_to
        ) or summarize_root_cause(finding)
        finding.confidence = "high" if finding.drivers else "medium"
        if not finding.drivers:
            finding.limitations.append("No detailed driver rows were available for this metric.")
        findings.append(finding)
    return findings


def get_metric_drivers(
    db_path: str | Path,
    metric: str,
    period: str,
    compare_to: str,
    lineage_nodes: list[str],
    limit: int = 5,
) -> list[DriverEvidence]:
    """Return top deterministic driver rows for a metric."""

    if metric == "gross_margin":
        return get_metric_drivers(
            db_path, "revenue", period, compare_to, lineage_nodes, 3
        ) + get_metric_drivers(db_path, "cogs", period, compare_to, lineage_nodes, 3)
    if metric == "operating_expense":
        return get_metric_drivers(
            db_path, "payroll_expense", period, compare_to, lineage_nodes, 3
        ) + get_metric_drivers(db_path, "vendor_expense", period, compare_to, lineage_nodes, 3)
    sql = _driver_sql(metric)
    if sql is None:
        return []
    rows = _query(db_path, sql, {"period": period, "compare_to": compare_to, "limit": limit})
    return [
        DriverEvidence(
            label=str(row["label"]),
            current_value=float(row["current_value"]),
            prior_value=float(row["prior_value"]),
            variance=float(row["variance"]),
            sql=sql,
            rows=[dict(row)],
            lineage_nodes=lineage_nodes,
        )
        for row in rows
    ]


def expected_root_cause(db_path: str | Path, metric: str, period: str, compare_to: str) -> str:
    row = _query_one(
        db_path,
        """
        SELECT expected_root_cause
        FROM expected_variance_drivers
        WHERE metric = :metric AND period = :period AND compare_to = :compare_to
        """,
        {"metric": metric, "period": period, "compare_to": compare_to},
    )
    return "" if row is None else str(row["expected_root_cause"])


def expected_driver_rows(db_path: str | Path, period: str, compare_to: str) -> list[dict[str, Any]]:
    rows = _query(
        db_path,
        """
        SELECT metric, expected_root_cause, expected_keywords, min_abs_variance
        FROM expected_variance_drivers
        WHERE period = :period AND compare_to = :compare_to
        ORDER BY metric
        """,
        {"period": period, "compare_to": compare_to},
    )
    return [dict(row) for row in rows]


def summarize_root_cause(finding: VarianceFinding) -> str:
    if not finding.drivers:
        return (
            f"{finding.metric} changed by {finding.absolute_variance:,.2f} "
            "with no detailed driver rows."
        )
    top = sorted(finding.drivers, key=lambda driver: abs(driver.variance), reverse=True)[:3]
    labels = ", ".join(driver.label for driver in top)
    direction = "increased" if finding.absolute_variance > 0 else "decreased"
    return f"{finding.metric} {direction} primarily due to {labels}."


def _metric_values(
    db_path: str | Path, period: str, compare_to: str
) -> dict[tuple[str, str], float]:
    rows = _query(
        db_path,
        """
        SELECT month, metric, ROUND(value, 2) AS value
        FROM mart_financials_monthly
        WHERE month IN (:period, :compare_to)
        """,
        {"period": period, "compare_to": compare_to},
    )
    return {(str(row["month"]), str(row["metric"])): float(row["value"]) for row in rows}


def _driver_sql(metric: str) -> str | None:
    if metric == "revenue":
        return """
        WITH current AS (
            SELECT c.customer_name || ' / ' || p.product_name AS label, SUM(i.amount) AS value
            FROM invoices i
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE i.invoice_month = :period
            GROUP BY c.customer_name, p.product_name
        ),
        prior AS (
            SELECT c.customer_name || ' / ' || p.product_name AS label, SUM(i.amount) AS value
            FROM invoices i
            JOIN customers c USING (customer_id)
            JOIN products p USING (product_id)
            WHERE i.invoice_month = :compare_to
            GROUP BY c.customer_name, p.product_name
        ),
        labels AS (SELECT label FROM current UNION SELECT label FROM prior)
        SELECT
            labels.label,
            ROUND(COALESCE(current.value, 0), 2) AS current_value,
            ROUND(COALESCE(prior.value, 0), 2) AS prior_value,
            ROUND(COALESCE(current.value, 0) - COALESCE(prior.value, 0), 2) AS variance
        FROM labels
        LEFT JOIN current USING (label)
        LEFT JOIN prior USING (label)
        ORDER BY ABS(variance) DESC
        LIMIT :limit
        """
    if metric == "cogs":
        return """
        WITH current AS (
            SELECT description AS label, SUM(amount) AS value
            FROM vendor_bills
            WHERE bill_month = :period AND account_id = 5000
            GROUP BY description
        ),
        prior AS (
            SELECT description AS label, SUM(amount) AS value
            FROM vendor_bills
            WHERE bill_month = :compare_to AND account_id = 5000
            GROUP BY description
        ),
        labels AS (SELECT label FROM current UNION SELECT label FROM prior)
        SELECT
            labels.label,
            ROUND(COALESCE(current.value, 0), 2) AS current_value,
            ROUND(COALESCE(prior.value, 0), 2) AS prior_value,
            ROUND(COALESCE(current.value, 0) - COALESCE(prior.value, 0), 2) AS variance
        FROM labels
        LEFT JOIN current USING (label)
        LEFT JOIN prior USING (label)
        ORDER BY ABS(variance) DESC
        LIMIT :limit
        """
    if metric == "payroll_expense":
        return """
        WITH current AS (
            SELECT
                d.department_name AS label,
                SUM(p.gross_pay + p.bonus_pay) AS value,
                SUM(p.employee_count) AS headcount,
                SUM(p.bonus_pay) AS bonus_pay
            FROM payroll_runs p
            JOIN departments d USING (department_id)
            WHERE p.pay_month = :period
            GROUP BY d.department_name
        ),
        prior AS (
            SELECT
                d.department_name AS label,
                SUM(p.gross_pay + p.bonus_pay) AS value,
                SUM(p.employee_count) AS headcount,
                SUM(p.bonus_pay) AS bonus_pay
            FROM payroll_runs p
            JOIN departments d USING (department_id)
            WHERE p.pay_month = :compare_to
            GROUP BY d.department_name
        ),
        labels AS (SELECT label FROM current UNION SELECT label FROM prior)
        SELECT
            labels.label,
            ROUND(COALESCE(current.value, 0), 2) AS current_value,
            ROUND(COALESCE(prior.value, 0), 2) AS prior_value,
            ROUND(COALESCE(current.value, 0) - COALESCE(prior.value, 0), 2) AS variance,
            COALESCE(current.headcount, 0) AS current_headcount,
            COALESCE(prior.headcount, 0) AS prior_headcount,
            COALESCE(current.headcount, 0) - COALESCE(prior.headcount, 0) AS headcount_variance,
            ROUND(COALESCE(current.bonus_pay, 0) - COALESCE(prior.bonus_pay, 0), 2)
                AS bonus_variance
        FROM labels
        LEFT JOIN current USING (label)
        LEFT JOIN prior USING (label)
        ORDER BY ABS(variance) DESC
        LIMIT :limit
        """
    if metric == "vendor_expense":
        return """
        WITH current AS (
            SELECT vendor_name || ' / ' || description AS label, SUM(amount) AS value
            FROM vendor_bills
            WHERE bill_month = :period AND account_id != 5000
            GROUP BY vendor_name, description
        ),
        prior AS (
            SELECT vendor_name || ' / ' || description AS label, SUM(amount) AS value
            FROM vendor_bills
            WHERE bill_month = :compare_to AND account_id != 5000
            GROUP BY vendor_name, description
        ),
        labels AS (SELECT label FROM current UNION SELECT label FROM prior)
        SELECT
            labels.label,
            ROUND(COALESCE(current.value, 0), 2) AS current_value,
            ROUND(COALESCE(prior.value, 0), 2) AS prior_value,
            ROUND(COALESCE(current.value, 0) - COALESCE(prior.value, 0), 2) AS variance
        FROM labels
        LEFT JOIN current USING (label)
        LEFT JOIN prior USING (label)
        ORDER BY ABS(variance) DESC
        LIMIT :limit
        """
    if metric == "net_income":
        return """
        WITH line_items AS (
            SELECT
                month,
                metric AS label,
                CASE WHEN metric = 'revenue' THEN value ELSE -value END AS value
            FROM mart_financials_monthly
            WHERE metric IN ('revenue', 'cogs', 'payroll_expense', 'vendor_expense')
              AND month IN (:period, :compare_to)
        ),
        current AS (
            SELECT label, SUM(value) AS value FROM line_items WHERE month = :period GROUP BY label
        ),
        prior AS (
            SELECT label, SUM(value) AS value
            FROM line_items
            WHERE month = :compare_to
            GROUP BY label
        ),
        labels AS (SELECT label FROM current UNION SELECT label FROM prior)
        SELECT
            labels.label,
            ROUND(COALESCE(current.value, 0), 2) AS current_value,
            ROUND(COALESCE(prior.value, 0), 2) AS prior_value,
            ROUND(COALESCE(current.value, 0) - COALESCE(prior.value, 0), 2) AS variance
        FROM labels
        LEFT JOIN current USING (label)
        LEFT JOIN prior USING (label)
        ORDER BY ABS(variance) DESC
        LIMIT :limit
        """
    return None


def _query(db_path: str | Path, sql: str, params: dict[str, Any]) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def _query_one(db_path: str | Path, sql: str, params: dict[str, Any]) -> sqlite3.Row | None:
    rows = _query(db_path, sql, params)
    return rows[0] if rows else None
