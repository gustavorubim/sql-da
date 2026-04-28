import sqlite3
from contextlib import closing
from pathlib import Path

from sql_da.config import ProjectConfig
from sql_da.lineage import LineageGraph
from sql_da.synthetic import init_demo_database, table_counts
from sql_da.variance import (
    compute_variances,
    expected_driver_rows,
    get_metric_drivers,
    summarize_root_cause,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> ProjectConfig:
    return ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")


def test_demo_database_builds_known_financials():
    config = _config()
    db_path = init_demo_database(config)

    counts = table_counts(db_path)
    assert counts["invoices"] > 20
    assert counts["gl_entries"] > counts["invoices"]

    with closing(sqlite3.connect(db_path)) as conn:
        rows = dict(
            conn.execute(
                "SELECT metric, value FROM mart_financials_monthly WHERE month = '2026-03'"
            ).fetchall()
        )

    assert rows["revenue"] == 199000.0
    assert rows["cogs"] == 48400.0
    assert rows["gross_margin"] == 150600.0
    assert rows["payroll_expense"] == 202000.0
    assert rows["vendor_expense"] == 55000.0
    assert rows["net_income"] == -106400.0
    assert init_demo_database(config.database.path, replace=True) == config.database.path


def test_compute_variances_detects_seeded_root_causes():
    config = _config()
    db_path = init_demo_database(config)
    lineage = LineageGraph.from_yaml(config.lineage_path)

    findings = compute_variances(
        db_path,
        "2026-03",
        "2026-02",
        config.analysis.metrics,
        config.analysis.materiality_abs,
        config.analysis.materiality_pct,
        lineage,
    )
    by_metric = {finding.metric: finding for finding in findings}

    assert set(by_metric) == set(config.analysis.metrics)
    assert by_metric["revenue"].absolute_variance == 57000.0
    assert "Delta Bank" in by_metric["revenue"].root_cause
    assert by_metric["cogs"].absolute_variance == 19840.0
    assert by_metric["net_income"].absolute_variance == -36840.0
    assert by_metric["vendor_expense"].drivers[0].label.startswith("ConsultCo")
    assert by_metric["operating_expense"].drivers[0].label == "Engineering"
    assert by_metric["operating_expense"].drivers[0].rows[0]["headcount_variance"] == 2
    assert by_metric["revenue"].lineage_path[0] == "mart_revenue_monthly"


def test_compute_variances_skips_non_material_and_handles_missing_drivers(monkeypatch):
    config = _config()
    db_path = init_demo_database(config)
    lineage = LineageGraph.from_yaml(config.lineage_path)

    assert (
        compute_variances(
            db_path,
            "2026-03",
            "2026-02",
            ("revenue",),
            999999.0,
            9.0,
            lineage,
        )
        == []
    )

    monkeypatch.setattr("sql_da.variance.get_metric_drivers", lambda *args, **kwargs: [])
    monkeypatch.setattr("sql_da.variance.expected_root_cause", lambda *args, **kwargs: "")

    finding = compute_variances(
        db_path,
        "2026-03",
        "2026-02",
        ("revenue",),
        1.0,
        0.0,
        lineage,
    )[0]

    assert finding.confidence == "medium"
    assert finding.limitations == ["No detailed driver rows were available for this metric."]
    assert "no detailed driver rows" in finding.root_cause


def test_driver_queries_and_fallback_summary():
    config = _config()
    db_path = init_demo_database(config)
    lineage = LineageGraph.from_yaml(config.lineage_path)

    revenue_drivers = get_metric_drivers(
        db_path, "revenue", "2026-03", "2026-02", lineage.trace_metric("revenue")
    )
    gross_margin_drivers = get_metric_drivers(
        db_path, "gross_margin", "2026-03", "2026-02", lineage.trace_metric("gross_margin")
    )
    unknown_drivers = get_metric_drivers(db_path, "unknown", "2026-03", "2026-02", [])

    assert revenue_drivers[0].label == "Delta Bank / ComplianceSuite"
    assert len(gross_margin_drivers) > len(revenue_drivers[:3])
    assert unknown_drivers == []
    assert "revenue" in summarize_root_cause(
        compute_variances(
            db_path,
            "2026-03",
            "2026-02",
            ("revenue",),
            1.0,
            0.0,
            lineage,
        )[0]
    )


def test_expected_driver_metadata_is_available():
    config = _config()
    db_path = init_demo_database(config)

    expected = expected_driver_rows(db_path, "2026-03", "2026-02")

    assert {row["metric"] for row in expected} == set(config.analysis.metrics)
    assert any("ConsultCo" in row["expected_keywords"] for row in expected)
