"""Deterministic synthetic finance database used by demos and tests."""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from sql_da.config import ProjectConfig

MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
RANDOM_SEED = 4242


def init_demo_database(config_or_path: ProjectConfig | str | Path, replace: bool = True) -> Path:
    """Create the deterministic SQLite demo database and return its path."""

    db_path = _path_from_config(config_or_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if replace and db_path.exists():
        db_path.unlink()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            _create_schema(conn)
            _seed_dimensions(conn)
            _seed_contracts_and_invoices(conn)
            _seed_bills_payroll_and_gl(conn)
            _create_views(conn)
            _seed_expected_drivers(conn)
    return db_path


def table_counts(db_path: Path) -> dict[str, int]:
    """Return row counts for the core demo tables."""

    tables = (
        "accounts",
        "departments",
        "customers",
        "products",
        "customer_contracts",
        "invoices",
        "vendor_bills",
        "payroll_runs",
        "gl_entries",
        "expected_variance_drivers",
    )
    with closing(sqlite3.connect(db_path)) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables
        }


def _path_from_config(config_or_path: ProjectConfig | str | Path) -> Path:
    if isinstance(config_or_path, ProjectConfig):
        return config_or_path.database.path
    return Path(config_or_path).expanduser().resolve()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_name TEXT NOT NULL,
            statement_line TEXT NOT NULL,
            normal_balance TEXT NOT NULL
        );
        CREATE TABLE departments (
            department_id INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL
        );
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            segment TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            gross_margin_profile TEXT NOT NULL,
            cogs_rate REAL NOT NULL
        );
        CREATE TABLE customer_contracts (
            contract_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            start_month TEXT NOT NULL,
            end_month TEXT,
            monthly_amount REAL NOT NULL,
            units INTEGER NOT NULL
        );
        CREATE TABLE invoices (
            invoice_id INTEGER PRIMARY KEY,
            invoice_month TEXT NOT NULL,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            contract_id INTEGER NOT NULL REFERENCES customer_contracts(contract_id),
            amount REAL NOT NULL,
            quantity INTEGER NOT NULL
        );
        CREATE TABLE vendor_bills (
            bill_id INTEGER PRIMARY KEY,
            bill_month TEXT NOT NULL,
            vendor_name TEXT NOT NULL,
            department_id INTEGER NOT NULL REFERENCES departments(department_id),
            account_id INTEGER NOT NULL REFERENCES accounts(account_id),
            amount REAL NOT NULL,
            description TEXT NOT NULL
        );
        CREATE TABLE payroll_runs (
            payroll_id INTEGER PRIMARY KEY,
            pay_month TEXT NOT NULL,
            department_id INTEGER NOT NULL REFERENCES departments(department_id),
            employee_count INTEGER NOT NULL,
            gross_pay REAL NOT NULL,
            bonus_pay REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE gl_entries (
            entry_id INTEGER PRIMARY KEY,
            entry_month TEXT NOT NULL,
            account_id INTEGER NOT NULL REFERENCES accounts(account_id),
            department_id INTEGER REFERENCES departments(department_id),
            source_table TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            description TEXT NOT NULL
        );
        CREATE TABLE expected_variance_drivers (
            metric TEXT NOT NULL,
            period TEXT NOT NULL,
            compare_to TEXT NOT NULL,
            expected_root_cause TEXT NOT NULL,
            expected_keywords TEXT NOT NULL,
            min_abs_variance REAL NOT NULL,
            PRIMARY KEY (metric, period, compare_to)
        );
        """
    )


def _seed_dimensions(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO accounts VALUES (?, ?, ?, ?)",
        [
            (4000, "Subscription Revenue", "revenue", "credit"),
            (5000, "Cloud COGS", "cogs", "debit"),
            (6000, "Payroll Expense", "payroll_expense", "debit"),
            (6100, "Software Expense", "vendor_expense", "debit"),
            (6200, "Facilities Expense", "vendor_expense", "debit"),
            (6300, "Professional Services", "vendor_expense", "debit"),
        ],
    )
    conn.executemany(
        "INSERT INTO departments VALUES (?, ?)",
        [(1, "Engineering"), (2, "Sales"), (3, "G&A")],
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [
            (1, "Acme Corp", "enterprise", "active"),
            (2, "Beta Health", "mid-market", "active"),
            (3, "Cloud Nine Labs", "smb", "churned"),
            (4, "Delta Bank", "enterprise", "active"),
            (5, "Echo Retail", "smb", "active"),
        ],
    )
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            (1, "CoreApp", "standard", 0.22),
            (2, "Analytics", "high_margin", 0.18),
            (3, "ComplianceSuite", "lower_margin", 0.35),
            (4, "Support", "high_margin", 0.12),
        ],
    )


def _seed_contracts_and_invoices(conn: sqlite3.Connection) -> None:
    contracts = [
        (1, 1, 1, "2026-01", None, 70000.0, 100),
        (2, 2, 1, "2026-01", None, 32000.0, 45),
        (3, 3, 4, "2026-01", "2026-02", 18000.0, 30),
        (4, 4, 3, "2026-03", None, 50000.0, 65),
        (5, 5, 2, "2026-01", None, 22000.0, 35),
        (6, 1, 2, "2026-03", None, 25000.0, 40),
    ]
    conn.executemany("INSERT INTO customer_contracts VALUES (?, ?, ?, ?, ?, ?, ?)", contracts)

    rng = random.Random(RANDOM_SEED)
    invoice_id = 1
    invoice_rows = []
    for month in MONTHS:
        for contract in contracts:
            contract_id, customer_id, product_id, start_month, end_month, amount, units = contract
            if month < start_month or (end_month is not None and month > end_month):
                continue
            quantity = units + rng.randint(-2, 2)
            invoice_rows.append(
                (invoice_id, month, customer_id, product_id, contract_id, amount, quantity)
            )
            invoice_id += 1
    conn.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?)", invoice_rows)


def _seed_bills_payroll_and_gl(conn: sqlite3.Connection) -> None:
    product_rates = {
        row["product_id"]: row["cogs_rate"]
        for row in conn.execute("SELECT product_id, cogs_rate FROM products").fetchall()
    }
    bill_rows: list[tuple[int, str, str, int, int, float, str]] = []
    bill_id = 1
    for month in MONTHS:
        product_revenue = defaultdict(float)
        for row in conn.execute(
            "SELECT product_id, amount FROM invoices WHERE invoice_month = ?", (month,)
        ).fetchall():
            product_revenue[row["product_id"]] += float(row["amount"])
        for product_id, revenue in sorted(product_revenue.items()):
            product_name = conn.execute(
                "SELECT product_name FROM products WHERE product_id = ?", (product_id,)
            ).fetchone()["product_name"]
            amount = round(revenue * product_rates[product_id], 2)
            bill_rows.append(
                (bill_id, month, "CloudCo", 1, 5000, amount, f"Cloud usage - {product_name}")
            )
            bill_id += 1
        for row in _base_vendor_bills(month, bill_id):
            bill_rows.append(row)
            bill_id = row[0] + 1
    conn.executemany("INSERT INTO vendor_bills VALUES (?, ?, ?, ?, ?, ?, ?)", bill_rows)

    payroll_rows = _payroll_rows()
    conn.executemany("INSERT INTO payroll_runs VALUES (?, ?, ?, ?, ?, ?)", payroll_rows)
    _seed_gl_entries(conn)


def _base_vendor_bills(
    month: str, start_id: int
) -> Iterable[tuple[int, str, str, int, int, float, str]]:
    rows = [
        (start_id, month, "DataDog", 1, 6100, 8000.0, "Observability platform"),
        (start_id + 1, month, "HQ Lease", 3, 6200, 12000.0, "Office lease"),
    ]
    if month == "2026-03":
        rows.append(
            (start_id + 2, month, "ConsultCo", 3, 6300, 35000.0, "ERP implementation kickoff")
        )
    return rows


def _payroll_rows() -> list[tuple[int, str, int, int, float, float]]:
    payroll_id = 1
    rows = []
    for month in MONTHS:
        engineering_headcount = 10 if month >= "2026-03" else 8
        sales_headcount = 5 if month >= "2026-03" else 4
        rows.extend(
            [
                (payroll_id, month, 1, engineering_headcount, engineering_headcount * 12000.0, 0.0),
                (payroll_id + 1, month, 2, sales_headcount, sales_headcount * 10000.0, 0.0),
                (payroll_id + 2, month, 3, 3, 27000.0, 5000.0 if month == "2026-03" else 0.0),
            ]
        )
        payroll_id += 3
    return rows


def _seed_gl_entries(conn: sqlite3.Connection) -> None:
    entry_id = 1
    gl_rows = []
    for row in conn.execute("SELECT * FROM invoices ORDER BY invoice_id").fetchall():
        gl_rows.append(
            (
                entry_id,
                row["invoice_month"],
                4000,
                2,
                "invoices",
                row["invoice_id"],
                row["amount"],
                "Revenue recognition",
            )
        )
        entry_id += 1
    for row in conn.execute("SELECT * FROM vendor_bills ORDER BY bill_id").fetchall():
        gl_rows.append(
            (
                entry_id,
                row["bill_month"],
                row["account_id"],
                row["department_id"],
                "vendor_bills",
                row["bill_id"],
                row["amount"],
                row["description"],
            )
        )
        entry_id += 1
    for row in conn.execute("SELECT * FROM payroll_runs ORDER BY payroll_id").fetchall():
        gl_rows.append(
            (
                entry_id,
                row["pay_month"],
                6000,
                row["department_id"],
                "payroll_runs",
                row["payroll_id"],
                row["gross_pay"] + row["bonus_pay"],
                "Department payroll",
            )
        )
        entry_id += 1
    conn.executemany("INSERT INTO gl_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", gl_rows)


def _create_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIEW mart_revenue_monthly AS
        SELECT invoice_month AS month, ROUND(SUM(amount), 2) AS revenue
        FROM invoices
        GROUP BY invoice_month;

        CREATE VIEW mart_cogs_monthly AS
        SELECT bill_month AS month, ROUND(SUM(amount), 2) AS cogs
        FROM vendor_bills
        WHERE account_id = 5000
        GROUP BY bill_month;

        CREATE VIEW mart_payroll_monthly AS
        SELECT pay_month AS month, ROUND(SUM(gross_pay + bonus_pay), 2) AS payroll_expense
        FROM payroll_runs
        GROUP BY pay_month;

        CREATE VIEW mart_vendor_spend_monthly AS
        SELECT bill_month AS month, ROUND(SUM(amount), 2) AS vendor_expense
        FROM vendor_bills
        WHERE account_id != 5000
        GROUP BY bill_month;

        CREATE VIEW mart_department_expense_monthly AS
        SELECT
            p.pay_month AS month,
            d.department_name,
            'payroll' AS expense_type,
            ROUND(SUM(p.gross_pay + p.bonus_pay), 2) AS amount
        FROM payroll_runs p
        JOIN departments d USING (department_id)
        GROUP BY p.pay_month, d.department_name
        UNION ALL
        SELECT
            v.bill_month AS month,
            d.department_name,
            a.account_name AS expense_type,
            ROUND(SUM(v.amount), 2) AS amount
        FROM vendor_bills v
        JOIN departments d USING (department_id)
        JOIN accounts a USING (account_id)
        WHERE v.account_id != 5000
        GROUP BY v.bill_month, d.department_name, a.account_name;

        CREATE VIEW mart_financials_monthly AS
        WITH months AS (
            SELECT invoice_month AS month FROM invoices
            UNION
            SELECT bill_month AS month FROM vendor_bills
            UNION
            SELECT pay_month AS month FROM payroll_runs
        ),
        joined AS (
            SELECT
                m.month,
                COALESCE(r.revenue, 0.0) AS revenue,
                COALESCE(c.cogs, 0.0) AS cogs,
                COALESCE(p.payroll_expense, 0.0) AS payroll_expense,
                COALESCE(v.vendor_expense, 0.0) AS vendor_expense
            FROM months m
            LEFT JOIN mart_revenue_monthly r USING (month)
            LEFT JOIN mart_cogs_monthly c USING (month)
            LEFT JOIN mart_payroll_monthly p USING (month)
            LEFT JOIN mart_vendor_spend_monthly v USING (month)
        )
        SELECT month, 'revenue' AS metric, revenue AS value FROM joined
        UNION ALL SELECT month, 'cogs', cogs FROM joined
        UNION ALL SELECT month, 'gross_margin', revenue - cogs FROM joined
        UNION ALL SELECT month, 'payroll_expense', payroll_expense FROM joined
        UNION ALL SELECT month, 'vendor_expense', vendor_expense FROM joined
        UNION ALL SELECT month, 'operating_expense', payroll_expense + vendor_expense FROM joined
        UNION ALL
        SELECT month, 'net_income', revenue - cogs - payroll_expense - vendor_expense
        FROM joined;
        """
    )


def _seed_expected_drivers(conn: sqlite3.Connection) -> None:
    rows = [
        (
            "revenue",
            "2026-03",
            "2026-02",
            "Revenue increased because Delta Bank launched ComplianceSuite and Acme Corp "
            "expanded into Analytics, more than offsetting Cloud Nine Labs churn.",
            "Delta Bank,Acme Corp,Cloud Nine Labs,ComplianceSuite,Analytics",
            50000.0,
        ),
        (
            "cogs",
            "2026-03",
            "2026-02",
            "COGS increased with the new ComplianceSuite workload and higher cloud usage "
            "from the March enterprise expansion.",
            "ComplianceSuite,cloud usage",
            15000.0,
        ),
        (
            "gross_margin",
            "2026-03",
            "2026-02",
            "Gross margin rose in dollars, but ComplianceSuite introduced a lower-margin "
            "revenue mix that increased COGS.",
            "ComplianceSuite,revenue,cogs",
            30000.0,
        ),
        (
            "payroll_expense",
            "2026-03",
            "2026-02",
            "Payroll increased because Engineering added two employees, Sales added one "
            "employee, and G&A paid a one-time bonus.",
            "Engineering,Sales,G&A,bonus",
            30000.0,
        ),
        (
            "vendor_expense",
            "2026-03",
            "2026-02",
            "Vendor expense increased because ConsultCo billed a one-time ERP "
            "implementation kickoff invoice.",
            "ConsultCo,ERP implementation",
            30000.0,
        ),
        (
            "operating_expense",
            "2026-03",
            "2026-02",
            "Operating expense increased from March payroll headcount growth plus the "
            "one-time ConsultCo implementation invoice.",
            "headcount,ConsultCo,payroll",
            70000.0,
        ),
        (
            "net_income",
            "2026-03",
            "2026-02",
            "Net income declined because revenue growth was outweighed by higher COGS, "
            "payroll, and the ConsultCo implementation invoice.",
            "revenue,cogs,payroll,ConsultCo",
            30000.0,
        ),
    ]
    conn.executemany("INSERT INTO expected_variance_drivers VALUES (?, ?, ?, ?, ?, ?)", rows)
