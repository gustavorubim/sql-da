# SQL Deep Agent Memory

This agent analyzes month-over-month financial variances for a synthetic SaaS finance database.

## Operating Principles

- Treat deterministic SQL and lineage evidence as the source of truth.
- Use the deep agent for investigation, synthesis, and claim review; do not use it for core accounting arithmetic.
- Never mutate the database. SQL must be read-only and bounded.
- Every report claim must cite both lineage nodes and concrete evidence rows.
- Prefer concise, finance-facing root-cause language over generic analytical prose.

## Demo Scenario

- March 2026 is compared against February 2026.
- Revenue variance is driven by Delta Bank launching ComplianceSuite, Acme expanding into Analytics, and Cloud Nine Labs churn.
- COGS variance is driven by cloud usage for ComplianceSuite and Analytics.
- Payroll variance is driven by Engineering and Sales headcount changes plus a G&A bonus.
- Vendor expense variance is driven by a one-time ConsultCo ERP implementation invoice.
- Net income variance should explain that revenue growth was outweighed by COGS, payroll, and vendor expense increases.

## Evidence Standard

When reviewing a report:

- Flag any explanation that is not directly supported by SQL rows.
- Flag any metric driver that is mathematically unrelated to the metric.
- Ask for better SQL evidence if headcount, bonus, churn, product mix, or one-time claims are only implied.
