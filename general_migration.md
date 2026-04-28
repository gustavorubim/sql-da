# General SQLite Migration Guide

This guide explains how to adapt the repo from the demo finance warehouse to another SQLite database while keeping the agent grounded in configured schema, relationship, and transformation specs.

## Current Architecture

The agent uses three sources of truth:

- SQLite schema and read-only SQL results from the configured database.
- `configs/lineage.yaml` for nodes, metric mappings, joins, and transformation specs.
- Deterministic Python/SQL workflows for variance detection, data-quality checks, and report generation.

The first two are now configuration-driven. The deterministic variance and report workflow is still finance-specific, so a new domain should either map its metrics into the existing variance workflow or add a new workflow module with its own deterministic checks.

## Point To Another SQLite Database

Create a new config file, for example `configs/customer_a.yaml`:

```yaml
database:
  uri: sqlite:///data/customer_a.db
  path: data/customer_a.db
lineage_path: configs/customer_a_lineage.yaml
report_dir: reports
model:
  name: google_genai:gemini-3-flash-preview
  temperature: 0.0
  enable_live_agent: true
  args: {}
agent:
  memory_paths:
    - memory/AGENTS.md
  skill_paths:
    - skills
analysis:
  materiality_abs: 5000.0
  materiality_pct: 0.05
  max_query_rows: 100
  metrics:
    - revenue
    - gross_margin
reward:
  output_path: reports/reward_score.json
```

Make sure `database.path` already exists before running `sql-da analyze`. The current workflow creates the demo database when the configured path is missing, which is useful for the sample but should be treated as a demo fallback.

## Define Lineage Nodes

Create a lineage file for the target database. Each physical table, view, and configured mart should be a node with a row grain and column list:

```yaml
version: 1
nodes:
  orders:
    type: raw_table
    grain: one row per order
    description: Customer order header.
    columns:
      - order_id
      - order_date
      - customer_id
      - total_amount
  customers:
    type: dimension
    grain: one row per customer
    description: Customer master data.
    columns:
      - customer_id
      - customer_name
      - segment
  mart_revenue_monthly:
    type: mart_view
    grain: one row per month
    description: Monthly booked revenue.
    upstream:
      - orders
      - customers
    columns:
      - month
      - revenue
```

`upstream` is table/view lineage. It tells the agent which objects feed a mart, but it does not by itself describe joins, multiplication, filters, or aggregation. Use the sections below for that.

## Define Join Relationships

Use top-level `relationships` to document join keys and cardinality:

```yaml
relationships:
  order_customer:
    description: Orders join to customers for customer and segment analysis.
    left: orders.customer_id
    right: customers.customer_id
    relationship_type: many_to_one
    join_type: inner
```

The loader validates that referenced nodes and columns exist in `nodes`. For schema-qualified node ids, define the node id exactly as you want to reference it, for example `analytics.orders`, then use `analytics.orders.customer_id`.

## Define Transformations

Use top-level `transformations` to document summaries, formulas, generated fields, filters, and multiplication rules:

```yaml
transformations:
  monthly_revenue:
    description: Monthly revenue is summarized from order amount by order month.
    output: mart_revenue_monthly.revenue
    operation: aggregate
    expression: ROUND(SUM(orders.total_amount), 2)
    grain: one row per order month
    source_nodes:
      - orders
    inputs:
      - orders.total_amount
      - orders.order_date
    group_by:
      - orders.order_date
```

For multiplication or generated values, make the formula explicit:

```yaml
transformations:
  allocated_shipping_cost:
    description: Shipping cost is allocated by order value and configured product rate.
    output: fact_order_cost.amount
    operation: multiply_then_insert
    expression: ROUND(SUM(orders.total_amount) * products.shipping_rate, 2)
    grain: one generated cost row per month and product
    source_nodes:
      - orders
      - products
      - fact_order_cost
    inputs:
      - orders.total_amount
      - orders.order_date
      - orders.product_id
      - products.product_id
      - products.shipping_rate
    group_by:
      - orders.order_date
      - orders.product_id
    filters:
      - fact_order_cost.cost_type = 'shipping'
```

The model receives compact semantic counts in the review payload and retrieves details through bounded tools:

- `search_semantic_specs`
- `get_node_context`
- `list_relationship_specs`
- `list_transformation_specs`

That is the configured semantic layer the agent should use before guessing joins or formulas. For large databases, do not put the full lineage in the prompt; require the agent to search and fetch node context for only the relevant metric, table, or column.

## Map Metrics

Metric mappings tell the deterministic variance workflow which lineage node and value column represent each metric:

```yaml
metric_mappings:
  revenue:
    node: mart_revenue_monthly
    value_column: revenue
```

For the current finance workflow, metrics also need supporting deterministic SQL in `src/sql_da/variance.py` when you want driver-level explanations. If a new database uses different metrics or business entities, add or replace driver SQL there before relying on root-cause reports.

## Migration Checklist

1. Create or obtain the SQLite database file.
2. Create a new project config that points `database.path` and `lineage_path` at the new files.
3. Add every relevant table/view to `nodes`, with accurate `grain` and `columns`.
4. Add `relationships` for join keys and cardinality.
5. Add `transformations` for summaries, formulas, filters, allocations, generated fields, and multiplication logic.
6. Add `metric_mappings` for metrics you want to analyze.
7. Update `analysis.metrics` in the project config.
8. Add deterministic metric and driver SQL if the existing finance variance logic does not fit.
9. Run `uv run sql-da validate-lineage --lineage <lineage.yaml>`.
10. Run `uv run sql-da analyze --config <config.yaml> --period <period> --compare-to <prior> --verbose`.

## Recommended Next Refactors

For a fully general agent, split the demo-specific finance pieces from the generic agent shell:

- Add a config flag such as `database.initialize_demo_if_missing: false` so non-demo configs never create synthetic data.
- Move finance-specific variance driver SQL into a domain plugin or YAML-backed query catalog.
- Allow config-driven data-quality checks.
- Add a generic `inspect-db` command that exports candidate node and column stubs from SQLite schema.
- Add tests with a second tiny SQLite fixture to prove the agent works outside the finance demo.

## Large Lineage Pattern

When lineage is too large for the model context window, keep the complete specs in YAML or another metadata store and expose retrieval tools. The live prompt should contain only counts, metric mappings, and instructions to retrieve relevant slices.

Use this sequence:

1. Search by metric, table, column, or business term with `search_semantic_specs`.
2. Fetch the focused node packet with `get_node_context`.
3. Expand upstream with `trace_upstream` using a small `max_depth`.
4. Fetch relationships or transformations for the node with small `limit` values.
5. Run read-only SQL only after the semantic slice identifies the relevant tables and fields.
