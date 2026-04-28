---
name: sql-lineage
description: Use this skill when tracing a financial metric through mart views, transformations, source tables, and evidence SQL.
---

# sql-lineage

## Instructions

1. Call lineage tools before writing SQL when a metric is unfamiliar.
2. Identify the mart node, upstream raw tables, dimensions, grain, and value column.
3. Use only read-only SQL through the provided SQL tool.
4. Keep queries narrow: select the compared periods, group by the likely driver dimensions, and order by absolute variance.
5. If a root-cause claim requires a field that is not in the current SQL output, request or write a better evidence query.
6. Never infer a database table, column, customer, vendor, or product that is not visible in schema or lineage.
