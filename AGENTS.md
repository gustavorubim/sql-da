# Repository Instructions

This repo implements a local SQL Deep Agent for finance variance analysis. It builds a deterministic synthetic SQLite database, validates YAML lineage, computes month-over-month variance findings with deterministic SQL, and optionally uses LangChain Deep Agents with Google Gemini to review root-cause narratives.

## Project Shape

- Python package: `src/sql_da`
- CLI entrypoint: `sql-da`
- Demo config: `configs/demo.yaml`
- Lineage graph: `configs/lineage.yaml`
- Runtime Deep Agents memory: `memory/AGENTS.md`
- Runtime Deep Agents skills: `skills/*/SKILL.md`
- Tests: `tests`
- Generated local artifacts: `data/demo_finance.db`, `reports/*`, `.coverage`, and cache folders

Keep implementation changes scoped to the existing module boundaries unless a task explicitly requires a larger refactor.

## Environment

Use the repo-local `uv` workflow and `.venv`.

```bash
uv venv
uv sync
uv pip install .
```

Do not assume global Python packages are available. Use `uv run ...` for commands.

Secrets belong in `.env`, which is gitignored. Never commit real `GOOGLE_API_KEY` values.

## Required Quality Gates

For every task, before reporting completion, ensure the formatter, linter, tests, and coverage gate are clean:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=src/sql_da --cov-fail-under=95
```

The project target is at least 95% line coverage. If a task cannot reasonably run the full suite, state exactly what was skipped and why, and prefer running the closest focused pytest command first.

When changes affect the CLI workflow, reports, lineage, SQL safety, reward scoring, synthetic data, or agent orchestration, also run:

```bash
uv run sql-da reward --config configs/demo.yaml
```

The reward command must not require `GOOGLE_API_KEY`; live Gemini checks should remain optional.

## README Discipline

Always check whether `README.md` should be updated before finishing a task.

Update the README when a change affects:

- setup or dependency commands
- CLI flags or examples
- generated report, evidence, trace, or reward artifacts
- config keys or lineage format
- model defaults, memory, skills, or subagent behavior
- verification commands or coverage expectations

If no README update is needed, mention that it was checked in the final response.

## Agent And SQL Safety

- Keep SQL access read-only and bounded by row limits.
- Preserve mutation blocking for statements such as `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, and `CREATE`.
- Deterministic SQL/math should identify material variances before live LLM review.
- Live Gemini review should gracefully skip when credentials are absent.
- Report findings should cite lineage nodes and SQL-backed evidence.
- Keep `agent_trace.csv` and `agent_trace.json` generation working for analysis and reward runs.

## Terminal Experience

Use Rich logging patterns already present in the repo for visible progress. Long-running commands should show stage information or streamed subprocess output where practical, especially reward and analysis workflows.

## Editing Guidance

- Prefer small, focused patches.
- Use structured parsers for YAML, JSON, and SQL-related data instead of ad hoc text parsing when possible.
- Keep generated artifacts out of source changes unless a task explicitly asks to update checked-in examples.
- Do not revert user changes or unrelated work in the tree.
