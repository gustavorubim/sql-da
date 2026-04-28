"""Command-line interface for sql-da."""

from __future__ import annotations

import argparse
from pathlib import Path

from sql_da.config import ProjectConfig
from sql_da.lineage import LineageGraph
from sql_da.reward import run_reward
from sql_da.synthetic import init_demo_database
from sql_da.terminal import TerminalReporter, configure_logging, print_json
from sql_da.workflow import analyze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sql-da")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-demo-db", help="Build the synthetic SQLite demo DB.")
    init_parser.add_argument("--config", default="configs/demo.yaml")
    init_parser.add_argument("--verbose", action="store_true", help="Show Rich stage output.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze month-over-month variances.")
    analyze_parser.add_argument("--config", default="configs/demo.yaml")
    analyze_parser.add_argument("--period", required=True)
    analyze_parser.add_argument("--compare-to", required=True)
    analyze_parser.add_argument("--run-id")
    analyze_parser.add_argument("--verbose", action="store_true", help="Show Rich stage output.")
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final JSON payload even when --verbose is enabled.",
    )

    lineage_parser = subparsers.add_parser("validate-lineage", help="Validate lineage YAML.")
    lineage_parser.add_argument("--lineage", default="configs/lineage.yaml")
    lineage_parser.add_argument("--verbose", action="store_true", help="Show Rich stage output.")

    reward_parser = subparsers.add_parser("reward", help="Run the verifiable reward suite.")
    reward_parser.add_argument("--config", default="configs/demo.yaml")
    reward_parser.add_argument("--verbose", action="store_true", help="Show Rich stage output.")
    reward_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final JSON payload even when --verbose is enabled.",
    )
    reward_parser.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Skip subprocess ruff/pytest checks and run deterministic checks only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    reporter = TerminalReporter(args.verbose)
    if args.command == "init-demo-db":
        config = ProjectConfig.from_yaml(args.config)
        with reporter.status("Building synthetic SQLite demo database"):
            db_path = init_demo_database(config)
        print_json({"created": True, "database_path": str(db_path)})
        return 0
    if args.command == "analyze":
        config = ProjectConfig.from_yaml(args.config)
        result = analyze(
            config,
            args.period,
            args.compare_to,
            run_id=args.run_id,
            reporter=reporter,
        )
        payload = result.to_dict()
        if args.verbose and not args.json:
            reporter.stage(_analysis_summary(payload))
        else:
            print_json(payload)
        return 0
    if args.command == "validate-lineage":
        with reporter.status("Validating lineage YAML"):
            graph = LineageGraph.from_yaml(Path(args.lineage))
        errors = graph.validate()
        if errors:
            print_json({"valid": False, "errors": errors})
            return 1
        print_json({"valid": True, "nodes": len(graph.nodes)})
        return 0
    if args.command == "reward":
        config = ProjectConfig.from_yaml(args.config)
        payload = run_reward(
            config,
            include_quality_gates=not args.skip_quality_gates,
            reporter=reporter,
        )
        reporter.checks_table(payload)
        if args.verbose and not args.json:
            reporter.stage(_reward_summary(payload, config.reward.output_path))
        else:
            print_json(payload)
        return 0 if payload["passed"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")  # pragma: no cover


def _analysis_summary(payload: dict) -> str:
    cost = payload.get("llm_cost_estimate", {})
    total_cost = cost.get("total_cost_usd")
    cost_text = "unknown" if total_cost is None else f"${total_cost:.8f}"
    return (
        "Analysis complete: "
        f"report={payload.get('report_path')}; "
        f"trace={payload.get('trace_json_path')}; "
        f"llm_cost={cost_text}"
    )


def _reward_summary(payload: dict, output_path: Path) -> str:
    status = "passed" if payload.get("passed") else "failed"
    return f"Reward {status}: score={payload.get('score')}; output={output_path}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
