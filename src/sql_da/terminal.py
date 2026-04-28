"""Rich terminal output and logging helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table

stderr_console = Console(stderr=True)
stdout_console = Console()


def configure_logging(verbose: bool = False) -> None:
    """Configure Python logging through Rich."""

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=stderr_console, rich_tracebacks=True, markup=True)],
        force=True,
    )


class TerminalReporter:
    """Small Rich wrapper for optional verbose stage output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def stage(self, message: str) -> None:
        if self.verbose:
            stderr_console.log(f"[bold cyan]{escape(message)}[/]")

    def trace_event(self, event: dict[str, Any]) -> None:
        if not self.verbose:
            return
        if event["phase"] == "live_agent":
            self._live_agent_event(event)
            return
        tool = f" [dim]tool={escape(str(event.get('tool', '')))}[/]" if event.get("tool") else ""
        stderr_console.log(
            "[dim]trace[/] "
            f"[bold]{escape(str(event['phase']))}[/] "
            f"{escape(str(event['actor']))} -> {escape(str(event['action']))}{tool} "
            f"[{_status_color(event['status'])}]{escape(str(event['status']))}[/]"
        )

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        if self.verbose:
            with stderr_console.status(message):
                yield
            return
        yield

    def checks_table(self, payload: dict[str, Any]) -> None:
        if not self.verbose:
            return
        table = Table(title="Reward Checks")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for check in payload.get("checks", []):
            status = "[green]pass[/]" if check.get("passed") else "[red]fail[/]"
            table.add_row(str(check.get("name")), status, str(check.get("detail", ""))[:120])
        stderr_console.print(table)

    def _live_agent_event(self, event: dict[str, Any]) -> None:
        action = str(event.get("action", ""))
        tool = str(event.get("tool", ""))
        status = str(event.get("status", "ok"))
        output = _clip(str(event.get("output_summary", "")), 150)
        input_summary = _clip(str(event.get("input_summary", "")), 120)
        if action == "estimate llm cost":
            metadata = event.get("metadata", {})
            cost = metadata.get("total_cost_usd")
            cost_text = "unknown" if cost is None else f"${cost:.8f}"
            stderr_console.log(
                "[dim]llm[/] "
                f"[bold]cost[/] {escape(cost_text)} "
                f"tokens={metadata.get('total_tokens', 0)} "
                f"input={metadata.get('input_tokens', 0)} "
                f"output={metadata.get('output_tokens', 0)} "
                f"[{_status_color(status)}]{escape(status)}[/]"
            )
            return
        if action == "tool call":
            stderr_console.log(
                "[dim]llm[/] "
                f"{escape(str(event.get('actor', 'deep-agent')))} -> "
                f"[bold]tool call[/] {escape(tool or 'unknown')} "
                f"{escape(input_summary)} "
                f"[{_status_color(status)}]{escape(status)}[/]"
            )
            return
        if action == "tool result":
            stderr_console.log(
                "[dim]llm[/] "
                f"[bold]tool result[/] {escape(tool or 'unknown')} "
                f"{escape(output)} "
                f"[{_status_color(status)}]{escape(status)}[/]"
            )
            return
        stderr_console.log(
            "[dim]llm[/] "
            f"{escape(str(event.get('actor', 'deep-agent')))} -> "
            f"{escape(action)} {escape(output)} "
            f"[{_status_color(status)}]{escape(status)}[/]"
        )

    def trace_table(self, events: list[dict[str, Any]]) -> None:
        if not self.verbose or not events:
            return
        table = Table(title="Agent Research Trace")
        table.add_column("#", justify="right")
        table.add_column("Phase")
        table.add_column("Actor")
        table.add_column("Action")
        table.add_column("Output")
        for event in events[:20]:
            table.add_row(
                str(event["sequence"]),
                str(event["phase"]),
                str(event["actor"]),
                str(event["action"]),
                str(event["output_summary"])[:90],
            )
        stderr_console.print(table)


def print_json(payload: Any) -> None:
    """Print JSON using Rich while preserving stdout as the machine-readable channel."""

    stdout_console.print_json(data=payload)


def _status_color(status: str) -> str:
    if status == "ok":
        return "green"
    if status == "error":
        return "red"
    return "yellow"


def _clip(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
