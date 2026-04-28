"""Rich terminal output and logging helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
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
            stderr_console.log(f"[bold cyan]{message}[/]")

    def trace_event(self, event: dict[str, Any]) -> None:
        if not self.verbose:
            return
        stderr_console.log(
            "[dim]trace[/] "
            f"[bold]{event['phase']}[/] "
            f"{event['actor']} -> {event['action']} "
            f"[{_status_color(event['status'])}]{event['status']}[/]"
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
