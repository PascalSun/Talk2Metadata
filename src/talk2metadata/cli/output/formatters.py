"""Output formatting utilities for CLI."""

from __future__ import annotations

from typing import Any, Dict, List

import click


class OutputFormatter:
    """Format output for CLI display.

    Provides consistent formatting for different types of CLI output,
    including success messages, errors, warnings, and structured data.

    Example:
        >>> out = OutputFormatter()
        >>> out.success("Operation completed")
        >>> out.error("Something went wrong")
        >>> out.stats({"records": 100, "tables": 5})
    """

    @staticmethod
    def success(message: str) -> None:
        """Display success message with checkmark.

        Args:
            message: Success message to display
        """
        click.echo(f"✓ {message}")

    @staticmethod
    def error(message: str, abort: bool = False) -> None:
        """Display error message.

        Args:
            message: Error message to display
            abort: Whether to abort command execution after displaying error
        """
        click.echo(f"❌ {message}", err=True)
        if abort:
            raise click.Abort()

    @staticmethod
    def warning(message: str) -> None:
        """Display warning message.

        Args:
            message: Warning message to display
        """
        click.echo(f"⚠️  {message}")

    @staticmethod
    def info(message: str) -> None:
        """Display info message.

        Args:
            message: Info message to display
        """
        click.echo(f"ℹ️  {message}")

    @staticmethod
    def section(title: str) -> None:
        """Display section header.

        Args:
            title: Section title
        """
        click.echo(f"\n{title}")

    @staticmethod
    def stats(stats_dict: Dict[str, Any], indent: str = "   ") -> None:
        """Display statistics dictionary in key: value format.

        Args:
            stats_dict: Dictionary of statistics to display
            indent: Indentation string for each line
        """
        for key, value in stats_dict.items():
            click.echo(f"{indent}{key}: {value}")

    @staticmethod
    def list_items(items: List[str], indent: str = "   ", bullet: str = "-") -> None:
        """Display list of items with bullets.

        Args:
            items: List of strings to display
            indent: Indentation string for each line
            bullet: Bullet character to use
        """
        for item in items:
            click.echo(f"{indent}{bullet} {item}")

    @staticmethod
    def table_summary(
        table_name: str,
        row_count: int,
        col_count: int,
        primary_key: str = None,
        indent: str = "   ",
    ) -> None:
        """Display table summary in consistent format.

        Args:
            table_name: Name of the table
            row_count: Number of rows
            col_count: Number of columns
            primary_key: Primary key column name (if any)
            indent: Indentation string
        """
        pk_str = f", PK={primary_key}" if primary_key else ""
        click.echo(
            f"{indent}✓ {table_name}: {row_count} rows, {col_count} columns{pk_str}"
        )

    @staticmethod
    def progress_start(message: str) -> None:
        """Display progress start message.

        Args:
            message: Progress message to display
        """
        click.echo(f"\n🔍 {message}")

    @staticmethod
    def step(message: str, step_num: int = None) -> None:
        """Display a numbered step.

        Args:
            message: Step description
            step_num: Optional step number
        """
        if step_num is not None:
            click.echo(f"\n{step_num}. {message}")
        else:
            click.echo(f"\n• {message}")

    @staticmethod
    def next_steps(title: str, steps: List[str]) -> None:
        """Display next steps section.

        Args:
            title: Section title
            steps: List of next step descriptions
        """
        click.echo(f"\n{title}")
        for step in steps:
            click.echo(f"   - {step}")
