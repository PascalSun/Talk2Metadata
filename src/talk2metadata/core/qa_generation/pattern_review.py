"""Interactive pattern review and editing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import click

from talk2metadata.core.qa_generation.pattern_review_web import review_patterns_web
from talk2metadata.core.qa_generation.patterns import PathPattern
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def review_patterns_interactive(
    patterns: List[PathPattern],
    patterns_path: Optional[Path] = None,
    use_web: bool = True,
) -> List[PathPattern]:
    """Interactively review and edit path patterns.

    Args:
        patterns: List of path patterns to review
        patterns_path: Optional path to save edited patterns
        use_web: If True, use web interface; otherwise use CLI interface

    Returns:
        List of reviewed/edited PathPattern objects
    """
    if use_web and patterns_path:
        return review_patterns_web(patterns, patterns_path)

    # Fallback to CLI interface
    return _review_patterns_cli(patterns, patterns_path)


def _review_patterns_cli(
    patterns: List[PathPattern],
    patterns_path: Optional[Path] = None,
) -> List[PathPattern]:
    """CLI-based pattern review (fallback)."""
    click.echo("\n📝 Pattern Review Mode")
    click.echo("=" * 80)
    click.echo(f"Total patterns: {len(patterns)}")
    click.echo("\nCommands:")
    click.echo("  [Enter] - Next pattern")
    click.echo("  e - Edit current pattern")
    click.echo("  d - Delete current pattern")
    click.echo("  a - Add new pattern")
    click.echo("  s - Save and continue")
    click.echo("  q - Quit (save changes)")
    click.echo("  x - Exit without saving")
    click.echo("=" * 80)

    edited_patterns = patterns.copy()
    current_idx = 0

    while current_idx < len(edited_patterns):
        pattern = edited_patterns[current_idx]
        click.echo(f"\n[{current_idx + 1}/{len(edited_patterns)}] Pattern:")
        click.echo(f"  Path: {' -> '.join(pattern.pattern)}")
        click.echo(f"  Semantic: {pattern.semantic}")
        click.echo(f"  Template: {pattern.question_template}")
        click.echo(f"  Difficulty: {pattern.difficulty}")
        click.echo(f"  Answer Type: {pattern.answer_type}")
        if pattern.description:
            click.echo(f"  Description: {pattern.description}")

        action = (
            click.prompt("\nAction", default="", show_default=False).strip().lower()
        )

        if action == "e":
            # Edit pattern
            edited_patterns[current_idx] = _edit_pattern_interactive(pattern)
        elif action == "d":
            # Delete pattern
            if click.confirm("Delete this pattern?"):
                edited_patterns.pop(current_idx)
                click.echo("Pattern deleted")
                continue  # Don't increment index
        elif action == "a":
            # Add new pattern
            new_pattern = _create_pattern_interactive()
            if new_pattern:
                edited_patterns.insert(current_idx + 1, new_pattern)
                click.echo("Pattern added")
        elif action == "s":
            # Save and continue
            if patterns_path:
                _save_patterns(edited_patterns, patterns_path)
                click.echo(f"Saved to {patterns_path}")
            else:
                click.echo("No save path specified, skipping save")
        elif action == "q":
            # Quit and save
            if patterns_path:
                _save_patterns(edited_patterns, patterns_path)
                click.echo(f"Saved to {patterns_path}")
            break
        elif action == "x":
            # Exit without saving
            if click.confirm("Exit without saving changes?"):
                return patterns  # Return original patterns
            continue

        current_idx += 1

    if current_idx >= len(edited_patterns):
        click.echo("\n✅ Finished reviewing all patterns")

    return edited_patterns


def _edit_pattern_interactive(pattern: PathPattern) -> PathPattern:
    """Interactively edit a single pattern.

    Args:
        pattern: Pattern to edit

    Returns:
        Edited pattern
    """
    click.echo("\nEditing pattern (press Enter to keep current value):")

    # Edit path
    current_path = " -> ".join(pattern.pattern)
    new_path_str = click.prompt("Path", default=current_path, show_default=True)
    if new_path_str != current_path:
        new_path = [t.strip() for t in new_path_str.split("->")]
        pattern.pattern = new_path

    # Edit semantic
    new_semantic = click.prompt("Semantic", default=pattern.semantic, show_default=True)
    if new_semantic:
        pattern.semantic = new_semantic

    # Edit template
    new_template = click.prompt(
        "Question Template", default=pattern.question_template, show_default=True
    )
    if new_template:
        pattern.question_template = new_template

    # Edit difficulty
    new_difficulty = click.prompt(
        "Difficulty (easy/medium/hard)",
        default=pattern.difficulty,
        show_default=True,
        type=click.Choice(["easy", "medium", "hard"], case_sensitive=False),
    )
    pattern.difficulty = new_difficulty.lower()

    # Edit answer type
    new_answer_type = click.prompt(
        "Answer Type (single/multiple/aggregate)",
        default=pattern.answer_type,
        show_default=True,
        type=click.Choice(["single", "multiple", "aggregate"], case_sensitive=False),
    )
    pattern.answer_type = new_answer_type.lower()

    # Edit description
    new_description = click.prompt(
        "Description", default=pattern.description or "", show_default=True
    )
    pattern.description = new_description if new_description else None

    return pattern


def _create_pattern_interactive() -> Optional[PathPattern]:
    """Interactively create a new pattern.

    Returns:
        New PathPattern or None if cancelled
    """
    click.echo("\nCreating new pattern:")

    try:
        path_str = click.prompt("Path (table1 -> table2 -> ...)", type=str)
        path = [t.strip() for t in path_str.split("->")]

        semantic = click.prompt("Semantic description", type=str)
        template = click.prompt("Question template", type=str)
        difficulty = click.prompt(
            "Difficulty",
            type=click.Choice(["easy", "medium", "hard"], case_sensitive=False),
            default="medium",
        )
        answer_type = click.prompt(
            "Answer Type",
            type=click.Choice(
                ["single", "multiple", "aggregate"], case_sensitive=False
            ),
            default="multiple",
        )
        description = click.prompt(
            "Description (optional)", default="", show_default=False
        )

        return PathPattern(
            pattern=path,
            semantic=semantic,
            question_template=template,
            difficulty=difficulty.lower(),
            answer_type=answer_type.lower(),
            description=description if description else None,
        )
    except (click.Abort, KeyboardInterrupt):
        click.echo("\nCancelled")
        return None


def _save_patterns(patterns: List[PathPattern], path: Path) -> None:
    """Save patterns to file.

    Args:
        patterns: List of patterns to save
        path: Path to save file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load existing file to preserve metadata
    target_table = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_table = data.get("target_table")
        except Exception:
            pass

    data = {
        "target_table": target_table or "unknown",
        "total_patterns": len(patterns),
        "patterns": [p.to_dict() for p in patterns],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(patterns)} patterns to {path}")
