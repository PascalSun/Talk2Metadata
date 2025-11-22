"""Schema visualization for foreign key relationships."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from talk2metadata.core.schema import ForeignKey, SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def generate_html_visualization(
    schema: SchemaMetadata,
    output_path: str | Path,
    title: str = "Schema Visualization",
) -> Path:
    """Generate HTML visualization of schema with foreign key relationships.

    Args:
        schema: SchemaMetadata object
        output_path: Path to save HTML file
        title: Title for the visualization

    Returns:
        Path to generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate graph data
    nodes = []
    edges = []

    # Add nodes (tables)
    for table_name, table_meta in schema.tables.items():
        is_target = table_name == schema.target_table
        node = {
            "id": table_name,
            "label": table_name,
            "title": f"Table: {table_name}\nRows: {table_meta.row_count}\nColumns: {len(table_meta.columns)}",
            "group": "target" if is_target else "normal",
            "row_count": table_meta.row_count,
            "column_count": len(table_meta.columns),
            "primary_key": table_meta.primary_key or "None",
        }
        nodes.append(node)

    # Add edges (foreign keys)
    for fk in schema.foreign_keys:
        edge = {
            "from": fk.child_table,
            "to": fk.parent_table,
            "label": f"{fk.child_column} → {fk.parent_column}",
            "title": f"FK: {fk.child_table}.{fk.child_column} → {fk.parent_table}.{fk.parent_column}\nCoverage: {fk.coverage:.1%}",
            "coverage": fk.coverage,
            "color": {
                "color": "#ff6b6b" if fk.coverage < 0.9 else "#51cf66",
                "highlight": "#ff8787" if fk.coverage < 0.9 else "#69db7e",
            },
        }
        edges.append(edge)

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4dabf7;
            padding-bottom: 10px;
        }}
        .info {{
            background: #e7f5ff;
            border-left: 4px solid #4dabf7;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .info-item {{
            margin: 5px 0;
        }}
        .info-label {{
            font-weight: 600;
            color: #495057;
        }}
        #network {{
            width: 100%;
            height: 600px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            background: #fff;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .table-details {{
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}
        .table-card {{
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            background: #fff;
        }}
        .table-card.target {{
            border-color: #4dabf7;
            border-width: 2px;
            background: #e7f5ff;
        }}
        .table-card h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .table-card .pk {{
            color: #e03131;
            font-weight: 600;
        }}
        .fk-list {{
            margin-top: 10px;
        }}
        .fk-item {{
            padding: 5px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .fk-item.low-coverage {{
            background: #fff5f5;
            border-left: 3px solid #ff6b6b;
        }}
        .coverage {{
            font-weight: 600;
            color: #495057;
        }}
        .coverage.high {{
            color: #51cf66;
        }}
        .coverage.low {{
            color: #ff6b6b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="info">
            <div class="info-item">
                <span class="info-label">Target Table:</span> <strong>{schema.target_table}</strong>
            </div>
            <div class="info-item">
                <span class="info-label">Total Tables:</span> {len(schema.tables)}
            </div>
            <div class="info-item">
                <span class="info-label">Foreign Keys:</span> {len(schema.foreign_keys)}
            </div>
        </div>

        <div id="network"></div>

        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #4dabf7;"></span>
                Target Table
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #868e96;"></span>
                Normal Table
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #51cf66;"></span>
                FK Coverage ≥ 90%
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #ff6b6b;"></span>
                FK Coverage < 90%
            </div>
        </div>

        <div class="table-details">
            {_generate_table_cards(schema)}
        </div>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('network');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                shape: 'box',
                font: {{
                    size: 14,
                    face: 'Arial'
                }},
                borderWidth: 2,
                shadow: true,
                color: {{
                    border: '#2b343a',
                    background: '#868e96',
                    highlight: {{
                        border: '#2b343a',
                        background: '#4dabf7'
                    }}
                }},
                widthConstraint: {{
                    maximum: 200
                }},
                heightConstraint: {{
                    minimum: 50,
                    maximum: 100
                }}
            }},
            edges: {{
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 0.8
                    }}
                }},
                font: {{
                    size: 11,
                    align: 'middle'
                }},
                smooth: {{
                    type: 'curvedCW',
                    roundness: 0.2
                }},
                width: 2
            }},
            physics: {{
                enabled: true,
                stabilization: {{
                    iterations: 200
                }},
                barnesHut: {{
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }}
        }};

        // Customize node colors based on group
        nodes.forEach(function(node) {{
            if (node.group === 'target') {{
                node.color = {{
                    border: '#1971c2',
                    background: '#4dabf7',
                    highlight: {{
                        border: '#1971c2',
                        background: '#74c0fc'
                    }}
                }};
            }}
        }});

        var network = new vis.Network(container, data, options);

        // Add click event to show table details
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var element = document.getElementById('table-' + nodeId);
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    element.style.animation = 'pulse 0.5s';
                    setTimeout(function() {{
                        element.style.animation = '';
                    }}, 500);
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated schema visualization at {output_path}")
    return output_path


def _generate_table_cards(schema: SchemaMetadata) -> str:
    """Generate HTML for table detail cards."""
    cards = []

    for table_name, table_meta in schema.tables.items():
        is_target = table_name == schema.target_table
        target_class = "target" if is_target else ""

        # Get foreign keys for this table
        outgoing_fks = [
            fk for fk in schema.foreign_keys if fk.child_table == table_name
        ]
        incoming_fks = [
            fk for fk in schema.foreign_keys if fk.parent_table == table_name
        ]

        fk_html = ""
        if outgoing_fks:
            fk_html += '<div class="fk-list"><strong>Outgoing FKs:</strong>'
            for fk in outgoing_fks:
                coverage_class = "high" if fk.coverage >= 0.9 else "low"
                low_coverage_class = "low-coverage" if fk.coverage < 0.9 else ""
                fk_html += f'''
                    <div class="fk-item {low_coverage_class}">
                        {fk.child_table}.{fk.child_column} → {fk.parent_table}.{fk.parent_column}
                        <span class="coverage {coverage_class}">({fk.coverage:.1%})</span>
                    </div>
                '''
            fk_html += "</div>"

        if incoming_fks:
            fk_html += '<div class="fk-list"><strong>Incoming FKs:</strong>'
            for fk in incoming_fks:
                coverage_class = "high" if fk.coverage >= 0.9 else "low"
                fk_html += f'''
                    <div class="fk-item">
                        {fk.child_table}.{fk.child_column} → {fk.parent_table}.{fk.parent_column}
                        <span class="coverage {coverage_class}">({fk.coverage:.1%})</span>
                    </div>
                '''
            fk_html += "</div>"

        columns_list = ", ".join(table_meta.columns.keys())

        card = f"""
            <div class="table-card {target_class}" id="table-{table_name}">
                <h3>{table_name}{" (Target)" if is_target else ""}</h3>
                <p><strong>Rows:</strong> {table_meta.row_count}</p>
                <p><strong>Columns:</strong> {len(table_meta.columns)}</p>
                <p><strong>Primary Key:</strong> <span class="pk">{table_meta.primary_key or "None"}</span></p>
                <p><strong>Columns:</strong> {columns_list}</p>
                {fk_html}
            </div>
        """
        cards.append(card)

    return "\n".join(cards)


def validate_schema(schema: SchemaMetadata) -> Dict[str, List[str]]:
    """Validate schema and return errors/warnings.

    Args:
        schema: SchemaMetadata to validate

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []

    # Check if target table exists
    if schema.target_table not in schema.tables:
        errors.append(f"Target table '{schema.target_table}' not found in tables")

    # Check foreign keys
    for fk in schema.foreign_keys:
        # Check child table exists
        if fk.child_table not in schema.tables:
            errors.append(
                f"FK references non-existent child table: {fk.child_table}"
            )

        # Check parent table exists
        if fk.parent_table not in schema.tables:
            errors.append(
                f"FK references non-existent parent table: {fk.parent_table}"
            )

        # Check child column exists
        if fk.child_table in schema.tables:
            if fk.child_column not in schema.tables[fk.child_table].columns:
                errors.append(
                    f"FK references non-existent child column: "
                    f"{fk.child_table}.{fk.child_column}"
                )

        # Check parent column exists
        if fk.parent_table in schema.tables:
            if fk.parent_column not in schema.tables[fk.parent_table].columns:
                errors.append(
                    f"FK references non-existent parent column: "
                    f"{fk.parent_table}.{fk.parent_column}"
                )

        # Check coverage
        if fk.coverage < 0.5:
            errors.append(
                f"FK {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} has very low coverage "
                f"({fk.coverage:.1%}), may be incorrect"
            )
        elif fk.coverage < 0.9:
            warnings.append(
                f"FK {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} has low coverage "
                f"({fk.coverage:.1%})"
            )

    # Check for tables with no relationships
    table_names = set(schema.tables.keys())
    related_tables = set()
    for fk in schema.foreign_keys:
        related_tables.add(fk.child_table)
        related_tables.add(fk.parent_table)

    isolated_tables = table_names - related_tables
    if len(isolated_tables) > 1:  # Allow target table to be isolated
        warnings.append(
            f"Found {len(isolated_tables)} isolated tables (no FK relationships): "
            f"{', '.join(isolated_tables)}"
        )

    return {"errors": errors, "warnings": warnings}


def export_schema_for_review(
    schema: SchemaMetadata,
    output_path: str | Path,
    include_validation: bool = True,
) -> Path:
    """Export schema in a standardized, human-readable format for review.

    This format is designed to be easy to read, understand, and edit manually.

    Args:
        schema: SchemaMetadata to export
        output_path: Path to save the review file
        include_validation: Whether to include validation results

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("SCHEMA REVIEW FILE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This file contains the detected schema in a standardized format.")
    lines.append("You can review and modify this file, then use it with:")
    lines.append("  talk2metadata ingest <source> --schema <this_file>")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Basic info
    lines.append("BASIC INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Target Table: {schema.target_table}")
    lines.append(f"Total Tables: {len(schema.tables)}")
    lines.append(f"Foreign Keys: {len(schema.foreign_keys)}")
    lines.append("")

    # Validation results
    if include_validation:
        validation_result = validate_schema(schema)
        lines.append("VALIDATION RESULTS")
        lines.append("-" * 80)
        if validation_result["errors"]:
            lines.append("❌ ERRORS:")
            for error in validation_result["errors"]:
                lines.append(f"  - {error}")
            lines.append("")
        else:
            lines.append("✓ No errors found")
            lines.append("")

        if validation_result["warnings"]:
            lines.append("⚠️  WARNINGS:")
            for warning in validation_result["warnings"]:
                lines.append(f"  - {warning}")
            lines.append("")
        else:
            lines.append("✓ No warnings")
            lines.append("")

    # Tables
    lines.append("TABLES")
    lines.append("-" * 80)
    for table_name, table_meta in schema.tables.items():
        is_target = " (TARGET)" if table_name == schema.target_table else ""
        lines.append(f"\nTable: {table_name}{is_target}")
        lines.append(f"  Rows: {table_meta.row_count}")
        lines.append(f"  Columns: {len(table_meta.columns)}")
        lines.append(f"  Primary Key: {table_meta.primary_key or 'None (inferred)'}")
        lines.append("  Column Details:")
        for col_name, col_type in table_meta.columns.items():
            pk_marker = " [PK]" if col_name == table_meta.primary_key else ""
            lines.append(f"    - {col_name}: {col_type}{pk_marker}")
    lines.append("")

    # Foreign Keys
    lines.append("FOREIGN KEY RELATIONSHIPS")
    lines.append("-" * 80)
    if schema.foreign_keys:
        for i, fk in enumerate(schema.foreign_keys, 1):
            coverage_status = "✓" if fk.coverage >= 0.9 else "⚠"
            lines.append(f"\nFK #{i}: {coverage_status}")
            lines.append(f"  Child:  {fk.child_table}.{fk.child_column}")
            lines.append(f"  Parent: {fk.parent_table}.{fk.parent_column}")
            lines.append(f"  Coverage: {fk.coverage:.1%}")
            if fk.coverage < 0.9:
                lines.append(f"  ⚠️  Low coverage - please verify this relationship")
    else:
        lines.append("No foreign key relationships detected.")
    lines.append("")

    # JSON export (for programmatic use)
    lines.append("=" * 80)
    lines.append("JSON EXPORT (for programmatic use)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(json.dumps(schema.to_dict(), indent=2))

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported schema review file to {output_path}")
    return output_path

