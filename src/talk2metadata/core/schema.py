"""Schema detection and foreign key inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForeignKey:
    """Foreign key relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float  # 0.0-1.0 (percentage of child values found in parent)

    def __repr__(self) -> str:
        return (
            f"FK({self.child_table}.{self.child_column} -> "
            f"{self.parent_table}.{self.parent_column}, coverage={self.coverage:.2f})"
        )


@dataclass
class TableMetadata:
    """Metadata for a single table."""

    name: str
    columns: Dict[str, str]  # column_name -> dtype
    primary_key: Optional[str] = None
    row_count: int = 0
    sample_values: Dict[str, List[str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TableMetadata({self.name}, columns={len(self.columns)}, rows={self.row_count})"


@dataclass
class SchemaMetadata:
    """Complete schema metadata."""

    tables: Dict[str, TableMetadata]
    foreign_keys: List[ForeignKey]
    target_table: str

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved schema metadata to {path}")

    @classmethod
    def load(cls, path: str | Path) -> SchemaMetadata:
        """Load metadata from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            SchemaMetadata instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "tables": {
                name: {
                    "columns": meta.columns,
                    "primary_key": meta.primary_key,
                    "row_count": meta.row_count,
                    "sample_values": meta.sample_values,
                }
                for name, meta in self.tables.items()
            },
            "foreign_keys": [
                {
                    "child_table": fk.child_table,
                    "child_column": fk.child_column,
                    "parent_table": fk.parent_table,
                    "parent_column": fk.parent_column,
                    "coverage": fk.coverage,
                }
                for fk in self.foreign_keys
            ],
            "target_table": self.target_table,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> SchemaMetadata:
        """Create from dictionary.

        Args:
            data: Dictionary with schema data

        Returns:
            SchemaMetadata instance
        """
        tables = {
            name: TableMetadata(
                name=name,
                columns=meta["columns"],
                primary_key=meta.get("primary_key"),
                row_count=meta.get("row_count", 0),
                sample_values=meta.get("sample_values", {}),
            )
            for name, meta in data["tables"].items()
        }

        foreign_keys = [
            ForeignKey(
                child_table=fk["child_table"],
                child_column=fk["child_column"],
                parent_table=fk["parent_table"],
                parent_column=fk["parent_column"],
                coverage=fk["coverage"],
            )
            for fk in data["foreign_keys"]
        ]

        return cls(
            tables=tables,
            foreign_keys=foreign_keys,
            target_table=data["target_table"],
        )

    def get_related_tables(self, table_name: str) -> List[str]:
        """Get all tables related to the given table via foreign keys.

        Args:
            table_name: Table name

        Returns:
            List of related table names
        """
        related = set()

        for fk in self.foreign_keys:
            if fk.child_table == table_name:
                related.add(fk.parent_table)
            elif fk.parent_table == table_name:
                related.add(fk.child_table)

        return list(related)

    def get_foreign_keys_for_table(
        self, table_name: str, direction: str = "both"
    ) -> List[ForeignKey]:
        """Get foreign keys involving a table.

        Args:
            table_name: Table name
            direction: 'outgoing' (child), 'incoming' (parent), or 'both'

        Returns:
            List of ForeignKey objects
        """
        if direction == "outgoing":
            return [fk for fk in self.foreign_keys if fk.child_table == table_name]
        elif direction == "incoming":
            return [fk for fk in self.foreign_keys if fk.parent_table == table_name]
        else:  # both
            return [
                fk
                for fk in self.foreign_keys
                if fk.child_table == table_name or fk.parent_table == table_name
            ]

    def __repr__(self) -> str:
        return (
            f"SchemaMetadata(tables={len(self.tables)}, "
            f"fks={len(self.foreign_keys)}, target={self.target_table})"
        )


class SchemaDetector:
    """Schema detection with foreign key inference."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize schema detector.

        Args:
            config: Configuration dict (uses global config if None)
        """
        self.config = config or get_config().get("schema", {})
        self.fk_config = self.config.get("fk_detection", {})
        self.min_coverage = self.fk_config.get("min_coverage", 0.9)
        self.tolerance = self.fk_config.get("inclusion_tolerance", 0.1)

    def detect(
        self,
        tables: Dict[str, pd.DataFrame],
        target_table: str,
        provided_schema: Optional[Dict] = None,
    ) -> SchemaMetadata:
        """Detect schema and infer foreign keys.

        Args:
            tables: Dict mapping table_name -> DataFrame
            target_table: Name of the target table
            provided_schema: Optional schema dict with FK information

        Returns:
            SchemaMetadata object

        Example:
            >>> tables = {"orders": orders_df, "customers": customers_df}
            >>> detector = SchemaDetector()
            >>> metadata = detector.detect(tables, target_table="orders")
        """
        logger.info(f"Detecting schema for {len(tables)} tables")

        if target_table not in tables:
            raise ValueError(
                f"Target table '{target_table}' not found in tables: {list(tables.keys())}"
            )

        # 1. Extract table metadata
        table_metadata = self._extract_table_metadata(tables)

        # 2. Detect foreign keys
        if provided_schema and "foreign_keys" in provided_schema:
            logger.info("Using provided schema for foreign keys")
            fks = self._parse_provided_fks(provided_schema["foreign_keys"], tables)
        else:
            logger.info("Inferring foreign keys from data")
            fks = self._detect_foreign_keys(tables, table_metadata, target_table)

        logger.info(f"Detected {len(fks)} foreign key relationships")
        for fk in fks:
            logger.debug(f"  {fk}")

        return SchemaMetadata(
            tables=table_metadata,
            foreign_keys=fks,
            target_table=target_table,
        )

    def _extract_table_metadata(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, TableMetadata]:
        """Extract metadata from tables.

        Args:
            tables: Dict of DataFrames

        Returns:
            Dict of TableMetadata objects
        """
        metadata = {}

        for name, df in tables.items():
            # Collect sample values for each column (first 3 non-null values)
            sample_values = {}
            for col in df.columns:
                non_null = df[col].dropna().head(3).astype(str).tolist()
                if non_null:
                    sample_values[col] = non_null

            metadata[name] = TableMetadata(
                name=name,
                columns={col: str(dtype) for col, dtype in df.dtypes.items()},
                primary_key=self._infer_primary_key(df),
                row_count=len(df),
                sample_values=sample_values,
            )

            logger.debug(
                f"Table {name}: {len(df.columns)} columns, {len(df)} rows, "
                f"PK={metadata[name].primary_key}"
            )

        return metadata

    def _infer_primary_key(self, df: pd.DataFrame) -> Optional[str]:
        """Infer primary key column.

        Args:
            df: DataFrame

        Returns:
            Primary key column name or None
        """
        # Priority 1: Column named 'id'
        if "id" in df.columns and df["id"].is_unique and not df["id"].isna().any():
            return "id"

        # Priority 2: Column ending with '_id' that is unique
        for col in df.columns:
            if col.endswith("_id") and df[col].is_unique and not df[col].isna().any():
                return col

        # Priority 3: Any unique column without nulls
        for col in df.columns:
            if df[col].is_unique and not df[col].isna().any():
                return col

        return None

    def _detect_foreign_keys(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: str,
    ) -> List[ForeignKey]:
        """Detect foreign key relationships using hybrid strategy.

        Combines rule-based detection with AI agent analysis for fuzzy patterns.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Name of the target table (for deduplication priority)

        Returns:
            List of ForeignKey objects
        """
        # Phase 1: Rule-based detection
        logger.info("Running rule-based FK detection...")
        rule_based_fks = self._detect_fks_rule_based(tables, table_metadata)
        logger.info(f"Rule-based detection found {len(rule_based_fks)} FKs")

        # Phase 2: Decide whether to use agent
        use_agent = self.fk_config.get("use_agent", True)
        agent_trigger = self.fk_config.get("agent_trigger", "auto")
        agent_threshold = self.fk_config.get("agent_threshold", 2)

        should_use_agent = False
        if not use_agent:
            logger.info("Agent-based FK detection disabled in config")
        elif agent_trigger == "never":
            logger.info("Agent trigger set to 'never'")
        elif agent_trigger == "always":
            should_use_agent = True
            logger.info("Agent trigger set to 'always'")
        elif agent_trigger == "auto":
            if len(rule_based_fks) < agent_threshold:
                should_use_agent = True
                logger.info(
                    f"Rule-based found {len(rule_based_fks)} FKs (< threshold {agent_threshold}), "
                    "triggering agent-based detection"
                )
            else:
                logger.info(
                    f"Rule-based found {len(rule_based_fks)} FKs (>= threshold {agent_threshold}), "
                    "skipping agent-based detection"
                )

        # Phase 3: Agent-based detection (if triggered)
        agent_fks = []
        if should_use_agent:
            agent_fks = self._detect_fks_with_agent(
                tables, table_metadata, rule_based_fks, target_table
            )

        # Combine results
        all_fks = rule_based_fks + agent_fks

        # Phase 4: Deduplicate and prioritize FKs
        all_fks = self._deduplicate_fks(all_fks, table_metadata, target_table)

        logger.info(
            f"Total FKs detected: {len(all_fks)} "
            f"(rule-based: {len(rule_based_fks)}, agent: {len(agent_fks)})"
        )

        return all_fks

    def _detect_fks_rule_based(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
    ) -> List[ForeignKey]:
        """Rule-based FK detection (original heuristic method).

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata

        Returns:
            List of ForeignKey objects
        """
        fks = []

        for child_name, child_df in tables.items():
            for child_col in child_df.columns:
                # Heuristic 1: Column name ends with _id or _key
                if not (child_col.endswith("_id") or child_col.endswith("_key")):
                    continue

                # Skip if this is the table's own primary key
                if child_col == table_metadata[child_name].primary_key:
                    continue

                # Heuristic 2: Extract potential parent table name
                parent_candidates = self._get_parent_candidates(
                    child_col, list(tables.keys())
                )

                for parent_name in parent_candidates:
                    if parent_name == child_name:  # Skip self-references
                        continue

                    parent_df = tables[parent_name]
                    parent_pk = table_metadata[parent_name].primary_key

                    if parent_pk is None or parent_pk not in parent_df.columns:
                        continue

                    # Check inclusion dependency
                    coverage = self._check_inclusion(
                        child_df[child_col],
                        parent_df[parent_pk],
                    )

                    if coverage >= self.min_coverage:
                        fks.append(
                            ForeignKey(
                                child_table=child_name,
                                child_column=child_col,
                                parent_table=parent_name,
                                parent_column=parent_pk,
                                coverage=coverage,
                            )
                        )
                        break  # Found FK, stop searching for this column

        return fks

    def _get_parent_candidates(
        self, column_name: str, table_names: List[str]
    ) -> List[str]:
        """Get potential parent table names from column name.

        Args:
            column_name: Column name (e.g., 'customer_id')
            table_names: Available table names

        Returns:
            List of candidate parent table names
        """
        # Remove suffixes
        base_name = (
            column_name.replace("_id", "").replace("_key", "").replace("_fk", "")
        )

        candidates = []

        # Exact match
        if base_name in table_names:
            candidates.append(base_name)

        # Plural forms
        if base_name + "s" in table_names:
            candidates.append(base_name + "s")

        # Try removing common prefixes
        for table_name in table_names:
            if table_name.lower().startswith(base_name.lower()):
                candidates.append(table_name)

        return candidates

    def _check_inclusion(
        self,
        child_values: pd.Series,
        parent_values: pd.Series,
    ) -> float:
        """Check inclusion dependency (child ⊆ parent).

        Args:
            child_values: Child column values
            parent_values: Parent column values

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        child_set = set(child_values.dropna())
        parent_set = set(parent_values.dropna())

        if not child_set:
            return 0.0

        overlap = child_set & parent_set
        return len(overlap) / len(child_set)

    def _parse_provided_fks(
        self, fk_list: List[Dict], tables: Dict[str, pd.DataFrame]
    ) -> List[ForeignKey]:
        """Parse foreign keys from provided schema.

        Args:
            fk_list: List of FK dicts
            tables: Dict of DataFrames (for validation)

        Returns:
            List of ForeignKey objects
        """
        fks = []

        for fk_dict in fk_list:
            child_table = fk_dict["child_table"]
            child_column = fk_dict["child_column"]
            parent_table = fk_dict["parent_table"]
            parent_column = fk_dict["parent_column"]

            # Validate
            if child_table not in tables:
                logger.warning(f"Child table {child_table} not found, skipping FK")
                continue

            if parent_table not in tables:
                logger.warning(f"Parent table {parent_table} not found, skipping FK")
                continue

            # Calculate coverage
            coverage = self._check_inclusion(
                tables[child_table][child_column],
                tables[parent_table][parent_column],
            )

            fks.append(
                ForeignKey(
                    child_table=child_table,
                    child_column=child_column,
                    parent_table=parent_table,
                    parent_column=parent_column,
                    coverage=coverage,
                )
            )

        return fks

    def _find_column_candidates(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        target_table: Optional[str] = None,
    ) -> List[Dict]:
        """Find candidate column pairs with high value overlap.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            target_table: Target/center table name (prioritized in star schema)

        Returns:
            List of candidate dicts with overlap info, sorted by target table priority
        """
        candidates = []
        min_overlap = self.fk_config.get("min_overlap_ratio", 0.8)

        # Find all column pairs with potential relationships
        for child_name, child_df in tables.items():
            # Skip target table itself as child
            if child_name == target_table:
                continue

            for child_col in child_df.columns:
                # Note: A column CAN be both a PK and FK (foreign primary key pattern)
                # We only skip self-referential relationships (same table)

                # Check overlap with all other tables' primary keys
                for parent_name, parent_df in tables.items():
                    if parent_name == child_name:
                        continue

                    parent_pk = table_metadata[parent_name].primary_key
                    if parent_pk is None or parent_pk not in parent_df.columns:
                        continue

                    # Calculate overlap
                    coverage = self._check_inclusion(
                        child_df[child_col], parent_df[parent_pk]
                    )

                    if coverage >= min_overlap:
                        # Get sample values for agent analysis
                        child_sample = child_df[child_col].dropna().head(5).tolist()
                        parent_sample = parent_df[parent_pk].dropna().head(5).tolist()

                        # Mark if this is a target table relationship
                        is_target_relationship = parent_name == target_table

                        candidates.append(
                            {
                                "child_table": child_name,
                                "child_column": child_col,
                                "parent_table": parent_name,
                                "parent_column": parent_pk,
                                "coverage": coverage,
                                "child_sample": child_sample,
                                "parent_sample": parent_sample,
                                "child_unique": child_df[child_col].nunique(),
                                "parent_unique": parent_df[parent_pk].nunique(),
                                "is_target_relationship": is_target_relationship,
                            }
                        )

        # Sort candidates: target table relationships first, then by coverage
        candidates.sort(key=lambda x: (not x["is_target_relationship"], -x["coverage"]))

        return candidates

    def _detect_fks_with_agent(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
        rule_based_fks: List[ForeignKey],
        target_table: Optional[str] = None,
    ) -> List[ForeignKey]:
        """Use AI agent to detect foreign keys from fuzzy patterns.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata
            rule_based_fks: FKs already found by rule-based detection
            target_table: Target/center table name (for star schema prioritization)

        Returns:
            List of additional ForeignKey objects found by agent
        """
        try:
            from talk2metadata.agent import AgentWrapper
        except ImportError:
            logger.warning(
                "Agent module not available, skipping agent-based FK detection"
            )
            return []

        # Find candidates (prioritizing target table relationships)
        candidates = self._find_column_candidates(tables, table_metadata, target_table)

        if not candidates:
            logger.info("No candidate column pairs found for agent analysis")
            return []

        # Filter out candidates already found by rule-based detection
        existing_fks = {
            (fk.child_table, fk.child_column, fk.parent_table, fk.parent_column)
            for fk in rule_based_fks
        }
        candidates = [
            c
            for c in candidates
            if (
                c["child_table"],
                c["child_column"],
                c["parent_table"],
                c["parent_column"],
            )
            not in existing_fks
        ]

        if not candidates:
            logger.info(
                "All high-overlap candidates already detected by rule-based method"
            )
            return []

        logger.info(f"Analyzing {len(candidates)} candidate FKs with AI agent...")

        # Prepare prompt for agent (with target table context)
        prompt = self._build_agent_prompt(candidates, table_metadata, target_table)

        # Initialize agent wrapper
        try:
            agent = AgentWrapper()
        except Exception as e:
            logger.warning(f"Failed to initialize agent: {e}")
            logger.warning("Skipping agent-based FK detection")
            return []

        # Call agent
        try:
            response = agent.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=4096,
            )

            # Parse response
            agent_fks = self._parse_agent_response(response.content, candidates)

            logger.info(f"Agent detected {len(agent_fks)} additional FKs")
            return agent_fks

        except Exception as e:
            logger.error(f"Agent FK detection failed: {e}")
            return []

    def _build_agent_prompt(
        self,
        candidates: List[Dict],
        table_metadata: Dict[str, TableMetadata],
        target_table: Optional[str] = None,
    ) -> str:
        """Build prompt for agent FK detection.

        Args:
            candidates: List of candidate column pairs
            table_metadata: Dict of TableMetadata
            target_table: Target/center table name (for star schema context)

        Returns:
            Prompt string
        """
        # Build table schema summary
        schema_summary = "## Database Schema\n\n"
        if target_table:
            schema_summary += f"**Target/Center Table: `{target_table}`** (all tables should relate to this)\n\n"

        for table_name, meta in table_metadata.items():
            marker = " (TARGET)" if table_name == target_table else ""
            schema_summary += f"**{table_name}{marker}**\n"
            schema_summary += f"- Primary Key: {meta.primary_key}\n"
            schema_summary += f"- Columns: {', '.join(meta.columns.keys())}\n"
            schema_summary += f"- Row Count: {meta.row_count}\n\n"

        # Build candidates summary
        candidates_summary = "## Candidate Foreign Key Relationships\n\n"
        for i, c in enumerate(candidates, 1):
            # Mark if this is a target table relationship
            target_marker = (
                " ⭐ TARGET TABLE" if c.get("is_target_relationship") else ""
            )
            candidates_summary += f"### Candidate {i}{target_marker}\n"
            candidates_summary += f"- Child: `{c['child_table']}.{c['child_column']}`\n"
            candidates_summary += (
                f"- Parent: `{c['parent_table']}.{c['parent_column']}`\n"
            )
            candidates_summary += f"- Coverage: {c['coverage']:.2%}\n"
            candidates_summary += f"- Child unique values: {c['child_unique']}\n"
            candidates_summary += f"- Parent unique values: {c['parent_unique']}\n"
            candidates_summary += f"- Child sample values: {c['child_sample']}\n"
            candidates_summary += f"- Parent sample values: {c['parent_sample']}\n\n"

        # Build architecture context
        architecture_context = ""
        if target_table:
            architecture_context = f"""
## Architecture Context

This database follows a **star schema** pattern with `{target_table}` as the central table:
- All dimension tables should have foreign keys pointing to `{target_table}`
- Prefer relationships to the target table over intermediate tables
- If a column could reference multiple tables with the same primary key, choose the target table
"""

        prompt = f"""You are a database schema expert analyzing potential foreign key relationships.

{schema_summary}

{candidates_summary}
{architecture_context}

## Task

Analyze each candidate relationship and determine if it represents a valid foreign key.
Consider:
1. **Star schema architecture**: Prioritize relationships to the target table (`{target_table if target_table else 'N/A'}`)
2. Column name semantics (e.g., similar names across tables)
3. Data type compatibility
4. Value overlap coverage (higher is better)
5. Cardinality patterns (many-to-one relationships)
6. Domain knowledge (e.g., "ANumber" likely means assignment/accession number)

**IMPORTANT**: When multiple candidates point to tables with the same primary key values,
prefer the relationship to the target table (marked with ⭐).

## Output Format

For each candidate that IS a valid foreign key, output ONLY the candidate number (e.g., "1", "2", "3").
Put each number on a separate line.
If a candidate is NOT a valid FK, do not include its number.

Example output:
```
1
3
5
```

Begin your analysis:"""

        return prompt

    def _parse_agent_response(
        self, response_text: str, candidates: List[Dict]
    ) -> List[ForeignKey]:
        """Parse agent response and create ForeignKey objects.

        Args:
            response_text: Agent response text
            candidates: Original candidates list

        Returns:
            List of ForeignKey objects
        """
        import re

        # Extract candidate numbers from response
        # Look for lines with just numbers
        lines = response_text.strip().split("\n")
        selected_indices = []

        for line in lines:
            line = line.strip()
            # Match lines that are just numbers (possibly in code blocks)
            if re.match(r"^\d+$", line):
                selected_indices.append(int(line))

        # Create ForeignKey objects
        fks = []
        for idx in selected_indices:
            if 1 <= idx <= len(candidates):
                c = candidates[idx - 1]
                fks.append(
                    ForeignKey(
                        child_table=c["child_table"],
                        child_column=c["child_column"],
                        parent_table=c["parent_table"],
                        parent_column=c["parent_column"],
                        coverage=c["coverage"],
                    )
                )
            else:
                logger.warning(f"Agent returned invalid candidate index: {idx}")

        return fks

    def _deduplicate_fks(
        self,
        fks: List[ForeignKey],
        table_metadata: Dict[str, TableMetadata],
        target_table: str,
    ) -> List[ForeignKey]:
        """Deduplicate foreign keys and resolve conflicts.

        When a child column points to multiple parent columns with similar values,
        keep only the best relationship based on:
        1. Prefer relationships to the target table
        2. Prefer higher coverage
        3. Prefer parent table with matching primary key

        Args:
            fks: List of detected foreign keys
            table_metadata: Dict of TableMetadata
            target_table: Name of the target table (for priority)

        Returns:
            Deduplicated list of ForeignKey objects
        """
        if not fks:
            return fks

        # Group FKs by (child_table, child_column)
        fk_groups = {}
        for fk in fks:
            key = (fk.child_table, fk.child_column)
            if key not in fk_groups:
                fk_groups[key] = []
            fk_groups[key].append(fk)

        deduplicated = []

        for (child_table, child_column), group in fk_groups.items():
            if len(group) == 1:
                # Only one FK for this column, keep it
                deduplicated.append(group[0])
                continue

            # Multiple FKs for the same child column, need to choose
            logger.info(
                f"Found {len(group)} FK candidates for {child_table}.{child_column}, deduplicating..."
            )

            # Sort by priority:
            # 1. Target table first
            # 2. Higher coverage
            # 3. Parent table name (for determinism)
            def fk_priority(fk: ForeignKey) -> tuple:
                is_target = 1 if fk.parent_table == target_table else 0
                return (is_target, fk.coverage, fk.parent_table)

            group_sorted = sorted(group, key=fk_priority, reverse=True)
            best_fk = group_sorted[0]

            # Check if this looks like the same relationship pointing to different tables
            # (e.g., ANumber in abstracts vs wamex_reports)
            parent_columns = {fk.parent_column for fk in group}
            parent_tables = {fk.parent_table for fk in group}

            if len(parent_columns) == 1 and len(parent_tables) > 1:
                # Same column name in different parent tables
                # This might be a case where both parent tables represent the same entity
                logger.info(
                    f"  Detected redundant FK: {child_table}.{child_column} -> "
                    f"{', '.join(sorted(parent_tables))}.{list(parent_columns)[0]}"
                )
                logger.info(f"  Keeping only: {best_fk}")

            deduplicated.append(best_fk)

        logger.info(f"Deduplication: {len(fks)} -> {len(deduplicated)} FKs")
        return deduplicated
