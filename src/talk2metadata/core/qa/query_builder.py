"""Query builder for generating SQL queries based on difficulty strategies.

This module generates SQL queries by:
1. Randomly selecting tables and columns based on the strategy
2. Generating appropriate filter conditions
3. Building JOIN statements for path/intersection patterns
4. Executing the query to get answer record IDs
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from talk2metadata.core.qa.difficulty_classifier import (
    DifficultyClassifier,
    JoinPath,
    QueryPlan,
)
from talk2metadata.core.schema import ForeignKey, SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Filter:
    """Represents a filter condition."""

    table: str
    column: str
    operator: str  # '=', '>', '<', '>=', '<=', 'LIKE', 'IN'
    value: Any  # The filter value

    def to_sql(self) -> str:
        """Convert filter to SQL condition."""
        table_lower = self.table.lower()
        column_lower = self.column.lower()
        if isinstance(self.value, str):
            return f"{table_lower}.{column_lower} {self.operator} '{self.value}'"
        else:
            return f"{table_lower}.{column_lower} {self.operator} {self.value}"


@dataclass
class QuerySpec:
    """Specification for a generated query."""

    strategy: str  # Difficulty code (e.g., "2iM")
    target_table: str
    join_paths: List[JoinPath]
    filters: List[Filter]
    sql: str
    involved_tables: List[str]
    involved_columns: List[str]  # table.column format
    answer_row_ids: List[Any]  # Answer record IDs from query execution


class QueryBuilder:
    """Builds SQL queries based on difficulty strategies."""

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        engine: Optional[Engine] = None,
        connection_string: Optional[str] = None,
    ):
        """Initialize query builder.

        Args:
            schema: Schema metadata
            tables: Dictionary mapping table names to DataFrames
            engine: Optional SQLAlchemy engine for SQL validation
            connection_string: Optional database connection string for SQL validation
        """
        self.schema = schema
        self.target_table = schema.target_table
        self.classifier = DifficultyClassifier()

        # Get primary key for target table
        self.target_pk = schema.tables[self.target_table].primary_key

        # Normalize tables: convert table names and column names to lowercase
        # to match the schema metadata (which was normalized during database import)
        self.tables = {}
        for table_name, df in tables.items():
            table_name_lower = table_name.lower()
            df_normalized = df.copy()
            # Convert column names to lowercase
            df_normalized.columns = [col.lower() for col in df_normalized.columns]
            self.tables[table_name_lower] = df_normalized

        logger.debug(
            f"Normalized {len(self.tables)} tables to lowercase: {list(self.tables.keys())}"
        )

        # Set up database engine for SQL validation
        self.engine = engine
        if connection_string and not engine:
            self.engine = create_engine(connection_string)

    def build_query(self, strategy: str, max_attempts: int = 10) -> Optional[QuerySpec]:
        """Build a query based on the difficulty strategy.

        Args:
            strategy: Difficulty code (e.g., "2iM")
            max_attempts: Maximum number of attempts to generate a valid query

        Returns:
            QuerySpec object, or None if generation failed
        """
        failure_reasons = []
        no_result_count = 0
        validation_failed_count = 0
        last_attempt_info = None  # Store info from last attempt for analysis

        for attempt in range(max_attempts):
            try:
                # Parse strategy
                pattern, difficulty = self._parse_strategy(strategy)

                # Generate JOIN structure
                join_paths = self._generate_join_structure(pattern)

                # Generate filters
                filters = self._generate_filters(difficulty, join_paths)

                # Build SQL
                sql = self._build_sql(join_paths, filters)

                # Execute query to get answer IDs
                answer_row_ids = self._execute_query_from_spec(join_paths, filters)

                # Check if we got valid results
                if not answer_row_ids or len(answer_row_ids) == 0:
                    no_result_count += 1
                    # Store info from this attempt for later analysis
                    involved_tables = self._get_involved_tables(join_paths)
                    last_attempt_info = {
                        "pattern": pattern,
                        "difficulty": difficulty,
                        "join_paths": join_paths,
                        "filters": filters,
                        "involved_tables": involved_tables,
                        "num_filters": len(filters),
                    }
                    logger.debug("Query returned no results, retrying...")
                    continue

                # Get involved tables and columns
                involved_tables = self._get_involved_tables(join_paths)
                involved_columns = [f"{f.table}.{f.column}" for f in filters]

                query_spec = QuerySpec(
                    strategy=strategy,
                    target_table=self.target_table,
                    join_paths=join_paths,
                    filters=filters,
                    sql=sql,
                    involved_tables=involved_tables,
                    involved_columns=involved_columns,
                    answer_row_ids=answer_row_ids,
                )

                # Validate the generated query
                if self._validate_query(query_spec):
                    logger.debug(
                        f"Successfully generated query for {strategy} with {len(answer_row_ids)} results"
                    )
                    return query_spec
                else:
                    validation_failed_count += 1

            except ValueError as e:
                # These are expected errors (e.g., cannot build chain, no foreign keys)
                error_msg = str(e)
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                continue
            except Exception as e:
                # Unexpected errors
                error_msg = f"Unexpected error: {str(e)}"
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                continue

        # Build failure message with reasons in plain language
        reason_parts = []
        if failure_reasons:
            # Show unique failure reasons (limit to most common)
            # Prefer errors that contain detailed explanations
            detailed_reasons = [
                r for r in failure_reasons if " - " in r or "because:" in r
            ]
            if detailed_reasons:
                # Use the most detailed reason, clean up "because:" to " - "
                most_detailed = detailed_reasons[0].replace("because:", "-")
                reason_parts.append(most_detailed)
            else:
                unique_reasons = list(set(failure_reasons))[:2]
                reason_parts.append(f"issues: {', '.join(unique_reasons)}")

        if no_result_count > 0:
            # Provide professional analysis for empty results
            analysis = self._analyze_empty_result_reason(
                strategy, last_attempt_info, no_result_count
            )
            reason_parts.append(analysis)

        if validation_failed_count > 0:
            reason_parts.append(
                f"{validation_failed_count} attempts failed validation checks"
            )

        reason_str = f" ({'; '.join(reason_parts)})" if reason_parts else ""

        logger.warning(
            f"Unable to generate query for strategy {strategy} after {max_attempts} attempts{reason_str}"
        )
        return None

    def _analyze_empty_result_reason(
        self, strategy: str, last_attempt_info: Optional[Dict], no_result_count: int
    ) -> str:
        """Analyze why queries returned empty results and provide professional error message.

        Args:
            strategy: Strategy code
            last_attempt_info: Information from last attempt (if available)
            no_result_count: Number of attempts that returned no results

        Returns:
            Professional error message explaining the failure
        """
        if not last_attempt_info:
            return f"all {no_result_count} attempts returned empty result sets"

        pattern = last_attempt_info.get("pattern", "")
        num_filters = last_attempt_info.get("num_filters", 0)
        involved_tables = last_attempt_info.get("involved_tables", [])

        # Analyze the pattern in plain language
        if pattern[-1] == "i":
            # Intersection pattern
            branches = int(pattern[0]) if pattern[0].isdigit() else 0
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
                f"query requires connecting {branches} different tables to find matching records",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"but no records satisfy all {num_filters} filter conditions at the same time"
                )
            else:
                analysis_parts.append(
                    "but the table connections don't match any records in the data"
                )
        elif pattern[-1] == "p":
            # Path pattern
            hops = int(pattern[0]) if pattern[0].isdigit() else 0
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
                f"query requires following a {hops}-step path through {len(involved_tables)} connected tables",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"but no records satisfy all {num_filters} filter conditions along this path"
                )
            else:
                analysis_parts.append(
                    "but this connection path doesn't match any records in the data"
                )
        else:
            # Direct query
            analysis_parts = [
                f"all {no_result_count} attempts returned no results",
            ]
            if num_filters > 0:
                analysis_parts.append(
                    f"no records satisfy all {num_filters} filter conditions"
                )

        return "; ".join(analysis_parts)

    def _parse_strategy(self, strategy: str) -> Tuple[str, str]:
        """Parse strategy into pattern and difficulty.

        Args:
            strategy: Difficulty code (e.g., "2iM")

        Returns:
            Tuple of (pattern, difficulty) (e.g., ("2i", "M"))
        """
        if strategy[0].isdigit():
            if len(strategy) >= 2 and strategy[1] in ["p", "i"]:
                return strategy[:2], strategy[2:]
            else:
                return strategy[0], strategy[1:]
        else:
            return strategy[:2], strategy[2:]

    def _generate_join_structure(self, pattern: str) -> List[JoinPath]:
        """Generate JOIN structure based on pattern.

        Args:
            pattern: Pattern code (e.g., "2i", "1p")

        Returns:
            List of JoinPath objects
        """
        if pattern == "0":
            # No JOINs
            return []

        # Extract number and type
        if pattern[-1] == "p":
            # Path pattern (chain)
            hops = int(pattern[0])
            return self._generate_chain_joins(hops)
        elif pattern[-1] == "i":
            # Intersection pattern (star)
            branches = int(pattern[0])
            return self._generate_star_joins(branches)
        else:
            # Mixed or other patterns - not implemented yet
            raise NotImplementedError(f"Pattern {pattern} not yet implemented")

    def _generate_chain_joins(self, hops: int) -> List[JoinPath]:
        """Generate chain JOIN paths.

        Args:
            hops: Number of hops (JOINs)

        Returns:
            List of JoinPath objects representing the chain

        Raises:
            ValueError: If chain cannot be generated with detailed reason
        """
        # Try multiple attempts to build a valid chain
        max_attempts = 5
        failure_reasons = []
        no_fk_count = 0
        cycle_count = 0

        for attempt in range(max_attempts):
            try:
                # Build a chain from target table
                current_table = self.target_table
                path_tables = [current_table]

                for hop in range(hops):
                    # Get foreign keys from current table
                    fks = self.schema.get_foreign_keys_for_table(
                        current_table, direction="outgoing"
                    )

                    if not fks:
                        # No outgoing FKs, try incoming
                        fks = self.schema.get_foreign_keys_for_table(
                            current_table, direction="incoming"
                        )

                    if not fks:
                        no_fk_count += 1
                        raise ValueError(
                            f"No foreign keys found for table {current_table} "
                            f"(at hop {hop + 1}/{hops})"
                        )

                    # Filter out FKs that would create cycles
                    valid_fks = []
                    for fk in fks:
                        next_table = (
                            fk.parent_table
                            if fk.child_table == current_table
                            else fk.child_table
                        )
                        if next_table not in path_tables:
                            valid_fks.append(fk)

                    if not valid_fks:
                        cycle_count += 1
                        raise ValueError(
                            f"No valid FKs to continue chain from {current_table} "
                            f"(all would create cycles, at hop {hop + 1}/{hops})"
                        )

                    # Randomly select one valid FK
                    fk = random.choice(valid_fks)

                    # Add next table to path
                    if fk.child_table == current_table:
                        next_table = fk.parent_table
                    else:
                        next_table = fk.child_table

                    path_tables.append(next_table)
                    current_table = next_table

                # Successfully built a chain
                return [JoinPath(tables=path_tables, join_type="chain")]

            except ValueError as e:
                error_msg = str(e)
                if error_msg not in failure_reasons:
                    failure_reasons.append(error_msg)
                logger.debug(f"Chain generation attempt {attempt + 1} failed: {e}")
                continue

        # Build detailed error message in plain language
        reason_parts = []
        if no_fk_count > 0:
            reason_parts.append(
                f"cannot find enough table relationships to build a {hops}-hop chain "
                f"(tried {no_fk_count} times but no valid connections found)"
            )
        elif cycle_count > 0:
            reason_parts.append(
                f"all possible table connection paths would create circular references "
                f"(tried {cycle_count} different paths, all would loop back to previous tables)"
            )
        elif failure_reasons:
            # Show the most common failure reason
            most_common = max(set(failure_reasons), key=failure_reasons.count)
            if "No foreign keys" in most_common:
                reason_parts.append(
                    f"cannot find enough table relationships to build a {hops}-hop chain"
                )
            elif "cycles" in most_common:
                reason_parts.append(
                    "all possible table connection paths would create circular references"
                )
            else:
                reason_parts.append(most_common)

        reason_str = f" - {', '.join(reason_parts)}" if reason_parts else ""

        raise ValueError(f"Unable to generate {hops}-hop chain query{reason_str}")

    def _generate_star_joins(self, branches: int) -> List[JoinPath]:
        """Generate star JOIN paths (intersection pattern).

        Args:
            branches: Number of branches (JOINs from target table)

        Returns:
            List of JoinPath objects, each representing one branch
        """
        # Get all foreign keys involving the target table
        fks = self.schema.get_foreign_keys_for_table(
            self.target_table, direction="both"
        )

        if len(fks) < branches:
            raise ValueError(
                f"Cannot generate {branches}-way intersection because target table "
                f"'{self.target_table}' has only {len(fks)} foreign key relationship(s), "
                f"but {branches} are required"
            )

        # Randomly select branches
        selected_fks = random.sample(fks, branches)

        join_paths = []
        for fk in selected_fks:
            # Determine the related table
            if fk.child_table == self.target_table:
                related_table = fk.parent_table
            else:
                related_table = fk.child_table

            # Each branch is a 2-table path
            join_paths.append(
                JoinPath(tables=[self.target_table, related_table], join_type="star")
            )

        return join_paths

    def _generate_filters(
        self, difficulty: str, join_paths: List[JoinPath]
    ) -> List[Filter]:
        """Generate filter conditions based on difficulty level.

        Args:
            difficulty: Difficulty level (E/M/H)
            join_paths: List of JOIN paths

        Returns:
            List of Filter objects
        """
        # Determine number of filter columns based on difficulty
        if difficulty == "E":
            min_cols, max_cols = 1, 2
        elif difficulty == "M":
            min_cols, max_cols = 3, 5
        elif difficulty == "H":
            min_cols, max_cols = 6, 8
        else:
            min_cols, max_cols = 1, 2

        num_filters = random.randint(min_cols, max_cols)

        # Get all involved tables
        involved_tables = self._get_involved_tables(join_paths)
        if not involved_tables:
            involved_tables = [self.target_table]

        # Generate filters
        filters = []
        used_columns: Set[str] = set()

        attempts = 0
        max_attempts = num_filters * 3  # Allow multiple attempts

        while len(filters) < num_filters and attempts < max_attempts:
            attempts += 1

            # Randomly select a table
            table = random.choice(involved_tables)

            # Get available columns (exclude primary keys and already used columns)
            table_meta = self.schema.tables[table]
            available_cols = [
                col
                for col in table_meta.columns
                if col != table_meta.primary_key
                and f"{table}.{col}" not in used_columns
            ]

            if not available_cols:
                continue

            # Randomly select a column
            column = random.choice(available_cols)
            used_columns.add(f"{table}.{column}")

            # Generate a filter condition
            filter_obj = self._generate_filter_condition(table, column)
            if filter_obj:
                filters.append(filter_obj)

        return filters

    def _generate_filter_condition(self, table: str, column: str) -> Optional[Filter]:
        """Generate a filter condition for a specific column.

        Args:
            table: Table name
            column: Column name

        Returns:
            Filter object, or None if generation failed
        """
        try:
            df = self.tables[table]
            values = df[column].dropna()

            if len(values) == 0:
                return None

            # Randomly select a value
            value = random.choice(values.tolist())

            # Determine operator based on column type
            dtype = values.dtype

            if pd.api.types.is_numeric_dtype(dtype):
                # Numeric column - use =, >, <, >=, <=
                operator = random.choice(["=", ">", "<", ">=", "<="])
            else:
                # Non-numeric column - use =
                operator = "="

            return Filter(table=table, column=column, operator=operator, value=value)

        except Exception as e:
            logger.debug(f"Failed to generate filter for {table}.{column}: {e}")
            return None

    def _get_involved_tables(self, join_paths: List[JoinPath]) -> List[str]:
        """Get all involved tables from JOIN paths.

        Args:
            join_paths: List of JOIN paths

        Returns:
            List of unique table names (always includes target_table)
        """
        tables = {self.target_table}  # Always include target table
        for path in join_paths:
            tables.update(path.tables)
        return list(tables)

    def _build_sql(self, join_paths: List[JoinPath], filters: List[Filter]) -> str:
        """Build SQL query from JOIN paths and filters.

        Args:
            join_paths: List of JOIN paths
            filters: List of filters

        Returns:
            SQL query string (all table and column names in lowercase)
        """
        # Convert table and column names to lowercase
        target_table_lower = self.target_table.lower()
        target_pk_lower = self.target_pk.lower()

        # SELECT clause - select primary key from target table
        sql = f"SELECT {target_table_lower}.{target_pk_lower} FROM {target_table_lower}"

        # JOIN clauses
        if join_paths:
            joined_tables = {target_table_lower}
            for path in join_paths:
                for i in range(len(path.tables) - 1):
                    from_table = path.tables[i].lower()
                    to_table = path.tables[i + 1].lower()

                    if to_table in joined_tables:
                        continue

                    # Find the foreign key relationship (using original case for lookup)
                    fk = self._find_foreign_key(path.tables[i], path.tables[i + 1])
                    if fk:
                        child_col_lower = fk.child_column.lower()
                        parent_col_lower = fk.parent_column.lower()
                        if fk.child_table.lower() == from_table:
                            sql += f"\nJOIN {to_table} ON {from_table}.{child_col_lower} = {to_table}.{parent_col_lower}"
                        else:
                            sql += f"\nJOIN {to_table} ON {from_table}.{parent_col_lower} = {to_table}.{child_col_lower}"
                        joined_tables.add(to_table)

        # WHERE clause
        if filters:
            where_conditions = [f.to_sql().lower() for f in filters]
            sql += "\nWHERE " + " AND ".join(where_conditions)

        return sql

    def _find_foreign_key(self, table1: str, table2: str) -> Optional[ForeignKey]:
        """Find foreign key relationship between two tables.

        Args:
            table1: First table name (can be original case or lowercase)
            table2: Second table name (can be original case or lowercase)

        Returns:
            ForeignKey object, or None if not found
        """
        # Normalize to lowercase for comparison
        table1_lower = table1.lower()
        table2_lower = table2.lower()

        for fk in self.schema.foreign_keys:
            fk_child_lower = fk.child_table.lower()
            fk_parent_lower = fk.parent_table.lower()
            if (fk_child_lower == table1_lower and fk_parent_lower == table2_lower) or (
                fk_child_lower == table2_lower and fk_parent_lower == table1_lower
            ):
                return fk
        return None

    def _execute_query_from_spec(
        self, join_paths: List[JoinPath], filters: List[Filter]
    ) -> List[Any]:
        """Execute query using join paths and filters on pandas DataFrames.

        Args:
            join_paths: List of JOIN paths
            filters: List of filter conditions

        Returns:
            List of target table row IDs
        """
        try:
            # Start with target table (already normalized to lowercase)
            result_df = self.tables[self.target_table].copy()

            # Add suffix to avoid column name conflicts
            result_df = result_df.add_suffix(f"_{self.target_table}")
            result_df = result_df.rename(
                columns={f"{self.target_pk}_{self.target_table}": self.target_pk}
            )

            joined_tables = {self.target_table}

            # Perform JOINs
            for path in join_paths:
                for i in range(len(path.tables) - 1):
                    # Normalize table names to lowercase (schema already uses lowercase)
                    from_table = path.tables[i].lower()
                    to_table = path.tables[i + 1].lower()

                    if to_table in joined_tables:
                        continue

                    # Find FK relationship (using original case for lookup, but FK is already lowercase)
                    fk = self._find_foreign_key(path.tables[i], path.tables[i + 1])
                    if not fk:
                        logger.warning(
                            f"No FK found between {from_table} and {to_table}"
                        )
                        continue

                    # Get the table to join (already normalized to lowercase)
                    join_df = self.tables[to_table].copy()
                    join_df = join_df.add_suffix(f"_{to_table}")

                    # Determine join columns (FK columns are already lowercase)
                    if fk.child_table == from_table:
                        left_on = (
                            f"{fk.child_column}_{from_table}"
                            if from_table != self.target_table
                            else fk.child_column
                        )
                        right_on = f"{fk.parent_column}_{to_table}"
                    else:
                        left_on = (
                            f"{fk.parent_column}_{from_table}"
                            if from_table != self.target_table
                            else fk.parent_column
                        )
                        right_on = f"{fk.child_column}_{to_table}"

                    # Perform the join
                    result_df = result_df.merge(
                        join_df, left_on=left_on, right_on=right_on, how="inner"
                    )

                    joined_tables.add(to_table)

            # Apply filters (filter_obj.table and filter_obj.column are already lowercase)
            for filter_obj in filters:
                # Column names have suffix added, so we need to match that
                if filter_obj.table == self.target_table:
                    # For target table, column already has suffix from add_suffix
                    col_name = f"{filter_obj.column}_{self.target_table}"
                else:
                    # For joined tables, column has suffix from join
                    col_name = f"{filter_obj.column}_{filter_obj.table}"

                # Make sure column exists
                if col_name not in result_df.columns:
                    # Try without suffix (fallback)
                    if filter_obj.column in result_df.columns:
                        col_name = filter_obj.column
                    else:
                        logger.debug(
                            f"Column {col_name} not found in result, skipping filter"
                        )
                        continue

                # Apply filter based on operator
                if filter_obj.operator == "=":
                    result_df = result_df[result_df[col_name] == filter_obj.value]
                elif filter_obj.operator == ">":
                    result_df = result_df[result_df[col_name] > filter_obj.value]
                elif filter_obj.operator == "<":
                    result_df = result_df[result_df[col_name] < filter_obj.value]
                elif filter_obj.operator == ">=":
                    result_df = result_df[result_df[col_name] >= filter_obj.value]
                elif filter_obj.operator == "<=":
                    result_df = result_df[result_df[col_name] <= filter_obj.value]

            # Get primary key values from target table
            if self.target_pk in result_df.columns:
                return result_df[self.target_pk].dropna().unique().tolist()
            else:
                logger.warning(f"Primary key {self.target_pk} not found in result")
                return []

        except Exception as e:
            logger.debug(f"Failed to execute query: {e}")
            return []

    def _validate_query(self, query_spec: QuerySpec) -> bool:
        """Validate that the generated query matches the strategy.

        Args:
            query_spec: QuerySpec object

        Returns:
            True if valid, False otherwise
        """
        # Create QueryPlan for classification
        filter_columns = set(query_spec.involved_columns)

        query_plan = QueryPlan(
            target_table=query_spec.target_table,
            join_paths=query_spec.join_paths,
            filter_columns=filter_columns,
        )

        # Classify the query
        classified_strategy = self.classifier.classify(query_plan)

        # Check if it matches the target strategy
        if classified_strategy != query_spec.strategy:
            logger.debug(
                f"Strategy mismatch: expected {query_spec.strategy}, "
                f"got {classified_strategy}"
            )
            return False

        # Check if we have valid answers
        if not query_spec.answer_row_ids or len(query_spec.answer_row_ids) == 0:
            logger.debug("Query returned no results")
            return False

        return True

    def validate_sql_execution(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """Validate that SQL query can be executed successfully.

        Args:
            sql_query: SQL query string to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if SQL executed successfully, False otherwise
            - error_message: Error message if execution failed, None if successful
        """
        if not self.engine:
            # No engine available, skip validation
            logger.debug("No database engine available for SQL validation")
            return True, None

        try:
            # Normalize SQL to lowercase before validation
            # (since we preprocess all table/column names to lowercase)
            sql_query_lower = sql_query.lower()

            # Try to execute the query with LIMIT 0 to avoid fetching data
            test_query = sql_query_lower.rstrip(";")
            if "limit" not in test_query:
                test_query += " limit 0"
            else:
                # Replace existing LIMIT with 0
                test_query = re.sub(
                    r"limit\s+\d+", "limit 0", test_query, flags=re.IGNORECASE
                )

            with self.engine.connect() as conn:
                conn.execute(text(test_query))
                logger.debug(f"SQL validation successful: {sql_query[:100]}...")
                return True, None

        except Exception as e:
            error_msg = str(e)
            logger.debug(f"SQL validation failed: {error_msg}")
            return False, error_msg
