"""Base classes for Text2SQL retrievers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from talk2metadata.agent.factory import LLMProviderFactory
from talk2metadata.core.modes.registry import BaseRetriever
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Text2SQLSearchResult:
    """Search result from text2sql mode.

    Attributes:
        rank: Rank of the result
        table: Table name (or "multiple" if query spans multiple tables)
        data: Result data (list of dicts representing rows)
        sql_query: The SQL query that was executed
        row_count: Number of rows returned
        score: Confidence score (always 1.0 for SQL results)
    """

    rank: int
    table: str
    data: List[Dict[str, Any]]
    sql_query: str
    row_count: int
    score: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Text2SQLSearchResult(rank={self.rank}, table={self.table}, "
            f"rows={self.row_count}, sql={self.sql_query[:50]}...)"
        )


class BaseText2SQLRetriever(BaseRetriever):
    """Base class for text2sql retrievers."""

    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        connection_string: Optional[str] = None,
        engine: Optional[Engine] = None,
        **kwargs: Any,
    ):
        """Initialize text2sql retriever.

        Args:
            schema_metadata: Schema metadata with table structures
            connection_string: Database connection string (SQLAlchemy format)
            engine: Optional pre-created SQLAlchemy engine
            **kwargs: Additional configuration
        """
        self.schema_metadata = schema_metadata
        self.target_table = schema_metadata.target_table

        # Get database connection
        config = get_config()
        if engine:
            self.engine = engine
            self._own_engine = False
        elif connection_string:
            self.engine = create_engine(connection_string)
            self._own_engine = True
        else:
            # Try to get from config
            ingest_config = config.get("ingest", {})
            source_path = ingest_config.get("source_path")
            data_type = ingest_config.get("data_type", "csv")

            if data_type in ("database", "db") and source_path:
                logger.info(f"Using connection string from config: {source_path}")
                self.engine = create_engine(source_path)
                self._own_engine = True
            else:
                raise ValueError(
                    "Either connection_string, engine, or database config must be provided"
                )

        # Initialize LLM provider
        # Use same config resolution logic as AgentWrapper
        agent_config = config.get("agent", {})
        provider = agent_config.get("provider", "openai")
        model = agent_config.get("model") or agent_config.get(provider, {}).get("model")

        # Build provider kwargs (merge config, provider-specific config, and keys)
        import os

        agent_kwargs = agent_config.get("config", {}).copy()

        # Merge provider-specific config
        provider_config = agent_config.get(provider, {})
        for key, value in provider_config.items():
            if key != "model":  # model handled separately
                agent_kwargs[key] = value

        # Merge API keys from keys section
        keys_config = agent_config.get("keys", {})
        if f"{provider}_api_key" in keys_config:
            agent_kwargs["api_key"] = keys_config[f"{provider}_api_key"]

        # Expand environment variables in string values
        for key, value in agent_kwargs.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value:
                    agent_kwargs[key] = env_value

        logger.info(f"Initializing LLM provider: {provider}, model: {model}")
        self.llm = LLMProviderFactory.create_provider(
            provider=provider, model=model, **agent_kwargs
        )

        # Load context from config (if available)
        # Try to get context from mode-specific config first, then fallback to general text2sql config
        mode_config = kwargs.get("mode_name", "text2sql")
        mode_retriever_config = config.get(f"modes.{mode_config}.retriever", {})
        self.context = mode_retriever_config.get("context", "").strip()
        if not self.context:
            # Fallback to general text2sql config
            general_config = config.get("modes.text2sql.retriever", {})
            self.context = general_config.get("context", "").strip()

        if self.context:
            logger.info(f"Loaded context for text2sql ({len(self.context)} chars)")

    def _format_schema_for_prompt(self) -> str:
        """Format schema metadata for LLM prompt.

        Returns:
            Formatted schema string
        """
        parts = ["# Database Schema\n"]
        parts.append(
            "IMPORTANT: Use EXACT table and column names as shown below. Case sensitivity matters!\n"
        )

        # Add table information with emphasis on exact names
        for table_name, table_meta in self.schema_metadata.tables.items():
            parts.append(f"## Table: {table_name}")
            if table_name == self.target_table:
                parts.append(
                    "  ⭐ THIS IS THE TARGET TABLE (use this as the main table)"
                )
            if table_meta.row_count > 0:
                parts.append(f"  Row count: {table_meta.row_count}")
            if table_meta.primary_key:
                parts.append(f"  Primary Key: {table_meta.primary_key}")

            # Format columns with data types - emphasize exact names
            if isinstance(table_meta.columns, dict):
                # columns is a dict: {column_name: dtype}
                column_list = []
                for col_name, dtype in table_meta.columns.items():
                    column_list.append(f"{col_name} ({dtype})")
                parts.append(f"  Columns: {', '.join(column_list)}")
            else:
                # Fallback: columns is a list
                parts.append(f"  Columns: {', '.join(table_meta.columns)}")

            # Include more sample values for better understanding
            if table_meta.sample_values:
                parts.append(
                    "  Sample values (use EXACT format including spaces and case):"
                )
                # Show more columns (up to 5) with more samples
                for col, vals in list(table_meta.sample_values.items())[:5]:
                    # Truncate long values but show more samples
                    sample_strs = []
                    for v in vals[:5]:
                        val_str = str(v)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        sample_strs.append(f"'{val_str}'")
                    sample = ", ".join(sample_strs)
                    parts.append(f"    {col}: {sample}")
            parts.append("")

        # Add foreign key relationships with more context
        if self.schema_metadata.foreign_keys:
            parts.append("## Foreign Key Relationships\n")
            parts.append("Use these EXACT relationships to JOIN tables:\n")
            for fk in self.schema_metadata.foreign_keys:
                parts.append(
                    f"  {fk.child_table}.{fk.child_column} = "
                    f"{fk.parent_table}.{fk.parent_column}"
                )
            parts.append("")

        # Add user-provided context if available
        if hasattr(self, "context") and self.context:
            parts.append("# Additional Context\n")
            parts.append(self.context)
            parts.append("")

        return "\n".join(parts)

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response.

        Args:
            response: LLM response text

        Returns:
            Extracted SQL query
        """
        # Try to extract SQL from code blocks
        sql_match = re.search(r"```(?:sql)?\s*\n(.*?)\n```", response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # Try to find SELECT statement (more flexible pattern)
        # Match SELECT ... until semicolon or end of string
        select_match = re.search(
            r"(SELECT.*?)(?:;|$)", response, re.DOTALL | re.IGNORECASE
        )
        if select_match:
            return select_match.group(1).strip()

        # If no SQL found, return the response as-is (might be plain SQL)
        return response.strip()

    def _get_target_id_column(self) -> Optional[str]:
        """Get the ID column name for the target table.

        Returns:
            Primary key column name, or None if not found
        """
        if self.target_table not in self.schema_metadata.tables:
            return None

        table_meta = self.schema_metadata.tables[self.target_table]
        # Try primary key first
        if table_meta.primary_key:
            return table_meta.primary_key

        # Fallback: look for common ID column names
        common_id_names = ["id", "Id", "ID", "row_id", "RowId", "ROW_ID"]
        for col_name in common_id_names:
            if col_name in table_meta.columns:
                return col_name

        return None

    def _ensure_id_column_in_select(self, sql_query: str) -> str:
        """Ensure target table's ID column is included in SELECT clause if target table is involved.

        Args:
            sql_query: SQL query string

        Returns:
            Modified SQL query with ID column if needed
        """
        id_column = self._get_target_id_column()
        if not id_column:
            # No ID column found, return as-is
            return sql_query

        sql_upper = sql_query.upper()

        # Check if target table is in the query
        target_table_pattern = rf"\b{re.escape(self.target_table.upper())}\b"
        if not re.search(target_table_pattern, sql_upper):
            return sql_query

        # Check if ID column is already in SELECT
        # Match SELECT ... FROM pattern (more flexible)
        select_match = re.search(
            r"(SELECT\s+)(.*?)(\s+FROM)", sql_query, re.DOTALL | re.IGNORECASE
        )
        if not select_match:
            return sql_query

        select_clause = select_match.group(2).strip()
        # Check if ID column (or table.ID_column) is already selected
        id_pattern = (
            r"\b(?:"
            + re.escape(id_column)
            + r"|"
            + re.escape(self.target_table)
            + r"\."
            + re.escape(id_column)
            + r")\b"
        )
        if re.search(id_pattern, select_clause, re.IGNORECASE):
            return sql_query

        # Add ID column to SELECT clause
        # Handle different cases: SELECT * vs SELECT col1, col2
        if "*" in select_clause:
            # Replace * with specific columns including ID column
            # Handle both SELECT * and SELECT table.*
            if f"{self.target_table}.*" in select_clause:
                new_select = select_clause.replace(
                    f"{self.target_table}.*",
                    f"{self.target_table}.{id_column}, {self.target_table}.*",
                )
            elif "*" in select_clause:
                new_select = select_clause.replace(
                    "*", f"{self.target_table}.{id_column}, *"
                )
            else:
                new_select = f"{self.target_table}.{id_column}, {select_clause}"
        else:
            # Add ID column at the beginning
            new_select = f"{self.target_table}.{id_column}, {select_clause}"

        # Reconstruct the query
        return (
            sql_query[: select_match.start(2)]
            + new_select
            + sql_query[select_match.end(2) :]
        )

    def _validate_and_fix_sql(
        self, sql_query: str, query: str, max_retries: int = 2
    ) -> str:
        """Validate SQL query and attempt to fix common issues.

        Args:
            sql_query: SQL query to validate
            query: Original natural language query
            max_retries: Maximum number of retry attempts

        Returns:
            Fixed SQL query
        """
        # Try to execute and catch errors
        for attempt in range(max_retries + 1):
            try:
                # Test execute (with LIMIT 0 to avoid fetching data)
                test_query = sql_query.rstrip(";")
                if "LIMIT" not in test_query.upper():
                    test_query += " LIMIT 0"
                else:
                    # Replace existing LIMIT with 0
                    test_query = re.sub(
                        r"LIMIT\s+\d+", "LIMIT 0", test_query, flags=re.IGNORECASE
                    )

                with self.engine.connect() as conn:
                    conn.execute(text(test_query))

                # If successful, return the query
                return sql_query

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    # Try to fix common issues
                    sql_query = self._attempt_sql_fix(sql_query, str(e), query)
                else:
                    logger.error(
                        f"SQL validation failed after {max_retries + 1} attempts: {e}"
                    )
                    # Return original query, let execution handle the error
                    return sql_query

        return sql_query

    def _validate_table_column_names(self, sql_query: str) -> tuple[bool, list[str]]:
        """Validate that all table and column names in SQL exist in schema.

        Args:
            sql_query: SQL query to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        sql_upper = sql_query.upper()

        # Extract table names from SQL (FROM, JOIN clauses)
        table_pattern = r"(?:FROM|JOIN)\s+(\w+)"
        mentioned_tables = set(re.findall(table_pattern, sql_upper))

        # Check if mentioned tables exist in schema
        schema_tables_upper = {t.upper(): t for t in self.schema_metadata.tables.keys()}
        for table_upper in mentioned_tables:
            if table_upper not in schema_tables_upper:
                # Try to find similar table name
                similar = [
                    t
                    for t in schema_tables_upper.keys()
                    if table_upper in t or t in table_upper
                ]
                if similar:
                    issues.append(
                        f"Table '{table_upper}' not found. Did you mean '{schema_tables_upper[similar[0]]}'?"
                    )
                else:
                    issues.append(f"Table '{table_upper}' not found in schema")

        # Extract column references (table.column or just column)
        # This is a simplified check - full parsing would be more complex
        column_pattern = r"(\w+)\.(\w+)"
        column_refs = re.findall(column_pattern, sql_query)

        for table_ref, col_ref in column_refs:
            table_upper = table_ref.upper()
            if table_upper in schema_tables_upper:
                actual_table = schema_tables_upper[table_upper]
                table_meta = self.schema_metadata.tables[actual_table]
                columns_upper = {c.upper(): c for c in table_meta.columns.keys()}
                col_upper = col_ref.upper()
                if col_upper not in columns_upper:
                    similar = [
                        c
                        for c in columns_upper.keys()
                        if col_upper in c or c in col_upper
                    ]
                    if similar:
                        issues.append(
                            f"Column '{table_ref}.{col_ref}' not found. Did you mean '{actual_table}.{columns_upper[similar[0]]}'?"
                        )
                    else:
                        issues.append(
                            f"Column '{table_ref}.{col_ref}' not found in table '{actual_table}'"
                        )

        return len(issues) == 0, issues

    def _attempt_sql_fix(
        self, sql_query: str, error_msg: str, original_query: str
    ) -> str:
        """Attempt to fix SQL query based on error message.

        Args:
            sql_query: SQL query with error
            error_msg: Error message from database
            original_query: Original natural language query

        Returns:
            Potentially fixed SQL query
        """
        error_lower = error_msg.lower()
        fixed_query = sql_query

        # Common fixes
        # 1. Missing table qualifier for ambiguous columns
        if "ambiguous" in error_lower or "ambiguous column" in error_lower:
            # Try to add table qualifiers - this is complex, might need LLM help
            logger.debug("Ambiguous column detected, may need table qualifiers")

        # 2. Invalid column name - log for debugging
        if "no such column" in error_lower or "invalid column" in error_lower:
            logger.debug(f"Invalid column detected: {error_msg}")
            # Note: Specific fixes should be handled via context in config.yml
            # or through schema validation which provides suggestions

        # 3. Syntax errors - try basic fixes
        if "syntax error" in error_lower:
            # Remove trailing semicolons if present multiple times
            fixed_query = fixed_query.rstrip(";")
            # Ensure proper spacing
            fixed_query = re.sub(r"\s+", " ", fixed_query)

        # 4. Validate table/column names and suggest fixes
        is_valid, issues = self._validate_table_column_names(fixed_query)
        if not is_valid:
            logger.debug(f"Schema validation issues: {issues}")

        return fixed_query

    def _execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results.

        Args:
            sql_query: SQL query string

        Returns:
            DataFrame with query results

        Raises:
            Exception: If SQL execution fails
        """
        logger.debug(f"Executing SQL: {sql_query[:100]}...")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = result.keys()

                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                logger.debug(f"Query returned {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

    def _convert_dataframe_to_results(
        self, df: pd.DataFrame, sql_query: str, rank: int = 1
    ) -> Text2SQLSearchResult:
        """Convert DataFrame to Text2SQLSearchResult.

        Args:
            df: DataFrame with query results
            sql_query: The SQL query that was executed
            rank: Rank of the result

        Returns:
            Text2SQLSearchResult object
        """
        # Determine table name (use target table or "multiple" if joins)
        table_name = self.target_table
        if "JOIN" in sql_query.upper() or "FROM" in sql_query.upper():
            # Try to detect if multiple tables are involved
            from_match = re.search(r"FROM\s+(\w+)", sql_query, re.IGNORECASE)
            if from_match:
                table_name = from_match.group(1)

        # Convert DataFrame to list of dicts
        data = df.to_dict("records")

        return Text2SQLSearchResult(
            rank=rank,
            table=table_name,
            data=data,
            sql_query=sql_query,
            row_count=len(data),
            score=1.0,
        )

    def close(self):
        """Close database connection."""
        if self._own_engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
