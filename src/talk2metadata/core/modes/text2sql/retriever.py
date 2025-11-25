"""Text2SQL Retriever - convert natural language to SQL and execute queries."""

from __future__ import annotations

import json
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
from talk2metadata.utils.timing import timed

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

    def _format_schema_for_prompt(self) -> str:
        """Format schema metadata for LLM prompt.

        Returns:
            Formatted schema string
        """
        parts = ["# Database Schema\n"]

        # Add table information
        for table_name, table_meta in self.schema_metadata.tables.items():
            parts.append(f"## Table: {table_name}")
            if table_meta.row_count > 0:
                parts.append(f"Row count: {table_meta.row_count}")
            if table_meta.primary_key:
                parts.append(f"Primary Key: {table_meta.primary_key}")

            # Format columns with data types
            if isinstance(table_meta.columns, dict):
                # columns is a dict: {column_name: dtype}
                column_list = []
                for col_name, dtype in table_meta.columns.items():
                    column_list.append(f"{col_name} ({dtype})")
                parts.append(f"Columns: {', '.join(column_list)}")
            else:
                # Fallback: columns is a list
                parts.append(f"Columns: {', '.join(table_meta.columns)}")

            # Include more sample values for better understanding
            if table_meta.sample_values:
                parts.append("Sample values:")
                # Show more columns (up to 5) with more samples
                for col, vals in list(table_meta.sample_values.items())[:5]:
                    # Truncate long values but show more samples
                    sample_strs = []
                    for v in vals[:5]:
                        val_str = str(v)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        sample_strs.append(val_str)
                    sample = ", ".join(sample_strs)
                    parts.append(f"  {col}: {sample}")
            parts.append("")

        # Add foreign key relationships with more context
        if self.schema_metadata.foreign_keys:
            parts.append("## Foreign Key Relationships\n")
            parts.append("Use these relationships to JOIN tables when needed:\n")
            for fk in self.schema_metadata.foreign_keys:
                parts.append(
                    f"  {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column}"
                )
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

        # Common fixes
        # 1. Missing table qualifier for ambiguous columns
        if "ambiguous" in error_lower or "ambiguous column" in error_lower:
            # Try to add table qualifiers - this is complex, might need LLM help
            logger.debug("Ambiguous column detected, may need table qualifiers")

        # 2. Invalid column name
        if "no such column" in error_lower or "invalid column" in error_lower:
            # Extract column name from error and try to suggest fix
            logger.debug(f"Invalid column detected: {error_msg}")

        # 3. Syntax errors - try basic fixes
        if "syntax error" in error_lower:
            # Remove trailing semicolons if present multiple times
            sql_query = sql_query.rstrip(";")
            # Ensure proper spacing
            sql_query = re.sub(r"\s+", " ", sql_query)

        return sql_query

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


class DirectText2SQLRetriever(BaseText2SQLRetriever):
    """Direct text2sql retriever: Question + schema → SQL → results.

    This approach directly generates SQL from the question and schema information
    without first locating relevant columns/tables.
    """

    @timed("text2sql_direct.search")
    def search(self, query: str, top_k: int = 5) -> List[Text2SQLSearchResult]:
        """Search by converting question to SQL and executing it.

        Args:
            query: Natural language question
            top_k: Maximum number of results (used as LIMIT in SQL)

        Returns:
            List of Text2SQLSearchResult objects
        """
        # Log question
        logger.info("=" * 80)
        logger.info(f"🔍 QUESTION: {query}")
        logger.info("=" * 80)

        # Format schema for prompt
        schema_text = self._format_schema_for_prompt()

        # Create enhanced prompt with few-shot examples
        system_prompt = """You are an expert SQL query generator. Your task is to convert natural language questions into accurate, executable SQL queries.

CRITICAL REQUIREMENTS:
1. ALWAYS include the target table's primary key or ID column in your SELECT clause (e.g., SELECT {target_table}.id, ...)
2. Use proper table qualifiers when columns might be ambiguous (e.g., table_name.column_name)
3. Pay careful attention to text matching - use exact string matching with proper escaping
4. When joining tables, use the foreign key relationships provided
5. Use proper SQL syntax for the database type
6. Limit results to {top_k} rows using LIMIT clause

EXAMPLES:

Example 1:
Question: "Show me reports where the author is 'SMITH J' and target commodity is 'GOLD'"
SQL: SELECT {target_table}.id FROM {target_table} WHERE {target_table}.AuthorNames = 'SMITH J' AND {target_table}.TargetCommoditiesNames = 'GOLD' LIMIT {top_k}

Example 2:
Question: "Find reports that mention 'iron ore' in the abstract"
SQL: SELECT {target_table}.id FROM {target_table} JOIN abstracts ON {target_table}.id = abstracts.report_id WHERE abstracts.Abstract LIKE '%iron ore%' LIMIT {top_k}

Example 3:
Question: "Get reports where ID is greater than 100 and operator ID is '12345'"
SQL: SELECT {target_table}.id FROM {target_table} WHERE {target_table}.Id > 100 AND {target_table}.OperatorIds = '12345' LIMIT {top_k}

IMPORTANT:
- Always SELECT the target table's primary key or ID column along with any other columns needed
- Use exact string matching with single quotes: column = 'value'
- For text search, use LIKE with wildcards: column LIKE '%text%'
- When joining, specify table names: table1.column = table2.column
- Return ONLY the SQL query, no explanations"""

        user_prompt = f"""{schema_text}

Question: {query}

Generate a SQL query to answer this question. Remember to:
1. Include the target table's primary key or ID column in the SELECT clause
2. Use proper table qualifiers for columns
3. Match the question requirements exactly
4. Use LIMIT {top_k} to limit results

SQL Query:"""

        # Generate SQL
        logger.info("💾 Generating SQL query...")
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt.format(
                target_table=self.target_table, top_k=top_k
            ),
            temperature=0.0,
            max_tokens=2048,
        )

        sql_query = self._extract_sql_from_response(response.content)
        logger.info("-" * 80)
        logger.info(f"💾 SQL (initial): {sql_query}")
        logger.info("-" * 80)

        # Validate and fix SQL
        sql_query = self._validate_and_fix_sql(sql_query, query, max_retries=1)

        # Ensure ID column is selected (for evaluation purposes)
        sql_query = self._ensure_id_column_in_select(sql_query)

        # Add LIMIT if not present
        if "LIMIT" not in sql_query.upper():
            sql_query = f"{sql_query.rstrip(';')} LIMIT {top_k}"

        logger.info(f"💾 SQL (final): {sql_query}")
        logger.info("-" * 80)

        # Execute SQL with retry logic
        max_execution_retries = 2
        for attempt in range(max_execution_retries + 1):
            try:
                df = self._execute_sql(sql_query)
                result = self._convert_dataframe_to_results(df, sql_query, rank=1)

                # Log results
                logger.info("📊 RESULTS:")
                logger.info(f"   Returned {len(df)} row(s)")
                if len(df) > 0:
                    logger.info(f"   Columns: {', '.join(df.columns.tolist()[:5])}")
                    if len(df.columns) > 5:
                        logger.info(f"   ... and {len(df.columns) - 5} more columns")
                logger.info("=" * 80)

                return [result]
            except Exception as e:
                if attempt < max_execution_retries:
                    logger.warning(
                        f"SQL execution failed (attempt {attempt + 1}/{max_execution_retries + 1}): {e}"
                    )
                    # Try to fix and regenerate
                    sql_query = self._attempt_sql_fix(sql_query, str(e), query)
                    sql_query = self._ensure_id_column_in_select(sql_query)
                    if "LIMIT" not in sql_query.upper():
                        sql_query = f"{sql_query.rstrip(';')} LIMIT {top_k}"
                    logger.info(f"Retrying with fixed SQL: {sql_query[:200]}...")
                else:
                    logger.error("=" * 80)
                    logger.error(
                        f"❌ ERROR: Failed to execute SQL after {max_execution_retries + 1} attempts: {e}"
                    )
                    logger.error("=" * 80)
                    # Return error result
                    return [
                        Text2SQLSearchResult(
                            rank=1,
                            table=self.target_table,
                            data=[],
                            sql_query=sql_query,
                            row_count=0,
                            score=0.0,
                        )
                    ]


class TwoStepText2SQLRetriever(BaseText2SQLRetriever):
    """Two-step text2sql retriever: Question → locate columns/tables → SQL → results.

    This approach first analyzes the question to identify relevant columns and tables,
    then generates SQL using only the relevant schema information.
    """

    @timed("text2sql_two_step.search")
    def search(self, query: str, top_k: int = 5) -> List[Text2SQLSearchResult]:
        """Search using two-step approach: locate relevant schema, then generate SQL.

        Args:
            query: Natural language question
            top_k: Maximum number of results (used as LIMIT in SQL)

        Returns:
            List of Text2SQLSearchResult objects
        """
        # Log question
        logger.info("=" * 80)
        logger.info(f"🔍 QUESTION: {query}")
        logger.info("=" * 80)

        # Step 1: Locate relevant columns and tables
        logger.info("🔎 Step 1: Locating relevant schema elements...")
        relevant_schema = self._locate_relevant_schema(query)
        logger.info(f"   Located: {relevant_schema}")

        # Step 2: Generate SQL using relevant schema
        schema_text = self._format_relevant_schema(relevant_schema)

        system_prompt = """You are an expert SQL query generator. Your task is to convert natural language questions into accurate, executable SQL queries.

CRITICAL REQUIREMENTS:
1. ALWAYS include the target table's primary key or ID column in your SELECT clause (e.g., SELECT {target_table}.id, ...)
2. Use proper table qualifiers when columns might be ambiguous (e.g., table_name.column_name)
3. Pay careful attention to text matching - use exact string matching with proper escaping
4. When joining tables, use the foreign key relationships provided
5. Use proper SQL syntax for the database type
6. Limit results to {top_k} rows using LIMIT clause
7. Use ONLY the tables and columns provided in the relevant schema above

EXAMPLES:

Example 1:
Question: "Show me reports where the author is 'SMITH J'"
SQL: SELECT {target_table}.id FROM {target_table} WHERE {target_table}.AuthorNames = 'SMITH J' LIMIT {top_k}

Example 2:
Question: "Find reports that mention 'gold' in the abstract"
SQL: SELECT {target_table}.id FROM {target_table} JOIN abstracts ON {target_table}.id = abstracts.report_id WHERE abstracts.Abstract LIKE '%gold%' LIMIT {top_k}

IMPORTANT:
- Always SELECT the target table's primary key or ID column along with any other columns needed
- Use exact string matching with single quotes: column = 'value'
- For text search, use LIKE with wildcards: column LIKE '%text%'
- When joining, specify table names: table1.column = table2.column
- Return ONLY the SQL query, no explanations"""

        user_prompt = f"""{schema_text}

Question: {query}

Generate a SQL query to answer this question using ONLY the tables and columns provided above. Remember to:
1. Include the target table's primary key or ID column in the SELECT clause
2. Use proper table qualifiers for columns
3. Match the question requirements exactly
4. Use LIMIT {top_k} to limit results

SQL Query:"""

        # Generate SQL
        logger.info("💾 Step 2: Generating SQL query with relevant schema...")
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt.format(
                target_table=self.target_table, top_k=top_k
            ),
            temperature=0.0,
            max_tokens=2048,
        )

        sql_query = self._extract_sql_from_response(response.content)
        logger.info("-" * 80)
        logger.info(f"💾 SQL (initial): {sql_query}")
        logger.info("-" * 80)

        # Validate and fix SQL
        sql_query = self._validate_and_fix_sql(sql_query, query, max_retries=1)

        # Ensure ID column is selected (for evaluation purposes)
        sql_query = self._ensure_id_column_in_select(sql_query)

        # Add LIMIT if not present
        if "LIMIT" not in sql_query.upper():
            sql_query = f"{sql_query.rstrip(';')} LIMIT {top_k}"

        logger.info(f"💾 SQL (final): {sql_query}")
        logger.info("-" * 80)

        # Execute SQL with retry logic
        max_execution_retries = 2
        for attempt in range(max_execution_retries + 1):
            try:
                df = self._execute_sql(sql_query)
                result = self._convert_dataframe_to_results(df, sql_query, rank=1)

                # Log results
                logger.info("📊 RESULTS:")
                logger.info(f"   Returned {len(df)} row(s)")
                if len(df) > 0:
                    logger.info(f"   Columns: {', '.join(df.columns.tolist()[:5])}")
                    if len(df.columns) > 5:
                        logger.info(f"   ... and {len(df.columns) - 5} more columns")
                logger.info("=" * 80)

                return [result]
            except Exception as e:
                if attempt < max_execution_retries:
                    logger.warning(
                        f"SQL execution failed (attempt {attempt + 1}/{max_execution_retries + 1}): {e}"
                    )
                    # Try to fix and regenerate
                    sql_query = self._attempt_sql_fix(sql_query, str(e), query)
                    sql_query = self._ensure_id_column_in_select(sql_query)
                    if "LIMIT" not in sql_query.upper():
                        sql_query = f"{sql_query.rstrip(';')} LIMIT {top_k}"
                    logger.info(f"Retrying with fixed SQL: {sql_query[:200]}...")
                else:
                    logger.error("=" * 80)
                    logger.error(
                        f"❌ ERROR: Failed to execute SQL after {max_execution_retries + 1} attempts: {e}"
                    )
                    logger.error("=" * 80)
                    # Return error result
                    return [
                        Text2SQLSearchResult(
                            rank=1,
                            table=self.target_table,
                            data=[],
                            sql_query=sql_query,
                            row_count=0,
                            score=0.0,
                        )
                    ]

    def _locate_relevant_schema(self, query: str) -> Dict[str, List[str]]:
        """Locate relevant tables and columns for the query.

        Args:
            query: Natural language question

        Returns:
            Dict mapping table_name -> list of relevant column names
        """
        # Format full schema for analysis
        full_schema = self._format_schema_for_prompt()

        system_prompt = """You are a database schema analyzer. Analyze the question and identify which tables and columns are relevant.
Return a JSON object with table names as keys and lists of relevant column names as values.
Example: {{"customers": ["id", "name", "email"], "orders": ["id", "customer_id", "total"]}}
If a table is relevant but you're not sure about specific columns, include all columns from that table."""

        user_prompt = f"""{full_schema}

Question: {query}

Identify which tables and columns are relevant to answer this question. Return only the JSON object."""

        logger.info("Locating relevant schema elements...")
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1024,
            response_format="json",
        )

        try:
            relevant_schema = json.loads(response.content)
            if not isinstance(relevant_schema, dict):
                raise ValueError("Response is not a dictionary")
            return relevant_schema
        except Exception as e:
            logger.warning(
                f"Failed to parse relevant schema JSON: {e}, using full schema"
            )
            # Fallback: return all tables and columns
            return {
                table_name: table_meta.columns
                for table_name, table_meta in self.schema_metadata.tables.items()
            }

    def _format_relevant_schema(self, relevant_schema: Dict[str, List[str]]) -> str:
        """Format relevant schema information for SQL generation prompt.

        Args:
            relevant_schema: Dict mapping table_name -> list of column names

        Returns:
            Formatted schema string with only relevant information
        """
        parts = ["# Relevant Database Schema\n"]

        for table_name in relevant_schema.keys():
            if table_name not in self.schema_metadata.tables:
                logger.warning(f"Table {table_name} not found in schema metadata")
                continue

            table_meta = self.schema_metadata.tables[table_name]
            relevant_columns = relevant_schema[table_name]

            parts.append(f"## Table: {table_name}")
            if table_meta.primary_key:
                parts.append(f"Primary Key: {table_meta.primary_key}")
            parts.append(f"Relevant Columns: {', '.join(relevant_columns)}")

            # Include sample values for relevant columns
            if table_meta.sample_values:
                parts.append("Sample values:")
                for col in relevant_columns[:5]:  # Limit to 5 columns
                    if col in table_meta.sample_values:
                        vals = table_meta.sample_values[col]
                        sample = ", ".join(str(v) for v in vals[:3])
                        parts.append(f"  {col}: {sample}")
            parts.append("")

        # Include relevant foreign keys
        if self.schema_metadata.foreign_keys:
            parts.append("## Relevant Foreign Key Relationships\n")
            relevant_tables = set(relevant_schema.keys())
            for fk in self.schema_metadata.foreign_keys:
                if (
                    fk.child_table in relevant_tables
                    or fk.parent_table in relevant_tables
                ):
                    parts.append(
                        f"{fk.child_table}.{fk.child_column} -> "
                        f"{fk.parent_table}.{fk.parent_column}"
                    )
            parts.append("")

        return "\n".join(parts)
