"""Two-step Text2SQL Retriever - Question → locate columns/tables → SQL → results."""

from __future__ import annotations

import json
from typing import Dict, List

from talk2metadata.core.modes.text2sql.base import (
    BaseText2SQLRetriever,
    Text2SQLSearchResult,
)
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import timed

logger = get_logger(__name__)


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

        target_table_name = self.target_table
        id_column = self._get_target_id_column() or "id"

        system_prompt = f"""You are an expert SQL query generator. Your task is to convert natural language questions into accurate, executable SQL queries.

CRITICAL REQUIREMENTS:
1. ALWAYS include the target table's primary key or ID column in your SELECT clause (e.g., SELECT {target_table_name}.{id_column}, ...)
2. Use EXACT table and column names from the relevant schema provided - case sensitivity matters!
3. Use proper table qualifiers when columns might be ambiguous (e.g., table_name.column_name)
4. Pay careful attention to text matching - use EXACT string matching with proper escaping (column = 'value' with exact case and spacing)
5. When joining tables, use the EXACT foreign key relationships provided
6. Use proper SQL syntax for the database type
7. Limit results to {top_k} rows using LIMIT clause
8. Use ONLY the tables and columns provided in the relevant schema above

STRING MATCHING RULES:
- Use EXACT match (=) for specific values mentioned in the question: column = 'EXACT VALUE'
- Use LIKE only when the question asks for "contains", "mentions", or similar fuzzy matching
- Preserve exact case and spacing in string values as they appear in the data

GENERAL EXAMPLES:

Example 1 - Simple filter:
Question: "Show me records where status is 'ACTIVE'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.status = 'ACTIVE' LIMIT {top_k}

Example 2 - Join with text search:
Question: "Find records that mention 'important' in the description"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} JOIN related_table ON {target_table_name}.{id_column} = related_table.foreign_key WHERE related_table.description LIKE '%important%' LIMIT {top_k}

IMPORTANT:
- Always SELECT the target table's primary key or ID column ({target_table_name}.{id_column})
- Use EXACT table and column names from the relevant schema provided
- Use exact string matching (=) unless the question explicitly asks for partial matching
- Preserve exact case and spacing in string values
- When joining, use the EXACT foreign key relationships shown in the schema
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
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=2048,
        )

        sql_query = self._extract_sql_from_response(response.content)
        logger.info("-" * 80)
        logger.info(f"💾 SQL (initial): {sql_query}")
        logger.info("-" * 80)

        # Validate table/column names before execution
        is_valid, issues = self._validate_table_column_names(sql_query)
        if not is_valid:
            logger.warning(f"Schema validation issues detected: {issues}")
            # Try to fix common mistakes
            sql_query = self._attempt_sql_fix(sql_query, "; ".join(issues), query)

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
Example: {"customers": ["id", "name", "email"], "orders": ["id", "customer_id", "total"]}
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
        parts.append(
            "IMPORTANT: Use EXACT table and column names as shown below. Case sensitivity matters!\n"
        )

        for table_name in relevant_schema.keys():
            if table_name not in self.schema_metadata.tables:
                logger.warning(f"Table {table_name} not found in schema metadata")
                continue

            table_meta = self.schema_metadata.tables[table_name]
            relevant_columns = relevant_schema[table_name]

            parts.append(f"## Table: {table_name}")
            if table_name == self.target_table:
                parts.append(
                    "  ⭐ THIS IS THE TARGET TABLE (use this as the main table)"
                )
            if table_meta.primary_key:
                parts.append(f"  Primary Key: {table_meta.primary_key}")
            parts.append(f"  Relevant Columns: {', '.join(relevant_columns)}")

            # Include sample values for relevant columns
            if table_meta.sample_values:
                parts.append(
                    "  Sample values (use EXACT format including spaces and case):"
                )
                for col in relevant_columns[:5]:  # Limit to 5 columns
                    if col in table_meta.sample_values:
                        vals = table_meta.sample_values[col]
                        sample_strs = [f"'{str(v)}'" for v in vals[:3]]
                        sample = ", ".join(sample_strs)
                        parts.append(f"    {col}: {sample}")
            parts.append("")

        # Include relevant foreign keys
        if self.schema_metadata.foreign_keys:
            parts.append("## Relevant Foreign Key Relationships\n")
            parts.append("Use these EXACT relationships to JOIN tables:\n")
            relevant_tables = set(relevant_schema.keys())
            for fk in self.schema_metadata.foreign_keys:
                if (
                    fk.child_table in relevant_tables
                    or fk.parent_table in relevant_tables
                ):
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
