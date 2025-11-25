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

QUERY PATTERN UNDERSTANDING:
All questions follow the same pattern: they ask you to FIND/FILTER records from the TARGET TABLE ({target_table_name}) based on various conditions.
- The goal is ALWAYS to return records (identified by {target_table_name}.{id_column}) from the target table
- You may need to JOIN other tables to apply filters, but the final result must be records from {target_table_name}
- Think of it as: "Which records in {target_table_name} match these conditions?"

CRITICAL REQUIREMENTS:
1. ALWAYS SELECT {target_table_name}.{id_column} (the target table's primary key) - this identifies the records
2. ALWAYS query FROM {target_table_name} (the target table) - this is your main table
3. Use EXACT table and column names from the relevant schema provided - case sensitivity matters!
4. Use proper table qualifiers when columns might be ambiguous: table_name.column_name
5. Pay careful attention to text matching - use EXACT string matching with proper escaping
6. When joining tables, use the EXACT foreign key relationships provided in the schema
7. Use proper SQL syntax (SQLite-compatible)
8. Limit results to {top_k} rows using LIMIT clause
9. Use ONLY the tables and columns provided in the relevant schema above

STRING MATCHING RULES:
- DEFAULT: Use LIKE with wildcards for text matching to handle variations: column LIKE '%value%'
- This provides fuzzy matching that handles:
  * Case variations (uppercase/lowercase)
  * Spacing differences
  * Punctuation variations
  * Partial matches
- Use EXACT match (=) ONLY for:
  * Numeric comparisons (id = 123)
  * Boolean/enum values where exact match is critical (status = 'active')
  * When the question explicitly requires exact matching
- For text fields (titles, names, descriptions, etc.), ALWAYS use LIKE '%value%' for better search results
- Pay attention to sample values in the schema - they show the format, but LIKE will match variations

QUERY PATTERNS:

Pattern 1 - Direct filter on target table:
Question: "Show me records where status is 'Active'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.status LIKE '%active%' LIMIT {top_k}
Note: Use LIKE for text fields to handle case/spacing variations. Use = only for exact enum/boolean values.

Pattern 2 - Filter on target table + JOIN to filter by related table:
Question: "Find records that have associated items with category 'Electronics'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} JOIN items_table ON {target_table_name}.{id_column} = items_table.parent_id WHERE items_table.category LIKE '%electronics%' LIMIT {top_k}
Note: The JOIN is used to FILTER the target table records, not to return data from the joined table. Use LIKE for text matching.

Pattern 3 - Multiple conditions (target table + related tables):
Question: "Find records where status = 'Active' AND associated notes contain 'urgent'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} JOIN notes_table ON {target_table_name}.{id_column} = notes_table.record_id WHERE {target_table_name}.status LIKE '%active%' AND notes_table.content LIKE '%urgent%' LIMIT {top_k}
Note: Use LIKE for text fields. For enum/boolean fields, you may use = if exact match is required.

Pattern 4 - Numeric comparison:
Question: "Find records where price is greater than 100"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.price > 100 LIMIT {top_k}

IMPORTANT REMINDERS:
- The question is asking: "Which {target_table_name} records match these conditions?"
- Always SELECT {target_table_name}.{id_column} to identify the matching records
- Always FROM {target_table_name} (your main table)
- JOINs are used to FILTER, not to return data from joined tables
- Use column descriptions in the schema to understand what each column represents
- Use LIKE '%value%' for text fields to handle variations in case, spacing, and punctuation
- Return ONLY the SQL query, no explanations or markdown"""

        user_prompt = f"""{schema_text}

Question: {query}

Your task: Find which records in {target_table_name} match the conditions described in the question.

Generate a SQL query using ONLY the tables and columns provided above. Follow these steps:
1. SELECT {target_table_name}.{id_column} (to identify the matching records)
2. FROM {target_table_name} (the target table - this is your main table)
3. JOIN other tables ONLY if needed to apply filters (use foreign key relationships from schema)
4. WHERE conditions: Match the question requirements
   - Use LIKE '%value%' for text fields (titles, names, descriptions, etc.) for fuzzy matching
   - Use = for numeric comparisons and exact enum/boolean values when needed
   - Pay attention to column descriptions to understand what each column represents
   - LIKE handles case, spacing, and punctuation variations automatically
5. LIMIT {top_k} (to limit results)

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

        # Convert SQL to lowercase before storing and executing
        sql_query = self._normalize_sql_to_lowercase(sql_query)

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
                    # Convert SQL to lowercase before retrying
                    sql_query = self._normalize_sql_to_lowercase(sql_query)
                    logger.info(f"Retrying with fixed SQL: {sql_query[:200]}...")
                else:
                    logger.error("=" * 80)
                    logger.error(
                        f"❌ ERROR: Failed to execute SQL after {max_execution_retries + 1} attempts: {e}"
                    )
                    logger.error("=" * 80)
                    # Convert SQL to lowercase before storing error result
                    sql_query = self._normalize_sql_to_lowercase(sql_query)
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
                table_name: list(table_meta.columns.keys())
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

            # Ensure relevant_columns is a list (handle case where it might be a dict)
            if isinstance(relevant_columns, dict):
                relevant_columns = list(relevant_columns.keys())
            elif not isinstance(relevant_columns, list):
                relevant_columns = (
                    list(relevant_columns)
                    if hasattr(relevant_columns, "__iter__")
                    else []
                )

            parts.append(f"## Table: {table_name}")
            if table_name == self.target_table:
                parts.append(
                    "  ⭐ THIS IS THE TARGET TABLE (use this as the main table)"
                )
            # Add table description if available
            if table_meta.description:
                parts.append(f"  Description: {table_meta.description}")
            if table_meta.primary_key:
                parts.append(f"  Primary Key: {table_meta.primary_key}")

            # Format columns with descriptions
            parts.append("  Relevant Columns:")
            for col in relevant_columns:
                col_info = f"{col}"
                if col in table_meta.columns:
                    col_info += f" ({table_meta.columns[col]})"
                # Add column description if available
                if col in table_meta.column_descriptions:
                    col_info += f" - {table_meta.column_descriptions[col]}"
                parts.append(f"    - {col_info}")

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
