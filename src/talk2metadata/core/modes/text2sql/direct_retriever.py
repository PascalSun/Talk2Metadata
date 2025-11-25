"""Direct Text2SQL Retriever - Question + schema → SQL → results."""

from __future__ import annotations

from typing import List

from talk2metadata.core.modes.text2sql.base import (
    BaseText2SQLRetriever,
    Text2SQLSearchResult,
)
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import timed

logger = get_logger(__name__)


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

        # Create enhanced prompt with generic examples
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
3. Use EXACT table and column names from the schema - case sensitivity matters!
4. Use proper table qualifiers when columns might be ambiguous: table_name.column_name
5. Pay careful attention to text matching - use LIKE '%value%' for fuzzy matching on text fields
6. When joining tables, use the EXACT foreign key relationships provided in the schema
7. Use proper SQL syntax (SQLite-compatible)
8. Limit results to {top_k} rows using LIMIT clause

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

Pattern 4 - Complex text matching (fuzzy match):
Question: "Find records with name 'John Smith'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.name LIKE '%john smith%' LIMIT {top_k}
Note: Use LIKE for text fields to handle variations. LIKE '%value%' matches the value anywhere in the field, handling case and spacing differences.

Pattern 5 - Numeric comparison:
Question: "Find records where price is greater than 100 and quantity is less than 50"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.price > 100 AND {target_table_name}.quantity < 50 LIMIT {top_k}

IMPORTANT REMINDERS:
- The question is asking: "Which {target_table_name} records match these conditions?"
- Always SELECT {target_table_name}.{id_column} to identify the matching records
- Always FROM {target_table_name} (your main table)
- JOINs are used to FILTER, not to return data from joined tables
- Use column descriptions in the schema to understand what each column represents
- Match text values EXACTLY as they appear in sample values (case, spacing, punctuation)
- Return ONLY the SQL query, no explanations or markdown"""

        user_prompt = f"""{schema_text}

Question: {query}

Your task: Find which records in {target_table_name} match the conditions described in the question.

Generate a SQL query following these steps:
1. SELECT {target_table_name}.{id_column} (to identify the matching records)
2. FROM {target_table_name} (the target table - this is your main table)
3. JOIN other tables ONLY if needed to apply filters (use foreign key relationships from schema)
4. WHERE conditions: Match the question requirements exactly
   - Use EXACT text matching (=) unless the question asks for partial matching
   - Pay attention to column descriptions to understand what each column represents
   - Match sample values exactly (case, spacing, punctuation)
5. LIMIT {top_k} (to limit results)

SQL Query:"""

        # Generate SQL
        logger.info("💾 Generating SQL query...")
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
