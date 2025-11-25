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

CRITICAL REQUIREMENTS:
1. ALWAYS include the target table's primary key or ID column in your SELECT clause (e.g., SELECT {target_table_name}.{id_column}, ...)
2. Use EXACT table and column names from the schema - case sensitivity matters!
3. Use proper table qualifiers when columns might be ambiguous (e.g., table_name.column_name)
4. Pay careful attention to text matching - use EXACT string matching with proper escaping (column = 'value' with exact case and spacing)
5. When joining tables, use the EXACT foreign key relationships provided
6. Use proper SQL syntax for the database type
7. Limit results to {top_k} rows using LIMIT clause

STRING MATCHING RULES:
- Use EXACT match (=) for specific values mentioned in the question: column = 'EXACT VALUE'
- Use LIKE only when the question asks for "contains", "mentions", or similar fuzzy matching
- Preserve exact case and spacing in string values as they appear in the data

GENERAL EXAMPLES:

Example 1 - Simple filter:
Question: "Show me records where status is 'ACTIVE' and category is 'PREMIUM'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.status = 'ACTIVE' AND {target_table_name}.category = 'PREMIUM' LIMIT {top_k}

Example 2 - Join with text search:
Question: "Find records that mention 'important' in the description"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} JOIN related_table ON {target_table_name}.{id_column} = related_table.foreign_key WHERE related_table.description LIKE '%important%' LIMIT {top_k}

Example 3 - Numeric filter:
Question: "Get records where id is greater than 100 and value is '12345'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} WHERE {target_table_name}.id > 100 AND {target_table_name}.value = '12345' LIMIT {top_k}

Example 4 - Join with exact match:
Question: "Find records associated with file 'document.pdf'"
SQL: SELECT {target_table_name}.{id_column} FROM {target_table_name} JOIN files_table ON {target_table_name}.{id_column} = files_table.foreign_key WHERE files_table.filename = 'document.pdf' LIMIT {top_k}

IMPORTANT:
- Always SELECT the target table's primary key or ID column ({target_table_name}.{id_column})
- Use EXACT table and column names from the schema provided
- Use exact string matching (=) unless the question explicitly asks for partial matching
- Preserve exact case and spacing in string values
- When joining, use the EXACT foreign key relationships shown in the schema
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
