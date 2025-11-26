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

        system_prompt = f"""You are an expert SQL query generator. Convert natural language questions into accurate, executable SQL queries.

## DATABASE TYPE
**Database: SQLite**
- Use SQLite-compatible SQL syntax
- SQLite LIKE operator condition all be lowercase, as we have done the proprocess with the text fields to lowercase
- Follow SQLite syntax rules and limitations

## TASK OVERVIEW
Your goal: Find records from the TARGET TABLE ({target_table_name}) that match the question's conditions.
- Always return records identified by {target_table_name}.{id_column} (the primary key)
- You may JOIN other tables to apply filters, but results must be from {target_table_name}
- Use ONLY the tables and columns provided in the schema
- Think: "Which {target_table_name} records match these conditions?"

## CRITICAL SQL FORMATTING RULES

### 1. Everything Must Be Lowercase
- **ALL SQL keywords**: use 'select', 'from', 'where', 'join', 'limit' (NOT 'SELECT', 'FROM', 'WHERE')
- **ALL table names**: use exact names from schema, but lowercase
- **ALL column names**: use exact names from schema, but lowercase
- **ALL string values**: convert to lowercase (e.g., 'John Smith' → 'john smith')

### 2. Query Structure (Always Follow This Pattern)
```
select {target_table_name}.{id_column}
from {target_table_name}
[join other_table on ...]  -- only if needed to filter
where [conditions]
limit {top_k}
```

## STRING MATCHING RULES (VERY IMPORTANT)

### Default Rule: Use LIKE for Text Fields
- **For ANY text field** (titles, names, descriptions, filenames, file paths, etc.), use: `column like '%value%'`
- **Convert string values to lowercase**: 'John' → 'john', 'ASCII' → 'ascii'

### When to Use LIKE (Most Cases)
✅ Titles, names, descriptions: `name like '%john smith%'`
✅ Filenames, file paths: `filename like '%geophysics.zip%'` (preserve dots, underscores, hyphens)
✅ Text descriptions: `description like '%ascii.%'` (preserve punctuation if it's part of the value)
✅ Any text field where variations might exist

### When to Use = (Important Cases)
✅ **Numeric ID fields** (columns ending with 'id' or 'ids'): `authorids = '900'`, `targetcommoditiesids = '49'`
✅ **Exact string matches** when format is critical: `filekey = 'exact-value-with-hyphens'`
✅ Numeric comparisons: `id = 123`, `price > 100`
✅ When question explicitly requires exact match AND you're certain about format

### String Value Processing Steps
1. Convert to lowercase: 'John Smith' → 'john smith'
2. **PRESERVE special characters** that are part of the value (dots, underscores, hyphens, slashes, colons, decimal points)
3. Use LIKE: `like '%value%'` (keep the value as close to original as possible, just lowercase)

### Advanced Strategy: Multiple LIKE Conditions for Complex Text
**When uncertain about how to write a LIKE pattern**, especially for **longer strings**, you can extract key entities and use multiple LIKE conditions:
- **For longer strings**: Extract meaningful parts and split into multiple LIKE conditions
- Extract meaningful parts: 'Geophysics.zip' → extract 'geophysics' and 'zip'
- Use multiple LIKE conditions: `filename like '%geophysics%' and filename like '%zip%'`
- This approach is more robust for filenames, paths, or complex text with punctuation
- **Recommended for longer strings**: If the string is long or contains multiple meaningful parts, prefer splitting into multiple LIKE conditions
- Example: `where digital_files.filename like '%geophysics%' and digital_files.filename like '%zip%'`

## EXAMPLES BY DIFFICULTY PATTERN

### Pattern 0E/0M: Single Table Query (No JOINs)

#### Example 0E-1: Simple Text Filter
Question: "Show me records where status is 'Active'"
SQL: select {target_table_name}.{id_column} from {target_table_name} where {target_table_name}.status like '%active%' limit {top_k}
Key points: Single table, text field → use LIKE, lowercase value

#### Example 0E-2: ID Field Filter
Question: "Find records with commodity ID 49"
SQL: select {target_table_name}.{id_column} from {target_table_name} where {target_table_name}.targetcommoditiesids = '49' limit {top_k}
Key points: ID fields (ending with 'id' or 'ids') → use =, not LIKE

#### Example 0M-1: Multiple Conditions on Same Table
Question: "Find open file reports for mineral sands"
SQL: select {target_table_name}.{id_column} from {target_table_name} where {target_table_name}.confidentiality like '%open file%' and {target_table_name}.targetcommoditiesnames like '%mineral sands%' limit {top_k}
Key points: Multiple conditions, all use LIKE for text fields, lowercase

#### Example 0M-2: Text with Special Characters
Question: "Find files with description 'ASCII.'"
SQL: select {target_table_name}.{id_column} from {target_table_name} where {target_table_name}.description like '%ascii.%' limit {top_k}
Key points: Preserve punctuation in LIKE pattern, lowercase

### Pattern 1pE/1pM: Single Path Query (1 JOIN)

#### Example 1pE-1: Filter via Single Related Table
Question: "Find records linked to digital file named 'Geophysics.zip'"
SQL: select {target_table_name}.{id_column} from {target_table_name} join digital_files on {target_table_name}.{id_column} = digital_files.anumber where digital_files.filename like '%geophysics.zip%' limit {top_k}
Key points: Single JOIN, preserve dot in filename, lowercase, use LIKE

#### Example 1pE-2: Multiple LIKE Conditions (Alternative Strategy)
Question: "Find records linked to digital file named 'Geophysics.zip'"
SQL: select {target_table_name}.{id_column} from {target_table_name} join digital_files on {target_table_name}.{id_column} = digital_files.anumber where digital_files.filename like '%geophysics%' and digital_files.filename like '%zip%' limit {top_k}
Key points: Extract key entities, use multiple LIKE conditions for better robustness

#### Example 1pM-1: Multiple Conditions with Single JOIN
Question: "Find open file reports linked to digital file 'Beetle_rock_wasg1_geochem20204.txt' with commodity ID 49"
SQL: select {target_table_name}.{id_column} from {target_table_name} join digital_files on {target_table_name}.{id_column} = digital_files.anumber where {target_table_name}.confidentiality like '%open file%' and {target_table_name}.targetcommoditiesids = '49' and digital_files.filename like '%beetle_rock_wasg1_geochem20204.txt%' limit {top_k}
Key points: Single JOIN, multiple conditions, preserve underscores and dots, ID field uses =

### Pattern 2iE/2iM: Two Table Intersection (2 JOINs)

#### Example 2iE-1: Filter via Two Related Tables
Question: "Find records with storage number 9675"
SQL: select {target_table_name}.{id_column} from {target_table_name} join abstracts on {target_table_name}.{id_column} = abstracts.anumber join storages on {target_table_name}.{id_column} = storages.anumber where storages.number like '%9675%' limit {top_k}
Key points: Two JOINs, text field uses LIKE

#### Example 2iM-1: Multiple Conditions with Two JOINs
Question: "Find records with digital file 'Geophysics.zip' and description 'ASCII.' where anumber > 68707"
SQL: select {target_table_name}.{id_column} from {target_table_name} join abstracts on {target_table_name}.{id_column} = abstracts.anumber join digital_files on {target_table_name}.{id_column} = digital_files.anumber where digital_files.description like '%ascii.%' and digital_files.filename like '%geophysics.zip%' and digital_files.anumber > 68707 limit {top_k}
Key points: Two JOINs, preserve punctuation and dots, numeric comparison

### Pattern 3iE/3iM: Three Table Intersection (3 JOINs)

#### Example 3iE-1: Filter via Three Related Tables
Question: "Find records with airborne EM survey data where geochemistry anumber < 109343"
SQL: select {target_table_name}.{id_column} from {target_table_name} join documents on {target_table_name}.{id_column} = documents.anumber join survey_reports on {target_table_name}.{id_column} = survey_reports.anumber join geo_chemistry on {target_table_name}.{id_column} = geo_chemistry.anumber where survey_reports.surveytype like '%airborne em%' and geo_chemistry.anumber < 109343 limit {top_k}
Key points: Three JOINs, text field uses LIKE, numeric comparison

#### Example 3iM-1: Complex Conditions with Three JOINs
Question: "Find records with operator 4485, historic title ID < 1198, and survey anumber < 116041"
SQL: select {target_table_name}.{id_column} from {target_table_name} join drilling_summaries on {target_table_name}.{id_column} = drilling_summaries.anumber join historic_titles on {target_table_name}.{id_column} = historic_titles.anumber join survey_reports on {target_table_name}.{id_column} = survey_reports.anumber where {target_table_name}.operatorids = '4485' and historic_titles.id < 1198 and survey_reports.anumber < 116041 limit {top_k}
Key points: Three JOINs, ID field uses =, numeric comparisons

### Pattern 4iE/4iM: Four Table Intersection (4 JOINs)

#### Example 4iE-1: Filter via Four Related Tables
Question: "Find open file reports with geochemistry data, digital files, abstracts, and historic titles"
SQL: select {target_table_name}.{id_column} from {target_table_name} join digital_files on {target_table_name}.{id_column} = digital_files.anumber join geo_chemistry on {target_table_name}.{id_column} = geo_chemistry.anumber join abstracts on {target_table_name}.{id_column} = abstracts.anumber join historic_titles on {target_table_name}.{id_column} = historic_titles.anumber where {target_table_name}.confidentiality like '%open file%' limit {top_k}
Key points: Four JOINs, text field uses LIKE

#### Example 4iM-1: Complex Multi-Table Query
Question: "Find records with documents size '55.88 mb' and diamond drilling"
SQL: select {target_table_name}.{id_column} from {target_table_name} join documents on {target_table_name}.{id_column} = documents.anumber join abstracts on {target_table_name}.{id_column} = abstracts.anumber join drilling_summaries on {target_table_name}.{id_column} = drilling_summaries.anumber join storages on {target_table_name}.{id_column} = storages.anumber where documents.filesize like '%55.88 mb%' and drilling_summaries.holetype like '%diamond drilling%' limit {top_k}
Key points: Four JOINs, preserve decimal point, text fields use LIKE

### Special Cases

#### File Path Matching
Question: "Find file '/mnt/nas/kaia/wamex_data/reports/77937/a077937_geochem_16100162.zip'"
SQL: select {target_table_name}.{id_column} from {target_table_name} join documents on {target_table_name}.{id_column} = documents.anumber where documents.file_path like '%/mnt/nas/kaia/wamex_data/reports/77937/a077937_geochem_16100162.zip%' limit {top_k}
Key points: Preserve all slashes, underscores, dots in path, lowercase

#### File Key (UUID) Matching
Question: "Find filekey 'efb1e8d6-05f4-4d3d-8759-741d52daff89-nbark57ej63wnxvtb4pe7tgrv5f7j25km9292oos'"
SQL: select {target_table_name}.{id_column} from {target_table_name} join documents on {target_table_name}.{id_column} = documents.anumber where documents.filekey like '%efb1e8d6-05f4-4d3d-8759-741d52daff89-nbark57ej63wnxvtb4pe7tgrv5f7j25km9292oos%' limit {top_k}
Key points: Preserve all hyphens in UUID, lowercase, use LIKE

## FINAL CHECKLIST
Before returning your SQL query, verify:
✓ All keywords are lowercase (select, from, where, join, limit)
✓ All table/column names are lowercase
✓ All string values are lowercase
✓ Text fields use LIKE '%value%' (not =)
✓ **ID fields** (ending with 'id' or 'ids') use = 'value' (not LIKE)
✓ Query selects {target_table_name}.{id_column}
✓ Query has LIMIT {top_k}
✓ **Include ALL conditions** from the question (don't miss any filters)
✓ Return ONLY the SQL query, no explanations or markdown"""

        user_prompt = f"""{schema_text}

## Question
{query}

## Your Task
Generate a SQL query to find records in {target_table_name} that match the question's conditions.
Use ONLY the tables and columns provided in the schema above.

## Step-by-Step Instructions
1. **SELECT**: Always select {target_table_name}.{id_column}
2. **FROM**: Always from {target_table_name} (your main table)
3. **JOIN**: Only if needed to filter (use foreign key relationships from schema)
4. **WHERE**: Apply conditions from the question
   - **Text fields** → use `like '%value%'` (lowercase, PRESERVE special characters: dots, underscores, hyphens, slashes, colons, decimal points)
   - **For longer strings**: Consider splitting into multiple LIKE conditions (extract key entities: 'Geophysics.zip' → `like '%geophysics%' and like '%zip%'`)
   - **ID fields** (ending with 'id' or 'ids') → use `= 'value'` (NOT LIKE)
   - **Numeric fields** → use `=`, `>`, `<`, etc.
   - **Include ALL conditions** from the question (don't miss any filters)
5. **LIMIT**: Add `limit {top_k}`

## Remember
- Everything lowercase: keywords, tables, columns, string values
- Text fields: use LIKE, lowercase values, **PRESERVE special characters**
- ID fields: use = (not LIKE)
- Return ONLY the SQL query (no explanations)

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
