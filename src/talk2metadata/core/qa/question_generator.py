"""Question generator for converting SQL queries to natural language questions.

Uses LLM to rewrite SQL queries into human-like natural language questions.
"""

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.query_builder import QuerySpec
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class QuestionGenerator:
    """Generates natural language questions from SQL queries using LLM."""

    def __init__(self, agent: AgentWrapper, schema: SchemaMetadata):
        """Initialize question generator.

        Args:
            agent: AgentWrapper instance for LLM calls
            schema: Schema metadata for context
        """
        self.agent = agent
        self.schema = schema

    def generate(self, query_spec: QuerySpec) -> str:
        """Generate a natural language question from a query specification.

        Args:
            query_spec: QuerySpec object containing the SQL query and metadata

        Returns:
            Natural language question string
        """
        # Build the prompt for LLM
        prompt = self._build_prompt(query_spec)

        # Call LLM to generate question
        try:
            response = self.agent.generate(prompt)
            question = response.content.strip()

            # Clean up the question
            question = self._clean_question(question)

            logger.debug(f"Generated question: {question}")
            return question

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            # Fallback to a simple template-based question
            return self._generate_fallback_question(query_spec)

    def _build_prompt(self, query_spec: QuerySpec) -> str:
        """Build the prompt for LLM to generate a question.

        Args:
            query_spec: QuerySpec object

        Returns:
            Prompt string
        """
        # Get table and column information
        target_table = query_spec.target_table
        involved_tables = query_spec.involved_tables
        filters = query_spec.filters

        # Build filter descriptions
        filter_descriptions = []
        for f in filters:
            table_meta = self.schema.tables[f.table]
            # Get sample values for context
            sample_vals = table_meta.sample_values.get(f.column, [])
            sample_str = f", examples: {sample_vals[:3]}" if sample_vals else ""

            filter_descriptions.append(
                f"  - {f.table}.{f.column} {f.operator} {f.value}{sample_str}"
            )

        filter_text = "\n".join(filter_descriptions)

        # Build JOIN description
        if query_spec.join_paths:
            join_descriptions = []
            for path in query_spec.join_paths:
                if path.join_type == "chain":
                    join_descriptions.append(f"  - Chain: {' → '.join(path.tables)}")
                else:
                    join_descriptions.append(
                        f"  - Star: {path.tables[0]} ↔ {path.tables[1]}"
                    )
            join_text = "\n".join(join_descriptions)
        else:
            join_text = "  - No JOINs (direct query on target table)"

        # Build the prompt
        prompt = f"""You are a data analyst helping to create natural language questions for a database query.

**Task**: Convert the following SQL query into a natural, human-like question that a user might ask.

**Target Table**: {target_table} (this is what the user wants to find records from)

**Database Schema**:
Tables involved: {', '.join(involved_tables)}

**JOIN Structure**:
{join_text}

**Filter Conditions**:
{filter_text}

**SQL Query**:
```sql
{query_spec.sql}
```

**Requirements**:
1. The question should be natural and conversational (like a real user would ask)
2. Focus on WHAT the user is looking for (records from {target_table})
3. Include the filter conditions in a natural way
4. Don't mention technical terms like "JOIN", "WHERE", "SQL"
5. The question should make sense to someone who doesn't know SQL
6. Keep it concise but complete

**Example transformations**:
- SQL: SELECT * FROM orders WHERE status = 'completed'
  Question: "Find all completed orders"

- SQL: SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.industry = 'Healthcare'
  Question: "What orders are from Healthcare industry customers?"

- SQL: SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id JOIN products ON orders.product_id = products.id WHERE customers.industry = 'Healthcare' AND products.category = 'Software'
  Question: "Find orders from Healthcare customers that purchased Software products"

Now generate a natural question for the given query. Output ONLY the question, nothing else."""

        return prompt

    def _clean_question(self, question: str) -> str:
        """Clean up the generated question.

        Args:
            question: Raw question string from LLM

        Returns:
            Cleaned question string
        """
        # Remove quotes
        question = question.strip('"').strip("'")

        # Remove "Question:" prefix if present
        if question.lower().startswith("question:"):
            question = question[9:].strip()

        # Ensure it ends with a question mark
        if not question.endswith("?"):
            question += "?"

        return question

    def _generate_fallback_question(self, query_spec: QuerySpec) -> str:
        """Generate a fallback template-based question if LLM fails.

        Args:
            query_spec: QuerySpec object

        Returns:
            Template-based question string
        """
        target_table = query_spec.target_table
        filters = query_spec.filters

        if not filters:
            return f"Find all {target_table}?"

        # Build filter descriptions
        filter_parts = []
        for f in filters:
            if f.operator == "=":
                filter_parts.append(f"{f.column} is {f.value}")
            elif f.operator == ">":
                filter_parts.append(f"{f.column} greater than {f.value}")
            elif f.operator == "<":
                filter_parts.append(f"{f.column} less than {f.value}")
            else:
                filter_parts.append(f"{f.column} {f.operator} {f.value}")

        filter_text = " and ".join(filter_parts)

        # Include JOIN information
        if query_spec.join_paths:
            related_tables = [
                t
                for path in query_spec.join_paths
                for t in path.tables
                if t != target_table
            ]
            if related_tables:
                return f"Find {target_table} from {', '.join(related_tables)} where {filter_text}?"

        return f"Find {target_table} where {filter_text}?"
