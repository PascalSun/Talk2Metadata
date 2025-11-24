"""Path pattern generator using LLM to analyze schema."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa_generation.patterns import PathPattern
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class PathPatternGenerator:
    """Generate meaningful path patterns using LLM analysis of schema."""

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize pattern generator.

        Args:
            agent: Optional AgentWrapper instance (creates one if None)
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
        """
        if agent is None:
            self.agent = AgentWrapper(provider=provider, model=model)
        else:
            self.agent = agent

    def generate_patterns(
        self,
        schema: SchemaMetadata,
        num_patterns: int = 15,
        min_patterns: int = 10,
    ) -> List[PathPattern]:
        """Generate path patterns from schema.

        Args:
            schema: Schema metadata
            num_patterns: Target number of patterns to generate
            min_patterns: Minimum number of patterns required

        Returns:
            List of PathPattern objects
        """
        logger.info(f"Generating {num_patterns} path patterns from schema...")

        # Build prompt
        prompt = self._build_pattern_generation_prompt(schema, num_patterns)

        # Call LLM
        try:
            response = self.agent.generate(prompt)
            patterns_json = self._parse_llm_response(response.content)

            # Convert to PathPattern objects
            patterns = []
            for p in patterns_json:
                try:
                    pattern = PathPattern(
                        pattern=p["pattern"],
                        semantic=p["semantic"],
                        question_template=p.get("question_template", ""),
                        answer_type=p.get("answer_type", "multiple"),
                        difficulty=p.get("difficulty", "medium"),
                        description=p.get("description"),
                    )
                    patterns.append(pattern)
                except KeyError as e:
                    logger.warning(f"Invalid pattern format, missing key: {e}")
                    continue

            if len(patterns) < min_patterns:
                logger.warning(
                    f"Only generated {len(patterns)} patterns, "
                    f"less than minimum {min_patterns}"
                )

            logger.info(f"Generated {len(patterns)} path patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to generate patterns: {e}")
            # Return fallback patterns
            return self._generate_fallback_patterns(schema)

    def _build_pattern_generation_prompt(
        self, schema: SchemaMetadata, num_patterns: int
    ) -> str:
        """Build prompt for LLM to generate path patterns.

        Args:
            schema: Schema metadata
            num_patterns: Number of patterns to generate

        Returns:
            Prompt string
        """
        # Format tables information
        tables_info = []
        for name, meta in schema.tables.items():
            is_target = " (TARGET TABLE)" if name == schema.target_table else ""
            columns_str = ", ".join(meta.columns.keys())
            tables_info.append(
                f"- {name}{is_target}:\n"
                f"  - Primary Key: {meta.primary_key}\n"
                f"  - Columns: {columns_str}\n"
                f"  - Row Count: {meta.row_count}"
            )

        # Format foreign keys
        fk_info = []
        if schema.foreign_keys:
            for fk in schema.foreign_keys:
                fk_info.append(
                    f"- {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column} "
                    f"(coverage: {fk.coverage:.1%})"
                )
        else:
            # If no explicit FKs, infer from ANumber/Anumber columns
            fk_info.append(
                "- All tables are connected via ANumber/Anumber columns "
                "to the target table wamex_reports"
            )

        prompt = f"""You are a database query expert. Given the following database schema, generate {num_patterns} meaningful query path patterns for generating evaluation questions.

## Database Schema

**Target Table:** {schema.target_table} (Primary Key: {schema.tables[schema.target_table].primary_key})
This is the table we want to retrieve records from.

**Related Tables:**
{chr(10).join(tables_info)}

**Foreign Key Relationships:**
{chr(10).join(fk_info) if fk_info else "No explicit foreign keys defined. Tables are connected via ANumber/Anumber columns."}

## Task

Generate {num_patterns} meaningful query path patterns. Each path pattern should:

1. **Start from any table** and **end at the target table** ({schema.target_table})
2. **Path length**: 1-3 hops
   - 1 hop: Direct connection to target table (e.g., ["historic_titles", "wamex_reports"])
   - 2 hops: Through one intermediate table (e.g., ["abstracts", "wamex_reports", "documents"])
   - 3 hops: Through multiple tables (e.g., ["historic_titles", "wamex_reports", "abstracts", "drilling_summaries"])
3. **Semantic clarity**: Users might actually ask questions following this pattern
4. **Question template**: Include a template with {{placeholder}} for attribute values
5. **Diversity**: Cover different query types:
   - Direct queries (querying target table directly)
   - Single-hop queries (through one FK)
   - Multi-hop queries (through multiple FKs)
   - Conditional queries (with specific conditions)

## Output Format

Return a JSON array, where each element is a path pattern object with:
- `pattern`: Array of table names, e.g., ["historic_titles", "wamex_reports"]
- `semantic`: Semantic description of what this path represents (in English)
- `question_template`: Question template with {{placeholder}} for values, in English, e.g., "Which reports have the title '{{historic_title}}'?"
- `answer_type`: "single" (one answer) | "multiple" (multiple answers) | "aggregate" (aggregated answer)
- `difficulty`: "easy" | "medium" | "hard"
- `description`: Optional additional description

## Example Output

```json
[
  {{
    "pattern": ["historic_titles", "wamex_reports"],
    "semantic": "Find reports by historic title",
    "question_template": "Which reports have the title '{{historic_title}}'?",
    "answer_type": "multiple",
    "difficulty": "easy",
    "description": "Simple query through historic titles"
  }},
  {{
    "pattern": ["abstracts", "wamex_reports"],
    "semantic": "Find reports by abstract content",
    "question_template": "Which reports contain abstracts about '{{topic}}'?",
    "answer_type": "multiple",
    "difficulty": "medium",
    "description": "Query through abstract content"
  }}
]
```

Now generate {num_patterns} diverse and meaningful path patterns:"""

        return prompt

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract JSON patterns.

        Args:
            response: LLM response string

        Returns:
            List of pattern dictionaries
        """
        # Try to extract JSON from response
        # LLM might wrap JSON in markdown code blocks
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            patterns = json.loads(response)
            if isinstance(patterns, list):
                return patterns
            elif isinstance(patterns, dict) and "patterns" in patterns:
                return patterns["patterns"]
            else:
                logger.warning(f"Unexpected response format: {type(patterns)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response[:500]}")
            return []

    def _generate_fallback_patterns(self, schema: SchemaMetadata) -> List[PathPattern]:
        """Generate fallback patterns when LLM fails.

        Args:
            schema: Schema metadata

        Returns:
            List of basic PathPattern objects
        """
        logger.info("Generating fallback patterns...")
        patterns = []

        target_table = schema.target_table

        # Generate direct patterns (1 hop)
        for table_name, meta in schema.tables.items():
            if table_name == target_table:
                continue

            # Check if table has ANumber/Anumber column
            has_anumber = "ANumber" in meta.columns or "Anumber" in meta.columns

            if has_anumber:
                patterns.append(
                    PathPattern(
                        pattern=[table_name, target_table],
                        semantic=f"Query reports through {table_name}",
                        question_template=f"哪些报告与{table_name}相关？",
                        answer_type="multiple",
                        difficulty="easy",
                    )
                )

        return patterns[:10]  # Limit to 10 fallback patterns
