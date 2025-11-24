"""Question generator for converting paths to natural language questions."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa_generation.patterns import PathInstance
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class QuestionGenerator:
    """Generate natural language questions from path instances."""

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_llm: bool = True,
    ):
        """Initialize question generator.

        Args:
            agent: Optional AgentWrapper instance
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            use_llm: Whether to use LLM for question generation (if False, uses template)
        """
        self.use_llm = use_llm
        if use_llm:
            if agent is None:
                self.agent = AgentWrapper(provider=provider, model=model)
            else:
                self.agent = agent

    def generate(
        self, path_instance: PathInstance, prefer_template: bool = False
    ) -> str:
        """Generate a natural language question from a path instance.

        Args:
            path_instance: Path instance
            prefer_template: If True, prefer template-based generation over LLM

        Returns:
            Natural language question string
        """
        if prefer_template or not self.use_llm:
            return self._generate_from_template(path_instance)
        else:
            return self._generate_with_llm(path_instance)

    def _generate_from_template(self, path_instance: PathInstance) -> str:
        """Generate question using template.

        Args:
            path_instance: Path instance

        Returns:
            Question string
        """
        template = path_instance.pattern.question_template
        if not template:
            # Fallback to simple template
            return self._create_simple_template(path_instance)

        # Extract attribute values from path nodes
        placeholders = self._extract_placeholders(template)
        values = {}

        for placeholder in placeholders:
            # Try to find value in path nodes
            value = self._extract_value_for_placeholder(
                placeholder, path_instance
            )
            if value:
                values[placeholder] = value

        # Fill template
        question = template
        for placeholder, value in values.items():
            question = question.replace(f"{{{placeholder}}}", str(value))

        # Clean up any remaining placeholders
        question = re.sub(r"\{[^}]+\}", "[信息]", question)

        return question

    def _generate_with_llm(self, path_instance: PathInstance) -> str:
        """Generate question using LLM.

        Args:
            path_instance: Path instance

        Returns:
            Question string
        """
        prompt = self._build_question_generation_prompt(path_instance)

        try:
            response = self.agent.generate(prompt)
            question = response.content.strip()

            # Clean up response (remove quotes if present)
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            if question.startswith("'") and question.endswith("'"):
                question = question[1:-1]

            return question
        except Exception as e:
            logger.warning(f"LLM question generation failed: {e}, falling back to template")
            return self._generate_from_template(path_instance)

    def _build_question_generation_prompt(
        self, path_instance: PathInstance
    ) -> str:
        """Build prompt for LLM to generate question.

        Args:
            path_instance: Path instance

        Returns:
            Prompt string
        """
        # Extract key information from path
        path_info = []
        for i, node in enumerate(path_instance.nodes):
            path_info.append(
                f"  {i+1}. {node.table} (row_id: {node.row_id})"
            )

        # Extract key attribute values
        key_values = self._extract_key_values(path_instance)

        prompt = f"""Given a path through a database knowledge graph, generate a natural language question that a user might ask.

## Path Information

Path Pattern: {' -> '.join(path_instance.pattern.pattern)}
Semantic: {path_instance.pattern.semantic}

Path Nodes:
{chr(10).join(path_info)}

Key Attribute Values:
{chr(10).join(f'  - {k}: {v}' for k, v in key_values.items())}

## Task

Generate a natural, conversational question in Chinese that:
1. Reflects the semantic meaning of this path
2. Incorporates the key attribute values naturally
3. Asks about the target table ({path_instance.pattern.pattern[-1]})
4. Is clear and specific enough to be answerable

## Example

If the path is ["historic_titles", "wamex_reports"] and historic_title is "Annual Report 1969-1970", 
the question might be: "哪些报告的标题是'Annual Report 1969-1970'？"

Now generate the question for this path:"""

        return prompt

    def _extract_placeholders(self, template: str) -> List[str]:
        """Extract placeholder names from template.

        Args:
            template: Template string with {{placeholder}} syntax

        Returns:
            List of placeholder names
        """
        pattern = r"\{([^}]+)\}"
        return re.findall(pattern, template)

    def _extract_value_for_placeholder(
        self, placeholder: str, path_instance: PathInstance
    ) -> Optional[str]:
        """Extract value for a placeholder from path instance.

        Args:
            placeholder: Placeholder name
            path_instance: Path instance

        Returns:
            Value string or None
        """
        # Try to find value in node data
        for node in path_instance.nodes:
            # Check column names (case-insensitive)
            for col_name, value in node.data.items():
                col_lower = col_name.lower()
                placeholder_lower = placeholder.lower()

                # Direct match
                if col_lower == placeholder_lower:
                    if value and not pd.isna(value):
                        return str(value)[:100]  # Truncate long values

                # Partial match (e.g., "historic_title" matches "HistoricTitle")
                if placeholder_lower.replace("_", "") in col_lower.replace(
                    "_", ""
                ):
                    if value and not pd.isna(value):
                        return str(value)[:100]

        return None

    def _extract_key_values(
        self, path_instance: PathInstance
    ) -> Dict[str, str]:
        """Extract key attribute values from path instance.

        Args:
            path_instance: Path instance

        Returns:
            Dictionary of key-value pairs
        """
        values = {}

        for node in path_instance.nodes:
            # Extract important columns (non-ID columns)
            for col_name, value in node.data.items():
                if col_name.lower() in ["id", "anumber", "anumber"]:
                    continue

                if value and not pd.isna(value):
                    # Use first non-empty value for each column type
                    col_type = col_name.lower()
                    if col_type not in values:
                        value_str = str(value)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        values[col_type] = value_str

        return values

    def _create_simple_template(self, path_instance: PathInstance) -> str:
        """Create a simple question template when none exists.

        Args:
            path_instance: Path instance

        Returns:
            Simple question string
        """
        pattern = path_instance.pattern.pattern
        if len(pattern) == 2:
            source_table = pattern[0]
            return f"哪些{pattern[1]}与{source_table}相关？"
        else:
            return f"哪些{pattern[-1]}符合以下路径：{' -> '.join(pattern)}？"

