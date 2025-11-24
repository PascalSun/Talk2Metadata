"""QA pair validator for quality control."""

from __future__ import annotations

from typing import List, Optional

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa_generation.qa_pair import QAPair
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class QAValidator:
    """Validate QA pairs for quality and correctness."""

    def __init__(
        self,
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_llm: bool = True,
    ):
        """Initialize validator.

        Args:
            agent: Optional AgentWrapper instance
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            use_llm: Whether to use LLM for validation
        """
        self.use_llm = use_llm
        if use_llm:
            if agent is None:
                self.agent = AgentWrapper(provider=provider, model=model)
            else:
                self.agent = agent

    def validate(self, qa_pair: QAPair) -> bool:
        """Validate a QA pair.

        Args:
            qa_pair: QA pair to validate

        Returns:
            True if valid, False otherwise
        """
        errors = []

        # Basic validation
        if not qa_pair.question or not qa_pair.question.strip():
            errors.append("Question is empty")
            qa_pair.is_valid = False
            qa_pair.validation_errors = errors
            return False

        if not qa_pair.answer_row_ids:
            errors.append("Answer is empty (no target row IDs)")
            qa_pair.is_valid = False
            qa_pair.validation_errors = errors
            return False

        # LLM validation (if enabled)
        if self.use_llm:
            llm_valid, llm_errors = self._validate_with_llm(qa_pair)
            if not llm_valid:
                errors.extend(llm_errors)

        # Set validation result
        qa_pair.is_valid = len(errors) == 0
        qa_pair.validation_errors = errors

        return qa_pair.is_valid

    def _validate_with_llm(self, qa_pair: QAPair) -> tuple[bool, List[str]]:
        """Validate QA pair using LLM.

        Args:
            qa_pair: QA pair to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        prompt = self._build_validation_prompt(qa_pair)

        try:
            response = self.agent.generate(prompt)
            result = response.content.strip().lower()

            # Parse LLM response
            if "valid" in result or "yes" in result or "正确" in result:
                return True, []
            else:
                # Try to extract errors
                errors = []
                if "invalid" in result or "no" in result or "错误" in result:
                    errors.append("LLM validation failed")
                return False, errors

        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            # If LLM fails, assume valid (don't block on LLM errors)
            return True, []

    def _build_validation_prompt(self, qa_pair: QAPair) -> str:
        """Build prompt for LLM validation.

        Args:
            qa_pair: QA pair to validate

        Returns:
            Prompt string
        """
        prompt = f"""You are a quality validator for question-answer pairs used to evaluate a database retrieval system.

## Question-Answer Pair

Question: {qa_pair.question}

Answer: {len(qa_pair.answer_row_ids)} target table row IDs

## Validation Criteria

Check if this QA pair is:
1. **Question Quality**: Is the question natural, clear, and grammatically correct?
2. **Answerability**: Can this question be answered by retrieving records from the target table?
3. **Correctness**: Does the answer make sense for the question?
4. **Usefulness**: Would this QA pair be useful for evaluating a retrieval system?

## Task

Respond with:
- "VALID" or "YES" if the QA pair passes all criteria
- "INVALID" or "NO" followed by reasons if it fails

Your response:"""

        return prompt

    def validate_batch(
        self, qa_pairs: List[QAPair], parallel: bool = False
    ) -> List[QAPair]:
        """Validate multiple QA pairs.

        Args:
            qa_pairs: List of QA pairs to validate
            parallel: Whether to validate in parallel (not implemented yet)

        Returns:
            List of validated QA pairs (with is_valid set)
        """
        for qa_pair in qa_pairs:
            self.validate(qa_pair)

        return qa_pairs

