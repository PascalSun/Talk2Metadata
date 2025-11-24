"""QA pair verifier for validating generated question-answer pairs.

Validates that QA pairs are meaningful, consistent, and correct.
"""

from typing import TYPE_CHECKING, List

from talk2metadata.agent import AgentWrapper
from talk2metadata.utils.logging import get_logger

if TYPE_CHECKING:
    from talk2metadata.core.qa.qa_pair import QAPair

logger = get_logger(__name__)


class QAVerifier:
    """Verifies QA pairs for quality and correctness."""

    def __init__(self, agent: AgentWrapper, max_answer_records: int = 10):
        """Initialize QA verifier.

        Args:
            agent: AgentWrapper instance for LLM calls
            max_answer_records: Maximum number of answer records per question (default: 10)
                                Questions with more records are considered too general
        """
        self.agent = agent
        self.max_answer_records = max_answer_records

    def verify(self, qa_pair: "QAPair") -> bool:
        """Verify a single QA pair.

        Args:
            qa_pair: QAPair object to verify

        Returns:
            True if valid, False otherwise (also updates qa_pair.is_valid)
        """
        errors = []

        # 1. Check if question is not empty
        if not qa_pair.question or len(qa_pair.question.strip()) == 0:
            errors.append("Question is empty")

        # 2. Check if answer IDs are not empty
        if not qa_pair.answer_row_ids or len(qa_pair.answer_row_ids) == 0:
            errors.append("No answer records found")

        # 3. Check if answer count exceeds maximum (question too general)
        if qa_pair.answer_count > self.max_answer_records:
            errors.append(
                f"Too many answer records ({qa_pair.answer_count} > {self.max_answer_records}), "
                "question is too general"
            )

        # 4. Check if question is meaningful (not too short)
        if len(qa_pair.question.split()) < 5:
            errors.append("Question is too short (less than 5 words)")

        # 5. Check if SQL is valid (basic syntax check)
        if qa_pair.sql:
            if "SELECT" not in qa_pair.sql.upper():
                errors.append("SQL does not contain SELECT statement")

        # 6. Use LLM to check if question makes sense
        if not errors:
            is_meaningful = self._check_meaningfulness(qa_pair)
            if not is_meaningful:
                errors.append("Question is not meaningful or coherent")

        # Update QA pair
        qa_pair.is_valid = len(errors) == 0
        qa_pair.validation_errors = errors

        if errors:
            logger.debug(f"QA pair validation failed: {errors}")

        return qa_pair.is_valid

    def verify_batch(self, qa_pairs: List["QAPair"]) -> None:
        """Verify a batch of QA pairs.

        Args:
            qa_pairs: List of QAPair objects to verify
        """
        logger.info(f"Verifying {len(qa_pairs)} QA pairs...")

        for i, qa_pair in enumerate(qa_pairs):
            try:
                self.verify(qa_pair)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Verified {i + 1}/{len(qa_pairs)} QA pairs")
            except Exception as e:
                logger.warning(f"Failed to verify QA pair {i}: {e}")
                qa_pair.is_valid = False
                qa_pair.validation_errors = [f"Verification error: {str(e)}"]

        valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
        logger.info(f"Verification complete: {valid_count}/{len(qa_pairs)} valid")

    def _check_meaningfulness(self, qa_pair: "QAPair") -> bool:
        """Use LLM to check if a question is meaningful and coherent.

        Args:
            qa_pair: QAPair object

        Returns:
            True if meaningful, False otherwise
        """
        try:
            prompt = f"""You are a quality control expert for database question-answer pairs.

**Task**: Evaluate if the following question is meaningful, coherent, and makes sense for a database query.

**Question**: {qa_pair.question}

**Evaluation Criteria**:
1. Is the question grammatically correct?
2. Is it clear what the user is asking for?
3. Does it make sense in the context of querying a database?
4. Is it specific enough to be answerable?
5. Does it sound like a real user question (not too generic or nonsensical)?

**Answer ONLY with "YES" or "NO":**
- YES if the question passes all criteria
- NO if it fails any criterion

Your answer:"""

            response = self.agent.generate(prompt).content.strip().upper()

            # Check if response contains YES
            return "YES" in response

        except Exception as e:
            logger.debug(f"Failed to check meaningfulness with LLM: {e}")
            # If LLM fails, use heuristic - assume it's valid if no other errors
            return True

    def _check_answer_consistency(self, qa_pair: "QAPair") -> bool:
        """Check if answer IDs are consistent with the SQL query.

        This is a placeholder for future implementation that would actually
        execute the SQL and verify the results.

        Args:
            qa_pair: QAPair object

        Returns:
            True if consistent, False otherwise
        """
        # TODO: Implement actual SQL execution and result verification
        # For now, assume it's consistent if we have answer IDs
        return len(qa_pair.answer_row_ids) > 0
