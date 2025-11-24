"""QA generation module for automatic question-answer pair generation.

This module generates QA pairs for evaluating retrieval strategies by:
1. Converting relational database to knowledge graph
2. Using LLM to generate meaningful path patterns
3. Instantiating paths from real data
4. Generating natural language questions
5. Extracting answers (target table row IDs)
"""

from talk2metadata.core.qa_generation.generator import QAGenerator
from talk2metadata.core.qa_generation.patterns import PathPattern, PathInstance
from talk2metadata.core.qa_generation.qa_pair import QAPair

__all__ = ["QAGenerator", "PathPattern", "PathInstance", "QAPair"]

