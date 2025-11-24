# QA Generation Module

## Overview

The QA generation module automatically generates question-answer pairs for evaluating retrieval strategies. It converts your relational database into a knowledge graph and uses LLM to generate meaningful query patterns.

## How It Works

1. **Path Pattern Generation**: Uses LLM to analyze your database schema and generate meaningful path patterns
2. **Path Instantiation**: Samples concrete paths from your real data following these patterns
3. **Question Generation**: Converts paths to natural language questions
4. **Answer Extraction**: Extracts target table row IDs as answers
5. **Validation**: Validates QA pairs for quality and correctness

## Configuration

All settings are configured in `config.yml`:

```yaml
# QA generation configuration
qa_generation:
  num_patterns: 15  # Number of path patterns to generate
  instances_per_pattern: 5  # Number of instances per pattern
  validate: true  # Validate QA pairs
  filter_valid: true  # Filter out invalid QA pairs
  auto_save: true  # Auto-save to run directory (qa/kg_paths.json and qa/qa_pairs.json)

# Agent configuration (for LLM-based pattern generation)
agent:
  provider: "anthropic"  # LLM provider
  model: "claude-sonnet-4-5-20250929"  # LLM model
```

## Usage

The QA generation workflow consists of three commands:

### 1. Generate Path Patterns

```bash
# Generate path patterns with settings from config.yml
talk2metadata qa path-generate

# Save to custom location
talk2metadata qa path-generate --output custom_patterns.json
```

### 2. Review Patterns (Optional)

```bash
# Review and edit patterns in web browser
talk2metadata qa review

# Review patterns from custom file
talk2metadata qa review --patterns-file custom_patterns.json
```

### 3. Generate QA Pairs

```bash
# Generate QA pairs from patterns
talk2metadata qa generate

# Use custom patterns file
talk2metadata qa generate --patterns-file custom_patterns.json

# Save to custom location
talk2metadata qa generate --output custom_qa_pairs.json
```

### Complete Workflow

```bash
# Step 1: Generate patterns
talk2metadata qa path-generate

# Step 2: Review and edit (optional)
talk2metadata qa review

# Step 3: Generate QA pairs
talk2metadata qa generate
```

### Command Options

**qa path-generate:**
- `--output, -o`: Output file path (default: `qa/kg_paths.json` in run directory)

**qa review:**
- `--patterns-file, -p`: Path to patterns file (default: `qa/kg_paths.json` in run directory)

**qa generate:**
- `--patterns-file, -p`: Path to patterns file (default: `qa/kg_paths.json` in run directory)
- `--output, -o`: Output file path for QA pairs (default: `qa/qa_pairs.json` in run directory)

All other settings are read from `config.yml`:
- Schema file: Auto-detected from `run_id` and `data.metadata_dir`
- Data directory: From `data.raw_dir`
- Pattern generation: From `qa_generation.*` settings
- LLM provider/model: From `agent.*` settings

## Path Patterns

Path patterns are automatically saved to `data/{run_id}/qa/kg_paths.json` (if `auto_save: true` in config).
This allows you to:
1. Review LLM-generated patterns before generating QA pairs
2. Edit patterns manually if needed
3. Reuse patterns across multiple QA generation runs

### Reviewing Patterns

Use the `qa review` command to open a web-based editor:

```bash
# Review patterns from default location
talk2metadata qa review
```

The web interface allows you to:
- View all path patterns
- Edit pattern details (path, semantic, template, difficulty, etc.)
- Add new patterns
- Delete patterns
- Save changes (automatically overwrites `kg_paths.json`)

### Patterns File Format

The patterns file format:
```json
{
  "target_table": "wamex_reports",
  "total_patterns": 20,
  "patterns": [
    {
      "pattern": ["historic_titles", "wamex_reports"],
      "semantic": "通过历史标题查找报告",
      "question_template": "哪些报告的标题包含'{historic_title}'？",
      "answer_type": "multiple",
      "difficulty": "easy",
      "description": "Simple query through historic titles"
    }
  ]
}
```

### Loading Patterns

Patterns are automatically loaded from `data/{run_id}/qa/kg_paths.json` if they exist.
You can also load from a custom file:

```bash
# Use saved patterns from custom location
talk2metadata qa-generate --load-patterns custom_patterns.json
```

## Output Format

The generated QA pairs are saved as JSON:

```json
{
  "target_table": "wamex_reports",
  "total_qa_pairs": 75,
  "valid_qa_pairs": 72,
  "qa_pairs": [
    {
      "question": "哪些报告的标题包含'Annual Report 1969-1970'？",
      "answer_row_ids": [1, 2, 3],
      "answer_count": 3,
      "difficulty": "easy",
      "is_valid": true,
      "validation_errors": [],
      "metadata": {
        "pattern": ["historic_titles", "wamex_reports"],
        "semantic": "通过历史标题查找报告"
      }
    }
  ]
}
```

## Python API

You can also use the QA generation module programmatically:

```python
from talk2metadata.core.qa_generation import QAGenerator
from talk2metadata.core.schema import SchemaMetadata
import pandas as pd

# Load schema and tables
schema = SchemaMetadata.load("data/metadata/schema.json")
tables = {
    "wamex_reports": pd.read_csv("data/wamex/wamex_reports.csv"),
    "historic_titles": pd.read_csv("data/wamex/historic_titles.csv"),
    # ... other tables
}

# Create generator
generator = QAGenerator(schema=schema, tables=tables)

# Generate QA pairs
qa_pairs = generator.generate(
    num_patterns=15,
    instances_per_pattern=5,
    validate=True,
    filter_valid=True
)

# Save results
generator.save(qa_pairs, "qa_pairs.json")
```

## Architecture

The module consists of several components:

- **PathPatternGenerator**: Generates path patterns using LLM
- **PathInstantiator**: Instantiates paths from real data
- **QuestionGenerator**: Converts paths to natural language
- **AnswerExtractor**: Extracts target table row IDs
- **QAValidator**: Validates QA pairs for quality
- **QAGenerator**: Main class coordinating all components

## Tips

1. **More patterns = more diversity**: Increase `--num-patterns` for more diverse query types
2. **More instances = more QA pairs**: Increase `--instances-per-pattern` for larger datasets
3. **Validation improves quality**: Keep validation enabled unless you need faster generation
4. **LLM configuration**: Make sure your LLM provider is configured in `config.yml`

## Limitations

- Requires LLM access for pattern generation and question rewriting
- Path instantiation depends on data quality and FK relationships
- Generated questions may need manual review for domain-specific accuracy

