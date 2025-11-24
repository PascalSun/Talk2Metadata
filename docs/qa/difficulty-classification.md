# QA Difficulty Classification System

## Overview

This document defines the difficulty classification system for question-answer (QA) pair generation in Talk2Metadata. The system is designed specifically for **record localization tasks**, where the goal is to locate records in a target table by filtering through related tables via JOIN operations.

### Design Goals

1. **Quantifiable**: Each difficulty level corresponds to a clear numeric score
2. **Intuitive**: Encoding format is human-readable and self-explanatory
3. **Comprehensive**: Covers simple to expert-level queries
4. **Theoretically Grounded**: Based on Query Graph frameworks from Knowledge Graph research
5. **Practical**: Directly applicable to relational database schema and SQL generation

---

## Format Specification

### Encoding Format

```
{Pattern}{Difficulty}
```

- **Pattern**: Describes the JOIN structure (e.g., `0`, `1p`, `2i`)
- **Difficulty**: Describes the filter complexity (e.g., `E`, `M`, `H`)

### Examples

- `0E`: Direct query, Easy
- `1pM`: Single-hop path, Medium
- `2iE`: Two-way intersection, Easy
- `2iH`: Two-way intersection, Hard
- `3iM`: Three-way intersection, Medium

---

## Pattern Types

Pattern types describe the **JOIN structure** of the query, inspired by Query Graph notation from Knowledge Graph QA research.

### Pattern Notation

| Code | Name                   | Description                      | JOIN Type | Example                                                         |
| ---- | ---------------------- | -------------------------------- | --------- | --------------------------------------------------------------- |
| `0`  | Direct                 | No JOIN, query only target table | None      | Find orders with status='completed'                             |
| `1p` | Single-hop Path        | 1 JOIN in a chain                | Chain     | Find orders from Healthcare customers                           |
| `2p` | Two-hop Path           | 2 JOINs in a chain               | Chain     | Find orders from US-West region customers                       |
| `3p` | Three-hop Path         | 3 JOINs in a chain               | Chain     | Orders → Customers → Regions → Countries                        |
| `2i` | Two-way Intersection   | 2 JOINs in a star                | Star      | Find orders: Healthcare customers × Software products           |
| `3i` | Three-way Intersection | 3 JOINs in a star                | Star      | Orders: Healthcare customers × Software products × Senior sales |
| `4i` | Four-way Intersection  | 4 JOINs in a star                | Star      | Orders with 4 independent filter dimensions                     |
| `Xm` | Mixed                  | Complex combination              | Mixed     | Chain + Star combinations                                       |

### Path vs Intersection

#### Path Queries (`Xp`)

**Structure**: Chain/Sequential JOINs
```
Target → Table1 → Table2 → ... → TableN
```

**Characteristics**:

- **Transitive reasoning** required
- Each table depends on the previous one
- Example schema: `Orders → Customers → Regions`

**SQL Pattern**:

```sql
SELECT o.*
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN regions r ON c.region_id = r.id
WHERE r.name = 'US-West'
```

**Cognitive Complexity**:

- Understanding indirect relationships
- Multi-step inference
- Harder to trace the reasoning path

---

#### Intersection Queries (`Xi`)

**Structure**: Star/Parallel JOINs
```
      Table1
        ↓
    Target (center)
        ↓
      Table2
```

**Characteristics**:

- **Conjunction reasoning** required
- Independent filter conditions
- Example schema: `Orders` connects to both `Customers` and `Products`

**SQL Pattern**:

```sql
SELECT o.*
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE c.industry = 'Healthcare'
  AND p.category = 'Software'
```

**Cognitive Complexity**:

- Understanding multi-dimensional constraints
- Combining independent conditions
- Requires awareness of multiple relationships simultaneously

---

### Pattern Hierarchy

```
Easy:    0
         ↓
Medium:  1p
         ↓
Hard:    2p, 2i
         ↓
Expert:  3p, 3i, 4i, Xm
```

**Note**: At the same hop count, chain (`Xp`) and star (`Xi`) have similar base difficulty, but chain queries typically require slightly more complex transitive reasoning.

---

## Difficulty Levels

Difficulty levels describe the **filter complexity** based on the number of distinct columns used in filter conditions.

### Difficulty Notation

| Code | Name   | Column Count | Description                                        |
| ---- | ------ | ------------ | -------------------------------------------------- |
| `E`  | Easy   | 1-2 columns  | Simple, single-condition or dual-condition filters |
| `M`  | Medium | 3-5 columns  | Multiple filter conditions across tables           |
| `H`  | Hard   | 6+ columns   | Complex multi-dimensional filtering                |

### Column Counting Rules

**Count distinct columns used in WHERE conditions** across all tables involved:

#### Example 1: `1pE`

```sql
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.industry = 'Healthcare'
```
- Columns: `c.industry` (1 column) → **E** (Easy)

#### Example 2: `1pM`

```sql
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.industry = 'Healthcare'
  AND c.annual_revenue > 1000000
  AND o.amount > 10000
```
- Columns: `c.industry`, `c.annual_revenue`, `o.amount` (3 columns) → **M** (Medium)

#### Example 3: `2iH`

```sql
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE c.industry = 'Healthcare'
  AND c.region = 'US-West'
  AND c.annual_revenue > 1000000
  AND p.category = 'Software'
  AND p.price > 10000
  AND o.amount > 50000
```
- Columns: 6 distinct columns → **H** (Hard)

---

## Complete Difficulty Matrix

### Difficulty Codes and Scores

| Code    | Pattern    | Difficulty | Score | Tier   | Description                 |
| ------- | ---------- | ---------- | ----- | ------ | --------------------------- |
| **0E**  | Direct     | Easy       | 0.0   | Easy   | Direct filter, 1-2 columns  |
| **0M**  | Direct     | Medium     | 0.3   | Easy   | Direct filter, 3-5 columns  |
| **0H**  | Direct     | Hard       | 0.6   | Easy   | Direct filter, 6+ columns   |
| **1pE** | 1-hop Path | Easy       | 1.0   | Medium | Single JOIN, simple filter  |
| **1pM** | 1-hop Path | Medium     | 1.3   | Medium | Single JOIN, medium filter  |
| **1pH** | 1-hop Path | Hard       | 1.6   | Medium | Single JOIN, complex filter |
| **2pE** | 2-hop Path | Easy       | 2.0   | Hard   | 2-hop chain, simple         |
| **2pM** | 2-hop Path | Medium     | 2.3   | Hard   | 2-hop chain, medium         |
| **2pH** | 2-hop Path | Hard       | 2.6   | Hard   | 2-hop chain, complex        |
| **2iE** | 2-way Star | Easy       | 2.0   | Hard   | 2-way intersection, simple  |
| **2iM** | 2-way Star | Medium     | 2.3   | Hard   | 2-way intersection, medium  |
| **2iH** | 2-way Star | Hard       | 2.6   | Hard   | 2-way intersection, complex |
| **3pE** | 3-hop Path | Easy       | 3.0   | Expert | 3-hop chain, simple         |
| **3pM** | 3-hop Path | Medium     | 3.3   | Expert | 3-hop chain, medium         |
| **3pH** | 3-hop Path | Hard       | 3.6   | Expert | 3-hop chain, complex        |
| **3iE** | 3-way Star | Easy       | 3.0   | Expert | 3-way intersection, simple  |
| **3iM** | 3-way Star | Medium     | 3.3   | Expert | 3-way intersection, medium  |
| **3iH** | 3-way Star | Hard       | 3.6   | Expert | 3-way intersection, complex |
| **4iE** | 4-way Star | Easy       | 4.0   | Expert | 4-way intersection, simple  |
| **4iM** | 4-way Star | Medium     | 4.3   | Expert | 4-way intersection, medium  |
| **4iH** | 4-way Star | Hard       | 4.6   | Expert | 4-way intersection, complex |

### Tier Boundaries

- **Easy Tier**: Score 0.0 - 0.9 (0E, 0M, 0H)
- **Medium Tier**: Score 1.0 - 1.9 (1pE, 1pM, 1pH)
- **Hard Tier**: Score 2.0 - 2.9 (2pE, 2pM, 2pH, 2iE, 2iM, 2iH)
- **Expert Tier**: Score 3.0+ (3p*, 3i*, 4i*, Xm*)

---

## Concrete Examples

### Schema Context

```
customers (10 rows)
├─ id, name, industry, region, annual_revenue, created_date

orders (20 rows) [TARGET TABLE]
├─ id, customer_id (FK), product_id (FK), amount, quantity,
│  order_date, status, sales_rep_id (FK)

products (10 rows)
├─ id, name, category, price, description

sales_reps (5 rows)
├─ id, name, seniority, region

regions (4 rows)
├─ id, name, country, tax_rate
```

### Example Questions by Difficulty

#### Easy Tier (0.0 - 0.9)

**0E** (Score: 0.0)

- Question: "Find all completed orders"
- Filter: `status = 'completed'`
- SQL: `SELECT * FROM orders WHERE status = 'completed'`

**0M** (Score: 0.3)

- Question: "Find completed orders with amount greater than $10,000"
- Filters: `status`, `amount` (2 columns)
- SQL: `SELECT * FROM orders WHERE status = 'completed' AND amount > 10000`

**0H** (Score: 0.6)

- Question: "Find completed orders with amount > $10,000, quantity > 5, placed after 2024-01-01"
- Filters: `status`, `amount`, `quantity`, `order_date` (4 columns)

---

#### Medium Tier (1.0 - 1.9)

**1pE** (Score: 1.0)

- Question: "Find all orders from Healthcare industry customers"
- Pattern: Orders → Customers
- Filters: `customers.industry` (1 column)
- SQL:
  ```sql
  SELECT o.* FROM orders o
  JOIN customers c ON o.customer_id = c.id
  WHERE c.industry = 'Healthcare'
  ```

**1pM** (Score: 1.3)

- Question: "Find orders from Healthcare customers with annual revenue > $1M"
- Pattern: Orders → Customers
- Filters: `customers.industry`, `customers.annual_revenue` (2 columns)
- SQL:
  ```sql
  SELECT o.* FROM orders o
  JOIN customers c ON o.customer_id = c.id
  WHERE c.industry = 'Healthcare'
    AND c.annual_revenue > 1000000
  ```

**1pH** (Score: 1.6)

- Question: "Find high-value orders from US Healthcare customers with revenue > $1M created after 2020"
- Pattern: Orders → Customers
- Filters: `customers.industry`, `customers.region`, `customers.annual_revenue`,
           `customers.created_date`, `orders.amount` (5 columns)

---

#### Hard Tier (2.0 - 2.9)

**2pE** (Score: 2.0)

- Question: "Find all orders from US-West region customers"
- Pattern: Orders → Customers → Regions (chain)
- Filters: `regions.name` (1 column)
- SQL:
  ```sql
  SELECT o.* FROM orders o
  JOIN customers c ON o.customer_id = c.id
  JOIN regions r ON c.region_id = r.id
  WHERE r.name = 'US-West'
  ```

**2iE** (Score: 2.0)

- Question: "Find orders from Healthcare customers buying Software products"
- Pattern: Orders ← Customers, Orders ← Products (star)
- Filters: `customers.industry`, `products.category` (2 columns)
- SQL:
  ```sql
  SELECT o.* FROM orders o
  JOIN customers c ON o.customer_id = c.id
  JOIN products p ON o.product_id = p.id
  WHERE c.industry = 'Healthcare'
    AND p.category = 'Software'
  ```

**2iM** (Score: 2.3)

- Question: "Find orders from US Healthcare customers buying Enterprise Software"
- Pattern: Orders ← Customers, Orders ← Products (star)
- Filters: `customers.industry`, `customers.region`,
           `products.category`, `products.name` (4 columns)

**2iH** (Score: 2.6)

- Question: "Find high-value orders from US Healthcare customers with revenue > $1M buying expensive Enterprise Software"
- Pattern: Star with 2 foreign tables
- Filters: 6+ columns across customers, products, and orders

---

#### Expert Tier (3.0+)

**3iE** (Score: 3.0)

- Question: "Find orders from Healthcare customers buying Software, handled by Senior sales reps"
- Pattern: Orders ← Customers, Orders ← Products, Orders ← Sales_Reps (3-way star)
- Filters: `customers.industry`, `products.category`, `sales_reps.seniority` (3 columns)
- SQL:
  ```sql
  SELECT o.* FROM orders o
  JOIN customers c ON o.customer_id = c.id
  JOIN products p ON o.product_id = p.id
  JOIN sales_reps s ON o.sales_rep_id = s.id
  WHERE c.industry = 'Healthcare'
    AND p.category = 'Software'
    AND s.seniority = 'Senior'
  ```

**3pE** (Score: 3.0)

- Question: "Find orders from customers in California, USA"
- Pattern: Orders → Customers → Regions → Countries (3-hop chain)
- Filters: `countries.name`, `regions.name` (2 columns)
- Requires: Extended schema with country-level data

---

## Implementation Guide

### Difficulty Calculator Class

```python
from dataclasses import dataclass
from typing import List, Set
from enum import Enum
import math

class PatternType(Enum):
    DIRECT = "0"
    PATH_1 = "1p"
    PATH_2 = "2p"
    PATH_3 = "3p"
    INTERSECTION_2 = "2i"
    INTERSECTION_3 = "3i"
    INTERSECTION_4 = "4i"
    MIXED = "Xm"

class DifficultyLevel(Enum):
    EASY = "E"
    MEDIUM = "M"
    HARD = "H"

@dataclass
class QueryPlan:
    """Represents a parsed query plan"""
    target_table: str
    target_columns: Set[str]
    join_paths: List['JoinPath']
    filter_columns: Set[str]  # All columns used in WHERE clause

@dataclass
class JoinPath:
    """Represents a JOIN path"""
    tables: List[str]
    join_type: str  # 'chain' or 'star'

class DifficultyClassifier:

    def classify(self, query_plan: QueryPlan) -> str:
        """
        Classify a query plan into difficulty code

        Returns:
            Difficulty code (e.g., "2iM")
        """
        pattern = self._identify_pattern(query_plan)
        difficulty = self._assess_difficulty(query_plan)

        return f"{pattern.value}{difficulty.value}"

    def _identify_pattern(self, query_plan: QueryPlan) -> PatternType:
        """Identify the pattern type"""
        num_joins = len(query_plan.join_paths)

        if num_joins == 0:
            return PatternType.DIRECT

        # Check if all JOINs are direct to target (star pattern)
        is_star = all(
            len(path.tables) == 2 for path in query_plan.join_paths
        )

        if is_star:
            # Star/Intersection pattern
            if num_joins == 1:
                return PatternType.PATH_1  # Actually still a path
            elif num_joins == 2:
                return PatternType.INTERSECTION_2
            elif num_joins == 3:
                return PatternType.INTERSECTION_3
            elif num_joins >= 4:
                return PatternType.INTERSECTION_4
        else:
            # Path/Chain pattern
            max_depth = max(len(path.tables) for path in query_plan.join_paths)
            if max_depth == 2:
                return PatternType.PATH_1
            elif max_depth == 3:
                return PatternType.PATH_2
            elif max_depth >= 4:
                return PatternType.PATH_3

        return PatternType.MIXED

    def _assess_difficulty(self, query_plan: QueryPlan) -> DifficultyLevel:
        """Assess difficulty based on filter complexity"""
        num_filter_columns = len(query_plan.filter_columns)

        if num_filter_columns <= 2:
            return DifficultyLevel.EASY
        elif num_filter_columns <= 5:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD

    def get_score(self, difficulty_code: str) -> float:
        """Convert difficulty code to numeric score"""
        # Parse pattern and difficulty
        if difficulty_code[0].isdigit():
            if len(difficulty_code) >= 2 and difficulty_code[1] in ['p', 'i']:
                pattern_str = difficulty_code[:2]
                diff_str = difficulty_code[2:]
            else:
                pattern_str = difficulty_code[0]
                diff_str = difficulty_code[1:]
        else:
            pattern_str = difficulty_code[:2]
            diff_str = difficulty_code[2:]

        # Pattern base scores
        pattern_scores = {
            "0": 0.0,
            "1p": 1.0,
            "2p": 2.0,
            "2i": 2.0,
            "3p": 3.0,
            "3i": 3.0,
            "4i": 4.0,
            "Xm": 5.0,
        }

        # Difficulty modifiers
        difficulty_modifiers = {
            "E": 0.0,
            "M": 0.3,
            "H": 0.6,
        }

        base = pattern_scores.get(pattern_str, 0.0)
        modifier = difficulty_modifiers.get(diff_str, 0.0)

        return base + modifier
```

### Usage Example

```python
classifier = DifficultyClassifier()

# Example query plan
query_plan = QueryPlan(
    target_table="orders",
    target_columns={"id", "amount", "status"},
    join_paths=[
        JoinPath(tables=["orders", "customers"], join_type="star"),
        JoinPath(tables=["orders", "products"], join_type="star")
    ],
    filter_columns={"customers.industry", "products.category"}
)

difficulty_code = classifier.classify(query_plan)
print(difficulty_code)  # Output: "2iE"

score = classifier.get_score(difficulty_code)
print(score)  # Output: 2.0
```

---

## Dataset Distribution Recommendations

### Balanced Distribution

For a balanced QA dataset suitable for training and evaluation:

| Tier       | Target % | Difficulty Codes                                 |
| ---------- | -------- | ------------------------------------------------ |
| **Easy**   | 30%      | 0E (15%), 0M (10%), 0H (5%)                      |
| **Medium** | 50%      | 1pE (20%), 1pM (20%), 1pH (10%)                  |
| **Hard**   | 15%      | 2pE (3%), 2pM (3%), 2iE (3%), 2iM (3%), 2iH (3%) |
| **Expert** | 5%       | 3pE (2%), 3iE (2%), 3iM (1%)                     |

### Progressive Distribution

For training models with curriculum learning:

**Phase 1 (Weeks 1-2)**: 80% Easy, 20% Medium
**Phase 2 (Weeks 3-4)**: 40% Easy, 50% Medium, 10% Hard
**Phase 3 (Weeks 5+)**: 20% Easy, 40% Medium, 30% Hard, 10% Expert

---

## Theoretical Foundation

### Query Graph Framework

This classification system is based on **Query Graph** frameworks from Knowledge Graph QA research, particularly:

1. **Path Queries**: Sequential edge traversal in knowledge graphs
   - Adapted to: Chain JOINs in relational databases
   - Notation: `1p`, `2p`, `3p`

2. **Intersection Queries**: Multiple edges converging at a node
   - Adapted to: Star schema JOINs
   - Notation: `2i`, `3i`, `4i`

3. **Union/Negation Queries**: (Not currently used)
   - Could be extended for `UNION`, `EXCEPT`, `NOT EXISTS` patterns

### Comparison to SQL Complexity Metrics

Traditional SQL complexity metrics (e.g., Spider, WikiSQL) focus on:

- Number of SELECT clauses
- Aggregation functions (COUNT, SUM, AVG)
- GROUP BY, HAVING, ORDER BY
- Subqueries and nested queries
- Window functions

**Our system differs** by:

- ✅ Focusing on **record localization** rather than complex SQL operations
- ✅ Emphasizing **JOIN structure** (graph topology)
- ✅ Separating **structural complexity** (pattern) from **filter complexity** (difficulty)
- ✅ Ignoring aggregation and sorting (not relevant for record retrieval)

### Why This Matters for Talk2Metadata

Talk2Metadata is a **semantic search system** for database records. Users want to:

- "Find orders from Healthcare customers" (1pE)
- "Find orders matching specific product and customer criteria" (2iE)

They typically **don't** need:

- Complex aggregations ("What is the average order value by region?")
- Window functions ("Rank customers by revenue")
- Subqueries ("Find customers with above-average orders")

Our classification aligns with this **retrieval-focused** use case.

---

## Extensions and Future Work

### Potential Extensions

1. **Mixed Patterns** (`Xm`)
   - Combination of chain + star (e.g., 2p + 2i)
   - Scoring formula: `base = max_hop_count + num_intersections * 0.5`

2. **Negation Patterns** (`Xn`)
   - Queries with `NOT EXISTS`, `EXCEPT`
   - Example: "Find customers who never bought Software products"

3. **Temporal Patterns** (`Xt`)
   - Time-based filtering requiring temporal reasoning
   - Example: "Find customers active in Q1 but not Q2"

4. **Aggregation Overlay** (Optional suffix)
   - `2iE-agg`: 2-way intersection with aggregation
   - Only if aggregation becomes important for your use case

### Schema-Specific Calibration

For specific database schemas, you may want to:

1. **Adjust column count thresholds** based on your typical table widths
   - Wide tables (50+ columns): E=1-3, M=4-8, H=9+
   - Narrow tables (<10 columns): E=1-2, M=3-4, H=5+

2. **Weight certain tables** as inherently more complex
   - Dimension tables (customers, products): Lower weight
   - Fact tables with many FKs: Higher weight

3. **Consider domain complexity**
   - Technical/medical domains: Adjust upward
   - Simple e-commerce: Keep as-is

---

## FAQ

### Q: Why distinguish 2p from 2i?

**A**: They represent fundamentally different reasoning patterns:

- **2p (chain)**: Requires transitive reasoning (A→B→C)
- **2i (star)**: Requires parallel filtering (A∧B from center)

Both are "hard" (score 2.0+), but the cognitive processes differ.

### Q: What if a query has both chain AND star?

**A**: Use the `Xm` (mixed) pattern or choose the dominant pattern:

- If primarily chain: Use `Xp`
- If primarily star: Use `Xi`
- If truly complex: `Xm` with custom scoring

### Q: Can I add intermediate levels like `VH` (Very Hard)?

**A**: Yes, but keep it simple. If needed:

- `E`: 1-2 columns
- `M`: 3-4 columns
- `H`: 5-6 columns
- `VH`: 7+ columns

Update the scoring formula accordingly (+0.2, +0.4, +0.6, +0.8).

### Q: How to handle self-joins?

**A**: Treat as an additional hop:

- Self-join on same table = +1 hop
- Example: "Find employees managed by senior managers" (employee → employee)

### Q: What about optional JOINs (LEFT JOIN)?

**A**: Same classification as INNER JOIN:

- Pattern is determined by structure, not JOIN type
- Optionality affects result cardinality, not reasoning complexity

---

## References

### Academic Papers

1. **Query2box: Reasoning over Knowledge Graphs using Box Embeddings**
   - Ren et al., ICLR 2020
   - Introduced path/intersection query notation

2. **Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**
   - Yu et al., EMNLP 2018
   - SQL complexity metrics for text-to-SQL

3. **KaggleDBQA: Realistic Large-Scale Database Question Answering**
   - Lee et al., NeurIPS 2021
   - Real-world database QA benchmarks

### Related Systems

- **Talk2Metadata**: Semantic search for relational databases (this project)
- **WikiSQL**: Simple table QA (single table focus)
- **Spider**: Complex SQL generation (aggregation/subquery focus)
- **KGQA**: Knowledge Graph Question Answering (graph reasoning focus)

---

## Changelog

### Version 1.0 (2025-11-24)

- Initial difficulty classification system
- Pattern types: 0, 1p, 2p, 2i, 3p, 3i, 4i
- Difficulty levels: E, M, H
- Scoring formula: base + modifier
- Complete documentation and examples

---

## License

This specification is part of the Talk2Metadata project.

---

## Contact

For questions or suggestions regarding this classification system, please open an issue in the Talk2Metadata repository.
