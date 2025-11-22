"""Search guide prompt for Talk2Metadata."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent


async def get_search_guide_prompt() -> GetPromptResult:
    """Return guidance on effective searching in Talk2Metadata."""
    return GetPromptResult(
        description="Guide for effective searching with Talk2Metadata",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "# Effective Searching with Talk2Metadata\n\n"
                        "## Search Modes\n\n"
                        "### Semantic Search (Default)\n"
                        "- Uses embeddings to understand meaning\n"
                        "- Great for conceptual queries\n"
                        "- Handles synonyms and related terms\n"
                        "- Example: 'VIP customers' finds high-value, premium, important customers\n\n"
                        "### Hybrid Search (Recommended)\n"
                        "- Combines BM25 (keyword) + semantic search\n"
                        "- Best overall results\n"
                        "- Good for both exact matches and conceptual search\n"
                        "- Set `hybrid=true` in search tool\n\n"
                        "## Query Tips\n\n"
                        "### Good Queries\n"
                        "✓ 'customers in healthcare industry'\n"
                        "✓ 'recent high-value orders'\n"
                        "✓ 'products frequently purchased together'\n"
                        "✓ 'employees with technical skills'\n\n"
                        "### Less Effective Queries\n"
                        "✗ Single keywords: 'healthcare'\n"
                        "✗ Too vague: 'find data'\n"
                        "✗ Too specific: 'customer_id = 12345' (use direct lookup)\n\n"
                        "## Understanding Results\n\n"
                        "Each result includes:\n"
                        "- `rank`: Position in results (1 = best match)\n"
                        "- `score`: Relevance score (higher = better match)\n"
                        "- `table`: Source table name\n"
                        "- `row_id`: Unique record identifier\n"
                        "- `data`: Complete record data\n\n"
                        "For hybrid search:\n"
                        "- `bm25_score`: Keyword matching score\n"
                        "- `semantic_score`: Embedding similarity score\n"
                        "- Final score combines both\n\n"
                        "## Best Practices\n\n"
                        "1. **Start broad, then refine**\n"
                        "   - Initial query: 'customers'\n"
                        "   - Refined: 'enterprise customers with active subscriptions'\n\n"
                        "2. **Use domain terminology**\n"
                        "   - Queries work better with terms from your data\n"
                        "   - Check sample_values in schema to see actual terms used\n\n"
                        "3. **Adjust top_k based on needs**\n"
                        "   - top_k=5: Quick overview\n"
                        "   - top_k=20: Comprehensive results\n"
                        "   - top_k=50: Exhaustive search\n\n"
                        "4. **Leverage foreign keys**\n"
                        "   - Query mentions related data (e.g., 'orders with premium customers')\n"
                        "   - System uses FKs to enrich results\n\n"
                        "## Troubleshooting\n\n"
                        "**No results?**\n"
                        "- Try broader terms\n"
                        "- Use hybrid search\n"
                        "- Increase top_k\n"
                        "- Check available tables with list_tables\n\n"
                        "**Irrelevant results?**\n"
                        "- Make query more specific\n"
                        "- Use exact terms from sample values\n"
                        "- Reduce top_k to see only best matches\n\n"
                        "**Mixed quality results?**\n"
                        "- Enable hybrid search\n"
                        "- Results are ranked by relevance - top results are best\n"
                    ),
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "search_guide",
    "description": "Learn how to effectively search and query data in Talk2Metadata",
}
