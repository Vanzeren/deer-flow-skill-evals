"""Memory tools for tool-driven memory mode.

Exposes memory_search, memory_add, memory_update, memory_delete as
LangChain @tool functions the model can call directly.

When memory.mode == "tool", these tools are registered on the agent
instead of appending MemoryMiddleware.  The model gains agency over
its own persistent memory: it decides what to remember, when to
search, and when to update or remove stale facts.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from deerflow.agents.memory.updater import (
    create_memory_fact,
    delete_memory_fact,
    search_memory_facts,
    update_memory_fact,
)
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)


def _resolve_scope() -> tuple[str | None, str]:
    """Resolve agent_name and user_id for tool handler scope.

    agent_name is None (global memory) for v1; per-agent scoping in
    tool mode is deferred. user_id is always the effective user from
    the request context.
    """
    return None, get_effective_user_id()


@tool("memory_search", parse_docstring=True)
def memory_search_tool(
    query: str,
    category: str | None = None,
    limit: int = 10,
) -> str:
    """Search existing facts by natural language query.

    Use this when you need to check what you already know about the user
    — their preferences, past corrections, context, or any stored facts.

    Args:
        query: Natural language query to match against fact content.
            Case-insensitive substring matching.
        category: Optional category filter (e.g. "preference", "correction",
            "context"). Only facts with this exact category are returned.
        limit: Maximum results to return (default 10).

    Returns:
        JSON string with "results" (list of fact objects) and "count".
        Each fact has id, content, category, confidence, createdAt.
    """
    agent_name, user_id = _resolve_scope()
    try:
        results = search_memory_facts(
            query,
            category=category,
            limit=limit,
            agent_name=agent_name,
            user_id=user_id,
        )
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("memory_search_tool failed")
        return json.dumps({"error": str(exc)})


@tool("memory_add", parse_docstring=True)
def memory_add_tool(
    content: str,
    category: str = "context",
    confidence: float = 0.7,
) -> str:
    """Store a new fact about the user or conversation context.

    Use this when the user shares something worth remembering for future
    conversations — preferences, corrections, personal details, work context.
    The fact persists across sessions and will be available via memory_search
    and automatic context injection.

    Args:
        content: The fact text to remember. Be specific and factual.
        category: Category label for organization (default "context").
            e.g. "preference", "correction", "behavior", "personal".
        confidence: How certain you are about this fact, 0.0-1.0
            (default 0.7). Use higher values for explicit user statements,
            lower for inferences.

    Returns:
        JSON string with "fact_id" and "status": "added".
        On duplicate content, returns "error" with explanation.
    """
    agent_name, user_id = _resolve_scope()
    try:
        updated_memory = create_memory_fact(
            content,
            category=category,
            confidence=confidence,
            agent_name=agent_name,
            user_id=user_id,
        )
        # Extract the newly created fact's id (last one in the list)
        facts = updated_memory.get("facts", [])
        fact_id = facts[-1]["id"] if facts else "unknown"
        return json.dumps({"fact_id": fact_id, "status": "added"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_add_tool failed")
        return json.dumps({"error": str(exc)})


@tool("memory_update", parse_docstring=True)
def memory_update_tool(
    fact_id: str,
    content: str | None = None,
    category: str | None = None,
    confidence: float | None = None,
) -> str:
    """Update an existing fact. Only provided fields are changed; omitted
    fields stay as-is.

    Use this when a stored fact is outdated, incorrect, or needs refinement.
    First use memory_search to find the fact_id, then update it.

    Args:
        fact_id: Fact ID from memory_search results (required).
        content: New fact text (unchanged if omitted).
        category: New category (unchanged if omitted).
        confidence: New confidence score 0.0-1.0 (unchanged if omitted).

    Returns:
        JSON string with "fact_id" and "status": "updated".
        On invalid fact_id, returns "error" with explanation.
    """
    agent_name, user_id = _resolve_scope()
    try:
        update_memory_fact(
            fact_id,
            content=content,
            category=category,
            confidence=confidence,
            agent_name=agent_name,
            user_id=user_id,
        )
        return json.dumps({"fact_id": fact_id, "status": "updated"})
    except KeyError:
        return json.dumps({"error": f"Fact not found: {fact_id}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_update_tool failed")
        return json.dumps({"error": str(exc)})


@tool("memory_delete", parse_docstring=True)
def memory_delete_tool(fact_id: str) -> str:
    """Delete a fact by its ID.

    Use this when a fact is no longer accurate or relevant. First use
    memory_search to find the fact_id, then delete it.

    Args:
        fact_id: Fact ID to delete (from memory_search results).

    Returns:
        JSON string with "fact_id" and "status": "deleted".
        On invalid fact_id, returns "error" with explanation.
    """
    agent_name, user_id = _resolve_scope()
    try:
        delete_memory_fact(fact_id, agent_name=agent_name, user_id=user_id)
        return json.dumps({"fact_id": fact_id, "status": "deleted"})
    except KeyError:
        return json.dumps({"error": f"Fact not found: {fact_id}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("memory_delete_tool failed")
        return json.dumps({"error": str(exc)})


def get_memory_tools() -> list:
    """Return all memory tools for agent registration.

    Called by agent factory when memory.mode == "tool".
    """
    return [
        memory_search_tool,
        memory_add_tool,
        memory_update_tool,
        memory_delete_tool,
    ]
