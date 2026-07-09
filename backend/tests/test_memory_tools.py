"""Tests for memory tool functions (tool-driven memory mode)."""

import json

from deerflow.agents.memory.tools import (
    get_memory_tools,
    memory_add_tool,
    memory_delete_tool,
    memory_search_tool,
    memory_update_tool,
)


class TestGetMemoryTools:
    """Tests for get_memory_tools registry."""

    def test_returns_four_tools(self):
        """Should return exactly 4 tools."""
        tools = get_memory_tools()
        assert len(tools) == 4

    def test_tools_have_unique_names(self):
        """All tools should have unique names."""
        tools = get_memory_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names))
        assert "memory_search" in names
        assert "memory_add" in names
        assert "memory_update" in names
        assert "memory_delete" in names


class TestMemorySearchTool:
    """Tests for memory_search tool handler."""

    def test_returns_json_with_results(self, monkeypatch):
        """Should return JSON with results and count."""
        mock_results = [
            {"id": "fact_abc123", "content": "User likes Python", "category": "preference", "confidence": 0.9, "createdAt": "2026-01-01T00:00:00Z"},
        ]

        def mock_search(query, category=None, limit=10, *, agent_name=None, user_id=None):
            return mock_results

        monkeypatch.setattr(
            "deerflow.agents.memory.tools.search_memory_facts",
            mock_search,
        )
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_search_tool.invoke({"query": "Python"})
        result = json.loads(result_json)
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "fact_abc123"

    def test_empty_results(self, monkeypatch):
        """Should return empty results for no matches."""
        monkeypatch.setattr(
            "deerflow.agents.memory.tools.search_memory_facts",
            lambda *a, **kw: [],
        )
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_search_tool.invoke({"query": "nothing"})
        result = json.loads(result_json)
        assert result["count"] == 0
        assert result["results"] == []

    def test_runtime_error_returns_error_json(self, monkeypatch):
        """Should return error JSON when search raises RuntimeError."""
        monkeypatch.setattr(
            "deerflow.agents.memory.tools.search_memory_facts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_search_tool.invoke({"query": "anything"})
        result = json.loads(result_json)
        assert "error" in result
        assert result["error"] == "boom"


class TestMemoryAddTool:
    """Tests for memory_add tool handler."""

    def test_adds_fact_and_returns_json(self, monkeypatch):
        """Should add a fact and return fact_id + status."""
        mock_memory = {"facts": [{"id": "fact_new123", "content": "User prefers dark mode"}]}

        def mock_create(content, category="context", confidence=0.5, agent_name=None, *, user_id=None):
            return mock_memory

        monkeypatch.setattr("deerflow.agents.memory.tools.create_memory_fact", mock_create)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_add_tool.invoke({"content": "User prefers dark mode", "category": "preference", "confidence": 0.9})
        result = json.loads(result_json)
        assert result["status"] == "added"
        assert "fact_id" in result

        # fact_id is derived from the last fact in the returned memory
        assert result["fact_id"] == "fact_new123"

    def test_duplicate_content_returns_error(self, monkeypatch):
        """Should return error JSON for duplicate content."""

        def mock_create(*a, **kw):
            raise ValueError("Duplicate fact")

        monkeypatch.setattr("deerflow.agents.memory.tools.create_memory_fact", mock_create)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_add_tool.invoke({"content": "duplicate"})
        result = json.loads(result_json)
        assert "error" in result

    def test_empty_content_returns_error(self, monkeypatch):
        """Should return error JSON for empty content."""

        def mock_create(*a, **kw):
            raise ValueError("content")

        monkeypatch.setattr("deerflow.agents.memory.tools.create_memory_fact", mock_create)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_add_tool.invoke({"content": ""})
        result = json.loads(result_json)
        assert "error" in result


class TestMemoryUpdateTool:
    """Tests for memory_update tool handler."""

    def test_updates_fact_and_returns_json(self, monkeypatch):
        """Should update a fact and return JSON."""
        mock_memory = {"facts": [{"id": "fact_abc", "content": "updated content"}]}

        def mock_update(fact_id, content=None, category=None, confidence=None, agent_name=None, *, user_id=None):
            return mock_memory

        monkeypatch.setattr("deerflow.agents.memory.tools.update_memory_fact", mock_update)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_update_tool.invoke({"fact_id": "fact_abc", "content": "updated content"})
        result = json.loads(result_json)
        assert result["status"] == "updated"
        assert result["fact_id"] == "fact_abc"

    def test_invalid_fact_id_returns_error(self, monkeypatch):
        """Should return error JSON for invalid fact_id."""

        def mock_update(*a, **kw):
            raise KeyError("fact_xxx")

        monkeypatch.setattr("deerflow.agents.memory.tools.update_memory_fact", mock_update)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_update_tool.invoke({"fact_id": "fact_xxx", "content": "nope"})
        result = json.loads(result_json)
        assert "error" in result
        assert "fact_xxx" in result["error"]


class TestMemoryDeleteTool:
    """Tests for memory_delete tool handler."""

    def test_deletes_fact_and_returns_json(self, monkeypatch):
        """Should delete a fact and return JSON."""
        mock_memory = {"facts": []}

        def mock_delete(fact_id, agent_name=None, *, user_id=None):
            return mock_memory

        monkeypatch.setattr("deerflow.agents.memory.tools.delete_memory_fact", mock_delete)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_delete_tool.invoke({"fact_id": "fact_abc"})
        result = json.loads(result_json)
        assert result["status"] == "deleted"
        assert result["fact_id"] == "fact_abc"

    def test_invalid_fact_id_returns_error(self, monkeypatch):
        """Should return error JSON for invalid fact_id."""

        def mock_delete(*a, **kw):
            raise KeyError("fact_xxx")

        monkeypatch.setattr("deerflow.agents.memory.tools.delete_memory_fact", mock_delete)
        monkeypatch.setattr("deerflow.agents.memory.tools.get_effective_user_id", lambda: "test-user")

        result_json = memory_delete_tool.invoke({"fact_id": "fact_xxx"})
        result = json.loads(result_json)
        assert "error" in result
        assert "fact_xxx" in result["error"]


class TestModeGating:
    """Integration tests for memory.mode exclusivity."""

    def test_tool_mode_registers_tools_not_middleware(self, monkeypatch):
        """When mode=tool, get_memory_tools are added to extra_tools and
        MemoryMiddleware is NOT in the chain."""
        from deerflow.agents.factory import _assemble_from_features
        from deerflow.agents.features import RuntimeFeatures
        from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
        from deerflow.config.memory_config import MemoryConfig

        tool_config = MemoryConfig(enabled=True, mode="tool")
        monkeypatch.setattr(
            "deerflow.config.memory_config.get_memory_config",
            lambda: tool_config,
        )

        feat = RuntimeFeatures(memory=True)
        chain, extra_tools = _assemble_from_features(feat, name="test-agent")

        middleware_types = [type(m) for m in chain]
        assert MemoryMiddleware not in middleware_types, "MemoryMiddleware should not be in the chain in tool mode"

        tool_names = [t.name for t in extra_tools]
        assert "memory_search" in tool_names
        assert "memory_add" in tool_names
        assert "memory_update" in tool_names
        assert "memory_delete" in tool_names

    def test_middleware_mode_appends_middleware_not_tools(self, monkeypatch):
        """When mode=middleware (default), MemoryMiddleware IS in the chain
        and memory tools are NOT in extra_tools."""
        from deerflow.agents.factory import _assemble_from_features
        from deerflow.agents.features import RuntimeFeatures
        from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
        from deerflow.config.memory_config import MemoryConfig

        mw_config = MemoryConfig(enabled=True, mode="middleware")
        monkeypatch.setattr(
            "deerflow.config.memory_config.get_memory_config",
            lambda: mw_config,
        )

        feat = RuntimeFeatures(memory=True)
        chain, extra_tools = _assemble_from_features(feat, name="test-agent")

        middleware_types = [type(m) for m in chain]
        assert MemoryMiddleware in middleware_types, "MemoryMiddleware should be in the chain in middleware mode"

        tool_names = [t.name for t in extra_tools]
        assert "memory_search" not in tool_names, "memory_search should not be registered in middleware mode"

    def test_memory_disabled_skips_both(self, monkeypatch):
        """When memory.enabled=False, middleware IS appended but no-ops at
        runtime (the enabled check is inside after_agent, not the factory).
        Tools are never registered because mode is middleware (default)."""
        from deerflow.agents.factory import _assemble_from_features
        from deerflow.agents.features import RuntimeFeatures
        from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
        from deerflow.config.memory_config import MemoryConfig

        disabled_config = MemoryConfig(enabled=False, mode="middleware")
        monkeypatch.setattr(
            "deerflow.config.memory_config.get_memory_config",
            lambda: disabled_config,
        )

        feat = RuntimeFeatures(memory=True)
        chain, extra_tools = _assemble_from_features(feat, name="test-agent")

        # Middleware is appended — it checks enabled internally in after_agent
        middleware_types = [type(m) for m in chain]
        assert MemoryMiddleware in middleware_types
        # Tools should NOT be registered in middleware mode regardless of enabled
        tool_names = [t.name for t in extra_tools]
        assert "memory_search" not in tool_names
