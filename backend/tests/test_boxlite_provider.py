"""Unit tests for the BoxLite community provider.
These run in CI without BoxLite installed: they cover the lazy-import error path,
provider lifecycle, the path-safety guards, and the warm pool lifecycle — none of
which need a live box.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from deerflow.community.boxlite.box import BoxliteBox
from deerflow.community.boxlite.provider import BoxliteProvider, _import_simplebox

# ── Fake BoxLite SDK ──────────────────────────────────────────────────


class _FakeBox:
    """A fake SimpleBox that records lifecycle calls without starting real VMs."""

    def __init__(self, *, image=None, name=None, memory_mib=None, cpus=None, **kwargs):
        self.id = name or "auto-gen-id"
        self.name = name
        self._image = image
        self._started = False
        self._stopped = False
        self._exec_history: list[tuple] = []

    async def start(self):
        self._started = True

    async def exec(self, *argv, env=None, timeout=None):
        self._exec_history.append((argv, env, timeout))
        _FakeResult = type("_FakeResult", (), {"stdout": "", "stderr": "", "exit_code": 0})
        # Health check: box.execute_command("echo ok") → exec("sh", "-lc", "echo ok")
        if len(argv) >= 3 and argv[0] == "sh" and argv[1] == "-lc" and argv[2] == "echo ok":
            return type("_FakeResult", (), {"stdout": "ok\n", "stderr": "", "exit_code": 0})()
        return _FakeResult()

    async def stop(self):
        self._stopped = True


def _fake_run(coro, *, timeout=None):
    """Sync runner that executes coroutines on a temporary event loop (no daemon thread)."""
    return asyncio.run(coro)


# ── Config stub ───────────────────────────────────────────────────────


def _stub_config(sandbox_attrs=None):
    """Stub get_app_config to return a config with given sandbox attrs."""
    attrs = sandbox_attrs or {}
    stub = types.SimpleNamespace(sandbox=types.SimpleNamespace(**attrs))
    return stub


def _no_boxlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import boxlite`` raise, regardless of whether it is installed."""
    monkeypatch.setitem(sys.modules, "boxlite", None)


def test_import_simplebox_missing_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_boxlite(monkeypatch)
    with pytest.raises(ImportError, match=r"pip install boxlite"):
        _import_simplebox()


def test_acquire_without_boxlite_raises_and_shuts_down_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub config so the provider constructs without a config.yaml on disk.
    stub = types.SimpleNamespace(sandbox=types.SimpleNamespace())
    monkeypatch.setattr("deerflow.community.boxlite.provider.get_app_config", lambda: stub)
    _no_boxlite(monkeypatch)

    provider = BoxliteProvider()
    try:
        with pytest.raises(ImportError, match=r"pip install boxlite"):
            provider.acquire("thread-1", user_id="u")
    finally:
        provider.shutdown()  # must not raise even though no box was ever created
    # Idempotent shutdown.
    provider.shutdown()


def test_guard_traversal() -> None:
    assert BoxliteBox._guard_traversal("/mnt/user-data/workspace/a.txt") == "/mnt/user-data/workspace/a.txt"
    assert BoxliteBox._guard_traversal("relative/ok.txt") == "relative/ok.txt"
    with pytest.raises(PermissionError):
        BoxliteBox._guard_traversal("/mnt/user-data/../etc/passwd")
    with pytest.raises(ValueError):
        BoxliteBox._guard_traversal("")


def test_download_file_guards_reject_before_touching_box() -> None:
    # ``run`` must never be called: both guards raise before any exec.
    def _fail_run(_coro: object) -> None:
        raise AssertionError("download_file must reject the path before running a command")

    box = BoxliteBox("box-id", box=object(), run=_fail_run)
    with pytest.raises(PermissionError):
        box.download_file("/etc/passwd")  # outside the /mnt/user-data prefix
    with pytest.raises(PermissionError):
        box.download_file("/mnt/user-data/../etc/passwd")  # traversal


def test_execute_command_rejects_invalid_env_key() -> None:
    def _fail_run(_coro: object) -> None:
        raise AssertionError("execute_command must reject a bad env key before running")

    box = BoxliteBox("box-id", box=object(), run=_fail_run)
    with pytest.raises(ValueError, match=r"POSIX"):
        box.execute_command("echo hi", env={"BAD KEY": "x"})
