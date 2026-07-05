"""``BoxliteProvider`` — DeerFlow :class:`SandboxProvider` backed by BoxLite.

Integrates `BoxLite <https://github.com/boxlite-ai/boxlite>`_ — a daemonless,
OCI-native micro-VM runtime — as a DeerFlow sandbox backend. See
https://github.com/bytedance/deer-flow/issues/3936.

Config is read off :class:`SandboxConfig` (``extra="allow"``), so BoxLite keys
may appear under ``sandbox:`` in ``config.yaml`` even though they are not declared
on the model — see this package's ``__init__`` docstring for the full set. The
provider creates one micro-VM per ``(user, thread)`` and reuses it within the
process.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import logging
import threading
import time
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, TypeVar

from deerflow.config import get_app_config
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.constants import DEFAULT_SKILLS_CONTAINER_PATH
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider

from .box import BoxliteBox

if TYPE_CHECKING:
    from boxlite import SimpleBox

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_IMAGE = "python:3.12-slim"
# DeerFlow's virtual prefixes, materialised on the box rootfs at start so the
# Sandbox file APIs (which address /mnt/user-data/...) resolve natively.
_VIRTUAL_DIRS = (
    f"{VIRTUAL_PATH_PREFIX}/workspace",
    f"{VIRTUAL_PATH_PREFIX}/uploads",
    f"{VIRTUAL_PATH_PREFIX}/outputs",
    DEFAULT_SKILLS_CONTAINER_PATH,
)


def _import_simplebox() -> type[SimpleBox]:
    """Import BoxLite's async ``SimpleBox`` lazily.

    Kept out of module import so the harness (and every other provider) installs
    without BoxLite; the dependency is only needed once this provider is selected.
    """
    try:
        from boxlite import SimpleBox
    except ImportError as e:  # pragma: no cover - depends on the optional dependency
        raise ImportError("BoxliteProvider requires the 'boxlite' package. Install it with: pip install boxlite.") from e
    return SimpleBox


class _EventLoopThread:
    """A private asyncio event loop running on a dedicated daemon thread.

    BoxLite is async-native and its box handles are loop-affine, while DeerFlow's
    ``Sandbox`` contract is synchronous and may be invoked from arbitrary
    ``asyncio.to_thread`` workers. Owning one loop here and marshalling every
    coroutine onto it via ``run_coroutine_threadsafe`` gives a stable, thread-safe
    bridge without BoxLite's greenlet sync facade (which refuses to run inside an
    async context and is thread-affine).
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="boxlite-loop", daemon=True)
        self._thread.start()

    def run(self, coro: Awaitable[T], *, timeout: float | None = None) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_running():
            self._loop.close()


class BoxliteProvider(SandboxProvider):
    """Run each DeerFlow sandbox as a BoxLite micro-VM."""

    uses_thread_data_mounts = False
    needs_upload_permission_adjustment = True

    # ── Warm pool constants (mirrors AioSandboxProvider) ─────────────────

    DEFAULT_IDLE_TIMEOUT = 600
    IDLE_CHECK_INTERVAL = 60
    DEFAULT_REPLICAS = 3

    @staticmethod
    def _sandbox_id(thread_id: str, user_id: str) -> str:
        """Deterministic sandbox ID from user/thread scope.

        Includes user_id so a box created for one user's bucket cannot be
        reclaimed by another user's thread with the same thread_id.
        """
        return hashlib.sha256(f"{user_id}:{thread_id}".encode()).hexdigest()[:8]

    # ── Provider ────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: dict[str, BoxliteBox] = {}
        self._thread_boxes: dict[tuple[str, str], str] = {}
        self._warm_pool: dict[str, tuple[BoxliteBox, float]] = {}
        self._acquire_locks: dict[str, threading.Lock] = {}
        self._idle_checker_stop = threading.Event()
        self._idle_checker_thread: threading.Thread | None = None
        self._shutdown_called = False
        self._config = self._load_config()
        self._loop = _EventLoopThread()
        atexit.register(self.shutdown)
        if self._config["idle_timeout"] > 0:
            self._idle_checker_thread = threading.Thread(target=self._idle_reaper_loop, name="boxlite-idle-reaper", daemon=True)
            self._idle_checker_thread.start()

    def _load_config(self) -> dict[str, Any]:
        sandbox_config = get_app_config().sandbox

        def _opt(name: str, default: Any = None) -> Any:
            return getattr(sandbox_config, name, default)

        # $VARS in config.yaml are already resolved by AppConfig.resolve_env_variables
        # (which raises on a missing var), so the environment dict is used as-is.
        replicas = _opt("replicas")
        idle_timeout = _opt("idle_timeout")
        return {
            "image": _opt("image") or DEFAULT_IMAGE,
            "memory_mib": _opt("memory_mib"),
            "cpus": _opt("cpus"),
            "environment": dict(_opt("environment") or {}),
            "replicas": replicas if replicas is not None else self.DEFAULT_REPLICAS,
            "idle_timeout": idle_timeout if idle_timeout is not None else self.DEFAULT_IDLE_TIMEOUT,
        }

    @staticmethod
    def _thread_key(thread_id: str, user_id: str | None) -> tuple[str, str]:
        return (user_id or "", thread_id)

    def _lock_for_sandbox(self, sandbox_id: str) -> threading.Lock:
        """Return the per-sandbox acquire lock for a deterministic sandbox id."""
        with self._lock:
            lock = self._acquire_locks.get(sandbox_id)
            if lock is None:
                lock = threading.Lock()
                self._acquire_locks[sandbox_id] = lock
            return lock

    # ── Idle reaper (Task 6) ─────────────────────────────────────────────

    def _idle_reaper_loop(self) -> None:
        """Daemon thread that periodically reaps expired warm-pool boxes."""
        while not self._idle_checker_stop.wait(self.IDLE_CHECK_INTERVAL):
            self._reap_expired_warm()

    def _reap_expired_warm(self) -> None:
        """Destroy warm-pool boxes that have been idle longer than the timeout."""
        timeout = self._config["idle_timeout"]
        if timeout <= 0:
            return
        now = time.time()
        expired: list[tuple[str, BoxliteBox]] = []
        with self._lock:
            for sid, (box, ts) in self._warm_pool.items():
                if now - ts > timeout:
                    expired.append((sid, box))
            for sid, _ in expired:
                self._warm_pool.pop(sid, None)

        for sid, box in expired:
            try:
                box.close()
                logger.info("Idle reaper destroyed expired warm-pool box %s", sid)
            except Exception as e:
                logger.warning("Error closing expired BoxLite box %s: %s", sid, e)

    # ── Replica enforcement (Task 7) ─────────────────────────────────────

    def _replica_count(self) -> tuple[int, int]:
        """Return configured replicas and current active + warm box count."""
        replicas = self._config.get("replicas", self.DEFAULT_REPLICAS)
        with self._lock:
            total = len(self._boxes) + len(self._warm_pool)
        return replicas, total

    def _evict_oldest_warm(self) -> str | None:
        """Evict and destroy the oldest warm-pool box (by timestamp).

        Only evicts from the warm pool — active boxes are never touched.
        """
        with self._lock:
            if not self._warm_pool:
                return None
            oldest_sid = min(self._warm_pool, key=lambda sid: self._warm_pool[sid][1])
            box, _ = self._warm_pool.pop(oldest_sid)

        try:
            box.close()
            logger.info("Replica enforcement evicted oldest warm-pool box %s", box.id)
        except Exception as e:
            logger.warning("Error closing evicted BoxLite box %s: %s", box.id, e)
        return oldest_sid

    # ── Acquire / release ────────────────────────────────────────────────

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        if thread_id is None:
            sandbox_id = str(uuid.uuid4())[:8]
            box = self._create_box(sandbox_id)
            with self._lock:
                self._boxes[box.id] = box
            return box.id

        key = self._thread_key(thread_id, user_id)
        sandbox_id = self._sandbox_id(thread_id, user_id)
        acquire_lock = self._lock_for_sandbox(sandbox_id)
        with acquire_lock:
            with self._lock:
                existing = self._thread_boxes.get(key)
                if existing is not None and existing in self._boxes:
                    return existing

            reclaimed = self._reclaim_warm_pool(sandbox_id)
            if reclaimed is not None:
                with self._lock:
                    self._thread_boxes[key] = reclaimed
                return reclaimed

            box = self._create_box(sandbox_id)
            with self._lock:
                self._boxes[box.id] = box
                self._thread_boxes[key] = box.id
            return box.id

    def _create_box(self, sandbox_id: str) -> BoxliteBox:
        # Enforce replica limit: evict oldest warm-pool box if active + warm boxes are at capacity.
        replicas, total = self._replica_count()
        if total >= replicas:
            self._evict_oldest_warm()

        simplebox_cls = _import_simplebox()
        mkdir_cmd = "mkdir -p " + " ".join(_VIRTUAL_DIRS)

        async def _make() -> SimpleBox:
            box = simplebox_cls(
                name=sandbox_id,
                image=self._config["image"],
                memory_mib=self._config["memory_mib"],
                cpus=self._config["cpus"],
            )
            await box.start()
            # Materialise DeerFlow's virtual prefixes so file ops resolve natively.
            await box.exec("sh", "-lc", mkdir_cmd)
            return box

        box = self._loop.run(_make())
        logger.info("Created BoxLite box %s (image=%s)", box.id, self._config["image"])
        return BoxliteBox(box.id, box, self._loop.run, default_env=self._config["environment"])

    def get(self, sandbox_id: str) -> Sandbox | None:
        with self._lock:
            return self._boxes.get(sandbox_id)

    def release(self, sandbox_id: str) -> None:
        """Release a sandbox into the warm pool — VM stays running.

        The box is moved from _boxes to _warm_pool; _thread_boxes entries are
        cleared so the thread no longer holds an active reference. The VM is
        NOT stopped unless shutdown has already begun.
        """
        close_box: BoxliteBox | None = None
        with self._lock:
            box = self._boxes.pop(sandbox_id, None)
            for key in [k for k, sid in self._thread_boxes.items() if sid == sandbox_id]:
                self._thread_boxes.pop(key, None)
            if box is None:
                return
            if self._shutdown_called:
                close_box = box
            else:
                self._warm_pool[sandbox_id] = (box, time.time())

        if close_box is not None:
            close_box.close()
            logger.info("Closed released sandbox %s because shutdown is in progress", sandbox_id)
        else:
            logger.info("Released sandbox %s to warm pool (VM still running)", sandbox_id)

    def _reclaim_warm_pool(self, sandbox_id: str) -> str | None:
        """Try to reclaim a warm-pool box by sandbox_id.

        Returns sandbox_id on success, None if not found or dead.
        """
        with self._lock:
            if sandbox_id not in self._warm_pool:
                return None
            box, _ = self._warm_pool[sandbox_id]

        # Health check: run a simple command to verify the VM is alive
        try:
            result = box.execute_command("echo ok")
            if "ok" not in result:
                logger.warning("Warm pool box %s health check failed: %s", sandbox_id, result)
                with self._lock:
                    self._warm_pool.pop(sandbox_id, None)
                box.close()
                return None
        except Exception as e:
            logger.warning("Warm pool box %s health check error: %s", sandbox_id, e)
            with self._lock:
                self._warm_pool.pop(sandbox_id, None)
            box.close()
            return None

        # Promote from warm pool to active
        with self._lock:
            warm_entry = self._warm_pool.pop(sandbox_id, None)
            if warm_entry is None:
                return None  # Raced with another thread
            box, _ = warm_entry
            self._boxes[sandbox_id] = box

        logger.info("Reclaimed warm-pool box %s", sandbox_id)
        return sandbox_id

    def reset(self) -> None:
        with self._lock:
            active = list(self._boxes.values())
            warm = [box for box, _ in self._warm_pool.values()]
            self._boxes.clear()
            self._warm_pool.clear()
            self._thread_boxes.clear()
            self._acquire_locks.clear()

        for box in active + warm:
            try:
                box.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Error closing BoxLite box during reset", exc_info=True)

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        self._idle_checker_stop.set()
        if self._idle_checker_thread is not None:
            self._idle_checker_thread.join(timeout=5)

        with self._lock:
            active = list(self._boxes.values())
            warm = [box for box, _ in self._warm_pool.values()]
            self._boxes.clear()
            self._warm_pool.clear()
            self._thread_boxes.clear()
            self._acquire_locks.clear()

        for box in active + warm:
            try:
                box.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Error closing BoxLite box %s during shutdown: %s", box.id, e)
        self._loop.close()
