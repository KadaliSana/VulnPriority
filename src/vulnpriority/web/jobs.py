"""In-process background jobs for the interactive server.

An analysis takes seconds to minutes, which is longer than a browser will politely hold a
socket open, so the HTTP layer starts a job and hands back an id. This module is that job
registry and nothing else: it knows how to start work on a thread, report its progress,
stop it when asked, and remember a bounded number of finished jobs.

Three properties are load-bearing and are tested directly:

*Thread safety.* Every field of every job is read and written under one re-entrant lock
owned by the store. Callers never touch a :class:`Job` attribute directly; they go through
the store, or through the :class:`JobContext` a worker is handed.

*No half-written results.* ``result`` is assigned exactly once, in the success branch,
after the target has returned. A job that raises, or that is cancelled midway, ends with
``result is None`` - never with a partially built payload that the page would render as if
it were a finished run.

*No global mutable state.* There is no module-level registry. A :class:`JobStore` is
created by whoever owns the server, which is what makes two servers (or two tests) in one
process independent of each other.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable

__all__ = [
    "JobStatus",
    "TERMINAL_STATUSES",
    "JobCancelled",
    "JobContext",
    "Job",
    "JobStore",
]

LOGGER = logging.getLogger("vulnpriority.web.jobs")


class JobStatus(str, Enum):
    """Where a job is.

    ``cancelled`` is deliberately distinct from ``failed``: one is the operator changing
    their mind, the other is the framework being unable to do what it was asked. Collapsing
    them would make the page lie about which happened.
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses from which a job never moves again.
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
)


class JobCancelled(Exception):
    """Raised inside a worker when cancellation was requested. Not an error."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="seconds")


@dataclass
class Job:
    """One unit of background work and everything the page needs to describe it.

    Fields are mutated only by :class:`JobStore` while holding its lock. Read them through
    :meth:`JobStore.snapshot`, which copies under the same lock, rather than directly.
    """

    job_id: str
    kind: str = "analyze"
    status: JobStatus = JobStatus.QUEUED
    phase: str = ""
    progress: float = 0.0
    message: str = ""
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def as_dict(self) -> dict[str, Any]:
        """A JSON-able snapshot. Call it under the store's lock."""
        return {
            "job_id": self.job_id,
            "mode": str(self.meta.get("mode", "")),
            "kind": self.kind,
            "status": self.status.value,
            "phase": self.phase,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "log": list(self.log),
            "error": self.error,
            "created_at": _iso(self.created_at) or "",
            "updated_at": _iso(self.updated_at) or "",
            "finished_at": _iso(self.finished_at),
            "has_result": self.result is not None,
        }


class JobContext:
    """The handle a worker uses to report progress and notice cancellation.

    A worker never sees the store or the job object, only this. That keeps the set of
    things a worker can mutate down to exactly the fields a progress report consists of.
    """

    def __init__(self, store: "JobStore", job: Job) -> None:
        self._store = store
        self._job_id = job.job_id
        self._job = job

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested

    def raise_if_cancelled(self) -> None:
        """The cancellation checkpoint. Workers call it between phases."""
        if self._job.cancel_requested:
            raise JobCancelled(self._job_id)

    def progress(
        self,
        phase: str | None = None,
        fraction: float | None = None,
        message: str | None = None,
        *,
        log: bool = True,
    ) -> None:
        """Record a phase transition, and stop the worker if cancellation was requested."""
        self.raise_if_cancelled()
        self._store.update(
            self._job_id,
            phase=phase,
            progress=fraction,
            message=message,
            log=message if (log and message) else None,
        )

    def log(self, line: str) -> None:
        self._store.update(self._job_id, log=line)


class JobStore:
    """A bounded, thread-safe registry of background jobs.

    ``max_history`` caps how many jobs are remembered. Eviction only ever removes
    *finished* jobs, oldest first: a live job is never dropped out from under the page
    that is polling it, however many new ones arrive.
    """

    def __init__(
        self,
        max_history: int = 32,
        max_log_lines: int = 400,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self.max_history = int(max_history)
        self.max_log_lines = int(max_log_lines)
        self._logger = logger or LOGGER
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()

    # -- creation and lookup -------------------------------------------------

    def create(self, kind: str = "analyze", meta: dict[str, Any] | None = None) -> Job:
        """Register a new queued job and evict finished history to stay within the cap."""
        job = Job(job_id=uuid.uuid4().hex, kind=kind, meta=dict(meta or {}))
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        """JSON-able state of one job, copied under the lock."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job is not None else None

    def result(self, job_id: str) -> Any:
        """The finished result, or ``None`` when the job is not done."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.DONE:
                return None
            return job.result

    def jobs(self) -> list[Job]:
        """Every remembered job, oldest first."""
        with self._lock:
            return [self._jobs[key] for key in self._order if key in self._jobs]

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    # -- mutation ------------------------------------------------------------

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        phase: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        log: str | Iterable[str] | None = None,
        error: str | None = None,
    ) -> Job | None:
        """Apply a partial update. ``None`` means "leave this field alone"."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if phase is not None:
                job.phase = phase
            if progress is not None:
                job.progress = min(1.0, max(0.0, float(progress)))
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if log is not None:
                lines = [log] if isinstance(log, str) else list(log)
                stamp = _now().strftime("%H:%M:%S")
                job.log.extend(f"{stamp}  {line}" for line in lines if str(line).strip())
                if len(job.log) > self.max_log_lines:
                    overflow = len(job.log) - self.max_log_lines
                    del job.log[:overflow]
            job.updated_at = _now()
            return job

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop. A queued job stops immediately; a running one at its next
        checkpoint. Returns the job, or ``None`` when the id is unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.finished:
                return job
            job._cancel.set()
            if job.status is JobStatus.QUEUED:
                # It never started, so there is no worker to notice the event.
                job.status = JobStatus.CANCELLED
                job.message = "cancelled before it started"
                job.error = "cancelled"
                job.finished_at = _now()
            else:
                job.message = "cancelling"
            job.updated_at = _now()
            return job

    # -- running -------------------------------------------------------------

    def submit(self, job: Job, target: Callable[[JobContext], Any]) -> Job:
        """Start ``target`` on its own thread, passing it a :class:`JobContext`."""
        with self._lock:
            if job.job_id not in self._jobs:
                raise KeyError(f"job {job.job_id} is not registered with this store")
            if job._thread is not None:
                raise RuntimeError(f"job {job.job_id} has already been submitted")
            thread = threading.Thread(
                target=self._run,
                args=(job, target),
                name=f"vulnpriority-job-{job.job_id[:8]}",
                daemon=True,
            )
            job._thread = thread
        thread.start()
        return job

    def _run(self, job: Job, target: Callable[[JobContext], Any]) -> None:
        with self._lock:
            if job.cancel_requested or job.finished:
                return
            job.status = JobStatus.RUNNING
            job.started_at = _now()
            job.updated_at = job.started_at
            job.message = job.message or "started"
        context = JobContext(self, job)
        try:
            context.raise_if_cancelled()
            produced = target(context)
            context.raise_if_cancelled()
        except JobCancelled:
            self._finish(job, JobStatus.CANCELLED, error="cancelled", message="cancelled")
        except BaseException as error:  # noqa: BLE001 - recorded, never re-raised into the socket
            # The detail the client sees is the exception's message; the traceback goes to
            # the server log only, because a stack trace is a map of the host filesystem.
            self._logger.exception("job %s failed", job.job_id)
            detail = f"{type(error).__name__}: {error}".strip()
            self._finish(job, JobStatus.FAILED, error=detail, message="failed")
        else:
            self._finish(
                job, JobStatus.DONE, result=produced, message="complete", progress=1.0
            )

    def _finish(
        self,
        job: Job,
        status: JobStatus,
        *,
        result: Any = None,
        error: str | None = None,
        message: str = "",
        progress: float | None = None,
    ) -> None:
        """The single place a job becomes terminal, and the only place ``result`` is set."""
        with self._lock:
            job.status = status
            job.result = result       # None for every non-DONE outcome, by construction
            job.error = error
            if message:
                job.message = message
            if progress is not None:
                job.progress = min(1.0, max(0.0, float(progress)))
            job.finished_at = _now()
            job.updated_at = job.finished_at
            if message:
                stamp = job.finished_at.strftime("%H:%M:%S")
                job.log.append(f"{stamp}  {message}")
                if len(job.log) > self.max_log_lines:
                    del job.log[: len(job.log) - self.max_log_lines]

    # -- lifecycle -----------------------------------------------------------

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        """Block until a job's thread ends. Used by tests and by shutdown, not by the server."""
        job = self.get(job_id)
        if job is None:
            return None
        thread = job._thread
        if thread is not None:
            thread.join(timeout)
        return job

    def shutdown(self, timeout: float = 5.0) -> None:
        """Ask every live job to stop, then wait briefly for the threads to unwind."""
        with self._lock:
            live = [job for job in self._jobs.values() if not job.finished]
        for job in live:
            self.cancel(job.job_id)
        # One deadline for the whole shutdown, not one per job: giving each thread the full
        # timeout made the worst case N x timeout, which is not "briefly" and is not what
        # the caller asked for.
        deadline = time.monotonic() + max(0.0, float(timeout))
        for job in live:
            thread = job._thread
            if thread is not None:
                thread.join(max(0.0, deadline - time.monotonic()))

    # -- internals -----------------------------------------------------------

    def _evict(self) -> None:
        """Drop finished jobs, oldest first, until the history fits. Live jobs stay."""
        while len(self._order) > self.max_history:
            victim: str | None = None
            for key in self._order:
                job = self._jobs.get(key)
                if job is None or job.finished:
                    victim = key
                    break
            if victim is None:
                return  # every remembered job is still live; the cap yields to that
            self._order.remove(victim)
            self._jobs.pop(victim, None)
