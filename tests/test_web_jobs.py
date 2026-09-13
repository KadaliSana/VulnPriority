"""The background job registry.

The HTTP layer hands work to :mod:`vulnprio.web.jobs` and then reports whatever it says, so
these tests are about the properties the rest of the application relies on rather than about
any particular analysis: a job goes through the states it claims to, a cancelled or failed
one never leaves a result behind, the history stays bounded without ever dropping live work,
and concurrent callers do not corrupt the store.

Everything here is offline and fast: the "work" is a callable the test supplies.
"""

from __future__ import annotations

import threading
import time

import pytest

from vulnprio.web.jobs import TERMINAL_STATUSES, Job, JobCancelled, JobStatus, JobStore


def _wait_for(store: JobStore, job_id: str, timeout: float = 5.0) -> dict:
    """Block until a job reaches a terminal state and return its snapshot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = store.snapshot(job_id)
        assert snapshot is not None, f"job {job_id} vanished from the store"
        if snapshot["status"] in {status.value for status in TERMINAL_STATUSES}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished: {store.snapshot(job_id)}")


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_a_job_runs_and_reports_its_result() -> None:
    store = JobStore()
    job = store.create(kind="analyze", meta={"mode": "demo"})
    assert job.status is JobStatus.QUEUED
    assert store.snapshot(job.job_id)["status"] == "queued"

    def work(context):
        context.progress("assess", 0.5, "halfway")
        return {"findings": 3}

    store.submit(job, work)
    snapshot = _wait_for(store, job.job_id)

    assert snapshot["status"] == "done"
    assert snapshot["progress"] == 1.0
    assert snapshot["error"] is None
    assert snapshot["has_result"] is True
    assert store.result(job.job_id) == {"findings": 3}


def test_progress_and_log_are_visible_while_running() -> None:
    """The page polls mid-run, so a partial snapshot has to be meaningful on its own."""
    store = JobStore()
    job = store.create()
    seen = threading.Event()
    release = threading.Event()

    def work(context):
        context.progress("enrich", 0.42, "working on it")
        seen.set()
        release.wait(5)
        return "done"

    store.submit(job, work)
    assert seen.wait(5), "the worker never started"

    snapshot = store.snapshot(job.job_id)
    assert snapshot["status"] == "running"
    assert snapshot["phase"] == "enrich"
    assert snapshot["progress"] == pytest.approx(0.42)
    assert any("working on it" in line for line in snapshot["log"])
    assert snapshot["has_result"] is False, "a running job must not advertise a result"

    release.set()
    _wait_for(store, job.job_id)


def test_progress_is_clamped_to_the_unit_interval() -> None:
    store = JobStore()
    job = store.create()
    store.update(job.job_id, progress=7.5)
    assert store.snapshot(job.job_id)["progress"] == 1.0
    store.update(job.job_id, progress=-3.0)
    assert store.snapshot(job.job_id)["progress"] == 0.0


# ---------------------------------------------------------------------------
# failure
# ---------------------------------------------------------------------------


def test_a_failure_is_captured_and_leaves_no_result() -> None:
    store = JobStore()
    job = store.create()

    def work(context):
        context.progress("rank", 0.6, "about to fall over")
        raise ValueError("the ranker could not be built")

    store.submit(job, work)
    snapshot = _wait_for(store, job.job_id)

    assert snapshot["status"] == "failed"
    assert "the ranker could not be built" in snapshot["error"]
    assert "ValueError" in snapshot["error"]
    assert snapshot["has_result"] is False
    assert store.result(job.job_id) is None, "a failed job must not leave a half-written result"


def test_a_failure_message_does_not_carry_a_traceback() -> None:
    """The error string reaches the browser, so it must be a message, not a stack trace."""
    store = JobStore()
    job = store.create()

    def work(context):
        raise RuntimeError("scanner refused the target")

    store.submit(job, work)
    snapshot = _wait_for(store, job.job_id)
    assert snapshot["error"] == "RuntimeError: scanner refused the target"
    assert "Traceback" not in snapshot["error"]
    assert "File \"" not in snapshot["error"]


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_cancelling_a_queued_job_stops_it_before_it_starts() -> None:
    store = JobStore()
    job = store.create()
    store.cancel(job.job_id)

    snapshot = store.snapshot(job.job_id)
    assert snapshot["status"] == "cancelled"
    assert snapshot["has_result"] is False

    started = threading.Event()
    store.submit(job, lambda context: started.set() or "result")
    time.sleep(0.15)
    assert not started.is_set(), "a job cancelled before submission must never run"
    assert store.result(job.job_id) is None


def test_cancelling_a_running_job_stops_it_at_its_next_checkpoint() -> None:
    store = JobStore()
    job = store.create()
    running = threading.Event()
    finished_body = threading.Event()

    def work(context):
        running.set()
        for _ in range(200):
            context.raise_if_cancelled()      # the checkpoint
            time.sleep(0.01)
        finished_body.set()
        return "should never be returned"

    store.submit(job, work)
    assert running.wait(5)
    store.cancel(job.job_id)

    snapshot = _wait_for(store, job.job_id)
    assert snapshot["status"] == "cancelled"
    assert snapshot["error"] == "cancelled"
    assert snapshot["has_result"] is False
    assert not finished_body.is_set(), "the worker body should not have run to completion"
    assert store.result(job.job_id) is None


def test_progress_raises_inside_a_cancelled_worker() -> None:
    """Cancellation unwinds through the progress call, which is what stops a live scan."""
    store = JobStore()
    job = store.create()
    raised: list[BaseException] = []
    ready = threading.Event()
    go = threading.Event()

    def work(context):
        ready.set()
        go.wait(5)
        try:
            context.progress("scan", 0.3, "still crawling")
        except JobCancelled as error:
            raised.append(error)
            raise
        return "unreachable"

    store.submit(job, work)
    assert ready.wait(5)
    store.cancel(job.job_id)
    go.set()

    snapshot = _wait_for(store, job.job_id)
    assert raised, "progress() should have raised JobCancelled"
    assert snapshot["status"] == "cancelled"


def test_cancelling_an_unknown_or_finished_job_is_harmless() -> None:
    store = JobStore()
    assert store.cancel("no-such-job") is None

    job = store.create()
    store.submit(job, lambda context: "ok")
    _wait_for(store, job.job_id)

    again = store.cancel(job.job_id)
    assert again is not None
    assert store.snapshot(job.job_id)["status"] == "done", "a finished job is not un-finished"
    assert store.result(job.job_id) == "ok"


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_is_capped_and_evicts_the_oldest_finished_job() -> None:
    store = JobStore(max_history=3)
    ids = []
    for _ in range(6):
        job = store.create()
        store.submit(job, lambda context: "ok")
        _wait_for(store, job.job_id)
        ids.append(job.job_id)

    assert len(store) == 3
    assert [store.snapshot(job_id) for job_id in ids[:3]] == [None, None, None]
    assert all(store.snapshot(job_id) is not None for job_id in ids[3:])


def test_the_cap_never_evicts_a_live_job() -> None:
    """A page polling a running job must not have it disappear because newer ones arrived."""
    store = JobStore(max_history=2)
    release = threading.Event()
    live = store.create()
    store.submit(live, lambda context: release.wait(5) and "ok")
    time.sleep(0.05)

    for _ in range(5):
        job = store.create()
        store.submit(job, lambda context: "ok")
        _wait_for(store, job.job_id)

    assert store.snapshot(live.job_id) is not None, "the running job was evicted"
    assert store.snapshot(live.job_id)["status"] == "running"
    release.set()
    _wait_for(store, live.job_id)


def test_log_lines_are_capped_keeping_the_most_recent() -> None:
    store = JobStore(max_log_lines=5)
    job = store.create()
    for index in range(25):
        store.update(job.job_id, log=f"line {index}")

    log = store.snapshot(job.job_id)["log"]
    assert len(log) == 5
    assert "line 24" in log[-1]
    assert not any("line 0" in line for line in log)


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


def test_concurrent_creates_produce_distinct_registered_jobs() -> None:
    store = JobStore(max_history=200)
    created: list[Job] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def spawn() -> None:
        barrier.wait(5)
        for _ in range(15):
            job = store.create()
            with lock:
                created.append(job)

    threads = [threading.Thread(target=spawn) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(created) == 120
    ids = {job.job_id for job in created}
    assert len(ids) == 120, "job ids collided"
    assert len(store) == 120
    assert all(store.snapshot(job_id) is not None for job_id in ids)


def test_concurrent_updates_do_not_lose_log_lines() -> None:
    store = JobStore(max_log_lines=10_000)
    job = store.create()
    barrier = threading.Barrier(6)

    def hammer(worker: int) -> None:
        barrier.wait(5)
        for index in range(50):
            store.update(job.job_id, log=f"w{worker}-{index}", progress=index / 50.0)

    threads = [threading.Thread(target=hammer, args=(worker,)) for worker in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    log = store.snapshot(job.job_id)["log"]
    assert len(log) == 300, "a concurrent append was lost"


def test_submitting_the_same_job_twice_is_refused() -> None:
    store = JobStore()
    job = store.create()
    store.submit(job, lambda context: "ok")
    with pytest.raises(RuntimeError):
        store.submit(job, lambda context: "again")
    _wait_for(store, job.job_id)


def test_submitting_an_unregistered_job_is_refused() -> None:
    store = JobStore()
    stranger = Job(job_id="not-in-this-store")
    with pytest.raises(KeyError):
        store.submit(stranger, lambda context: "ok")


def test_shutdown_cancels_live_work() -> None:
    store = JobStore()
    job = store.create()
    running = threading.Event()

    def work(context):
        running.set()
        for _ in range(500):
            context.raise_if_cancelled()
            time.sleep(0.01)
        return "should not finish"

    store.submit(job, work)
    assert running.wait(5)
    store.shutdown(timeout=5.0)

    snapshot = store.snapshot(job.job_id)
    assert snapshot["status"] == "cancelled"
    assert store.result(job.job_id) is None


def test_two_stores_are_independent() -> None:
    """No module-level registry: two servers in one process must not share jobs."""
    first, second = JobStore(), JobStore()
    job = first.create()
    assert second.snapshot(job.job_id) is None
    assert len(second) == 0
