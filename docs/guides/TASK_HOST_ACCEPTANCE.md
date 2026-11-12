# TP-09 — Cancellable Product Hosting for Model Planning

Historical acceptance for cancellable model planning hosts. Read together with [TP-02 decomposition](TASK_DECOMPOSITION_ACCEPTANCE.md), [TP-03 planning](TASK_PLANNING_ACCEPTANCE.md), [TP-04 guarded execution](TASK_EXECUTION_ACCEPTANCE.md), [TP-07 preview](TASK_PLAN_PREVIEW_ACCEPTANCE.md), and [TP-08 graph admission](TASK_ADMISSION_ACCEPTANCE.md).

**Status:** complete under the owner's focused-test instruction; all 9 host tests and 366 total planning checks passed (2026-11-12).  
**Scope:** Cancellable execution context for model decomposition and task planning in background threads and jobs.

---

## 1. Problem and Boundary

Previously, `_deadline(seconds)` in `dclab_rnd/tasks/decomposer.py` exclusively supported synchronous POSIX main-thread callers using `signal.setitimer(signal.ITIMER_REAL, ...)`. In worker threads—such as background worker jobs or `asyncio.to_thread` executions (e.g. `walk_project` in `dclab_rnd/orchestra/service.py`)—calling `_deadline` raised `_Refusal("deadline_unavailable")`, forcing an immediate fallback to deterministic graphs and plans.

This package provides a cancellable product host (`PlanningHost` in `dclab_rnd/tasks/host.py`) that enables bounded model planning on background threads without compromising safety:
- **Cancellable host context:** `with planning_host(...)` provides thread-local and contextvar tracking of cancellation state.
- **Jobs worker integration:** Automatically detects and wraps `dclab_rnd.jobs.worker.current()` controls, delegating cancellation checks to `checkpoint()`.
- **BaseException preservation:** When a job is stopped by a user, `checkpoint()` raises `Stopped("stopped")`. Because `Stopped` is a `BaseException`, it is never swallowed by internal `except Exception:` handlers in `decompose()` or `plan_task()`, allowing clean job termination.
- **Strict bounds:** Requests remain strictly bounded by `max_seconds` (via HTTP client `timeout_seconds`), `max_tokens`, `max_total_tokens` (including reasoning tokens), and `cap_eur`.
- **Unhosted thread safety:** Background threads without an explicit `planning_host` or active job control continue to refuse planning before invoking models, returning `_Refusal("deadline_unavailable")`.
- **Zero untracked requests:** Model request IDs are recorded in `finally:` blocks even when network calls fail or time out.

---

## 2. Implementation Architecture

### 2.1 Host Abstraction (`dclab_rnd/tasks/host.py`)

```python
with planning_host(name="worker_thread") as host:
    graph, report = decompose(...)
```

1. **`PlanningHost`:** Tracks thread-safe cancellation flags and optional cancel/check callbacks.
2. **`_JobWorkerHost`:** Subclasses `PlanningHost` to call `jobs.worker.checkpoint()`, ensuring that background jobs stopped via the Compute page or UI abort immediately.
3. **`current_host()`:** Returns the active host from contextvars, or auto-wraps `jobs.worker.current()` if running inside an active job.
4. **`is_cancellable()`:** Confirms whether the calling context has a valid cancellation mechanism (active host or POSIX main thread with inactive `ITIMER_REAL`).

### 2.2 Integration in `_deadline` (`dclab_rnd/tasks/decomposer.py`)

`_deadline(seconds)` checks `current_host()` first:
- If a host is active, it verifies cancellation on entry and on exit.
- If no host is active, it requires the POSIX main thread with available `setitimer`.
- If neither is available, it raises `_Refusal("deadline_unavailable")`.

### 2.3 Product Integration in `walk_project` (`dclab_rnd/orchestra/service.py`)

`walk_project` wraps the execution of `SolutionTeam` in `with planning_host(name=f"walk_{project_id}"):`, enabling model planning inside `asyncio.to_thread` worker threads while maintaining cancellation boundaries.

---

## 3. Verification & Acceptance Evidence

### 3.1 Focused Regression Tests (`tests/test_task_host.py`)

All 9 tests pass with zero skips:
1. `test_planning_host_context_manages_current_host_and_cancellability`: Host activation, context cleanup, and `PlanningCancelled` on check.
2. `test_planning_host_invokes_custom_cancel_callbacks`: Explicit cancel callback invocation.
3. `test_unhosted_worker_thread_refuses_with_deadline_unavailable`: Preserved refusal when worker thread lacks a host.
4. `test_hosted_worker_thread_allows_model_decomposition`: Successful model decomposition inside hosted worker threads.
5. `test_hosted_worker_thread_cancelled_before_call_falls_back_safely`: Safe deterministic fallback on cancelled host.
6. `test_hosted_worker_thread_allows_model_task_planning`: Successful model task planning inside hosted worker threads.
7. `test_hosted_worker_thread_cancelled_planner_falls_back_safely`: Safe deterministic plan fallback on cancelled host.
8. `test_job_worker_control_serves_as_cancellable_host_and_raises_stopped`: `Stopped` exception propagation across `decompose` and `plan_task`.
9. `test_untracked_requests_prevented_on_failure`: Request ID capture in report on provider failure.

### 3.2 Regression Suite

- **12 test files, 366 tests:** Passed in 44.4s with zero errors or regressions.
- All frozen fingerprints in `test_task_planner.py` preserved.
