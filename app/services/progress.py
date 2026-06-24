"""
app/services/progress.py — Server-Sent Events (SSE) helper for streaming pipeline
progress to the browser while a long-running request is still in flight.

Each "process" page (ingestion, risk analysis, design suggestions, innovation) runs
a multi-step pipeline that can take from a few seconds to a few minutes. Instead of
the browser waiting on one opaque request, the route streams a "step" event each time
a real pipeline stage starts or finishes, followed by one final "result" or "error"
event carrying the JSON payload that used to be the whole response body.

Usage in a route:
    async def work(on_step):
        on_step("embed", "active")
        ...
        on_step("embed", "done")
        return {"some": "json-serialisable result"}
    return StreamingResponse(stream_sse(work), media_type="text/event-stream")

on_step is also threaded down into service-layer pipeline functions (e.g.
run_patent_risk_pipeline) that run inside asyncio.to_thread — it is a plain
synchronous callback, safe to call from a worker thread.
"""
import asyncio
import json
import logging
import queue
from typing import Any, Awaitable, Callable, Tuple

log = logging.getLogger(__name__)

OnStep = Callable[[str, str], None]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def noop_on_step(_step: str, _status: str) -> None:
    """Default used by pipeline functions called outside of an SSE route (CLI, internal re-scoring calls)."""


async def stream_sse(work: Callable[[OnStep], Awaitable[Any]]):
    """
    Run `work(on_step)` to completion while yielding SSE 'step' events as it calls
    on_step(...) — including calls made from worker threads via asyncio.to_thread.
    Ends with one 'result' event (the value work() returned) or one 'error' event
    (str(exc) if work() raised). The HTTP status code is always 200 once streaming
    starts, so the frontend must check for an 'error' event rather than response.ok.
    """
    updates: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

    def on_step(step: str, status: str) -> None:
        updates.put(("step", {"step": step, "status": status}))

    async def run() -> None:
        try:
            result = await work(on_step)
            updates.put(("result", result))
        except Exception as exc:
            log.exception("SSE pipeline failed")
            updates.put(("error", {"detail": str(exc)}))

    task = asyncio.create_task(run())
    try:
        while True:
            try:
                kind, payload = await asyncio.to_thread(updates.get, True, 0.2)
            except queue.Empty:
                if task.done():
                    break
                continue
            yield _sse(kind, payload)
            if kind in ("result", "error"):
                break
    finally:
        if not task.done():
            await task
