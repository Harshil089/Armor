import json
import os
from typing import Literal

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from armor.api.schemas import InspectRequest, InspectResponse
from armor.observability import log_store
from armor.proxy.adapters import get_adapter
from armor.proxy.config import DEFAULT_CONFIG

router = APIRouter()

_HOP_BY_HOP = {"host", "content-length", "connection"}

# ponytail: approved batches are relayed live, so batches released before a
# later block may already have reached the client. A true zero-exposure
# guarantee requires buffering the whole response, defeating streaming. The
# current batch stays withheld until inspection and any review complete.
# Checking every chunk would add an ArmorBase forward pass (~30-50ms) per
# token-sized chunk; batching to whole clauses keeps inspections proportional
# to sentences, not tokens.
_STREAM_INSPECT_EVERY_CHARS = 40


def _drain_sse_events(buffer: bytes) -> tuple[list[tuple[str | None, dict]], bool, bytes]:
    """Splits a raw SSE byte buffer on event boundaries. Returns (event_type,
    payload) pairs for complete events (event_type is None for OpenAI-style
    frames with no `event:` line — Anthropic's `event: content_block_delta`
    etc. is preserved so reconstruction doesn't drop the field real SDKs
    key off), whether a `[DONE]` sentinel was seen, and the leftover
    (possibly incomplete) tail to prepend to the next chunk.

    ponytail: decodes the whole buffer as utf-8 per call rather than
    tracking incremental decode state — a multi-byte character split
    exactly across a chunk boundary can garble one character (errors are
    replaced, not raised). Rare in practice and self-heals on the next
    chunk; a full incremental-decoder state machine is overkill for what
    this buys here.
    """
    events: list[tuple[str | None, dict]] = []
    saw_done = False
    parts = buffer.decode("utf-8", errors="replace").split("\n\n")
    remainder = parts.pop()
    for part in parts:
        event_type = None
        for line in part.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    saw_done = True
                elif payload:
                    try:
                        events.append((event_type, json.loads(payload)))
                    except json.JSONDecodeError:
                        continue
    return events, saw_done, remainder.encode("utf-8")

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


def inspect_content(direction: Literal["prompt", "response"], content: str) -> InspectResponse:
    from armor.api.main import inspect  # lazy: avoids a circular import with armor.api.main

    return inspect(InspectRequest(direction=direction, content=content))


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


@router.post("/proxy/{provider}/{path:path}")
async def proxy(provider: str, path: str, request: Request):
    config = DEFAULT_CONFIG
    if provider not in config.upstreams:
        return JSONResponse({"error": f"unknown provider: {provider}"}, status_code=404)

    upstream = config.upstreams[provider]
    adapter = get_adapter(upstream.adapter)
    body = await request.json()

    prompt_text = adapter.extract_prompt_text(body)
    jev_reason = None
    try:
        prompt_result = inspect_content("prompt", prompt_text)
    except Exception:
        if not config.fail_open_prompt:
            log_store.log_entry(
                log_store.DB_PATH,
                source="proxy",
                provider=provider,
                direction="prompt",
                content=prompt_text,
                decision="block",
                reason="inspection_unavailable",
            )
            return JSONResponse(
                adapter.build_block_response("inspection_unavailable"),
                status_code=200,
                headers={"X-Armor-Decision": "block", "X-Armor-Reason": "inspection_unavailable"},
            )
        prompt_result = None

    if (prompt_result is not None and prompt_result.decision == "review"
            and os.environ.get("ARMOR_JEV_REVIEW_ENABLED") == "1"):
        from armor.proxy.jev import resolve_review

        jev_reason = await resolve_review(get_client(), prompt_result, prompt_text)
        if prompt_result.decision == "allow":
            log_store.log_entry(
                log_store.DB_PATH, source="proxy", provider=provider, direction="prompt",
                content=prompt_text, decision="allow", route=prompt_result.route,
                risk_score=prompt_result.risk_score, reason=jev_reason,
            )

    if prompt_result is not None and prompt_result.decision in ("block", "review"):
        held_for_review = prompt_result.decision == "review"
        log_store.log_entry(
            log_store.DB_PATH,
            source="proxy",
            provider=provider,
            direction="prompt",
            content=prompt_text,
            decision=prompt_result.decision,
            route=prompt_result.route,
            risk_score=prompt_result.risk_score,
            reason=jev_reason or ("review_required" if held_for_review else "prompt_blocked"),
        )
        return JSONResponse(
            adapter.build_review_response() if held_for_review else adapter.build_block_response("prompt_blocked"),
            status_code=200,
            headers={"X-Armor-Decision": prompt_result.decision, "X-Armor-Route": prompt_result.route},
        )

    url = f"{upstream.base_url.rstrip('/')}/{path}"
    fwd_headers = _forward_headers(request.headers)
    client = get_client()

    if body.get("stream"):
        # Chunked-with-kill-switch (proxy design doc section 5): relay
        # chunks live, but periodically inspect the accumulated response
        # text and stop the stream if it turns out to be a block — see
        # _STREAM_INSPECT_EVERY_CHARS above for the exposure-window tradeoff.
        async def relay():
            accumulated_text = ""
            unchecked_text = ""
            pending_frames: list[bytes] = []
            sse_buffer = b""
            saw_done = False
            async with client.stream(
                "POST", url, json=body, headers=fwd_headers, timeout=upstream.timeout_s
            ) as upstream_resp:
                async for chunk in upstream_resp.aiter_bytes():
                    sse_buffer += chunk
                    events, chunk_saw_done, sse_buffer = _drain_sse_events(sse_buffer)
                    saw_done = saw_done or chunk_saw_done
                    # Per-event granularity, not per-raw-chunk: a transport
                    # can bundle several SSE events into one network read
                    # (verified live — a slow upstream writing one event per
                    # flush still arrived at the proxy as one combined
                    # chunk), so checking after the whole chunk and yielding
                    # the whole chunk would ship every event bundled
                    # alongside the one that tripped the block. Check and
                    # yield one event at a time instead.
                    for event_type, event in events:
                        frame = f"data: {json.dumps(event)}\n\n"
                        if event_type is not None:
                            frame = f"event: {event_type}\n{frame}"
                        pending_frames.append(frame.encode())
                        delta = adapter.extract_stream_delta_text(event)
                        if delta:
                            accumulated_text += delta
                            unchecked_text += delta

                        if len(unchecked_text) >= _STREAM_INSPECT_EVERY_CHARS:
                            unchecked_text = ""
                            try:
                                result = inspect_content("response", accumulated_text)
                            except Exception:
                                if not config.fail_open_response:
                                    log_store.log_entry(
                                        log_store.DB_PATH,
                                        source="proxy",
                                        provider=provider,
                                        direction="response",
                                        content=accumulated_text,
                                        decision="block",
                                        reason="inspection_unavailable",
                                    )
                                    yield adapter.build_stream_kill_event("inspection_unavailable")
                                    return
                                result = None
                            if (result is not None and result.decision == "review"
                                    and os.environ.get("ARMOR_JEV_REVIEW_ENABLED") == "1"):
                                from armor.proxy.jev import resolve_review

                                jev_reason = await resolve_review(get_client(), result, accumulated_text)
                            else:
                                jev_reason = None
                            if result is not None and result.decision == "block":
                                log_store.log_entry(
                                    log_store.DB_PATH,
                                    source="proxy",
                                    provider=provider,
                                    direction="response",
                                    content=accumulated_text,
                                    decision="block",
                                    route=result.route,
                                    risk_score=result.risk_score,
                                    reason=jev_reason or "stream_response_blocked",
                                )
                                yield adapter.build_stream_kill_event("response_blocked")
                                return
                            for pending_frame in pending_frames:
                                yield pending_frame
                            pending_frames.clear()

            # Stream finished without tripping the kill switch — one final
            # inspection over the complete text so the log reflects the
            # whole response, not just whatever fell on an inspection
            # boundary (last partial segment may be under the char count).
            try:
                result = inspect_content("response", accumulated_text)
            except Exception:
                result = None
            if (result is not None and result.decision == "review"
                    and os.environ.get("ARMOR_JEV_REVIEW_ENABLED") == "1"):
                from armor.proxy.jev import resolve_review

                jev_reason = await resolve_review(get_client(), result, accumulated_text)
            else:
                jev_reason = None
            log_store.log_entry(
                log_store.DB_PATH,
                source="proxy",
                provider=provider,
                direction="response",
                content=accumulated_text,
                decision=result.decision if result is not None else "allow",
                route=result.route if result is not None else None,
                risk_score=result.risk_score if result is not None else 0.0,
                reason=jev_reason,
            )
            if result is not None and result.decision == "block":
                yield adapter.build_stream_kill_event("response_blocked")
                return
            for pending_frame in pending_frames:
                yield pending_frame
            if saw_done:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"X-Armor-Decision": "allow", "X-Armor-Route": "proxy-stream-kill-switch"},
        )

    upstream_resp = await client.post(url, json=body, headers=fwd_headers, timeout=upstream.timeout_s)
    resp_json = upstream_resp.json()

    response_text = adapter.extract_response_text(resp_json)
    try:
        response_result = inspect_content("response", response_text)
    except Exception:
        if not config.fail_open_response:
            log_store.log_entry(
                log_store.DB_PATH,
                source="proxy",
                provider=provider,
                direction="response",
                content=response_text,
                decision="block",
                reason="inspection_unavailable",
            )
            return JSONResponse(
                adapter.build_block_response("inspection_unavailable"),
                status_code=200,
                headers={"X-Armor-Decision": "block", "X-Armor-Reason": "inspection_unavailable"},
            )
        response_result = None

    jev_reason = None
    if (response_result is not None and response_result.decision == "review"
            and os.environ.get("ARMOR_JEV_REVIEW_ENABLED") == "1"):
        from armor.proxy.jev import resolve_review

        jev_reason = await resolve_review(get_client(), response_result, response_text)

    if response_result is not None and response_result.decision == "block":
        log_store.log_entry(
            log_store.DB_PATH,
            source="proxy",
            provider=provider,
            direction="response",
            content=response_text,
            decision="block",
            route=response_result.route,
            risk_score=response_result.risk_score,
            reason=jev_reason or "response_blocked",
        )
        return JSONResponse(
            adapter.build_block_response("response_blocked"),
            status_code=200,
            headers={"X-Armor-Decision": "block", "X-Armor-Route": response_result.route},
        )

    decision = response_result.decision if response_result is not None else "allow"
    log_store.log_entry(
        log_store.DB_PATH,
        source="proxy",
        provider=provider,
        direction="response",
        content=response_text,
        decision=decision,
        route=response_result.route if response_result is not None else None,
        risk_score=response_result.risk_score if response_result is not None else 0.0,
        reason=jev_reason,
    )
    return JSONResponse(
        resp_json,
        status_code=upstream_resp.status_code,
        headers={"X-Armor-Decision": decision},
    )
