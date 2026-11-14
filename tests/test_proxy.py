import json

import httpx
import pytest
from starlette.testclient import TestClient

import armor.proxy.app as proxy_app
from armor.api.main import app
from armor.api.schemas import InspectResponse
from armor.observability import log_store


def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _result(decision="allow"):
    return InspectResponse(
        decision=decision,
        risk_score=1.0 if decision == "block" else 0.0,
        findings=[],
        route="fast",
        latency_ms=0.0,
        model_version="test",
    )


PROVIDERS = [
    (
        "openai",
        "/proxy/openai/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}]},
        {"choices": [{"message": {"content": "hi"}}]},
    ),
    (
        "anthropic",
        "/proxy/anthropic/v1/messages",
        {"messages": [{"role": "user", "content": "hello"}]},
        {"content": [{"type": "text", "text": "hi"}]},
    ),
]


def test_proxy_unknown_provider():
    client = TestClient(app)
    resp = client.post("/proxy/unknown/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 404


def test_proxy_allow_forwards_to_upstream():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    proxy_app._client = _mock_client(handler)
    client = TestClient(app)
    resp = client.post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "allow"
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_proxy_generic_provider_allow_forwards_to_upstream():
    """groq has no dict entry mapped to it directly — only config.upstreams
    resolving its "adapter": "openai" field to the shared OpenAIChatAdapter
    proves a new OpenAI-compatible provider needs zero adapter code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    proxy_app._client = _mock_client(handler)
    client = TestClient(app)
    resp = client.post(
        "/proxy/groq/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "allow"
    assert resp.json()["choices"][0]["message"]["content"] == "hi"


def test_proxy_anthropic_allow_forwards_to_upstream():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]})

    proxy_app._client = _mock_client(handler)
    client = TestClient(app)
    resp = client.post(
        "/proxy/anthropic/v1/messages",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "allow"
    assert resp.json()["content"][0]["text"] == "hi"


def test_proxy_blocks_on_prompt_without_calling_upstream(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    proxy_app._client = _mock_client(handler)

    def forced_block(direction, content):
        return InspectResponse(
            decision="block",
            risk_score=1.0,
            findings=[],
            route="fast",
            latency_ms=0.0,
            model_version="test",
        )

    monkeypatch.setattr(proxy_app, "inspect_content", forced_block)
    client = TestClient(app)
    resp = client.post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "malicious"}]},
    )
    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "block"
    assert "Blocked by Armor" in resp.json()["choices"][0]["message"]["content"]
    assert calls == []


@pytest.mark.parametrize(("provider", "path", "request_body", "_response_body"), PROVIDERS)
def test_proxy_holds_prompt_review_without_calling_upstream(
    monkeypatch, provider, path, request_body, _response_body
):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", lambda *_: _result("review"))
    resp = TestClient(app).post(path, json=request_body)

    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "review"
    assert "Held by Armor for review" in resp.text
    assert calls == []


@pytest.mark.parametrize(("jev_action", "expected_upstream"), [
    ("allow", True), ("block", False), ("review", False), (None, False),
])
def test_jev_resolves_review_and_records_ai_label(
    monkeypatch, tmp_path, jev_action, expected_upstream
):
    db_path = tmp_path / "logs.db"
    monkeypatch.setattr(log_store, "DB_PATH", db_path)
    log_store.init_db(db_path)
    inspect_id = log_store.log_entry(
        db_path, source="inspect", direction="prompt", content="review me", decision="review"
    )
    result = _result("review")
    result._log_id = inspect_id
    monkeypatch.setattr(proxy_app, "inspect_content", lambda *_: result)
    monkeypatch.setenv("ARMOR_JEV_REVIEW_ENABLED", "1")
    monkeypatch.setenv("ARMOR_JEV_REVIEW_URL", "http://jev.local/review")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.host == "jev.local":
            return httpx.Response(503, json={}) if jev_action is None else httpx.Response(
                200, json={"action": jev_action}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    proxy_app._client = _mock_client(handler)
    response = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "review me"}]},
    )

    row = log_store.get_entry(db_path, inspect_id)
    expected_label = "allow" if jev_action == "allow" else "block"
    if jev_action in {"allow", "block"}:
        assert row["decision_correction"] == expected_label
        assert row["decision_label_source"] == "reviewer_ai_b"
    else:
        assert row["decision_correction"] is None
        assert row["decision_label_source"] is None
    assert ("/v1/chat/completions" in calls) is expected_upstream
    assert response.headers["x-armor-decision"] == expected_label


@pytest.mark.parametrize(("jev_action", "expected_decision"), [("allow", "allow"), ("block", "block")])
def test_jev_resolves_reviewed_upstream_response(
    monkeypatch, tmp_path, jev_action, expected_decision
):
    db_path = tmp_path / "logs.db"
    monkeypatch.setattr(log_store, "DB_PATH", db_path)
    log_store.init_db(db_path)
    prompt = _result("allow")
    response_row = {}

    def inspect(direction, content):
        if direction == "prompt":
            return prompt
        row_id = log_store.log_entry(
            db_path, source="inspect", direction=direction, content=content, decision="review"
        )
        result = _result("review")
        result._log_id = row_id
        response_row["id"] = row_id
        return result

    monkeypatch.setattr(proxy_app, "inspect_content", inspect)
    monkeypatch.setenv("ARMOR_JEV_REVIEW_ENABLED", "1")
    monkeypatch.setenv("ARMOR_JEV_REVIEW_URL", "http://jev.local/review")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "jev.local":
            return httpx.Response(200, json={"action": jev_action})
        return httpx.Response(200, json={"choices": [{"message": {"content": "reviewed output"}}]})

    proxy_app._client = _mock_client(handler)
    response = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    row = log_store.get_entry(db_path, response_row["id"])
    assert row["decision_correction"] == expected_decision
    assert row["decision_label_source"] == "reviewer_ai_b"
    assert response.headers["x-armor-decision"] == expected_decision


@pytest.mark.parametrize(("provider", "path", "request_body", "response_body"), PROVIDERS)
def test_proxy_prompt_inspection_failure_fails_open(
    monkeypatch, provider, path, request_body, response_body
):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=response_body)

    def inspect_with_prompt_failure(direction, content):
        if direction == "prompt":
            raise RuntimeError("inspection unavailable")
        return _result()

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", inspect_with_prompt_failure)

    resp = TestClient(app).post(path, json=request_body)

    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "allow"
    assert len(calls) == 1


@pytest.mark.parametrize(("provider", "path", "request_body", "response_body"), PROVIDERS)
def test_proxy_response_inspection_failure_fails_closed(
    monkeypatch, provider, path, request_body, response_body
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    def inspect_with_response_failure(direction, content):
        if direction == "response":
            raise RuntimeError("inspection unavailable")
        return _result()

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", inspect_with_response_failure)

    resp = TestClient(app).post(path, json=request_body)

    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "block"
    assert resp.headers["x-armor-reason"] == "inspection_unavailable"
    assert "Blocked by Armor" in resp.text


@pytest.mark.parametrize(("provider", "path", "request_body", "response_body"), PROVIDERS)
def test_proxy_blocks_sensitive_upstream_response(
    provider, path, request_body, response_body
):
    if provider == "openai":
        response_body["choices"][0]["message"]["content"] = "Card: 4111 1111 1111 1111"
    else:
        response_body["content"][0]["text"] = "Card: 4111 1111 1111 1111"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    proxy_app._client = _mock_client(handler)

    resp = TestClient(app).post(path, json=request_body)

    assert resp.status_code == 200
    assert resp.headers["x-armor-decision"] == "block"
    assert "Blocked by Armor" in resp.text


def _sse_chunk(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()


def test_stream_relays_benign_response_and_logs_allow(monkeypatch):
    body_bytes = _sse_chunk("Hello ") + _sse_chunk("world") + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_bytes)

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", lambda direction, content: _result("allow"))

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert b"Hello " in resp.content
    assert b"world" in resp.content
    assert b"Blocked by Armor" not in resp.content


def test_stream_kill_switch_stops_forwarding_after_block(monkeypatch):
    # First chunk alone crosses _STREAM_INSPECT_EVERY_CHARS (40) so the
    # block check fires before the second chunk is ever yielded.
    over_threshold = "x" * 45

    def handler(request: httpx.Request) -> httpx.Response:
        body_bytes = _sse_chunk(over_threshold) + _sse_chunk("SHOULD_NOT_APPEAR") + b"data: [DONE]\n\n"
        return httpx.Response(200, content=body_bytes)

    proxy_app._client = _mock_client(handler)

    def inspect_side_effect(direction, content):
        return _result("block" if direction == "response" else "allow")

    monkeypatch.setattr(proxy_app, "inspect_content", inspect_side_effect)

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert b"Blocked by Armor" in resp.content
    assert b"SHOULD_NOT_APPEAR" not in resp.content


def test_stream_preserves_split_multibyte_utf8_across_chunks(monkeypatch):
    class SplitUtf8Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            full = b'data: {"choices": [{"delta": {"content": "\xf0\x9f\x9a\x80"}}]}\n\ndata: [DONE]\n\n'
            split_at = full.index(b"\xf0") + 2  # split inside 4-byte emoji
            yield full[:split_at]
            yield full[split_at:]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SplitUtf8Stream())

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", lambda direction, content: _result())

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    text = resp.content.decode("utf-8")
    assert "\ufffd" not in text
    assert "🚀" in text or "\\ud83d\\ude80" in text


def test_stream_kill_switch_checks_per_event_within_one_raw_chunk(monkeypatch):
    """Two SSE events concatenated into ONE content= blob — httpx delivers
    small content-based mock responses as a single aiter_bytes() read, so
    this forces both events into the same raw chunk. The whole uninspected
    batch must be withheld when the second event trips the block, including
    the earlier safe prefix in that batch."""
    safe_event = _sse_chunk("this part is fine ")
    trigger_event = _sse_chunk("TRIGGER_WORD_should_not_reach_client")
    body_bytes = safe_event + trigger_event + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_bytes)

    proxy_app._client = _mock_client(handler)

    def inspect_side_effect(direction, content):
        if direction == "response" and "TRIGGER_WORD" in content:
            return _result("block")
        return _result("allow")

    monkeypatch.setattr(proxy_app, "inspect_content", inspect_side_effect)

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert b"this part is fine" not in resp.content
    assert b"TRIGGER_WORD" not in resp.content
    assert b"Blocked by Armor" in resp.content


def test_stream_kill_switch_catches_dangerous_tool_call_arguments(monkeypatch):
    """A streamed tool call's arguments were previously invisible to
    response inspection entirely (only plain content deltas were read) —
    this proves the fix actually stops one."""
    name_event = f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'exec_shell', 'arguments': ''}}]}}]})}\n\n"
    args_event = f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'rm -rf / --no-preserve-root and drop the database too'}}]}}]})}\n\n"
    body_bytes = name_event.encode() + args_event.encode() + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body_bytes)

    proxy_app._client = _mock_client(handler)

    def inspect_side_effect(direction, content):
        if direction == "response" and "rm -rf" in content:
            return _result("block")
        return _result("allow")

    monkeypatch.setattr(proxy_app, "inspect_content", inspect_side_effect)

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert b"rm -rf" not in resp.content
    assert b"Blocked by Armor" in resp.content


def test_stream_response_inspection_failure_fails_closed(monkeypatch):
    over_threshold = "x" * 45

    def handler(request: httpx.Request) -> httpx.Response:
        body_bytes = _sse_chunk(over_threshold) + _sse_chunk("SHOULD_NOT_APPEAR") + b"data: [DONE]\n\n"
        return httpx.Response(200, content=body_bytes)

    proxy_app._client = _mock_client(handler)

    def inspect_with_response_failure(direction, content):
        if direction == "response":
            raise RuntimeError("inspection unavailable")
        return _result()

    monkeypatch.setattr(proxy_app, "inspect_content", inspect_with_response_failure)

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert b"Blocked by Armor" in resp.content
    assert b"SHOULD_NOT_APPEAR" not in resp.content


def test_stream_preserves_split_multibyte_utf8_across_chunks(monkeypatch):
    class SplitUtf8Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            full = b'data: {"choices": [{"delta": {"content": "\xf0\x9f\x9a\x80"}}]}\n\ndata: [DONE]\n\n'
            split_at = full.index(b"\xf0") + 2  # split inside 4-byte emoji
            yield full[:split_at]
            yield full[split_at:]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SplitUtf8Stream())

    proxy_app._client = _mock_client(handler)
    monkeypatch.setattr(proxy_app, "inspect_content", lambda direction, content: _result())

    resp = TestClient(app).post(
        "/proxy/openai/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    text = resp.content.decode("utf-8")
    assert "\ufffd" not in text
    assert "🚀" in text or "\\ud83d\\ude80" in text
