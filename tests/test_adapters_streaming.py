import codecs
import json

from armor.proxy.adapters.anthropic import AnthropicMessagesAdapter
from armor.proxy.adapters.openai import OpenAIChatAdapter
from armor.proxy.app import _drain_sse_events, _drain_sse_text


def test_openai_extract_stream_delta_text():
    a = OpenAIChatAdapter()
    event = {"choices": [{"delta": {"content": "hello"}}]}
    assert a.extract_stream_delta_text(event) == "hello"
    assert a.extract_stream_delta_text({"choices": []}) == ""
    assert a.extract_stream_delta_text({"choices": [{"delta": {}}]}) == ""


def test_openai_extract_stream_delta_text_covers_tool_call_name_and_arguments():
    a = OpenAIChatAdapter()
    name_event = {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "exec_shell", "arguments": ""}}]}}]
    }
    assert a.extract_stream_delta_text(name_event) == "exec_shell"

    args_event = {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "rm -rf /"}}]}}]
    }
    assert a.extract_stream_delta_text(args_event) == "rm -rf /"


def test_anthropic_extract_stream_delta_text():
    a = AnthropicMessagesAdapter()
    event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}
    assert a.extract_stream_delta_text(event) == "hello"
    assert a.extract_stream_delta_text({"type": "message_stop"}) == ""
    assert a.extract_stream_delta_text({"type": "content_block_delta", "delta": {"type": "input_json_delta"}}) == ""


def test_anthropic_extract_stream_delta_text_covers_tool_use_name_and_input():
    a = AnthropicMessagesAdapter()
    start_event = {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "exec_shell"}}
    assert a.extract_stream_delta_text(start_event) == "exec_shell"
    assert a.extract_stream_delta_text(
        {"type": "content_block_start", "content_block": {"type": "text"}}
    ) == ""

    input_event = {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"cmd":"rm -rf /"}'}}
    assert a.extract_stream_delta_text(input_event) == '{"cmd":"rm -rf /"}'


def test_openai_build_stream_kill_event_contains_reason_and_done():
    a = OpenAIChatAdapter()
    event = a.build_stream_kill_event("response_blocked")
    assert b"response_blocked" in event
    assert b"[DONE]" in event


def test_anthropic_build_stream_kill_event_contains_reason_and_stop():
    a = AnthropicMessagesAdapter()
    event = a.build_stream_kill_event("response_blocked")
    assert b"response_blocked" in event
    assert b"message_stop" in event


def test_drain_sse_events_parses_complete_events_and_keeps_partial_tail():
    buffer = (
        b'data: {"a": 1}\n\n'
        b'data: {"a": 2}\n\n'
        b'data: [DONE]\n\n'
        b'data: {"a": 3'  # incomplete
    )
    events, saw_done, remainder = _drain_sse_events(buffer)
    assert events == [(None, {"a": 1}), (None, {"a": 2})]
    assert saw_done is True
    assert remainder == b'data: {"a": 3'


def test_drain_sse_events_reassembles_across_calls():
    events1, saw_done1, remainder = _drain_sse_events(b'data: {"a": 1')
    assert events1 == []
    assert saw_done1 is False
    events2, saw_done2, remainder2 = _drain_sse_events(remainder + b'}\n\n')
    assert events2 == [(None, {"a": 1})]
    assert saw_done2 is False
    assert remainder2 == b""


def test_drain_sse_events_preserves_event_type_lines():
    buffer = b'event: content_block_delta\ndata: {"a": 1}\n\n'
    events, _, _ = _drain_sse_events(buffer)
    assert events == [("content_block_delta", {"a": 1})]


def test_incremental_utf8_decoder_handles_split_multibyte_char():
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    payload = 'data: {"text": "🚀"}\n\n'.encode("utf-8")
    split_at = payload.index(b"\xf0") + 2  # split inside 4-byte emoji

    text = decoder.decode(payload[:split_at], final=False)
    events1, _, text = _drain_sse_text(text)
    assert events1 == []

    text += decoder.decode(payload[split_at:], final=False)
    events2, _, text = _drain_sse_text(text)
    assert events2 == [(None, {"text": "🚀"})]
    assert "\ufffd" not in events2[0][1]["text"]
