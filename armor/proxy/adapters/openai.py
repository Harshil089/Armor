import json


class OpenAIChatAdapter:
    """Adapter for OpenAI-shaped /v1/chat/completions requests and responses."""

    def extract_prompt_text(self, request_json: dict) -> str:
        """The newest user message, not the whole conversation.

        ponytail: this joined every message with "\\n" until 2026-09-20.
        Chat clients resend the full history on every turn, so what reached
        the classifier was a 100+ word blob in which the message the user
        had just typed was a rounding error — and the score tracked the
        accumulated history instead of the new input. Measured on live
        traffic through this proxy, that made the classifier stop
        discriminating entirely: "write me an exploit" and "how are you"
        both scored 0.99. Scored on their own, the same two are 0.650 and
        0.197.

        Only the newest user turn is new, unvetted input; earlier turns were
        already inspected when they were sent. Re-scanning them every turn
        costs tokens and destroys the signal.

        Known limitation: indirect injection carried in tool results or
        assistant turns is not covered by this. That needs its own handling
        (inspect tool outputs as their own direction), not a return to
        concatenating everything.
        """
        messages = request_json.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        # No user turn (or a non-standard shape) — fall back to the last
        # message so a malformed request is still inspected, never skipped.
        return str(messages[-1].get("content", "")) if messages else ""

    def extract_response_text(self, response_json: dict) -> str:
        """Plain content plus any tool calls — a non-streaming response
        with content=null and only tool_calls set (the common shape for a
        pure function-call turn) previously inspected an empty string,
        i.e. always allowed regardless of what the tool call actually did."""
        choices = response_json.get("choices", [])
        parts = []
        for choice in choices:
            message = choice.get("message") or {}
            parts.append(str(message.get("content", "")))
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                parts.append(str(function.get("name", "")))
                parts.append(str(function.get("arguments", "")))
        return "\n".join(parts)

    def build_block_response(self, reason: str) -> dict:
        return {
            "id": "armor-block",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Blocked by Armor: {reason}",
                    },
                    "finish_reason": "content_filter",
                }
            ],
        }

    def build_review_response(self) -> dict:
        response = self.build_block_response("review_required")
        response["id"] = "armor-review"
        response["choices"][0]["message"]["content"] = "Held by Armor for review: more context is required."
        return response

    def extract_stream_delta_text(self, event_json: dict) -> str:
        """Plain content deltas plus tool-call name/arguments deltas — a
        streamed tool call (e.g. a shell-exec function's arguments building
        up token by token) is a real attack surface. Mirrors the same fix
        in extract_response_text for the non-streaming shape."""
        choices = event_json.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        parts = [str(delta.get("content", ""))]
        for tool_call in delta.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            parts.append(str(function.get("name", "")))
            parts.append(str(function.get("arguments", "")))
        return "".join(parts)

    def build_stream_kill_event(self, reason: str) -> bytes:
        chunk = {
            "id": "armor-block",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n[Blocked by Armor: {reason}]"},
                    "finish_reason": "content_filter",
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
