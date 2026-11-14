import json


def _block_text(content) -> str:
    """Plain text blocks plus tool_use blocks' name+input — a response made
    entirely of a tool_use block (no text block at all, the common shape
    for a pure tool-call turn) previously inspected an empty string,
    i.e. always allowed regardless of what the tool call actually did."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                parts.append(str(block.get("name", "")))
                parts.append(json.dumps(block.get("input", {})))
            elif block.get("type") == "tool_result":
                parts.append(_block_text(block.get("content")))
            else:
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return ""


class AnthropicMessagesAdapter:
    """Adapter for Anthropic-shaped /v1/messages requests and responses."""

    def extract_prompt_text(self, request_json: dict) -> str:
        """The newest user message (including tool_result blocks), not the whole
        conversation — see armor/proxy/adapters/openai.py's extract_prompt_text
        for the measurement that motivated this (concatenating history collapsed
        the classifier's discrimination on live traffic)."""
        messages = request_json.get("messages", [])
        for message in reversed(messages):
            if message.get("role") == "user":
                return _block_text(message.get("content"))
        if messages:
            return _block_text(messages[-1].get("content"))
        if "system" in request_json:
            return _block_text(request_json.get("system"))
        return ""

    def extract_response_text(self, response_json: dict) -> str:
        return _block_text(response_json.get("content"))

    def build_block_response(self, reason: str) -> dict:
        return {
            "id": "armor-block",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": f"Blocked by Armor: {reason}"}],
            "stop_reason": "content_filter",
        }

    def build_review_response(self) -> dict:
        response = self.build_block_response("review_required")
        response["id"] = "armor-review"
        response["content"][0]["text"] = "Held by Armor for review: more context is required."
        return response

    def extract_stream_delta_text(self, event_json: dict) -> str:
        """Plain text deltas, plus tool-use name (at content_block_start)
        and its streamed JSON input (input_json_delta) — a streamed tool
        call was previously invisible to response inspection entirely (only
        text_delta was handled), the same gap OpenAI's adapter had for
        tool_calls."""
        event_type = event_json.get("type")
        if event_type == "content_block_start":
            block = event_json.get("content_block") or {}
            return str(block.get("name", "")) if block.get("type") == "tool_use" else ""
        if event_type == "content_block_delta":
            delta = event_json.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return str(delta.get("text", ""))
            if delta_type == "input_json_delta":
                return str(delta.get("partial_json", ""))
        return ""

    def build_stream_kill_event(self, reason: str) -> bytes:
        delta_event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"\n[Blocked by Armor: {reason}]"},
        }
        stop_event = {"type": "message_stop"}
        return (
            f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
            f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n"
        ).encode()
