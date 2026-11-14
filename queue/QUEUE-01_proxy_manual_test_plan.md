# QUEUE-01: Proxy manual test plan

## Status — initiated 2026-09-16

The deterministic proxy matrix is now covered in `tests/test_proxy.py` for
both providers: benign passthrough, prompt inspection fail-open, response
inspection fail-closed, and blocking a sensitive upstream response. The final
live-provider wire check remains pending because neither `OPENAI_API_KEY` nor
`ANTHROPIC_API_KEY` is available in the current environment.

Findings from the initial pass:

- The runnable ASGI entry point is `armor.api.main:app`; `armor.proxy.app`
  exports a router, not an app.
- A mock upstream already exists via `httpx.MockTransport`, so deterministic
  policy behavior belongs in pytest. A real key is only needed to prove the
  provider accepts the forwarded wire format.
- `fail_open_prompt=True` applies when inspection raises an exception. It does
  not turn a detected prompt attack into a flag. A `block` decision still
  stops the request before the upstream call.
- Prompt-injection text needs `ARMOR_USE_BASE_MODEL=1`; the default fast-only
  mode deterministically blocks sensitive-data patterns but does not classify
  prompt injection.

## Why a live check still remains

`armor/proxy/app.py` forwards to real upstreams. The provider list is now
config-driven (`configs/proxy/providers.json`, loaded by
`armor/proxy/config.py`) — 12 providers seeded (OpenAI, Anthropic, Groq,
Together, Fireworks, Mistral, DeepSeek, OpenRouter, Gemini's OpenAI-compat
endpoint, Perplexity, Cerebras, xAI). Adding a provider that speaks an
already-supported wire format (`"adapter": "openai"` — the vast majority)
is a JSON-only change, no code touch, verified in
`tests/test_proxy_config.py` and `tests/test_proxy.py`'s generic-provider
case. Policy behavior is tested with the existing mock-upstream seam, but
proving that each *real* provider accepts Armor's forwarded wire format
still needs a live API key and cannot run in CI. `scripts/smoke_test.sh`
(existing sibling work, already done) covers `/v1/inspect` directly; the
steps below cover the proxy around a real chat completion.

## Steps

1. Start proxy: `.venv/bin/uvicorn armor.api.main:app --port 8001`.
2. Pick any provider from `configs/proxy/providers.json` you hold a real
   key for (OpenAI and Anthropic have been used in earlier passes — any
   OpenAI-compatible provider, e.g. Groq, exercises the same generic
   adapter path with a different upstream). Not seeded yet? Add it at
   `http://localhost:8001/admin` (or `POST /admin/providers`) instead of
   editing the JSON by hand — takes effect immediately, no restart. Set
   that provider's API key env var — real key, own account, low-cost
   model to keep cost near zero.
3. Point that provider's own client `base_url` at
   `http://localhost:8001/proxy/<provider>/<path-that-provider-expects>`.
   Check `configs/proxy/providers.json` for that provider's registered
   `base_url` first — it may already include a path segment (e.g. Groq's
   is `https://api.groq.com/openai`), so the client path only needs
   what comes *after* that (`http://localhost:8001/proxy/groq/v1` for
   Groq, not `.../groq/openai/v1` — doubling `/openai` 404s with
   "Unknown request URL", caught live 2026-09-20). Send one normal chat
   request and confirm the response comes back unchanged.
   For openai/anthropic, this can be done with zero code change to the
   test client — set `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` (verified,
   `armor/proxy/env_overrides.py`) before launching it instead of passing
   `base_url` explicitly, and confirm the SDK picks it up on its own.
4. With `ARMOR_USE_BASE_MODEL=1`, send one prompt-leg attack (e.g. "Ignore
   previous instructions...") and record the model decision. Separately send
   a deterministic secret such as a test payment card and confirm a `block`
   decision prevents the upstream call.
5. Send one request designed to make the *response* leg risky (e.g. ask the
   model to repeat injected fake PII) and confirm the detected response is
   blocked. Response inspection failure is a separate case: with
   `fail_open_response=False`, an exception also blocks rather than passing
   through.
6. Repeat 3-5 against the Anthropic adapter specifically (`armor/proxy/
   adapters/anthropic.py`) — it's the one adapter with a genuinely
   different wire format from the shared OpenAI-compatible path, so it
   needs its own live check; a second OpenAI-compatible provider (step 2-5
   with a different provider name) does not need repeating once one has
   passed, since they share the same adapter code.

## Done when

The automated proxy matrix passes (including the config-driven multi-
provider tests) and the benign live-provider request works for both the
OpenAI-compatible path and the Anthropic adapter. The risky cases only
need to be repeated live if provider behavior differs from the mocked
wire format. Write any such finding back into this file or a new issue.

## Update — 2026-09-22: streaming and tool-call coverage added

Two gaps found and fixed this session, both covered by the mocked matrix
but not yet by a live check:

- **Streaming responses.** `armor/proxy/app.py`'s `relay()` previously
  passed streamed bytes straight through with zero response-side
  inspection — a blocked response leg could never actually stop a
  stream. It now parses SSE events per-event (not per raw network
  chunk — the two are not the same thing; a provider can bundle
  multiple events into one chunk), checks each one, and emits a kill
  event plus `[DONE]` if a block condition is hit mid-stream. Covered by
  `tests/test_proxy.py` and `tests/test_adapters_streaming.py` against
  the mock transport.
- **Tool-call / function-call content.** Both adapters previously
  extracted only `text`/`content` fields, so a response consisting
  purely of a tool call (OpenAI `tool_calls`, Anthropic `tool_use`) was
  inspected as an empty string and always allowed. Both adapters now
  extract tool name + arguments for streaming and non-streaming.
  Covered by `tests/test_adapters_prompt_extraction.py` and
  `tests/test_adapters_streaming.py`.

Add to steps 4-6 above: repeat with `stream=True` and with a tool-call-
only response, for both adapters. Still blocked on the same thing as the
rest of this queue item — no `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in the
current environment — so this remains mock-verified only.
