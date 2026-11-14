# Screenshots

UI evidence that text logs cannot capture. Capture manually from a browser
and reference from `plan.md`.

Worth having:

- `admin_logs_page.png` — `http://localhost:8001/admin/logs-page` with a
  decision row expanded, showing inspected content, decision, reason, and
  per-label findings.
- `client_block_message.png` — a chat client showing
  `Blocked by Armor: prompt_blocked` for a malicious prompt.
- `admin_providers_page.png` — `http://localhost:8001/admin/` showing the
  provider table and the proxy URL to point a client at.

The underlying decision data is preserved as text in
`../logs/inspection_log_sample.json` (same shape `/admin/logs/{id}`
returns), which diffs properly and does not go stale as a binary.
