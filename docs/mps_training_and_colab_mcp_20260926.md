# MPS Training Run and Colab MCP Findings — 2026-09-26

## MPS feature extraction changes

The long-context curriculum run uses the existing ModernBERT encoder as a frozen feature extractor and trains only the existing policy head. The encoder was not fine-tuned, and the production checkpoint was not modified.

The prior MPS run appeared stalled while the main thread waited for an MPS command buffer during the synchronous transfer of pooled features to CPU. The resumed run showed that these waits were encoder work completing on MPS; sparse progress output made long single-row forwards difficult to distinguish from a hang. Long examples are processed as singletons, so exact-length padding also caused frequent sequence-shape changes.

Changes in `armor/models/train.py`:

- Long-context inputs are padded to 128-token boundaries (maximum 8,192), reducing shape churn while preserving the full input and attention mask.
- MPS feature extraction reports progress every 25 rows, including the padded sequence length. CPU reports every 100 rows.
- MPS resumable feature snapshots are written every 100 rows; CPU keeps a 500-row interval.
- The pooled CLS vector is explicitly cloned before CPU transfer so singleton feature slices do not retain the full sequence activation storage.

The restarted MPS run resumed from feature batch 10,500 and completed all 12,218 batches. macOS reported no thermal or performance warnings. The completed feature cache was saved at `/tmp/armor_long_context_features_v3_20260925.pt` (about 63 MB) and is reusable when the training rows and base checkpoint hashes match.

## Candidate outcome

The encoder stayed frozen. The run trained one candidate head from the cached features, then compared the candidate against the existing full-depth regression set. The gate rejected it on `prompt_injection` and `obfuscation_encoding`; no candidate checkpoint was saved and the deployed `armor/models/checkpoints/head.pt` remains unchanged.

| Label | Existing head | Candidate head | Change |
| --- | ---: | ---: | ---: |
| prompt_injection | 0.7840 | 0.6790 | -0.1050 |
| jailbreak | 0.8669 | 0.8671 | +0.0002 |
| sensitive_data_exposure | 0.8387 | 0.8413 | +0.0026 |
| dangerous_tool_action | 0.7841 | 0.7872 | +0.0031 |
| harmful_content | 0.6011 | 0.6744 | +0.0733 |
| obfuscation_encoding | 0.8509 | 0.7549 | -0.0960 |

The current manifest status is `regression_gate_failed`. A follow-up head-only experiment can reuse the completed feature cache; it does not need to repeat ModernBERT feature extraction while the cache metadata remains valid. The failed label regressions need diagnosis before another candidate is accepted.

## Colab MCP assessment

Google announced an open-source Colab MCP server on 2026-03-17. It lets a local MCP-compatible agent control notebook cells while code executes in the connected Colab runtime. The intended cloud-training path is to open a Colab notebook, connect a GPU runtime, and run the training notebook there.

The official Colab MCP README lists Gemini CLI, Claude Code, and Windsurf as clients. It requires clients to handle `notifications/tools/list_changed`. The Codex issue tracker has an open report that Codex receives this notification but does not refresh the MCP tool list. This workspace currently has no Colab MCP server configured, so the integration was researched but not connected in this session.

References:

- [Google Colab MCP announcement](https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/)
- [Official Colab MCP server](https://github.com/googlecolab/colab-mcp)
- [Codex issue: support `notifications/tools/list_changed`](https://github.com/openai/codex/issues/10105)
- [Colab resource limits and runtime behavior](https://research.google.com/colaboratory/faq.html)
