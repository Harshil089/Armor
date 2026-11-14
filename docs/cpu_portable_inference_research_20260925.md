# CPU-portable inference research (2026-09-25)

## Recommendation

First test adaptive early exit inside the existing single ModernBERT encoder. Train small action heads at selected intermediate layers using the existing action-feedback corpus; stop only on a calibrated, high-confidence decision and otherwise continue through the current final layer. This avoids another encoder and a full encoder retrain. Keep the current full-depth path as the fallback. A smaller distilled encoder remains a later option only if early exit fails its latency and safety gates.

“CPU-portable” means the software works on supported CPUs without a required hardware feature. It cannot mean identical latency on every CPU; operations per request and measured latency still vary with the machine. Use the same request corpus and report latency ratios on each test machine rather than choosing one required production CPU.

## Evidence

- Armor uses `answerdotai/ModernBERT-base`: locally loaded config reports 22 layers, hidden size 768, native 8,192-token context, local window 128, and global attention every third layer. The classifier runs one full encoder pass and reuses its embedding for the policy head ([`armor_base.py`](../armor/models/armor_base.py)).
- On the current CPU-only path, seven locally logged 1,566–3,211-token prompts averaged 2.09 s, with a 4.06 s maximum. The 70-item long-text gate averaged 89 ms, p95 196 ms, and maximum 339 ms; 40/70 cases used the encoder and the rest were decided by Fast Scan. The gate's Jev labels are AI labels, not human ground truth ([latency report](performance_latency_cpu_20260925.md)).
- The original ModernBERT paper reports a native 8,192-token encoder and tests its efficiency on an NVIDIA RTX 4090. Its results do not establish CPU latency for Armor's deployment path. ModernBERT is already an optimized long-context encoder; switching to BERT-large would increase the work. [ModernBERT paper](https://arxiv.org/abs/2412.13663)
- Knowledge distillation is a well-established way to reduce encoder depth. DistilBERT reported a 40% smaller model and 60% higher inference speed while retaining most language-understanding performance; TinyBERT and Patient Knowledge Distillation report transferring intermediate-layer and task-specific knowledge into smaller students. Those benchmark numbers do not predict Armor's accuracy or speed, and standard BERT students must not be substituted as-is because Armor needs long context. [DistilBERT](https://arxiv.org/abs/1910.01108), [TinyBERT](https://arxiv.org/abs/1909.10351), [Patient Knowledge Distillation](https://arxiv.org/abs/1908.09355)
- Adaptive early exit is the first experiment to try here: FastBERT and patience-based exit attach classifiers to intermediate layers and stop easy inputs early. It keeps the current encoder and weights, but needs well-calibrated exits because a confident shallow-layer error could stop a correct full-depth decision. [FastBERT](https://arxiv.org/abs/2004.02178), [BERT Loses Patience](https://proceedings.neurips.cc/paper/2020/hash/d4dd111a4fd973394238aca5c05bebe9-Abstract.html)
- Runtime conversion and quantization are not universal speed switches. Armor's FP32 ONNX export was slower than PyTorch on the measured long-text gate; the temporary blanket dynamic INT8 export reduced AI-label accuracy from 87.1% to 70.0% without improving p95. ONNX Runtime documents both accuracy loss and hardware-dependent speed gains, and provides graph-level debugging and node exclusion for quantization. Revisit selective quantization only after the early-exit candidate is evaluated. [ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Proposed experiment

1. Extract CLS features from layers 6, 12, and 18 of the current encoder for the existing action-training rows; fit only small action heads. A first timing probe is recorded in [`early_exit_feature_extraction_probe_20260925.md`](early_exit_feature_extraction_probe_20260925.md).
2. Keep the full-depth action path for low-confidence outputs. Calibrate exit thresholds on training/validation splits only; do not use the frozen test gates to fit exits.
3. Compare against the same frozen AI-labeled gate plus a newly held-out, independently generated long-prompt set. Report per-action precision/recall/F1, benign-to-block errors, block misses, exit coverage, and decisions by exit depth. Do not call these human-ground-truth results.
4. Measure end-to-end mean, p50, p95, max, and memory by token-length bands (up to 512, 513–1,500, 1,501–3,000, 3,001–8,192). Run on available CPU systems as portability checks; do not bake CPU vendor checks into policy decisions.
5. Promote only if it improves p95 for long prompts without increasing safety misses or benign hard blocks. Otherwise retain the full-depth path and revisit a distilled single-encoder candidate separately.

Promotion gate: no new benign hard blocks or block misses on the frozen safety regression sets, no lower macro-F1 or block recall against their recorded AI labels, and at least 1.5× lower p95 on the 1,501–8,192-token band on the same machine. If early exit misses the quality gate, retain the current full-depth path; only then consider a distilled encoder or selective quantization as separate candidates.

Early exit also still tokenizes and inspects every input token. Its benefit comes from skipping later encoder blocks on requests that meet the calibrated exit condition; it will not lower worst-case latency when a request must run to the final layer.
