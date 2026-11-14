# Frozen 1K–8K full-depth baseline (2026-09-25)

Corpus: `armor_long_context_gate_v1_20260925` · 105 rows · SHA-256 `3883afe09563ccf625567804fed317340bf791b4e3e51e10f8c2ad16b21c1272`.
All labels are **synthetic scenario contracts**, not human-ground-truth labels. Metrics below are model-to-contract agreement.
Serving path: unchanged `/v1/inspect` logic, current ArmorBase checkpoint, PyTorch CPU, adaptive shadow/windowing off, local production policy action head on. Two warmups excluded. Database writes disabled so full prompt text is not duplicated in benchmark logs; response timing is the same pre-log timing returned by the API.

## Overall

n=105 · accuracy 34.3% · macro F1 21.3% · block recall 40.0% · block precision 46.2% · allow precision 56.2% · flag recall 0.0% · review rate 32.4%.
Compute paths: `{"fast_scan_decisive": 16, "full_depth_legacy": 89}`. Routes: `{"escalated": 102, "fast": 3}`.

| Bucket | n | tokens | mean ms | p50 ms | p95 ms | p99 ms | max ms | acc. | block recall | allow precision | review | tokenization mean | encoder mean | classifier head mean | policy heads mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 21 | 1003–1036 | 678.4 | 677.1 | 719.8 | 750.9 | 758.7 | 47.6% | 14.3% | 50.0% | 9.5% | 5.8 | 642.7 | 0.0 | 0.3 |
| 2k | 21 | 2039–2075 | 1401.1 | 1912.1 | 2082.9 | 2109.6 | 2116.3 | 42.9% | 27.3% | 66.7% | 28.6% | 7.2 | 1362.0 | 0.0 | 0.3 |
| 4k | 21 | 4108–4147 | 5227.6 | 6065.9 | 6467.8 | 7482.2 | 7735.8 | 19.0% | 12.5% | 75.0% | 61.9% | 14.9 | 5163.8 | 0.1 | 0.5 |
| 6k | 21 | 6177–6225 | 10479.3 | 12927.3 | 13421.8 | 14472.0 | 14734.5 | 28.6% | 60.0% | 0.0% | 33.3% | 21.1 | 10397.5 | 0.1 | 0.6 |
| 8k | 21 | 7850–7891 | 17537.9 | 20394.9 | 21345.2 | 26791.9 | 28153.5 | 33.3% | 77.8% | — | 28.6% | 27.0 | 17427.5 | 1.4 | 4.5 |

Stage timings are per-request trace values emitted by the current inspection path. Policy-head timing wraps the two existing small policy-head calls. `total_ms` is API compute/response latency before the separate SQLite audit-log write. The benchmark ran on this Apple M2, 8 GB development machine; it is not a production-host SLO.
