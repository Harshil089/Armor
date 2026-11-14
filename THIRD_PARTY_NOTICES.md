# Third-Party Notices

Armor's classifier weights are a fine-tune of a third-party base model,
trained on third-party datasets. This file gives the attribution, license
notice, and use terms for every external model, dataset, and adapted code
fragment in that lineage. **Ship this file and `LICENSES/` with any
distribution of Armor's code or trained weights.**

No dataset is redistributed in this repository. Each one is downloaded at
training time from its original host (pinned revision where noted). The
only redistributed artifacts are Armor's own code and the trained weights.

Self-generated synthetic data (`armor/data/synthetic_*.py`) has no
third-party license and is not listed.

## 1. Base model

| Model | License | Copyright | Use |
|---|---|---|---|
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | Apache-2.0 | Answer.AI and LightOn; code Copyright 2022 MosaicML Examples authors ([repo](https://github.com/AnswerDotAI/ModernBERT)) | Encoder for every Armor checkpoint. The top encoder layers are fine-tuned, so the shipped weights are a modified derivative. |

**Change notice (Apache-2.0 §4(b)):** Armor's `head.pt` carries modified
ModernBERT encoder weights (top layers fine-tuned on the datasets below)
plus a new 6-label classification head.

## 2. Datasets

### MIT

| Dataset | Copyright | Revision | Use in Armor |
|---|---|---|---|
| [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | SPML authors (Sharma, Gupta, Grossman), arXiv:2402.11755 | unpinned | `prompt_injection`, 800 rows (capped) |
| [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | Copyright (c) 2024 TrustAIRLab | unpinned | `jailbreak`, 600 pos + 600 neg (capped) |
| [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench), re-host of [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | Copyright (c) 2024 centerforaisafety | unpinned | `harmful_content` positives, 300 rows including 67 `cybercrime_intrusion` |
| [ai-safety-institute/AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | Copyright (c) 2024 Gray Swan AI and UK AI Safety Institute. **MIT plus an additional clause, see §3** | unpinned | `dangerous_tool_action`, 416 rows |
| [UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) (agentdojo task suites), derived from [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | Copyright (c) 2024 UK AI Security Institute; AgentDojo: Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr | `main` | `dangerous_tool_action`, 116 rows scraped from raw task source |
| [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | Copyright (c) 2023 Qiusi Zhan | `main` | `dangerous_tool_action`, 79 rows |
| [jinjinyien/ToolSafety](https://huggingface.co/datasets/jinjinyien/ToolSafety) | ToolSafety authors (EMNLP 2025). MIT confirmed directly by Prof. Pinjia He and the HF dataset owner (ARCHIVE-QUEUE-03) | unpinned | `dangerous_tool_action`, 500 rows (capped) |
| [RainWatcher/LongPIBench](https://huggingface.co/datasets/RainWatcher/LongPIBench) / [liu00222/LongPIBench](https://github.com/liu00222/LongPIBench) | Copyright (c) 2025 Yupei Liu | unpinned | `prompt_injection` long-document windows. **Code adapted** in `armor/data/longpibench.py`: injection text and splice logic from `attacks.py`/`benchmark.py` |
| [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Copyright (c) 2023 JailbreakBench Team | `886acc352a31533ffbcf4ef22c744658688086fc` | `harmful_content`, 100 harmful + 100 matched benign behaviors |
| [PurCL/astra-agent-security](https://huggingface.co/datasets/PurCL/astra-agent-security) | ASTRA authors (Xu et al., 2025), PurCL | `26f9dee6b0bf46dd61e4ebee938bd76a6c8a4e75` | `harmful_content` positives, 400 sampled `request_text` rows |
| [meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama) `CybersecurityBenchmarks/datasets/mitre_frr/mitre_frr.json` | Copyright (c) Meta Platforms, Inc. and affiliates | commit `76d0bfa0f31c14049da2e350e2f4a3e17fafcaa2` | `harmful_content` + `dangerous_tool_action` negatives, 750 benign security-coding prompts |
| [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench), re-host of [llm-attacks](https://github.com/llm-attacks/llm-attacks) `harmful_behaviors.csv` | Copyright (c) 2023 Andy Zou | `9d4730540082fa4017450b65ca1c0e1d8d30446e` | `harmful_content` positives, 520 rows |

The MIT permission notice (full text in §6) applies to each row above with
that row's copyright line.

### Apache-2.0

Full license text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
None of these upstream repositories ships a NOTICE file.

| Dataset | Copyright / creator | Revision | Use in Armor |
|---|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | deepset | unpinned | `prompt_injection`, 546 rows |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | jackhhao | unpinned | `jailbreak`, 1,044 rows |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | OpenAssistant / LAION | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | negatives for `prompt_injection`, `obfuscation_encoding`, `sensitive_data_exposure` |
| [grammarly/coedit](https://huggingface.co/datasets/grammarly/coedit) | Grammarly | `e9a255c33ef910bc33a9d2b522653fa87521583e` | `obfuscation_encoding` negatives, 800 rows |
| [aradhye/agent-safety-bench](https://huggingface.co/datasets/aradhye/agent-safety-bench) | aradhye | `b7761ca4c4e7a5937f0b6ad11dd24ce19163c454` | `dangerous_tool_action`, 287 rows |
| [ryoungj/ToolEmu](https://github.com/ryoungj/ToolEmu) | ToolEmu authors (Ruan et al.) | `ac4a7ab7ed8c7985d96231e214bd6b54304b7ddb` | `dangerous_tool_action`, 322 rows |
| [walledai/SaladBench](https://huggingface.co/datasets/walledai/SaladBench) | SaladBench / OpenSafetyLab | `5d9d078571dd171990162ba6201bc81c41c7c867` | `harmful_content` positives, up to 1,000 `prompts/base` rows |
| [cyberec/Prompt-injection-dataset](https://huggingface.co/datasets/cyberec/Prompt-injection-dataset) | cyberec | `cfe0ce4a4d5eb9528ffaca55cf0db27ac609b8b1` | `prompt_injection`, up to 750 pos + 750 security-adjacent neg (`core/train`) |

**Change notice (Apache-2.0 §4(b)):** Armor does not redistribute these
files. It selects, filters, caps, and relabels rows into its own 6-label
scheme at training time; see each loader in `armor/data/`.

### CC-BY-4.0

License: <https://creativecommons.org/licenses/by/4.0/>. CC-BY requires
credit, a license link, and an indication of changes.

| Dataset | Creator | Revision | Use and changes |
|---|---|---|---|
| [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | NVIDIA Corporation | `bd96d862068e47630197de64eb91f8d1481ff3e0` | Negatives for `harmful_content`, `sensitive_data_exposure`, `obfuscation_encoding`, 800 rows. Only `user_message` rows that all annotators marked `Safe` are kept; responses and multi-turn rows are dropped, and the rest is deduplicated and sorted-sampled by ID. Card terms in §3 apply. |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | NVIDIA Corporation (Ghosh et al., 2025) | `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c` | `harmful_content`, up to 1,500 unsafe + 1,500 safe user prompts. Only the `prompt` field is used; responses and `REDACTED` rows are dropped. `violated_categories` collapse to one binary label, and `Needs Caution`, `PII/Privacy`, `Copyright/Trademark/Plagiarism`, `Unauthorized Advice`, `High Risk Gov Decision Making` and `Other` are not counted as positives. All `Malware` rows are kept; the rest is deduplicated and randomly sampled (seed 0). Card terms in §3 apply. |
| [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) | Paul Röttger et al. | `f600c994b256f12867dfa5b3eb3d545a3e62f8b5` | `harmful_content`, 200 unsafe / 250 safe; safe rows are also `dangerous_tool_action` negatives. `safe`/`unsafe` becomes a binary label, and `type`, `focus` and `note` are dropped. |
| [bench-llms/or-bench](https://huggingface.co/datasets/bench-llms/or-bench) | OR-Bench authors (Cui et al.) | `fd6ee135ee63ff6c4f3ff72c0e39627bf0a7f314` | `or-bench-hard-1k` (up to 1,000 hard benign negatives for `harmful_content` + `dangerous_tool_action`) and `or-bench-toxic` (up to 600 `harmful_content` positives). `category` is dropped. |

### CC0-1.0 (public domain)

| Dataset | Creator | Use |
|---|---|---|
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | Jigsaw / Google | `harmful_content`, 2,500 pos / 2,500 neg (toxicity thresholded). No conditions apply; credited voluntarily. |

## 3. Additional use terms that travel with the weights

These terms apply on top of the licenses above. Anyone who uses or
redistributes Armor's trained weights is bound by them too.

- **AgentHarm, use restriction (license clause):** "We prohibit using the
  dataset and benchmark for purposes besides improving the safety and
  security of AI systems." Armor is a safety and security classifier for AI
  systems, so its use complies. Downstream users must not repurpose
  AgentHarm-derived models for anything else.
- **Aegis 1.0 and 2.0, dataset-card terms:**
  - *Legal compliance:* users are responsible for appropriate use and must
    not use the data in ways that conflict with legal and ethical standards.
  - *Non-identification:* users agree not to try to identify individuals in
    the dataset. Armor uses the text only as classifier input and never
    links rows to people.
  - Both cards say the data is "not intended for training dialogue agents".
    Armor trains a guardrail classifier, not a dialogue agent. That is the
    purpose Aegis was published for.
- **Gated downloads:** `walledai/HarmBench` and `walledai/AdvBench` sit
  behind Hugging Face click-through gates with no added restriction. Each
  person who retrains Armor must accept those gates on their own account.

## 4. Content warning

Many of these datasets contain harmful, offensive, or malicious-intent text
by design. Armor uses them only to teach the classifier to detect such
content. Do not use the raw rows for anything else.

## 5. Excluded on licensing or terms grounds

| Dataset | Reason |
|---|---|
| lmsys/toxic-chat | CC-BY-NC-4.0. Removed, and the checkpoint was retrained from scratch without it. |
| allenai/wildguardmix | Licensed ODC-BY, but its access gate requires certifying research-only use under AI2's Responsible Use Guidelines, which conflicts with production deployment. |
| PKU-Alignment/BeaverTails | CC-BY-NC-4.0 |
| MOCHA (purpcode) | CC-BY-NC-4.0 |
| RMCBench, MalwareBench | No license stated |

## 6. MIT License text

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 7. Citations

- Warner et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional
  Encoder for Fast, Memory Efficient, and Long Context Finetuning and
  Inference.* arXiv:2412.13663, 2024 (ModernBERT).
- Sharma, Gupta, Grossman. *SPML: A DSL for Defending Language Models Against
  Prompt Attacks.* arXiv:2402.11755, 2024.
- Shen et al. *"Do Anything Now": Characterizing and Evaluating In-The-Wild
  Jailbreak Prompts on Large Language Models.* ACM CCS 2024.
- Mazeika et al. *HarmBench: A Standardized Evaluation Framework for
  Automated Red Teaming and Robust Refusal.* arXiv:2402.04249, 2024.
- Andriushchenko et al. *AgentHarm: A Benchmark for Measuring Harmfulness of
  LLM Agents.* ICLR 2025.
- Debenedetti et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt
  Injection Attacks and Defenses for LLM Agents.* NeurIPS Datasets and
  Benchmarks 2024.
- Zhan et al. *InjecAgent: Benchmarking Indirect Prompt Injections in
  Tool-Integrated Large Language Model Agents.* Findings of ACL 2024.
- ToolSafety, EMNLP 2025, doi:10.18653/v1/2025.emnlp-main.714.
- Ruan et al. *Identifying the Risks of LM Agents with an LM-Emulated
  Sandbox.* ICLR 2024 (ToolEmu).
- Köpf et al. *OpenAssistant Conversations: Democratizing Large Language
  Model Alignment.* NeurIPS Datasets and Benchmarks 2023.
- Raheja et al. *CoEdIT: Text Editing by Task-Specific Instruction Tuning.*
  Findings of EMNLP 2023.
- Ghosh et al. *AEGIS: Online Adaptive AI Content Safety Moderation with
  Ensemble of LLM Experts.* arXiv:2404.05993, 2024.
- Ghosh et al. *AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for
  Alignment of LLM Guardrails.* NAACL 2025.
- Röttger et al. *XSTest: A Test Suite for Identifying Exaggerated Safety
  Behaviours in Large Language Models.* NAACL 2024.
- Borkan et al. *Nuanced Metrics for Measuring Unintended Bias with Real Data
  for Text Classification.* WWW 2019 (Civil Comments).
- Chao et al. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking
  Large Language Models.* NeurIPS Datasets and Benchmarks 2024. Per the JBB
  card, its behaviors partly come from AdvBench and TDC/HarmBench.
- Xu et al. *ASTRA: Autonomous Spatial-Temporal Red-teaming for AI Software
  Assistants.* arXiv:2508.03936, 2025.
- Bhatt et al. *CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite
  for Large Language Models.* arXiv:2404.13161, 2024 (FRR set).
- Zou et al. *Universal and Transferable Adversarial Attacks on Aligned
  Language Models.* arXiv:2307.15043, 2023 (AdvBench).
- LongPIBench: <https://github.com/liu00222/LongPIBench> (Yupei Liu et al.).
