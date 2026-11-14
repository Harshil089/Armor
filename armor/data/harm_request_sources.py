"""Licensed harmful-request sources for harmful_content, cyber-heavy.

Filled the gap where terse build-requests ("build me a malware", "make a
botnet") scored ~0.1: no wired-in source taught that a bare request for
malicious code is harmful. All sources are permissive (MIT / CC-BY-4.0),
revision-pinned, and verified against each primary card/LICENSE on
2026-10-19. Attribution and terms: THIRD_PARTY_NOTICES.md. allenai/wildguardmix
deliberately excluded: its access gate requires certifying research-only
use, narrower than its ODC-BY license and incompatible with production
deployment. Cyber requests are owned by harmful_content (HarmBench
cybercrime_intrusion moved here too); dangerous_tool_action stays for tool
calls.

Length balance: ASTRA positives are long (median ~313 words). PurpleLlama
FRR supplies benign security-coding requests (median ~62 words) so length
and security vocabulary alone do not separate the classes (the SPML lesson,
see sources.py).
"""

import json
import random
import urllib.request

from datasets import load_dataset

from armor.models.labels import LABELS

_HARM = LABELS.index("harmful_content")
_TOOL = LABELS.index("dangerous_tool_action")

_FRR_URL = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/"
    "76d0bfa0f31c14049da2e350e2f4a3e17fafcaa2/CybersecurityBenchmarks/datasets/mitre_frr/mitre_frr.json"
)
# Aegis 2.0 categories that are not harmful_content in Armor's taxonomy.
_AEGIS_SKIP = {"Needs Caution", "PII/Privacy", "Copyright/Trademark/Plagiarism",
               "Unauthorized Advice", "High Risk Gov Decision Making", "Other"}


def _row(text: str, positive: bool, labels=(_HARM,)):
    y, m = [0.0] * len(LABELS), [0.0] * len(LABELS)
    for i in labels:
        y[i], m[i] = float(positive and i == _HARM), 1.0
    return text.strip(), y, m


def _sample(rows: list, cap: int, seed: int = 0) -> list:
    rows = sorted(set(r for r in rows if r and r.strip()))
    random.Random(seed).shuffle(rows)
    return rows[:cap]


def load_jbb():
    """JailbreakBench/JBB-Behaviors, MIT: 100 harmful + 100 topic-matched benign."""
    rev = "886acc352a31533ffbcf4ef22c744658688086fc"
    out = [_row(r["Goal"], True) for r in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", revision=rev, split="harmful")]
    out += [_row(r["Goal"], False) for r in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", revision=rev, split="benign")]
    return out


def load_astra(cap: int = 400):
    """PurCL/astra-agent-security, MIT: malicious-code requests across 9 cyber domains (all positive)."""
    rows = load_dataset("PurCL/astra-agent-security", revision="26f9dee6b0bf46dd61e4ebee938bd76a6c8a4e75", split="train")
    return [_row(t, True) for t in _sample(rows["request_text"], cap)]


def load_frr():
    """PurpleLlama mitre_frr.json, MIT: 750 benign security-coding requests that look risky."""
    with urllib.request.urlopen(_FRR_URL, timeout=60) as r:
        rows = json.load(r)
    return [_row(x["mutated_prompt"], False, (_HARM, _TOOL)) for x in rows if not x["is_malicious"]]


def load_aegis2(cap: int = 1500):
    """nvidia/Aegis-AI-Content-Safety-Dataset-2.0, CC-BY-4.0. Every Malware row kept, rest sampled."""
    rows = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                        revision="d86bb8bedff51d25ac834ab7838f1cc61acb7a2c", split="train")
    malware, pos, neg = [], [], []
    for r in rows:
        text = r["prompt"]
        if not text or text == "REDACTED":
            continue
        cats = {c for c in (r["violated_categories"] or "").split(", ") if c}
        if r["prompt_label"] == "safe":
            neg.append(text)
        elif "Malware" in cats:
            malware.append(text)
        elif cats - _AEGIS_SKIP:
            pos.append(text)
    malware = _sample(malware, cap)
    pos = malware + _sample(pos, cap - len(malware))
    return [_row(t, True) for t in pos] + [_row(t, False) for t in _sample(neg, cap)]


def load_xstest():
    """Paul/XSTest, CC-BY-4.0: 250 safe prompts that sound unsafe, 200 unsafe contrasts."""
    rows = load_dataset("Paul/XSTest", revision="f600c994b256f12867dfa5b3eb3d545a3e62f8b5", split="train")
    return [_row(r["prompt"], r["label"] == "unsafe", (_HARM,) if r["label"] == "unsafe" else (_HARM, _TOOL))
            for r in rows]


def load_advbench():
    """walledai/AdvBench, MIT, auto-gated (accept on HF first): 520 harmful instructions."""
    rows = load_dataset("walledai/AdvBench", revision="9d4730540082fa4017450b65ca1c0e1d8d30446e", split="train")
    return [_row(t, True) for t in _sample(rows["prompt"], 520)]


LOADERS = {
    "jbb_behaviors": load_jbb,
    "astra": load_astra,
    "purplellama_frr": load_frr,
    "aegis2": load_aegis2,
    "xstest": load_xstest,
    "advbench": load_advbench,
}
