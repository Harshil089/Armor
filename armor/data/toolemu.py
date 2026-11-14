"""Matched dangerous-tool-action examples from ToolEmu.

Source: ryoungj/ToolEmu, Apache-2.0, pinned to a repository commit. Risky
actions are positive examples; expected safe achievements are negatives.
Only cases matching Armor's four weakest slices are used.
"""

import json
import re
import urllib.request
from collections import defaultdict

from armor.models.labels import LABELS

_REVISION = "ac4a7ab7ed8c7985d96231e214bd6b54304b7ddb"
_URL = f"https://raw.githubusercontent.com/ryoungj/ToolEmu/{_REVISION}/assets/all_cases.json"
_PER_SLICE_LIMIT = 60
_SLICE_ORDER = ("destructive", "exfiltration", "external", "financial")
_DTA_INDEX = LABELS.index("dangerous_tool_action")

_FINANCIAL_TOOLKITS = {
    "BankManager",
    "Binance",
    "EthereumManager",
    "InvestmentManager",
    "TDAmeritrade",
    "Venmo",
}
_EXTERNAL_TOOLKITS = {"FacebookManager", "Gmail", "Slack", "Twilio", "TwitterManager"}
_DESTRUCTIVE = re.compile(
    r"\b(?:delete|deletion|destroy|drop|erase|format|overwrite|purge|remove|removal|terminate|wipe)\b"
)
_EXFILTRATION = re.compile(
    r"\b(?:confidential|credential|data leak|disclose|disclosure|exfiltrate|exfiltration|"
    r"leak|leaking|personal information|privacy|private data|sensitive)\b"
)


def _matching_slices(case) -> set[str]:
    toolkits = set(case["Toolkits"])
    risk_text = " ".join(
        case["Potential Risky Outcomes"] + case["Potential Risky Actions"]
    ).lower()
    matches = set()
    if toolkits & _FINANCIAL_TOOLKITS:
        matches.add("financial")
    if toolkits & _EXTERNAL_TOOLKITS:
        matches.add("external")
    if _DESTRUCTIVE.search(risk_text):
        matches.add("destructive")
    if _EXFILTRATION.search(risk_text):
        matches.add("exfiltration")
    return matches


def load_examples(
    per_slice_per_label: int = _PER_SLICE_LIMIT,
) -> list[tuple[str, list[float], list[float]]]:
    with urllib.request.urlopen(_URL, timeout=30) as response:
        cases = json.loads(response.read().decode())

    buckets = defaultdict(list)
    for case in cases:
        matches = _matching_slices(case)
        for slice_name in _SLICE_ORDER:
            if slice_name not in matches:
                continue
            for value, field in (
                (1.0, "Potential Risky Actions"),
                (0.0, "Expected Achievements"),
            ):
                bucket = buckets[(slice_name, value)]
                for text in case[field]:
                    text = " ".join(text.split())
                    if text and len(bucket) < per_slice_per_label:
                        bucket.append(text)

    examples = []
    seen = set()
    for key in sorted(buckets):
        value = key[1]
        for text in buckets[key]:
            if (text, value) in seen:
                continue
            seen.add((text, value))
            labels = [0.0] * len(LABELS)
            mask = [0.0] * len(LABELS)
            labels[_DTA_INDEX] = value
            mask[_DTA_INDEX] = 1.0
            examples.append((text, labels, mask))
    return examples
