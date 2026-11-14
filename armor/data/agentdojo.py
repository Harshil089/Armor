"""AgentDojo training examples for dangerous_tool_action.

ponytail: raw GitHub file scrape, not a HF dataset — AgentDojo's tasks are
defined as Python classes (PROMPT = "..." for benign user tasks, GOAL = f"..."
for attacker-injected tool-misuse goals) inside UKGovernmentBEIS/inspect_evals,
not exposed as a flat dataset. Extracted via regex on the raw source, no need
to run the inspect_ai harness/Docker/live models to get the text.

License: MIT (inspect_evals repo root, no per-eval override checked for
agentdojo specifically).

Known limitation: some GOAL/PROMPT values are f-strings with embedded
variables (e.g. "{_ATTACKER_IBAN}") — extraction keeps the literal template
text, not the substituted value. Still clearly attack-shaped text; resolving
the placeholders would require importing and instantiating the task classes,
which defeats the point of not needing the full harness.
"""

import re
import urllib.request

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("dangerous_tool_action")

_BASE_URL = "https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/main/src/inspect_evals/agentdojo/task_suites"
_SUITES = ["banking", "slack", "travel", "workspace", "workspace_plus"]
_STRING_PATTERN = r'{var}\s*=\s*f?"""(.*?)"""|{var}\s*=\s*f?"(.*?)"(?<!\\")'


def _extract(content: str, var_name: str) -> list[str]:
    pattern = re.compile(_STRING_PATTERN.format(var=var_name), re.S)
    values = []
    for m in pattern.finditer(content):
        text = m.group(1) or m.group(2)
        if text:
            values.append(text.strip())
    return values


def load_examples() -> list[tuple[str, list[float], list[float]]]:
    examples = []
    for suite in _SUITES:
        for filename, var_name, positive_value in [
            ("user_tasks.py", "PROMPT", 0.0),
            ("injection_tasks.py", "GOAL", 1.0),
        ]:
            url = f"{_BASE_URL}/{suite}/{filename}"
            content = urllib.request.urlopen(url, timeout=15).read().decode()
            for text in _extract(content, var_name):
                label_vec = [0.0] * len(LABELS)
                mask_vec = [0.0] * len(LABELS)
                label_vec[_LABEL_INDEX] = positive_value
                mask_vec[_LABEL_INDEX] = 1.0
                examples.append((text, label_vec, mask_vec))
    return examples
