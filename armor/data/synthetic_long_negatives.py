"""Synthetic long multi-turn benign negatives.

ponytail: the other half of the out-of-distribution length defect that
armor/data/synthetic_short_negatives.py fixed at the short end. Every
training source tops out around 14 words, but the proxy's
OpenAIChatAdapter.extract_prompt_text() (armor/proxy/adapters/openai.py)
joins the *entire conversation history* into one newline-separated blob
on every turn — so real traffic arrives at 100+ words, far outside
anything the model ever saw.

Measured on the deployed v11 checkpoint against a 99-word benign
multi-turn chat blob (the exact shape a live groq-chat client sent
through the proxy 2026-09-20, captured in armor/observability/logs.db):

    prompt_injection        0.998
    obfuscation_encoding    0.970
    sensitive_data_exposure 0.156
    jailbreak               0.096
    dangerous_tool_action   0.089
    harmful_content         0.014

Two labels firing at ~0.99 on ordinary chat. Note those are the same two
labels that regressed on every retrain this session — they are the most
length-sensitive of the six, which is consistent with length being a
spurious feature they latched onto rather than a content signal.

Rows are built the way the adapter builds them (newline-joined turns) so
the training shape matches the serving shape. One label masked per row,
not all six: the all-six-at-once version of the short-negative booster
failed the regression gate at every volume tried, since it fires
gradients through all output heads per example.

Volume: a first pass at n=120 (~20 rows per label) fixed five of the six
labels outright on that same 99-word blob — obfuscation_encoding fell
0.970 -> 0.057 — but left prompt_injection at 0.801, still above the 0.5
finding threshold. That label is the hard case here because its own
positive class is long: SPML's system-prompt-violation texts are full
paragraphs, so length genuinely correlates with "positive" in its
training data and needs proportionally more counter-evidence to break.
Raised to n=480 (~80 per label) for that reason, not as blind volume.
"""

import random

from armor.models.labels import LABELS

# Deliberately disjoint from synthetic_short_negatives._SHORT_PHRASES and
# from every case in armor/eval/regression_set.py — the fix has to
# generalise to unseen benign chat, not memorise a fixed transcript.
_TURNS = [
    "Can you walk me through how this function works?",
    "Sure, it takes a list of records and groups them by date.",
    "What would you recommend for a beginner learning this?",
    "Start with the official tutorial, then build something small.",
    "I keep getting confused about when to use each approach.",
    "The difference mostly comes down to how often the data changes.",
    "Could you summarize that in simpler terms?",
    "Think of it like a filing cabinet with labelled drawers.",
    "What are the tradeoffs between those two options?",
    "One is faster to set up, the other scales better later on.",
    "Do you have an example I could look at?",
    "Here is a short snippet that shows the basic pattern.",
    "How long does something like this usually take?",
    "For a first pass, probably an afternoon of focused work.",
    "Is there anything I should watch out for?",
    "Mainly edge cases around empty input and timezone handling.",
    "That makes sense, thanks for explaining.",
    "Happy to help. Let me know if you want to go deeper on any part.",
    "What should I read next on this topic?",
    "The documentation has a good section on best practices.",
    "Can you help me plan my week?",
    "Start by listing everything, then pick the three that matter most.",
    "I want to improve at writing clearly.",
    "Shorter sentences and concrete examples go a long way.",
    "What is a good way to organize my notes?",
    "Group them by project rather than by date.",
    "How do I stay focused when there is a lot going on?",
    "Pick one thing, set a timer, and ignore the rest until it rings.",
    "Any suggestions for a weekend recipe?",
    "A simple roast chicken with vegetables is hard to get wrong.",
    "What is the weather usually like there this time of year?",
    "Mild during the day and noticeably cooler after sunset.",
    "Could you check my reasoning on this?",
    "Your logic holds up, though the second assumption needs a source.",
    "I am trying to decide between two job offers.",
    "List what each one gives you in two years, not just today.",
    "How do I get better at estimating how long work takes?",
    "Track your actual times for a few weeks and compare to your guesses.",
    "What is a reasonable goal for the first month?",
    "Something you can finish and show someone, however small.",
]


# Label weights. Uniform masking was tried first and does not work for
# prompt_injection: at n=120 (~20 rows to that label) the 99-word blob sat at
# 0.801, and at n=480 (~80 rows) it was 0.833 — four times the volume moved it
# the wrong way, so volume is not the lever.
#
# The asymmetry is in the positive class. prompt_injection carries ~1,797
# positives, most of them SPML system-prompt-violation texts that are full
# paragraphs, against ~80 long negatives — still ~22:1, so "long" remains a
# net-positive signal for that label no matter how much uniform volume is
# added. Every other label's positives are short template sentences, which is
# why ~20 uniform rows were enough to collapse obfuscation_encoding from 0.970
# to 0.057 on the same blob.
#
# So weight toward prompt_injection rather than raising n again.
_LABEL_WEIGHTS = {
    "prompt_injection": 10,
    "jailbreak": 1,
    "sensitive_data_exposure": 1,
    "dangerous_tool_action": 1,
    "harmful_content": 1,
    "obfuscation_encoding": 1,
}


def generate_examples(n: int = 480, min_turns: int = 6, max_turns: int = 16) -> list[tuple[str, list[float], list[float]]]:
    weights = [_LABEL_WEIGHTS.get(label, 1) for label in LABELS]
    examples = []
    for _ in range(n):
        turns = random.sample(_TURNS, random.randint(min_turns, min(max_turns, len(_TURNS))))
        text = "\n".join(turns)
        label_index = random.choices(range(len(LABELS)), weights=weights, k=1)[0]
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        mask_vec[label_index] = 1.0
        examples.append((text, label_vec, mask_vec))
    return examples
