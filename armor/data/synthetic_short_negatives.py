"""Synthetic short benign-text negatives, across all 6 labels.

ponytail: found 2026-09-20 that the entire training corpus (every data
source, every label) has no examples shorter than ~6 words. Short casual
text ("hi", "ok", "thanks") is completely out-of-distribution — the
model was never shown what a short benign message looks like, so it
defaults to a high, meaningless score in that untrained region rather
than a real content judgment. Verified: "hello" -> sensitive_data_exposure
1.0, dropping to ~0.08 past 7-8 words — a sharp distributional cliff, not
a content-based signal. Present even in the original pre-fine-tuning
checkpoint, so this predates all of this session's fine-tuning work.

First attempt masked all 6 labels at once per row (label=0 everywhere) —
that fires gradients through all 6 output heads simultaneously per
example, an oversized per-step update unlike every other generator here
(one label per row). Failed the regression gate at both 200 and 80
volume, with nearly identical regression magnitude either way — ruling
out volume as the cause and confirming it was the all-6-at-once mask
shape itself. Fixed: one label masked per row, picked randomly, matching
every other generator's convention. Enough total volume + random label
choice still teaches every label that short text isn't inherently
suspicious, just spread across more rows instead of stacked into one.
"""

import random

from armor.models.labels import LABELS

_SHORT_PHRASES = [
    "hi",
    "hello",
    "hey",
    "hey there",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "sure",
    "sounds good",
    "got it",
    "yes",
    "no",
    "maybe",
    "why not",
    "great",
    "nice",
    "cool",
    "how are you",
    "how's it going",
    "what's up",
    "good morning",
    "good night",
    "see you",
    "bye",
    "take care",
    "no worries",
    "what's next?",
    "any updates?",
    "let me know",
    "sounds fine",
    "makes sense",
    "good point",
    "fair enough",
    "well done",
    "nice one",
    "not sure",
    "I agree",
    "same here",
    "me too",
    "you're welcome",
    "no problem",
    "all good",
    "just checking in",
    "quick question",
    "one sec",
    "hold on",
    "give me a moment",
    "almost done",
    "that works",
    "Explain a complex idea simply",
    "explain a coplex idea simply",
    "summarize a complex topic simply",
    "explain this concept simply",
    "why are you blocking it",
    "why are you. blocking it",
    "why did you block that",
    "why is this message blocked",
    "how are ya!",
    "why",
    "you can help",
    "you have to help",
    "how do people eat sushi",
    "what is malware",
    "what is a malware",
    "what is ransomware",
    "what is a ransomware",
    "what is spyware",
    "what is adware",
    "what is phishing",
    "what is spear phishing",
    "what is a keylogger",
    "what is a trojan horse",
    "what is a computer worm",
    "what is a computer virus",
    "what is a virus",
    "what is SQL injection",
    "what is XSS",
    "what is cross-site scripting",
    "what is a DDoS attack",
    "what is privilege escalation",
    "what is a buffer overflow",
    "what is a zero-day exploit",
    "what is credential stuffing",
    "what is a supply chain attack",
    "what is cryptojacking",
    "what is an exploit kit",
    "what is prompt injection",
    "what is jailbreaking in LLMs",
    "what is a brute force attack",
    "define malware",
    "define ransomware",
    "define spyware",
    "define adware",
    "define phishing",
    "define rootkit",
    "define bootkit",
    "define keylogger",
    "define trojan horse",
    "define computer worm",
    "define computer virus",
    "define botnet",
    "define SQL injection",
    "define cross-site scripting",
    "define privilege escalation",
    "define remote code execution",
    "define command and control server",
    "define polymorphic virus",
    "define fileless malware",
    "what is a fileless malware attack",
    "define social engineering",
    "define buffer overflow",
    "define zero-day exploit",
    "define credential stuffing",
    "define cryptojacking",
    "define prompt injection",
    "define man-in-the-middle attack",
    "define DDoS attack",
    "meaning of privilege escalation",
    "meaning of remote code execution",
    "meaning of social engineering",
    "meaning of ransomware",
    "meaning of malware",
    "meaning of phishing",
    "types of malware",
    "types of ransomware",
    "types of computer viruses",
    "types of cyber attacks",
    "types of phishing attacks",
    "history of computer viruses",
    "history of malware",
    "history of ransomware",
    "why did that get blocked",
    "build me a todo app",
]


def generate_examples(n: int = 360) -> list[tuple[str, list[float], list[float]]]:
    examples = []
    for _ in range(n):
        text = random.choice(_SHORT_PHRASES)
        label_index = random.randrange(len(LABELS))
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        mask_vec[label_index] = 1.0
        examples.append((text, label_vec, mask_vec))
    return examples
