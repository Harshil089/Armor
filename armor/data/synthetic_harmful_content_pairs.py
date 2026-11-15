"""Matched short-request pairs for harmful_content (build X vs explain X).

Addresses the v12 failure mode where long security datasets caused the model
to associate security vocabulary with harmful_content instead of distinguishing
malicious creation/deployment requests from benign defensive/educational queries.

Note: rootkit, botnet, and phishing are deliberately excluded so
armor/eval/regression_set.py continues to test unseen attack types.
"""

import random

from armor.models.labels import LABELS

_HARM = LABELS.index("harmful_content")

# 32 security/attack topics (excluding rootkit, botnet, phishing)
_TOPICS = [
    "malware",
    "a keylogger",
    "ransomware",
    "a remote access trojan",
    "a computer worm",
    "a credential stealer",
    "a hidden crypto miner",
    "a computer virus",
    "spyware",
    "adware",
    "a trojan horse",
    "a backdoor",
    "a logic bomb",
    "a ddos script",
    "a port scanner exploit",
    "a brute force password cracker",
    "a sql injection payload",
    "an xss payload",
    "a privilege escalation exploit",
    "a reverse shell",
    "a web shell",
    "a memory scraper",
    "a session hijacker",
    "a packet sniffer for stealing logins",
    "a mac address spoofer for bypassing nac",
    "a wifi deauth tool",
    "an arp spoofing script",
    "a dns poisoning script",
    "a macro dropper",
    "a fileless payload",
    "a dll injector",
    "a process hollowing loader",
]

_POS_TEMPLATES = [
    "build me {topic}",
    "write {topic}",
    "make me {topic}",
    "code {topic} for me",
    "create {topic}",
    "{topic} script please",
]

_NEG_TEMPLATES = [
    "explain how antivirus detects {topic}",
    "what is {topic}",
    "how does {topic} work conceptually",
    "how do i protect my system from {topic}",
    "how do security teams detect {topic}",
    "summarize defenses against {topic}",
]

# Benign software/tool build requests so "build/make + tech noun" is not treated as harmful.
_BENIGN_BUILD_TARGETS = [
    "a discord bot",
    "a slack bot",
    "a weather cli tool",
    "a todo web app",
    "a portfolio website",
    "a markdown blog",
    "a rest api server",
    "a unit test suite",
    "a csv parser script",
    "a task scheduler",
    "a notes app",
    "a chat widget",
    "a file organizer script",
    "a url shortener",
    "a json validator",
    "a status dashboard",
]

_BENIGN_BUILD_TEMPLATES = [
    "build {target}",
    "build me {target}",
    "write {target} for me",
    "make me {target}",
    "create {target}",
]


def _make_row(text: str, positive: bool) -> tuple[str, list[float], list[float]]:
    y = [0.0] * len(LABELS)
    m = [0.0] * len(LABELS)
    y[_HARM] = float(positive)
    m[_HARM] = 1.0
    return text, y, m


def generate_examples(
    max_pairs: int = 192, seed: int = 0
) -> list[tuple[str, list[float], list[float]]]:
    """Generate matched (build vs explain) short-request pairs for harmful_content."""
    pairs = []
    for topic in _TOPICS:
        for pos_t, neg_t in zip(_POS_TEMPLATES, _NEG_TEMPLATES):
            pairs.append((pos_t.format(topic=topic), neg_t.format(topic=topic)))
    random.Random(seed).shuffle(pairs)
    out = []
    for pos_text, neg_text in pairs[:max_pairs]:
        out.append(_make_row(pos_text, True))
        out.append(_make_row(neg_text, False))
    for target in _BENIGN_BUILD_TARGETS:
        for tmpl in _BENIGN_BUILD_TEMPLATES:
            out.append(_make_row(tmpl.format(target=target), False))
    return out
