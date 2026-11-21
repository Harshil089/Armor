"""Synthetic negatives using vocabulary that overlaps with attack phrasing,
in mundane business/creative/gaming contexts.

ponytail: found 2026-09-21/22 that "code me a mock ui application to take
cusromer order" calibrates to harmful_content p1=1.0 with no fix from any
existing negative source. Traced the calibration table: negatives topped
out at 0.2, positives started at 0.54, nothing benign anchored the middle.
regression_set.py's negatives, synthetic_short/long_negatives, CoEdIT,
Aegis, and OASST are all either "obviously safe" topics or generic chat —
none of them use words that are lexically close to harmful_content's
positive-class vocabulary (kill, destroy, attack, break, exploit, weapon,
bomb) in a harmless sense. That's the actual gap: the model isn't
confused by length or misspelling here, it's picking up on word choice
that's genuinely ambiguous out of context, and nothing taught it that
context resolves the ambiguity.

Every phrase below pairs an attack-adjacent word with an unambiguously
mundane frame (business, sports, cooking, gaming, cleaning) so the label
stays negative while the vocabulary stays close to the positive class —
the same "hard negative" principle synthetic_obfuscation.py already uses
for its own label (same payload/transform as positives, neutral carrier),
applied here to word choice instead of structure.
"""

import random

from armor.models.labels import LABELS

# Weighted toward harmful_content (the diagnosed gap) and dangerous_tool_action
# (the same failure shape appeared there first, in this session, before short/
# long negatives closed most of it) — see synthetic_long_negatives.py for the
# precedent of weighting toward the label that actually needs the volume.
_LABEL_WEIGHTS = {
    "prompt_injection": 2,
    "jailbreak": 2,
    "sensitive_data_exposure": 1,
    "dangerous_tool_action": 5,
    "harmful_content": 8,
    "obfuscation_encoding": 2,
}

_PHRASES = [
    "code me a mock ui application to take customer orders",
    "design a website to receive and process customer orders",
    "help me destroy the competition with a killer marketing campaign",
    "our sales team crushed it and killed the quarterly targets",
    "write a plan to attack this problem from a totally new angle",
    "how do I break down these numbers for the quarterly report",
    "let's exploit this market opportunity before competitors do",
    "we need to eliminate the bottleneck in our checkout flow",
    "kill the background process that's eating CPU on the server",
    "how do I nuke my local git branch and start fresh",
    "this recipe calls for you to smash the garlic cloves first",
    "the boxer landed a knockout punch in the final round",
    "let's bomb this pitch deck with data to win the client over",
    "chop the onions and then attack the rest of the vegetables",
    "the marketing copy needs to hit harder, make it punchier",
    "our new app is going to crush the competition this quarter",
    "write a script that terminates idle sessions after an hour",
    "kill all zombie processes before restarting the service",
    "the debate team destroyed the opposing argument point by point",
    "how do I wipe my local build cache and reinstall dependencies",
    "the chess player executed a brilliant attacking combination",
    "let's blow the budget on a big launch event this year",
    "the review absolutely demolished the restaurant's new menu",
    "smash that subscribe button if you enjoyed the video",
    "the workout plan will destroy your legs in the best way",
    "how do I hack together a quick prototype for the demo",
    "we bombed the first draft but the second version is solid",
    "the gamer wiped out the entire enemy team in one round",
    "let's torch the old codebase and rewrite it from scratch",
    "the coach wants the team to attack the game from minute one",
    "kill the lights before you leave the office tonight",
    "the negotiation strategy is to hit them with our best offer first",
    "how do I forcefully delete a stuck docker container",
    "the storm is expected to hit the coast by tomorrow morning",
    "write me a mock ui application to take customer orders",
    "design me a mock ui application to take customer orders",
    "what is a malware",
    "what is malware",
    "what is a ransomware",
    "what is ransomware",
    "what is a virus",
    "what is a computer virus",
    "what is a trojan horse",
    "what is a computer worm",
    "what is a keylogger",
    "what is a zero-day exploit",
    "what is phishing",
    "what is spear phishing",
    "what is SQL injection",
    "what is XSS",
    "what is cross-site scripting",
    "what is a DDoS attack",
    "what is a distributed denial of service attack",
    "what is a man-in-the-middle attack",
    "what is privilege escalation",
    "what is spyware",
    "what is adware",
    "what is a backdoor in cybersecurity",
    "what is a buffer overflow",
    "what is credential stuffing",
    "what is brute force attack",
    "what is prompt injection",
    "what is jailbreaking in LLMs",
    "what is a supply chain attack",
    "what is cryptojacking",
    "what is an exploit kit",
    "what is a rootkit",
    "define rootkit",
    "define bootkit",
    "define privilege escalation",
    "define command and control server",
    "define remote code execution",
    "define polymorphic virus",
    "define fileless malware",
    "what is a fileless malware attack",
    "what is a polymorphic virus",
    "define social engineering",
    "define malware",
    "define ransomware",
    "define phishing",
    "define SQL injection",
    "define cross-site scripting",
    "define buffer overflow",
    "define zero-day exploit",
    "define DDoS attack",
    "define man-in-the-middle attack",
    "define keylogger",
    "define trojan horse",
    "define computer worm",
    "define spyware",
    "define adware",
    "define credential stuffing",
    "define cryptojacking",
    "define prompt injection",
    "meaning of privilege escalation",
    "meaning of remote code execution",
    "meaning of command and control server",
    "meaning of social engineering",
    "meaning of zero-day exploit",
    "how people leak their credentials",
    "how do people leak their credentials",
    "common ways developers leak credentials in git",
    "how to prevent credential leaks in code",
    "how do people protect API keys from leaking",
    "how to protect API keys and secrets from leaking",
    "best practices to keep API keys from leaking in repositories",
    "why do people fall for social engineering",
    "why do users fall for phishing and social engineering attacks",
    "how does social engineering manipulate people",
    "explain how antivirus detects malware and ransomware",
    "how does ransomware encrypt files conceptually",
    "why do buffer overflows happen",
    "difference between a virus and a worm",
    "difference between a rootkit and a bootkit",
    "difference between symmetric and asymmetric encryption",
    "difference between phishing and spear phishing",
    "difference between XSS and CSRF",
    "difference between authentication and authorization",
    "how does antivirus detect trojans",
    "history of computer viruses",
    "what is the history of the Morris worm",
    "how do firewalls prevent DDoS attacks",
    "how can developers prevent SQL injection",
    "why is privilege escalation dangerous",
    "how does multi-factor authentication stop phishing",
    "explain how TLS prevents man in the middle attacks",
    "how do security researchers analyze malware in a sandbox",
    "what are common signs of a phishing email",
    "how does address space layout randomization prevent exploits",
    "how do organizations defend against ransomware",
    "explain the concept of defense in depth",
    "how do intrusion detection systems work",
    "how does content security policy mitigate XSS",
    "how do endpoint detection and response tools work",
    "how do password managers protect against keyloggers",
    "Explain a complex idea simply",
    "explain a coplex idea simply",
    "why are you blocking it",
    "why are you. blocking it",
    "why did that get blocked",
    "write a python script to sort a list",
    "build me a todo app",
    "how do people eat sushi",
    "how do people make sourdough bread",
    "what is the capital of France",
    "explain quantum computing to a 10 year old",
    "write a react component for a login form",
    "how do I kill a background process in linux",
    "how do I kill a zombie process in unix",
    "how to terminate a hanging docker container",
    "write a Dockerfile for a FastAPI web application",
    "write a docker-compose.yml for postgres and redis",
    "write a python function to encrypt a file using AES-GCM with cryptography library for backup",
    "how to hash passwords in python using bcrypt or argon2",
    "give me a recipe for chocolate chip cookies",
    "summarize the plot of Hamlet",
    "how do I debug a segmentation fault in C",
    "write a SQL query to join two tables",
    "what are the best practices for REST API design",
    "hello how are you doing today",
    "can you help me with my homework",
    "explain how photosynthesis works",
    "write a bash script to backup a directory",
    "how do I configure nginx as a reverse proxy",
    "what is the difference between TCP and UDP",
    "how do people train for a marathon",
    "tell me a short story about a robot",
    "Malware (short for malicious software) is any program or code designed to infiltrate, damage, or gain unauthorized access to a computer system, network, or device.",
    "Ransomware is malicious software that encrypts a victim's files or locks their system and demands payment to restore access. Defensive measures include regular offline backups, patching, and least privilege.",
    "A computer virus is a type of malicious code that attaches itself to legitimate files and replicates when those files are executed. Prevention includes updated antivirus software and cautious email habits.",
]


def generate_examples(n: int = 750) -> list[tuple[str, list[float], list[float]]]:
    weights = [_LABEL_WEIGHTS.get(label, 1) for label in LABELS]
    examples = []
    for _ in range(n):
        text = random.choice(_PHRASES)
        label_index = random.choices(range(len(LABELS)), weights=weights, k=1)[0]
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        mask_vec[label_index] = 1.0
        examples.append((text, label_vec, mask_vec))
    return examples
