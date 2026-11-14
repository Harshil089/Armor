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
    "prompt_injection": 1,
    "jailbreak": 1,
    "sensitive_data_exposure": 1,
    "dangerous_tool_action": 4,
    "harmful_content": 6,
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
]


def generate_examples(n: int = 300) -> list[tuple[str, list[float], list[float]]]:
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
