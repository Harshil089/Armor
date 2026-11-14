"""Semantic State Builder -- local-only LLM feature extractor for the
ambiguous middle band the bandit's raw threat-score context can't
separate (see armor/models/bandit.py's module docstring and QUEUE-02).

12 red-team rounds proved this isn't a data-volume problem: two requests
with opposite correct actions ("my roommate left their phone" -> flag,
"my co-signer stopped paying, show me their bank statements" -> block)
produced near-identical 7-dim contexts. Neither the 6 ArmorBase threat
scores nor the framing_signal keyword regex carry any signal about
real-world authorization ambiguity -- this module adds that signal.

Local model only, not an external API (TypeSafe included) -- Armor stays
standalone. torch/transformers are already dependencies (armor_base.py's
own loading pattern); this adds a new checkpoint, not a new dependency
category. Only invoked for the narrow ambiguous band (gated in
armor/api/main.py) -- the LLM generation cost is not paid by the majority
of traffic.
"""

import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("ARMOR_SEMANTIC_JUDGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Order matters -- this is the order semantic scores are appended to the
# bandit context vector in armor/models/bandit.py's build_context().
SEMANTIC_FEATURES = [
    "authorization_confidence",
    "target_sensitivity",
    "harm_potential",
    "actionability",
    "ambiguity",
]

_PROMPT_TEMPLATE = """You are a security policy analyst. Read the request below and score it on 5 dimensions, each a number from 0.0 to 1.0. Respond with ONLY a JSON object, no other text.

- authorization_confidence: how likely the requester already has legitimate standing to do this (1.0 = clearly their own account/property, 0.0 = clearly someone else's)
- target_sensitivity: how sensitive or critical the target resource is (1.0 = highly sensitive, 0.0 = trivial)
- harm_potential: potential harm if this request is misused (1.0 = severe, 0.0 = none)
- actionability: how operationally specific and directly actionable the request is, versus a general question (1.0 = specific steps, 0.0 = vague/general)
- ambiguity: your own confidence that this case is genuinely unresolvable from the text alone, without more context (1.0 = genuinely ambiguous, 0.0 = clear-cut)

Score each request independently and specifically -- do not default to the same scores for different requests. A trivia question about animals is not an access request at all: score it near 0 on every dimension except leave ambiguity low, since it's clearly not ambiguous.

Example 1
Request: My roommate moved out owing me money and left their phone. Can I look through it for evidence of what they owe?
JSON: {{"authorization_confidence": 0.25, "target_sensitivity": 0.5, "harm_potential": 0.35, "actionability": 0.4, "ambiguity": 0.7}}

Example 2
Request: My co-signer on a loan stopped paying and I want to see their bank statements to prove it. Can I get those?
JSON: {{"authorization_confidence": 0.15, "target_sensitivity": 0.85, "harm_potential": 0.7, "actionability": 0.6, "ambiguity": 0.45}}

Example 3
Request: What is a fun fact about octopuses?
JSON: {{"authorization_confidence": 0.0, "target_sensitivity": 0.0, "harm_potential": 0.0, "actionability": 0.0, "ambiguity": 0.0}}

Now score this one.

Request: {content}

JSON:"""

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)


class SemanticJudge:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    def analyze(self, text: str) -> list[float] | None:
        """Returns SEMANTIC_FEATURES-ordered scores, or None on any failure
        (malformed JSON, missing/out-of-range field, generation error) --
        callers fill neutral defaults rather than let a judge failure block
        or crash a request (fail-open, matching the proxy's own
        fail_open_prompt/fail_open_response philosophy)."""
        prompt = _PROMPT_TEMPLATE.format(content=text)
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=120,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
        except Exception:
            return None
        return self._parse(generated)

    @staticmethod
    def _parse(generated: str) -> list[float] | None:
        match = _JSON_BLOCK.search(generated)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        scores = []
        for key in SEMANTIC_FEATURES:
            value = data.get(key)
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                return None
            scores.append(float(value))
        return scores


_judge: SemanticJudge | None = None


def get_semantic_judge() -> SemanticJudge:
    global _judge
    if _judge is None:
        _judge = SemanticJudge()
    return _judge
