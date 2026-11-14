"""Synthetic sensitive_data_exposure training examples.

ponytail: no external dataset used for this label. The two candidates found
(Isotonic/pii-masking-200k: cc-by-nc-4.0; bigcode/bigcode-pii-dataset-training:
gated, scraped real secrets) were both rejected — NC license and ethically
wrong source, respectively (see M1 source survey doc for the license-review
pattern this follows). Positives here are generated with real, verified
checksums (Luhn, IBAN mod-97, Verhoeff) — genuinely valid-shaped fake PII,
not just regex-matching noise, so the signal is as clean as a real dataset's
would be for these entity types.
"""

import random

from armor.models.labels import LABELS

_LABEL_INDEX = LABELS.index("sensitive_data_exposure")

_POSITIVE_TEMPLATES = [
    "Please update the record with {value}.",
    "For reference, {value} was used in the transaction.",
    "Can you confirm {value} is correct on file?",
    "I need to report an issue with {value}.",
    "Attached is the form showing {value} for verification.",
    "The customer's account lists {value} as the identifier.",
    "Send the invoice to the address on file and reference {value}.",
    "During onboarding we collected {value} from the applicant.",
]

_NEGATIVE_TEMPLATES = [
    "The quarterly report shows revenue increased by twelve percent.",
    "Please schedule the meeting for next Tuesday afternoon.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Our team finished the sprint ahead of schedule this week.",
    "The museum's new exhibit opens to the public next month.",
    "He enjoys hiking on weekends when the weather is nice.",
    "The committee will review the proposal at the next session.",
    "Traffic was light this morning so the commute was quick.",
    "The garden needs watering twice a week during summer.",
    "She finished reading the novel over the long weekend.",
]


def _luhn_check_digit(partial_digits: list[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(partial_digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10


def _gen_credit_card() -> str:
    partial = [4] + [random.randint(0, 9) for _ in range(14)]  # Visa, 16 digits
    check = _luhn_check_digit(partial)
    digits = partial + [check]
    return f"{''.join(map(str, digits[0:4]))} {''.join(map(str, digits[4:8]))} {''.join(map(str, digits[8:12]))} {''.join(map(str, digits[12:16]))}"


def _gen_iban() -> str:
    bban = "".join(str(random.randint(0, 9)) for _ in range(16))
    country = "GB"
    rearranged = bban + country + "00"
    numeric = "".join(str(ord(c) - ord("A") + 10) if c.isalpha() else c for c in rearranged)
    check = 98 - (int(numeric) % 97)
    return f"{country}{check:02d}{bban}"


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_check_digit(digits: list[int]) -> int:
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][d]]
    return _VERHOEFF_INV[c]


def _gen_aadhaar() -> str:
    body = [random.randint(2, 9)] + [random.randint(0, 9) for _ in range(10)]
    check = _verhoeff_check_digit(body)
    digits = body + [check]
    return f"{''.join(map(str, digits[0:4]))} {''.join(map(str, digits[4:8]))} {''.join(map(str, digits[8:12]))}"


def _gen_email() -> str:
    user = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=6))
    return f"{user}@{domain}.com"


def _gen_aws_key() -> str:
    return "AKIA" + "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=16))


_GENERATORS = [_gen_credit_card, _gen_iban, _gen_aadhaar, _gen_email, _gen_aws_key]


def generate_examples(n_positive: int = 400, n_negative: int = 400) -> list[tuple[str, list[float], list[float]]]:
    examples = []
    for _ in range(n_positive):
        value = random.choice(_GENERATORS)()
        template = random.choice(_POSITIVE_TEMPLATES)
        text = template.format(value=value)
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        label_vec[_LABEL_INDEX] = 1.0
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((text, label_vec, mask_vec))

    for _ in range(n_negative):
        text = random.choice(_NEGATIVE_TEMPLATES)
        label_vec = [0.0] * len(LABELS)
        mask_vec = [0.0] * len(LABELS)
        mask_vec[_LABEL_INDEX] = 1.0
        examples.append((text, label_vec, mask_vec))

    return examples
