from typing import Callable


def _luhn_checksum_ok(digits: list[int]) -> bool:
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    return _luhn_checksum_ok(digits)


def _matches_known_payment_network(digits: str) -> bool:
    """IIN/BIN prefix + length rules for major card networks (credit and
    debit share these ranges — Maestro/UnionPay are debit-heavy networks,
    Visa/Mastercard/Amex/Discover/Diners/JCB issue both). A number that
    passes Luhn but matches no known network's prefix+length is far more
    likely a coincidental random digit string than a real card."""
    n = len(digits)
    if digits[0] == "4" and n in (13, 16, 19):
        return True  # Visa
    if n == 16 and (51 <= int(digits[:2]) <= 55 or 2221 <= int(digits[:4]) <= 2720):
        return True  # Mastercard
    if n == 15 and digits[:2] in ("34", "37"):
        return True  # American Express
    if n == 16 and (
        digits[:4] == "6011"
        or digits[:2] == "65"
        or 644 <= int(digits[:3]) <= 649
        or 622126 <= int(digits[:6]) <= 622925
    ):
        return True  # Discover
    if n == 14 and (300 <= int(digits[:3]) <= 305 or digits[:2] in ("36", "38")):
        return True  # Diners Club
    if n == 16 and 3528 <= int(digits[:4]) <= 3589:
        return True  # JCB
    if 16 <= n <= 19 and digits[:2] == "62":
        return True  # UnionPay
    if 12 <= n <= 19 and (
        digits[:2] == "50"
        or digits[:2] in ("56", "57", "58")
        or digits[:4] in ("6304", "6390")
        or digits[:2] == "67"
    ):
        return True  # Maestro (debit-only network)
    return False


def _payment_card(value: str) -> bool:
    """Credit or debit card — Luhn checksum AND a known network's IIN/BIN
    prefix+length. Covers both card types: the number format doesn't
    distinguish credit from debit, only the issuing network's own systems
    know that, so this validates "is this a real payment card number",
    not which product it's tied to."""
    digits = "".join(c for c in value if c.isdigit())
    if not (12 <= len(digits) <= 19):
        return False
    if not _luhn_checksum_ok([int(c) for c in digits]):
        return False
    return _matches_known_payment_network(digits)


def _ca_sin(value: str) -> bool:
    """Canada SIN — 9 digits, Luhn checksum."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 9:
        return False
    return _luhn_checksum_ok(digits)


def _iban_mod97(value: str) -> bool:
    v = "".join(value.split()).upper()
    if not (15 <= len(v) <= 34) or not v[:2].isalpha() or not v[2:4].isdigit():
        return False
    rearranged = v[4:] + v[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            return False
    return int(numeric) % 97 == 1


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


def _verhoeff(value: str) -> bool:
    """Aadhaar (India) — 12-digit Verhoeff checksum (framework doc section 6).

    UIDAI never issues a number starting with 0 or 1 (first digit is 2-9) —
    checked here, not just in the entity regex, since the validator is the
    real gate per the framework design.
    """
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 12 or digits[0] < 2:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[i % 8][digit]]
    return checksum == 0


def _us_ssn_range(value: str) -> bool:
    """US SSN has no public checksum, but the SSA never issues certain area/group/
    serial ranges (000/666/900-999 area, 00 group, 0000 serial) — enough of a
    deterministic rule to reject obviously-fake numbers, short of a real checksum."""
    digits = [c for c in value if c.isdigit()]
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    area_n = int("".join(area))
    if area_n == 0 or area_n == 666 or area_n >= 900:
        return False
    if "".join(group) == "00" or "".join(serial) == "0000":
        return False
    return True


_GB_NINO_EXCLUDED_PREFIXES = {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}


def _gb_nino_prefix(value: str) -> bool:
    """UK NINO has no checksum, but HMRC's own manual (gov.uk, NIM39110)
    reserves seven specific two-letter prefixes that are never assigned,
    on top of the per-letter exclusions already enforced by the entity
    regex (D/F/I/Q/U/V, and O in the second position)."""
    v = value.strip().upper()
    if len(v) < 2:
        return False
    return v[:2] not in _GB_NINO_EXCLUDED_PREFIXES


def _cpf(value: str) -> bool:
    """Brazil CPF — 11 digits, two weighted mod-11 check digits."""

    def _check(digs: list[int], weight_start: int) -> int:
        total = sum(d * w for d, w in zip(digs, range(weight_start, 1, -1)))
        rem = (total * 10) % 11
        return 0 if rem >= 10 else rem

    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    d1 = _check(digits[:9], 10)
    d2 = _check(digits[:9] + [d1], 11)
    return digits[9] == d1 and digits[10] == d2


_CN_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_CN_ID_CHECK_MAP = ["1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2"]


def _cn_resident_id(value: str) -> bool:
    """China resident ID — 18 chars: 6-digit address code + 8-digit birth
    date (YYYYMMDD) + 3-digit sequence + ISO 7064 MOD 11-2 check character.
    Birth date is checked for plausibility (real month/day), same spirit as
    the US SSN range rule and South Africa ID date check."""
    v = value.strip().upper()
    if len(v) != 18:
        return False
    body = v[:17]
    if not body.isdigit():
        return False
    month = int(body[10:12])
    day = int(body[12:14])
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return False
    total = sum(int(d) * w for d, w in zip(body, _CN_ID_WEIGHTS))
    return v[17] == _CN_ID_CHECK_MAP[total % 11]


def _de_digit_repetition_ok(first10: list[int]) -> bool:
    """BZSt structural rule (de.wikipedia.org/wiki/Steuerliche_Identifikationsnummer):
    among the first 10 digits, either exactly one digit value appears twice
    and one other value appears zero times (pre-2016 rule), or exactly one
    value appears three times and two others appear zero times (post-2016),
    with the constraint that a triple-occurring digit's three positions must
    not all be mutually adjacent (a run of 3)."""
    from collections import Counter

    counts = Counter(first10)
    by_count = {2: 0, 3: 0}
    for c in counts.values():
        if c in by_count:
            by_count[c] += 1
    count_of_1 = sum(1 for c in counts.values() if c == 1)
    count_of_0 = 10 - len(counts)

    if by_count[2] == 1 and by_count[3] == 0 and count_of_1 == 8 and count_of_0 == 1:
        return True
    if by_count[3] == 1 and by_count[2] == 0 and count_of_1 == 7 and count_of_0 == 2:
        triple_value = next(v for v, c in counts.items() if c == 3)
        positions = [i for i, d in enumerate(first10) if d == triple_value]
        return positions[2] - positions[0] != 2  # not all three consecutive
    return False


def _de_steuerid(value: str) -> bool:
    """Germany Steuer-ID — 11 digits: BZSt iterative check-digit algorithm
    AND the structural digit-repetition rule on the first 10 digits."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 11 or digits[0] == 0:
        return False
    if not _de_digit_repetition_ok(digits[:10]):
        return False
    product = 10
    for d in digits[:10]:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (2 * s) % 11
    check = (11 - product) % 11
    if check == 10:
        check = 0
    return digits[10] == check


def _fr_nir(value: str) -> bool:
    """France NIR/INSEE number — 13 digits + 2-digit key, key = 97 - (N mod 97).
    Corsica: department 2A -> 19, 2B -> 18 before computing N.

    Formula confirmed against fr.wikipedia.org/wiki/Numéro_de_sécurité_sociale_en_France
    verbatim ("la clé est le complément à 97 du NIR modulo 97"; Corsica
    substitution matches exactly). Every third-party "worked example" found
    during research (four attempts, different sources) failed independent
    arithmetic re-verification — those are unreliable, not this formula.
    Test vectors are self-generated from the confirmed formula rather than
    trusting an unverifiable external number.
    """
    v = "".join(value.split()).upper()
    if len(v) < 15:
        return False
    body, key = v[:13], v[13:15]
    if not key.isdigit():
        return False
    dept = body[5:7]
    if dept == "2A":
        body = body[:5] + "19" + body[7:]
    elif dept == "2B":
        body = body[:5] + "18" + body[7:]
    if not body.isdigit():
        return False
    return int(key) == 97 - (int(body) % 97)


_AU_TFN_WEIGHTS = [1, 4, 3, 7, 5, 8, 6, 9, 10]


def _au_tfn(value: str) -> bool:
    """Australia TFN — 9-digit (modern) or 8-digit (legacy) format. Legacy
    numbers use the same weighted mod-11 check after left-padding with a
    zero to 9 digits (ATO developer guide: same weight table applies)."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) == 8:
        digits = [0] + digits
    if len(digits) != 9:
        return False
    total = sum(d * w for d, w in zip(digits, _AU_TFN_WEIGHTS))
    return total % 11 == 0


def _za_id(value: str) -> bool:
    """South Africa ID — 13 digits, Luhn checksum + a loose YYMMDD plausibility
    check on the first 6 digits (same spirit as the US SSN range rule)."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 13:
        return False
    month = digits[2] * 10 + digits[3]
    day = digits[4] * 10 + digits[5]
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return False
    return _luhn_checksum_ok(digits)


# ponytail: US SSN, UK NINO, and PAN (India) all have no checksum — US SSN
# and UK NINO have no check-digit position at all; PAN's "Checksum: No" is
# confirmed directly by Microsoft Purview's own SIT definition
# (learn.microsoft.com/en-us/purview/sit-defn-india-permanent-account-number),
# settling what secondary sources disagreed on. All three stay format-only
# ("structured" tier) with the strongest available rule-based check applied
# above (range/prefix rules), not a true checksum.
VALIDATORS: dict[str, Callable[[str], bool]] = {
    "luhn": _luhn,
    "payment_card": _payment_card,
    "iban_mod97": _iban_mod97,
    "verhoeff": _verhoeff,
    "ca_sin_luhn": _ca_sin,
    "gb_nino_prefix": _gb_nino_prefix,
    "us_ssn_range": _us_ssn_range,
    "cpf_mod11": _cpf,
    "cn_resident_id": _cn_resident_id,
    "de_steuerid": _de_steuerid,
    "fr_nir_mod97": _fr_nir,
    "au_tfn_mod11": _au_tfn,
    "za_id_luhn": _za_id,
}
