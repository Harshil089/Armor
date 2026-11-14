from armor.detectors.registry import default_registry

# Self-generated (ATO doesn't publish public test TFNs like card networks do):
# 8 arbitrary digits, 9th solved for so the weighted sum is divisible by 11.
VALID_TFN = "123456782"


def test_valid_tfn_passes_mod11():
    registry = default_registry()
    findings = registry.scan(f"TFN: {VALID_TFN} on the payroll form")
    au = [f for f in findings if f["entity_id"] == "au_tfn"]
    assert len(au) == 1
    assert au[0]["confidence_tier"] == "validated"


def test_invalid_tfn_checksum_rejected():
    registry = default_registry()
    bad = VALID_TFN[:-1] + str((int(VALID_TFN[-1]) + 1) % 10)
    findings = registry.scan(f"TFN: {bad} on the payroll form")
    au = [f for f in findings if f["entity_id"] == "au_tfn"]
    assert au == []


def test_legacy_8digit_tfn_passes_with_zero_padding():
    from armor.detectors.validators import VALIDATORS

    # self-generated: [0]+digits weighted-summed to 0 mod 11 (ATO's own
    # documented legacy-format rule: pad with a leading zero, same weights)
    assert VALIDATORS["au_tfn_mod11"]("12345606") is True
    assert VALIDATORS["au_tfn_mod11"]("12345607") is False
