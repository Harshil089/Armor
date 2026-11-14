from armor.detectors.registry import default_registry


def test_valid_steuerid_passes():
    registry = default_registry()
    findings = registry.scan("Steuer-ID: 86095742719 auf der Lohnsteuerkarte")
    de = [f for f in findings if f["entity_id"] == "de_steuerid"]
    assert len(de) == 1
    assert de[0]["confidence_tier"] == "validated"


def test_invalid_steuerid_checksum_rejected():
    registry = default_registry()
    findings = registry.scan("Steuer-ID: 86095742718 auf der Lohnsteuerkarte")
    de = [f for f in findings if f["entity_id"] == "de_steuerid"]
    assert de == []


def test_structurally_invalid_digit_pattern_rejected_despite_valid_checksum():
    from armor.detectors.validators import VALIDATORS

    # first 10 digits are all distinct (0-9 once each) — check digit is
    # correct per the BZSt math, but the repetition rule requires exactly
    # one digit twice/thrice and at least one absent, so this must fail.
    assert VALIDATORS["de_steuerid"]("12345678903") is False
