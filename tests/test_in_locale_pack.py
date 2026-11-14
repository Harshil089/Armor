from armor.detectors.registry import default_registry


def test_pan_valid_holder_type_matches_but_stays_structured():
    registry = default_registry()
    # 4th char 'P' = individual, a real valid holder-type code (Wikipedia:
    # Permanent_account_number). Check digit (10th char) has no public
    # algorithm, so this stays "structured" even though it matches.
    findings = registry.scan("employee PAN is ABCPE1234F on file")
    pan = [f for f in findings if f["entity_id"] == "in_pan"]
    assert len(pan) == 1
    assert pan[0]["confidence_tier"] == "structured"


def test_pan_invalid_holder_type_letter_does_not_match():
    registry = default_registry()
    # 4th char 'D' is not one of the 10 valid holder-type codes
    findings = registry.scan("employee PAN is ABCDE1234F on file")
    pan = [f for f in findings if f["entity_id"] == "in_pan"]
    assert pan == []


def test_valid_aadhaar_passes_verhoeff():
    registry = default_registry()
    findings = registry.scan("aadhaar number 2345 6789 0124 on the form")
    aadhaar = [f for f in findings if f["entity_id"] == "in_aadhaar"]
    assert len(aadhaar) == 1
    assert aadhaar[0]["confidence_tier"] == "validated"


def test_invalid_aadhaar_checksum_is_rejected():
    registry = default_registry()
    findings = registry.scan("aadhaar number 2345 6789 0125 on the form")
    aadhaar = [f for f in findings if f["entity_id"] == "in_aadhaar"]
    assert aadhaar == []


def test_aadhaar_rejects_leading_zero_or_one_even_with_valid_checksum():
    # 023456789014 has a valid Verhoeff checksum but UIDAI never issues a
    # number starting with 0 or 1 — the validator must reject it anyway.
    from armor.detectors.validators import VALIDATORS

    assert VALIDATORS["verhoeff"]("023456789014") is False


def test_in_entities_are_present_alongside_global_and_validated_packs():
    registry = default_registry()
    entity_ids = {e.entity_id for e in registry.entries}
    assert {"in_pan", "in_aadhaar", "email", "payment_card"} <= entity_ids
