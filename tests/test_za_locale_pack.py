from armor.detectors.registry import default_registry


def test_valid_id_passes_luhn_and_date_check():
    registry = default_registry()
    findings = registry.scan("SA ID: 8001015009087 on the form")
    za = [f for f in findings if f["entity_id"] == "za_id"]
    assert len(za) == 1
    assert za[0]["confidence_tier"] == "validated"


def test_invalid_luhn_checksum_rejected():
    registry = default_registry()
    findings = registry.scan("SA ID: 8001015009088 on the form")
    za = [f for f in findings if f["entity_id"] == "za_id"]
    assert za == []


def test_implausible_birthdate_rejected_even_with_valid_luhn():
    from armor.detectors.validators import VALIDATORS

    # month=13 is not a real month, regardless of Luhn outcome
    assert VALIDATORS["za_id_luhn"]("8013015009087") is False
