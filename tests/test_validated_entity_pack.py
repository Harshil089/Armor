from armor.detectors.registry import default_registry

VALID_NETWORK_NUMBERS = {
    "visa": "4111111111111111",
    "amex": "378282246310005",
    "mastercard": "5555555555554444",
    "discover": "6011111111111117",
    "maestro_debit": "6759649826438453",  # Maestro is a debit-only network
    "diners_club": "30569309025904",
    "jcb": "3530111333300000",
}


def test_visa_passes_luhn_and_network_check():
    registry = default_registry()
    findings = registry.scan("card on file: 4111 1111 1111 1111 exp 12/29")
    cc = [f for f in findings if f["entity_id"] == "payment_card"]
    assert len(cc) == 1
    assert cc[0]["confidence_tier"] == "validated"


def test_invalid_luhn_digit_sequence_is_rejected():
    registry = default_registry()
    # same shape, last digit changed so Luhn fails
    findings = registry.scan("card on file: 4111 1111 1111 1112 exp 12/29")
    cc = [f for f in findings if f["entity_id"] == "payment_card"]
    assert cc == []


def test_every_major_network_and_debit_network_detected():
    registry = default_registry()
    for network, number in VALID_NETWORK_NUMBERS.items():
        findings = registry.scan(f"number: {number} end")
        cc = [f for f in findings if f["entity_id"] == "payment_card"]
        assert len(cc) == 1, f"{network} ({number}) not detected"
        assert cc[0]["confidence_tier"] == "validated"


def test_luhn_valid_but_unknown_network_prefix_is_rejected():
    registry = default_registry()
    # 16 digits, passes Luhn, but 9999-prefix matches no known card network
    findings = registry.scan("random number: 9999999999999995 end")
    cc = [f for f in findings if f["entity_id"] == "payment_card"]
    assert cc == []


def test_valid_iban_passes_mod97():
    registry = default_registry()
    findings = registry.scan("wire to GB29NWBK60161331926819 please")
    iban = [f for f in findings if f["entity_id"] == "iban"]
    assert len(iban) == 1
    assert iban[0]["confidence_tier"] == "validated"


def test_invalid_iban_checksum_is_rejected():
    registry = default_registry()
    findings = registry.scan("wire to GB29NWBK60161331926818 please")
    iban = [f for f in findings if f["entity_id"] == "iban"]
    assert iban == []
