from armor.models.pii_ner import get_pii_ner


def test_detects_person_name():
    ner = get_pii_ner()
    findings = ner.predict("Please contact John Smith about the invoice.")
    entity_ids = {f["entity_id"] for f in findings}
    assert "person_name" in entity_ids


def test_detects_address_like_location():
    ner = get_pii_ner()
    findings = ner.predict("She moved to San Francisco last year.")
    entity_ids = {f["entity_id"] for f in findings}
    assert "address" in entity_ids


def test_benign_sentence_has_no_pii_findings():
    ner = get_pii_ner()
    findings = ner.predict("please review the attached document before the meeting")
    assert findings == []
