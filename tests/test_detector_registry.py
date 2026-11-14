import json

from armor.detectors.registry import DetectorRegistry


def _write_config(tmp_path, entities):
    path = tmp_path / "entities.json"
    path.write_text(json.dumps({"entities": entities}))
    return path


def test_scan_finds_match_and_masks_evidence(tmp_path):
    config = _write_config(
        tmp_path,
        [{"entity_id": "test_email", "pattern": r"[\w.+-]+@[\w-]+\.[\w.-]+", "severity": "high"}],
    )
    registry = DetectorRegistry.load(config)
    findings = registry.scan("contact me at admin@example.com please")

    assert len(findings) == 1
    assert findings[0]["entity_id"] == "test_email"
    assert findings[0]["confidence_tier"] == "structured"
    assert findings[0]["evidence"].endswith(".com")
    assert "admin@example.com" not in findings[0]["evidence"]


def test_locale_scoping(tmp_path):
    config = _write_config(
        tmp_path,
        [{"entity_id": "in_only", "locale": "IN", "pattern": r"XYZ\d{3}", "severity": "medium"}],
    )
    registry = DetectorRegistry.load(config)

    assert registry.scan("XYZ123", locales={"US"}) == []
    assert len(registry.scan("XYZ123", locales={"IN"})) == 1
    assert len(registry.scan("XYZ123")) == 1  # no locale filter = everything


def test_validator_gates_match_and_confidence_tier(tmp_path, monkeypatch):
    import armor.detectors.validators as validators_module

    monkeypatch.setitem(validators_module.VALIDATORS, "even_length", lambda v: len(v) % 2 == 0)
    config = _write_config(
        tmp_path,
        [{"entity_id": "even_code", "pattern": r"CODE\w+", "severity": "low", "validator": "even_length"}],
    )
    registry = DetectorRegistry.load(config)

    assert registry.scan("CODE1") == []  # len("CODE1")=5, odd -> validator fails
    findings = registry.scan("CODE12")  # len=6, even -> validator passes
    assert len(findings) == 1
    assert findings[0]["confidence_tier"] == "validated"
