from armor.detectors.registry import default_registry

# Self-generated via the documented formula (key = 97 - (N mod 97)), not an
# externally-sourced "known-valid" number — see validators.py docstring for
# why (four public worked examples found during research, from separate
# sources, each failed independent arithmetic re-verification; the formula
# itself is confirmed directly against fr.wikipedia.org's own text).
VALID_NIR = "185077511400154"
VALID_NIR_CORSICA = "290052A12304552"


def test_valid_nir_passes_mod97():
    registry = default_registry()
    findings = registry.scan(f"NIR: {VALID_NIR} sur la carte vitale")
    fr = [f for f in findings if f["entity_id"] == "fr_nir"]
    assert len(fr) == 1
    assert fr[0]["confidence_tier"] == "validated"


def test_invalid_nir_key_rejected():
    registry = default_registry()
    bad = VALID_NIR[:-2] + "00"
    findings = registry.scan(f"NIR: {bad} sur la carte vitale")
    fr = [f for f in findings if f["entity_id"] == "fr_nir"]
    assert fr == []


def test_corsica_department_substitution():
    registry = default_registry()
    findings = registry.scan(f"NIR: {VALID_NIR_CORSICA} sur la carte vitale")
    fr = [f for f in findings if f["entity_id"] == "fr_nir"]
    assert len(fr) == 1
    assert fr[0]["confidence_tier"] == "validated"
