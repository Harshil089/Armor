from armor.detectors.registry import default_registry


def test_us_ssn_valid_range_passes():
    registry = default_registry()
    findings = registry.scan("SSN on file: 219-09-9999")
    ssn = [f for f in findings if f["entity_id"] == "us_ssn"]
    assert len(ssn) == 1
    assert ssn[0]["confidence_tier"] == "validated"


def test_us_ssn_invalid_area_rejected():
    registry = default_registry()
    findings = registry.scan("SSN on file: 000-09-9999")
    ssn = [f for f in findings if f["entity_id"] == "us_ssn"]
    assert ssn == []


def test_us_ssn_space_separated_form_also_matches():
    # Microsoft Purview's own SSN SIT covers dash- and space-separated forms
    registry = default_registry()
    findings = registry.scan("SSN on file: 219 09 9999")
    ssn = [f for f in findings if f["entity_id"] == "us_ssn"]
    assert len(ssn) == 1


def test_gb_nino_valid_prefix_passes():
    registry = default_registry()
    findings = registry.scan("NINO is AB123456C for this employee")
    nino = [f for f in findings if f["entity_id"] == "gb_nino"]
    assert len(nino) == 1
    assert nino[0]["confidence_tier"] == "validated"


def test_gb_nino_grouped_format_also_matches():
    # standard printed format on the physical card, per Microsoft Purview's
    # NINO SIT definition (two possible patterns: compact and grouped)
    registry = default_registry()
    findings = registry.scan("NINO is AB 12 34 56 C for this employee")
    nino = [f for f in findings if f["entity_id"] == "gb_nino"]
    assert len(nino) == 1


def test_gb_nino_excluded_prefix_rejected():
    from armor.detectors.validators import VALIDATORS

    # BG is one of the 7 prefixes HMRC never assigns (NIM39110)
    assert VALIDATORS["gb_nino_prefix"]("BG123456C") is False


def test_ca_sin_valid_luhn_passes():
    registry = default_registry()
    # 046-454-286 is a commonly used valid test SIN (Luhn-valid)
    findings = registry.scan("SIN: 046-454-286")
    sin = [f for f in findings if f["entity_id"] == "ca_sin"]
    assert len(sin) == 1
    assert sin[0]["confidence_tier"] == "validated"


def test_ca_sin_invalid_luhn_rejected():
    registry = default_registry()
    findings = registry.scan("SIN: 046-454-287")
    sin = [f for f in findings if f["entity_id"] == "ca_sin"]
    assert sin == []


def test_br_cpf_valid_checksum_passes():
    registry = default_registry()
    findings = registry.scan("CPF: 111.444.777-35")
    cpf = [f for f in findings if f["entity_id"] == "br_cpf"]
    assert len(cpf) == 1
    assert cpf[0]["confidence_tier"] == "validated"


def test_br_cpf_invalid_checksum_rejected():
    registry = default_registry()
    findings = registry.scan("CPF: 111.444.777-36")
    cpf = [f for f in findings if f["entity_id"] == "br_cpf"]
    assert cpf == []


def test_cn_resident_id_valid_checksum_passes():
    registry = default_registry()
    findings = registry.scan("ID: 11010519491231002X")
    cn = [f for f in findings if f["entity_id"] == "cn_resident_id"]
    assert len(cn) == 1
    assert cn[0]["confidence_tier"] == "validated"


def test_cn_resident_id_invalid_checksum_rejected():
    registry = default_registry()
    findings = registry.scan("ID: 110105194912310021")
    cn = [f for f in findings if f["entity_id"] == "cn_resident_id"]
    assert cn == []


def test_cn_resident_id_implausible_month_rejected_despite_valid_checksum():
    from armor.detectors.validators import VALIDATORS

    # month=13 in the embedded birth date, checksum otherwise recomputed valid
    assert VALIDATORS["cn_resident_id"]("110105194913310021") is False


def test_multiple_locales_coexist_without_a_deployment_locale_filter():
    registry = default_registry()
    entity_ids = {e.entity_id for e in registry.entries}
    assert {"in_pan", "us_ssn", "gb_nino", "ca_sin", "br_cpf", "cn_resident_id"} <= entity_ids
