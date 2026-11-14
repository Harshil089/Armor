from armor.eval.agreement import adjudication_rate, agreement_by_class, cohens_kappa, full_report, raw_agreement

LABELS = ["allow", "flag", "block", "review"]


def test_raw_agreement_perfect():
    assert raw_agreement(["allow", "block"], ["allow", "block"]) == 1.0


def test_raw_agreement_none():
    assert raw_agreement(["allow", "block"], ["block", "allow"]) == 0.0


def test_raw_agreement_empty():
    assert raw_agreement([], []) == 0.0


def test_cohens_kappa_perfect_agreement_is_one():
    a = ["allow", "flag", "block", "review"] * 5
    b = list(a)
    assert cohens_kappa(a, b, LABELS) == 1.0


def test_cohens_kappa_zero_for_chance_level_agreement():
    # Both reviewers use "allow" and "block" evenly, but paired to
    # disagree exactly at the chance rate for a 2-outcome 50/50 split.
    a = ["allow", "block"] * 10
    b = ["block", "allow"] * 10
    kappa = cohens_kappa(a, b, LABELS)
    assert kappa < 0.1  # near-zero, well below any real agreement


def test_agreement_by_class_only_counts_relevant_examples():
    a = ["allow", "allow", "flag"]
    b = ["allow", "flag", "flag"]
    result = agreement_by_class(a, b, LABELS)
    assert result["allow"] == 0.5  # 1 of 2 examples involving "allow" agreed
    assert result["flag"] == 0.5   # 1 of 2 examples involving "flag" agreed
    assert result["block"] is None  # no examples involved "block"


def test_adjudication_rate():
    assert adjudication_rate([True, True, False, False]) == 0.5
    assert adjudication_rate([]) == 0.0


def test_full_report_shape():
    a = ["allow", "flag", "block"]
    b = ["allow", "block", "block"]
    report = full_report(a, b, LABELS)
    assert set(report.keys()) == {
        "raw_agreement", "agreement_by_class", "confusion_matrix", "cohens_kappa", "adjudication_rate"
    }
