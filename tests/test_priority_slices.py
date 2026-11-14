from armor.eval.priority_slices import SLICE_CASES, evaluate_slices


def test_every_slice_has_balanced_cases_and_reports_separation():
    for slices in SLICE_CASES.values():
        for cases in slices.values():
            assert len(cases["positive"]) == len(cases["negative"]) >= 2

    positives = {
        text
        for slices in SLICE_CASES.values()
        for cases in slices.values()
        for text in cases["positive"]
    }

    def predict(text):
        score = 0.9 if text in positives else 0.1
        return {label: score for label in SLICE_CASES}

    results = evaluate_slices(predict)

    assert set(results) == {
        f"{label}/{name}"
        for label, slices in SLICE_CASES.items()
        for name in slices
    }
    assert all(gap > 0 for gap in results.values())
