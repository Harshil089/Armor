import numpy as np

from armor.models import calibration


def test_pava_is_nondecreasing_on_violation():
    y = np.array([0.1, 0.9, 0.2, 0.8])
    fitted = calibration._pava(y, np.ones_like(y))
    assert all(fitted[i] <= fitted[i + 1] + 1e-9 for i in range(len(fitted) - 1))


def test_pava_already_monotonic_is_unchanged():
    y = np.array([0.1, 0.3, 0.5, 0.9])
    fitted = calibration._pava(y, np.ones_like(y))
    assert np.allclose(fitted, y)


def test_venn_abers_low_score_calibrates_low():
    # calibration set: low raw scores are negatives, high raw scores are positives
    x_cal = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    y_cal = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    p0, p1 = calibration.venn_abers(x_cal, y_cal, 0.08)
    assert p0 <= p1
    assert calibration.merge_venn_abers(p0, p1) < 0.5


def test_venn_abers_high_score_calibrates_high():
    x_cal = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    y_cal = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    p0, p1 = calibration.venn_abers(x_cal, y_cal, 0.92)
    assert calibration.merge_venn_abers(p0, p1) > 0.5


def test_merge_venn_abers_symmetric_interval_gives_midpoint():
    assert calibration.merge_venn_abers(0.4, 0.6) == 0.5


def test_calibrate_scores_uses_given_fingerprint_table(monkeypatch):
    fake_table = {
        "prompt_injection": (
            np.array([0.05, 0.1, 0.85, 0.9]),
            np.array([0.0, 0.0, 1.0, 1.0]),
        ),
    }
    monkeypatch.setitem(calibration._calibration_tables, "fake-fp", fake_table)
    result = calibration.calibrate_scores({"prompt_injection": 0.87}, "fake-fp")
    assert result["prompt_injection"]["score"] > 0.5
    assert "uncertainty" in result["prompt_injection"]


def test_calibrate_scores_passes_through_when_no_calibration_data(monkeypatch):
    monkeypatch.setitem(calibration._calibration_tables, "fake-fp-2", {})
    result = calibration.calibrate_scores({"jailbreak": 0.42}, "fake-fp-2")
    assert result["jailbreak"]["score"] == 0.42
    assert result["jailbreak"]["uncertainty"] == 0.0
