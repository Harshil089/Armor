import numpy as np

from armor.models.conformal import CONFORMAL_CLASSES, calibrate, nonconformity_scores, predict_set


def test_nonconformity_low_when_confidently_correct():
    probs = np.array([[0.9, 0.05, 0.05]])
    scores = nonconformity_scores(probs, np.array([0]))
    assert scores[0] < 0.15


def test_nonconformity_high_when_confidently_wrong():
    probs = np.array([[0.05, 0.05, 0.9]])
    scores = nonconformity_scores(probs, np.array([0]))
    assert scores[0] > 0.9


def test_calibrate_gives_higher_threshold_for_less_confident_calibration_set():
    confident = np.tile([0.9, 0.05, 0.05], (50, 1))
    unsure = np.tile([0.4, 0.35, 0.25], (50, 1))
    labels = np.zeros(50, dtype=int)
    t_confident = calibrate(confident, labels, alpha=0.1)
    t_unsure = calibrate(unsure, labels, alpha=0.1)
    assert t_unsure > t_confident


def test_predict_set_singleton_for_confident_prediction():
    threshold = 0.2
    probs = np.array([0.9, 0.05, 0.05])
    result = predict_set(probs, threshold)
    assert result == {"allow"}


def test_predict_set_multiple_for_ambiguous_prediction():
    threshold = 0.7
    probs = np.array([0.4, 0.35, 0.25])
    result = predict_set(probs, threshold)
    assert len(result) > 1
    assert result.issubset(set(CONFORMAL_CLASSES))
