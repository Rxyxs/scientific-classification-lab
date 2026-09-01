import numpy as np

from src.metrics import ams_score


def test_ams_zero_when_no_signal_predicted():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    assert ams_score(y_true, y_pred, weights) == 0.0


def test_ams_hand_computed_value():
    # s=2 (peso 1 c/u), b=1 (peso 1) -> AMS = sqrt(2*((2+1+10)*ln(1+2/11) - 2))
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    expected = np.sqrt(2 * ((2 + 1 + 10) * np.log(1 + 2 / 11) - 2))
    assert abs(ams_score(y_true, y_pred, weights) - expected) < 1e-9


def test_ams_increases_with_more_true_signal_caught():
    y_true = np.array([1, 1, 1, 0, 0])
    weights = np.ones(5)
    low = ams_score(y_true, np.array([1, 0, 0, 0, 0]), weights)
    high = ams_score(y_true, np.array([1, 1, 1, 0, 0]), weights)
    assert high > low
