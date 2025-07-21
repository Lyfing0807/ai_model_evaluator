import numpy as np
import pytest

from ai_eval_tool.utils.scoring_utils import normalize_metric_to_score


def test_normalize_metric_to_score_none_nan_inf():
    """
    测试当 value 为 None, NaN 或 Inf 时的行为。
    """
    assert normalize_metric_to_score(None) is None
    assert normalize_metric_to_score(np.nan) is None
    assert normalize_metric_to_score(np.inf) is None
    assert normalize_metric_to_score(-np.inf) is None


def test_normalize_metric_to_score_with_target_lower_is_better():
    """
    测试当使用 target 且 lower_is_better=True 时的行为。
    """
    # value <= target
    assert normalize_metric_to_score(5.0, target=10.0, lower_is_better=True) == 100.0
    assert normalize_metric_to_score(10.0, target=10.0, lower_is_better=True) == 100.0

    # value slightly higher than target (within 1.5x)
    assert normalize_metric_to_score(12.0, target=10.0, lower_is_better=True) == 80.0
    assert normalize_metric_to_score(15.0, target=10.0, lower_is_better=True) == 50.0

    # value significantly higher than target (beyond 2x)
    assert normalize_metric_to_score(20.0, target=10.0, lower_is_better=True) == 0.0
    assert normalize_metric_to_score(25.0, target=10.0, lower_is_better=True) == 0.0

    # target is ~0, value is higher
    assert normalize_metric_to_score(0.1, target=0.0, lower_is_better=True) == 0.0
    assert normalize_metric_to_score(0.0, target=0.0, lower_is_better=True) == 100.0


def test_normalize_metric_to_score_with_target_higher_is_better():
    """
    测试当使用 target 且 lower_is_better=False 时的行为。
    """
    # value >= target
    assert normalize_metric_to_score(10.0, target=5.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(5.0, target=5.0, lower_is_better=False) == 100.0

    # value slightly lower than target
    assert normalize_metric_to_score(4.0, target=5.0, lower_is_better=False) == 80.0
    assert normalize_metric_to_score(2.5, target=5.0, lower_is_better=False) == 50.0

    # value significantly lower than target
    assert normalize_metric_to_score(0.0, target=5.0, lower_is_better=False) == 0.0
    assert normalize_metric_to_score(1.0, target=5.0, lower_is_better=False) == 20.0

    # target is ~0, value is also ~0 or positive
    assert normalize_metric_to_score(0.0, target=0.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(0.1, target=0.0, lower_is_better=False) == 100.0


def test_normalize_metric_to_score_with_thresholds_lower_is_better():
    """
    测试当使用 good_threshold 和 bad_threshold 且 lower_is_better=True 时的行为。
    """
    # value <= good_threshold
    assert normalize_metric_to_score(5.0, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 100.0
    assert normalize_metric_to_score(10.0, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 100.0

    # value >= bad_threshold
    assert normalize_metric_to_score(20.0, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 0.0
    assert normalize_metric_to_score(25.0, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 0.0

    # good_threshold < value < bad_threshold
    assert normalize_metric_to_score(15.0, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 50.0
    assert normalize_metric_to_score(12.5, good_threshold=10.0, bad_threshold=20.0, lower_is_better=True) == 75.0


def test_normalize_metric_to_score_with_thresholds_higher_is_better():
    """
    测试当使用 good_threshold 和 bad_threshold 且 higher_is_better=True 时的行为。
    """
    # value >= good_threshold
    assert normalize_metric_to_score(20.0, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(10.0, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 100.0

    # value <= bad_threshold
    assert normalize_metric_to_score(5.0, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 0.0
    assert normalize_metric_to_score(0.0, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 0.0

    # bad_threshold < value < good_threshold
    assert normalize_metric_to_score(7.5, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 50.0
    assert normalize_metric_to_score(6.25, good_threshold=10.0, bad_threshold=5.0, lower_is_better=False) == 25.0


def test_normalize_metric_to_score_thresholds_equal():
    """
    测试当 good_threshold 等于 bad_threshold 时的行为。
    """
    assert normalize_metric_to_score(10.0, good_threshold=10.0, bad_threshold=10.0, lower_is_better=True) == 100.0
    assert normalize_metric_to_score(10.0, good_threshold=10.0, bad_threshold=10.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(5.0, good_threshold=10.0, bad_threshold=10.0, lower_is_better=True) == 100.0 # If good_threshold == bad_threshold, it's 100 if it meets the good condition
    assert normalize_metric_to_score(15.0, good_threshold=10.0, bad_threshold=10.0, lower_is_better=False) == 100.0 # Same here


def test_normalize_metric_to_score_is_0_1_rate():
    """
    测试当 is_0_1_rate_lower_better 参数被使用时的行为。
    """
    # lower_is_better=True (e.g., CV, error rate)
    assert normalize_metric_to_score(0.1, is_0_1_rate_lower_better=True) == pytest.approx(90.0)
    assert normalize_metric_to_score(0.5, is_0_1_rate_lower_better=True) == pytest.approx(50.0)
    assert normalize_metric_to_score(0.9, is_0_1_rate_lower_better=True) == pytest.approx(10.0)

    # lower_is_better=False (e.g., accuracy, consistency rate)
    assert normalize_metric_to_score(0.1, is_0_1_rate_lower_better=False) == 10.0
    assert normalize_metric_to_score(0.5, is_0_1_rate_lower_better=False) == 50.0
    assert normalize_metric_to_score(0.9, is_0_1_rate_lower_better=False) == 90.0


def test_normalize_metric_to_score_already_score():
    """
    测试当 value 已经在 0-100 范围内且没有其他规则适用时的行为。
    """
    assert normalize_metric_to_score(50.0) == 50.0
    assert normalize_metric_to_score(0.0) == 0.0
    assert normalize_metric_to_score(100.0) == 100.0


def test_normalize_metric_to_score_no_matching_rule():
    """
    测试当没有匹配的规则时，函数返回 0。
    """
    assert normalize_metric_to_score(150.0) == 0.0
    assert normalize_metric_to_score(-10.0) == 0.0


def test_normalize_metric_to_score_boundary_conditions():
    """
    测试分数是否始终在 0 到 100 之间。
    """
    # Values that should result in scores outside [0, 100] without clamping
    assert normalize_metric_to_score(0.0, target=10.0, lower_is_better=True) == 100.0
    assert normalize_metric_to_score(100.0, target=10.0, lower_is_better=True) == 0.0
    assert normalize_metric_to_score(10.0, target=0.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(0.0, target=10.0, lower_is_better=False) == 0.0

    # Test with thresholds that would result in scores outside [0, 100]
    assert normalize_metric_to_score(5.0, good_threshold=0.0, bad_threshold=10.0, lower_is_better=True) == 50.0
    assert normalize_metric_to_score(15.0, good_threshold=0.0, bad_threshold=10.0, lower_is_better=True) == 0.0
    assert normalize_metric_to_score(15.0, good_threshold=10.0, bad_threshold=0.0, lower_is_better=False) == 100.0
    assert normalize_metric_to_score(5.0, good_threshold=10.0, bad_threshold=0.0, lower_is_better=False) == 50.0
