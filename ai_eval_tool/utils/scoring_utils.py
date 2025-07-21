from typing import Optional

import numpy as np

from ..utils.logging_config import (
    get_logger,  # Relative import for utils within package
)

logger = get_logger(__name__)


def normalize_metric_to_score(
    value: Optional[float],
    target: Optional[float] = None,
    lower_is_better: bool = True,
    good_threshold: Optional[float] = None,
    bad_threshold: Optional[float] = None,
    # For values already 0-1 (like CV or rates), this flag indicates how to treat it
    # If True, (1-value)*100 (e.g. for CV). If False, value*100 (e.g. for accuracy rate)
    is_0_1_rate_lower_better: Optional[bool] = None,
) -> Optional[float]:
    """
    Normalizes a metric value to a 0-100 score.

    Args:
        value: The metric value to normalize.
        target: Ideal target value. If provided, score is based on proximity.
        lower_is_better: True if lower values of the metric are better (e.g., latency, error rate).
        good_threshold: Value at or beyond which the score is 100.
        bad_threshold: Value at or beyond which the score is 0.
        is_0_1_rate_lower_better: If the value is already a 0-1 rate (e.g. CV, accuracy),
                                  specifies if lower is better for this rate.
                                  True for CV (score = (1-CV)*100), False for accuracy (score = acc*100).

    Returns:
        A score between 0 and 100, or None if value is None.
    """
    if value is None or np.isnan(value) or np.isinf(value):
        return None

    score = 0.0
    if target is not None:
        if lower_is_better:
            if value <= target:
                score = 100.0
            elif target > 1e-9:
                ratio = value / target
                if ratio <= 1.5:
                    score = 100.0 - 100.0 * (ratio - 1.0)
                elif ratio <= 2.0:
                    score = 50.0 - 100.0 * (ratio - 1.5)
                else:
                    score = 0.0
            else:
                score = 0.0  # Target is ~0, value is higher, so score is 0
        else:  # Higher is better
            if value >= target:
                score = 100.0
            elif target > 1e-9:
                ratio = value / target
                score = ratio * 100.0
            else:  # Target is ~0, value is also ~0 or positive. If target=0, any positive value is infinitely better.
                # This case needs careful thought. If target is 0 and value is 0, score 100. If value > 0, score 100.
                # If target is very small positive, ratio can be huge.
                score = (
                    100.0 if value >= target else ((value / (target + 1e-9)) * 100.0)
                )

    elif good_threshold is not None and bad_threshold is not None:
        if good_threshold == bad_threshold:  # Avoid division by zero
            return (
                100.0
                if (lower_is_better and value <= good_threshold)
                or (not lower_is_better and value >= good_threshold)
                else 0.0
            )

        if lower_is_better:
            if value <= good_threshold:
                score = 100.0
            elif value >= bad_threshold:
                score = 0.0
            else:
                score = (
                    100.0 * (bad_threshold - value) / (bad_threshold - good_threshold)
                )
        else:  # Higher is better
            if value >= good_threshold:
                score = 100.0
            elif value <= bad_threshold:
                score = 0.0
            else:
                score = (
                    100.0 * (value - bad_threshold) / (good_threshold - bad_threshold)
                )

    elif is_0_1_rate_lower_better is not None and 0 <= value <= 1:
        if is_0_1_rate_lower_better:  # e.g. CV, error rate
            score = (1.0 - value) * 100.0
        else:  # e.g. accuracy, consistency rate
            score = value * 100.0

    elif (
        0 <= value <= 100 and target is None and good_threshold is None
    ):  # Assume it's already a score if no other rule applies
        score = value
    else:
        logger.debug(
            f"Cannot normalize score for value {value} with current rules. Returning 0."
        )
        return 0.0

    return max(0.0, min(100.0, score))
