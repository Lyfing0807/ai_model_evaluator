"""
Performance Evaluator: Calculates common performance metrics for any model type.
"""

from typing import Any, Dict

import numpy as np
import polars as pl

from ..config_manager import MainConfig
from ..utils.logging_config import get_logger
from ..utils.scoring_utils import normalize_metric_to_score
from ..utils.types import EvaluationResult
from .base import EvaluatorBase

logger = get_logger(__name__)


class PerformanceEvaluator(EvaluatorBase):
    """
    Calculates general performance metrics such as latency, throughput,
    and their stability (e.g., coefficient of variation).
    """

    def __init__(self, config: MainConfig):
        super().__init__(config)
        # Performance specific config can be accessed here if added to config structure
        logger.info("PerformanceEvaluator initialized.")

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        """
        Calculates performance metrics.

        Args:
            data_df: DataFrame containing performance-related columns
                     (e.g., 'total_time_ms', 'pre_time_ms', 'inference_time_ms',
                     'post_time_ms', 'loop', 'image_id').
                     Column names should be standardized by DataLoader.

        Returns:
            An EvaluationResult object with performance metrics.
        """
        logger.info("Starting performance evaluation...")
        metrics: Dict[str, Any] = {}
        plots: Dict[str, Any] = {}  # Placeholder for plot objects
        extra_data: Dict[str, Any] = {}  # Placeholder for data like outlier samples

        # Validate input DataFrame
        if data_df is None:
            raise ValueError("Input DataFrame cannot be None")

        if data_df.is_empty():
            logger.warning("Input DataFrame is empty for performance evaluation")
            return EvaluationResult(
                metrics={"warning": "Empty input data for performance evaluation"},
                plots={},
                extra_data={},
            )

        required_cols = [
            "total_time_ms",
            "pre_time_ms",
            "inference_time_ms",
            "post_time_ms",
            "loop",
            "image_id",  # loop and image_id for deduplication
        ]
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            logger.error(
                f"Missing required columns for performance evaluation: {missing_cols}"
            )
            raise ValueError(
                f"Missing required columns for performance evaluation: {missing_cols}"
            )

        # Validate data types for numeric columns
        numeric_cols = [
            "total_time_ms",
            "pre_time_ms",
            "inference_time_ms",
            "post_time_ms",
        ]
        for col in numeric_cols:
            if not data_df[col].dtype.is_numeric():
                logger.warning(
                    f"Column '{col}' is not numeric type: {data_df[col].dtype}. Attempting conversion."
                )
                try:
                    data_df = data_df.with_columns(
                        pl.col(col).cast(pl.Float64, strict=False)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert column '{col}' to numeric: {e}"
                    ) from e

        # Deduplicate based on (loop, image_id) to ensure each inference event is counted once
        # This is crucial if the input CSV has one row per detected object for detection models.
        try:
            perf_df = data_df.unique(
                subset=["loop", "image_id"], keep="first", maintain_order=True
            )
            logger.info(
                f"Performance data deduplicated by (loop, image_id): {data_df.shape[0]} -> {perf_df.shape[0]} rows."
            )
        except Exception as e:
            logger.error(f"Failed to deduplicate data: {e}", exc_info=True)
            raise ValueError(f"Data deduplication failed: {e}") from e

        if perf_df.is_empty():
            logger.warning(
                "No data available for performance evaluation after deduplication."
            )
            return EvaluationResult(
                metrics={"warning": "No data for performance eval"},
                plots={},
                extra_data={},
            )

        # Overall Performance Statistics (on total_time_ms)
        total_time_series = perf_df["total_time_ms"]
        if (
            total_time_series.is_empty()
            or total_time_series.null_count() == total_time_series.len()
        ):
            logger.warning("'total_time_ms' column is empty or all nulls.")
            return EvaluationResult(
                metrics={"warning": "'total_time_ms' is empty/all null"},
                plots={},
                extra_data={},
            )

        try:
            metrics["perf_mean_total_time_ms"] = total_time_series.mean()
            metrics["perf_median_total_time_ms"] = total_time_series.median()
            metrics["perf_std_total_time_ms"] = total_time_series.std()
            metrics["perf_min_total_time_ms"] = total_time_series.min()
            metrics["perf_max_total_time_ms"] = total_time_series.max()
            metrics["perf_p90_total_time_ms"] = total_time_series.quantile(
                0.90, interpolation="linear"
            )
            metrics["perf_p95_total_time_ms"] = total_time_series.quantile(
                0.95, interpolation="linear"
            )
            metrics["perf_p99_total_time_ms"] = total_time_series.quantile(
                0.99, interpolation="linear"
            )
        except Exception as e:
            logger.error(
                f"Failed to calculate basic performance statistics: {e}", exc_info=True
            )
            return EvaluationResult(
                metrics={"error": f"Performance calculation failed: {e}"},
                plots={},
                extra_data={},
            )

        if (
            metrics["perf_mean_total_time_ms"] is not None
            and metrics["perf_mean_total_time_ms"] > 0
        ):
            metrics["perf_avg_fps"] = 1000.0 / metrics["perf_mean_total_time_ms"]
        else:
            metrics["perf_avg_fps"] = 0

        # Performance Stability
        if (
            metrics["perf_mean_total_time_ms"] is not None
            and metrics["perf_std_total_time_ms"] is not None
            and metrics["perf_mean_total_time_ms"] > 0
        ):
            metrics["perf_cv_total_time_ms"] = (
                metrics["perf_std_total_time_ms"] / metrics["perf_mean_total_time_ms"]
            )
        else:
            metrics["perf_cv_total_time_ms"] = None

        # Jitter (mean absolute difference between consecutive total_time_ms values)
        # Ensure data is sorted if not already, e.g., by an implicit timestamp or sequence id if available
        # For now, assuming order in perf_df is meaningful or good enough
        if len(total_time_series) > 1:
            # Polars does not have a direct diff like pandas, calculate manually or use shift
            diffs = total_time_series - total_time_series.shift(1)
            metrics["perf_jitter_ms"] = diffs.abs().mean()
        else:
            metrics["perf_jitter_ms"] = 0

        # Worst-case amplification
        if (
            metrics["perf_median_total_time_ms"] is not None
            and metrics["perf_max_total_time_ms"] is not None
            and metrics["perf_median_total_time_ms"] > 0
        ):
            metrics["perf_worst_case_amplification"] = (
                metrics["perf_max_total_time_ms"] / metrics["perf_median_total_time_ms"]
            )
        else:
            metrics["perf_worst_case_amplification"] = None

        # High latency occurrence (e.g., > mean + 3*std)
        if (
            metrics["perf_mean_total_time_ms"] is not None
            and metrics["perf_std_total_time_ms"] is not None
        ):
            threshold = (
                metrics["perf_mean_total_time_ms"]
                + 3 * metrics["perf_std_total_time_ms"]
            )
            high_latency_count = total_time_series.filter(
                total_time_series > threshold
            ).len()
            metrics["perf_high_latency_rate"] = (
                high_latency_count / len(total_time_series)
                if len(total_time_series) > 0
                else 0
            )

            # Store top N high latency samples (image_id, loop, total_time_ms)
            # This is an example of 'extra_data'
            if high_latency_count > 0:
                top_slow_samples = (
                    perf_df.filter(pl.col("total_time_ms") > threshold)
                    .sort("total_time_ms", descending=True)
                    .head(10)
                    .select(
                        [
                            "image_id",
                            "loop",
                            "total_time_ms",
                            "pre_time_ms",
                            "inference_time_ms",
                            "post_time_ms",
                        ]
                    )
                )
                extra_data["perf_top_slow_samples_df"] = top_slow_samples
        else:
            metrics["perf_high_latency_rate"] = None

        # Time Composition Analysis
        time_components = ["pre_time_ms", "inference_time_ms", "post_time_ms"]
        component_metrics = {}
        total_mean_time = metrics.get("perf_mean_total_time_ms")

        for component in time_components:
            comp_series = perf_df[component]
            if comp_series.is_empty() or comp_series.null_count() == comp_series.len():
                logger.warning(f"Component series '{component}' is empty or all nulls.")
                mean_comp = None
                std_comp = None
                cv_comp = None
                percentage_comp = None
            else:
                mean_comp = comp_series.mean()
                std_comp = comp_series.std()
                cv_comp = (
                    (std_comp / mean_comp)
                    if mean_comp and mean_comp > 0 and std_comp is not None
                    else None
                )
                percentage_comp = (
                    (mean_comp / total_mean_time * 100)
                    if total_mean_time and total_mean_time > 0 and mean_comp is not None
                    else None
                )

            component_metrics[f"perf_mean_{component}"] = mean_comp
            component_metrics[f"perf_std_{component}"] = std_comp
            component_metrics[f"perf_cv_{component}"] = cv_comp
            if total_mean_time and total_mean_time > 0 and mean_comp is not None:
                component_metrics[f"perf_percentage_{component}"] = percentage_comp
            else:
                component_metrics[f"perf_percentage_{component}"] = None

        metrics.update(component_metrics)

        # Clean up None metrics to avoid issues with serialization or display later
        # metrics = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in metrics.items()}
        # Polars handles NaNs from operations like std on single element series as nulls, which is fine.
        # If we used numpy directly, we'd get np.nan, which might need conversion.

        # --- Calculate Performance Score ---
        self._calculate_performance_score(metrics)

        logger.info(
            f"Performance evaluation completed. Metrics calculated: {list(metrics.keys())}"
        )
        # Store the deduplicated df used for performance calculations for potential use in reporting
        extra_data["deduplicated_perf_df_for_charts"] = perf_df
        return EvaluationResult(metrics=metrics, plots=plots, extra_data=extra_data)

    def _calculate_performance_score(self, metrics: Dict[str, Any]):
        """Calculates overall performance score based on sub-metrics."""
        # Ensure self.config.evaluation_params.performance exists
        if (
            not hasattr(self.config.evaluation_params, "performance")
            or self.config.evaluation_params.performance is None
        ):
            logger.warning(
                "Performance evaluation parameters not found in config. Skipping performance score."
            )
            return

        params = self.config.evaluation_params.performance
        if not params.component_weights:
            logger.warning(
                "Performance component_weights not configured. Skipping performance score calculation."
            )
            return

        weights = params.component_weights

        p95_latency = metrics.get("perf_p95_total_time_ms")
        target_latency = params.target_latency_ms  # From PerformanceEvaluationParams

        # Score for throughput (based on p95 latency vs target)
        score_throughput = normalize_metric_to_score(
            value=p95_latency, target=target_latency, lower_is_better=True
        )
        if score_throughput is None:
            score_throughput = 0.0

        # Score for latency stability (based on CV vs target CV)
        cv_latency = metrics.get("perf_cv_total_time_ms")
        target_cv = params.cv_target_threshold  # From PerformanceEvaluationParams

        # For CV, it's a 0-1 rate, and lower is better.
        score_latency_stability = normalize_metric_to_score(
            value=cv_latency,
            # target=target_cv, # Option 1: treat target_cv as an ideal target
            # lower_is_better=True
            # Option 2: Use good/bad thresholds based on target_cv
            good_threshold=target_cv,  # Score 100 if cv <= target_cv
            bad_threshold=(
                target_cv * 5 if target_cv is not None else 0.5
            ),  # e.g. CV 5x target is 0 score. Default bad CV = 0.5
            lower_is_better=True,
            # Option 3: Directly use is_0_1_rate_lower_better if CV is guaranteed to be 0-1
            # is_0_1_rate_lower_better=True # If CV is the value
        )
        if score_latency_stability is None:
            score_latency_stability = 0.0

        s_performance = score_throughput * weights.get(
            "throughput", 0.0
        ) + score_latency_stability * weights.get("latency_stability", 0.0)

        # Ensure total weight is 1 if they are provided, otherwise scale.
        total_weight = weights.get("throughput", 0.0) + weights.get(
            "latency_stability", 0.0
        )
        if (
            total_weight > 1e-6 and abs(total_weight - 1.0) > 1e-6
        ):  # If weights provided but don't sum to 1
            logger.warning(
                f"Performance component weights ({weights}) do not sum to 1. Normalizing score."
            )
            s_performance = s_performance / total_weight
        s_performance = max(0.0, min(100.0, s_performance))

        metrics["perf_score_throughput"] = score_throughput
        metrics["perf_score_latency_stability"] = score_latency_stability
        metrics["perf_score_overall"] = s_performance  # This is S_performance
        logger.info(
            f"Performance Scores: Throughput={score_throughput:.2f}, LatencyStability={score_latency_stability:.2f}, S_Performance_Overall={s_performance:.2f}"
        )


if __name__ == "__main__":
    from pathlib import Path

    from ai_eval_tool.config_manager import load_config  # Relative import for testing

    # Create a dummy config
    dummy_config_content = """
project_info:
  project_name: "PerfEval Test"
  model_type: "detection" # Can be any, performance is generic
data_loader:
  field_mapping: # Placeholder, not directly used by PerfEval but good for config structure
    loop: "loop"
    image_id: "image_id"
    image_path: "image_path"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection:
      category_id: "category_id"
      score: "score"
      bbox: ["x","y","w","h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    dummy_config_path = Path("dummy_perf_eval_config.yaml")
    with open(dummy_config_path, "w") as f:
        f.write(dummy_config_content)

    config = load_config(dummy_config_path)
    perf_evaluator = PerformanceEvaluator(config)

    # Create a dummy DataFrame
    data = {
        "loop": [1, 1, 1, 2, 2],
        "image_id": [
            "img1",
            "img1",
            "img2",
            "img1",
            "img2",
        ],  # img1 in loop 1 is duplicated
        "pre_time_ms": [10, 10, 12, 11, 13],
        "inference_time_ms": [100, 100, 120, 110, 130],
        "post_time_ms": [5, 5, 6, 5, 7],
        "total_time_ms": [115, 115, 138, 126, 150],
    }
    df = pl.DataFrame(data)

    logger.info("--- Testing PerformanceEvaluator ---")
    results = perf_evaluator.evaluate(
        df.clone()
    )  # Clone df as evaluate might modify or subset it

    print("\nPerformance Metrics:")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")

    if results.extra_data.get("perf_top_slow_samples_df"):
        print("\nTop Slow Samples:")
        print(results.extra_data["perf_top_slow_samples_df"])

    # Basic assertions (deduplication check)
    # Original df has 5 rows. After unique by (loop, image_id):
    # (1, img1), (1, img2), (2, img1), (2, img2) -> 4 unique inference events
    # Expected mean of [115, 138, 126, 150]
    expected_mean = np.mean(
        [115, 138, 126, 150]
    )  # (115+138+126+150)/4 = 529/4 = 132.25
    assert (
        abs(results.metrics["perf_mean_total_time_ms"] - expected_mean) < 1e-6
    ), f"Mean total time mismatch. Expected {expected_mean}, Got {results.metrics['perf_mean_total_time_ms']}"
    logger.info("PerformanceEvaluator test completed.")

    if dummy_config_path.exists():
        dummy_config_path.unlink()

    # Test with empty dataframe
    logger.info("--- Testing PerformanceEvaluator with empty DataFrame ---")
    empty_df = df.clear()  # Creates an empty df with same schema
    # Or more robustly:
    empty_df_for_test = pl.DataFrame(
        {
            "loop": [],
            "image_id": [],
            "pre_time_ms": [],
            "inference_time_ms": [],
            "post_time_ms": [],
            "total_time_ms": [],
        },
        schema={
            "loop": pl.Int64,
            "image_id": pl.Utf8,
            "pre_time_ms": pl.Float64,
            "inference_time_ms": pl.Float64,
            "post_time_ms": pl.Float64,
            "total_time_ms": pl.Float64,
        },
    )

    results_empty = perf_evaluator.evaluate(empty_df_for_test)
    print("\nPerformance Metrics (Empty DF):")
    for k, v in results_empty.metrics.items():
        print(f"  {k}: {v}")
    assert "warning" in results_empty.metrics
    logger.info("PerformanceEvaluator empty DataFrame test completed.")

    # Test with all null total_time_ms
    logger.info("--- Testing PerformanceEvaluator with all null total_time_ms ---")
    null_df_data = {
        "loop": [1, 1],
        "image_id": ["img1", "img2"],
        "pre_time_ms": [10, 12],
        "inference_time_ms": [100, 120],
        "post_time_ms": [5, 6],
        "total_time_ms": [None, None],
    }
    null_df = pl.DataFrame(null_df_data).with_columns(
        pl.col("total_time_ms").cast(pl.Float64)
    )
    results_null = perf_evaluator.evaluate(null_df)
    print("\nPerformance Metrics (Null total_time_ms):")
    for k, v in results_null.metrics.items():
        print(f"  {k}: {v}")
    assert (
        "warning" in results_null.metrics
        or results_null.metrics.get("perf_mean_total_time_ms") is None
    )
    logger.info("PerformanceEvaluator all null total_time_ms test completed.")
