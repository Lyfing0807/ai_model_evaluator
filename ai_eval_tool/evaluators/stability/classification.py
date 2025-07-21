"""
Stability evaluator for Classification Models.
"""

from typing import Any, Dict, Optional, Set

import numpy as np
import polars as pl
import yaml  # For testing

from ...config_manager import ClassificationEvaluationParams, MainConfig
from ...utils.logging_config import get_logger
from ...utils.scoring_utils import normalize_metric_to_score
from ...utils.types import EvaluationResult
from ..base import StabilityEvaluatorBase
from .factory import StabilityEvaluatorFactory

logger = get_logger(__name__)


def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # Or 0.0, depending on definition for empty sets. For consistency, let's say 1.0 if both empty.
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


class ClassificationStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        # Ensure eval_params is correctly typed and present
        if not isinstance(self.eval_params, ClassificationEvaluationParams):
            logger.error(
                "Classification evaluation parameters not correctly configured or are of the wrong type."
            )
            raise ValueError(
                "Classification evaluation parameters missing or incorrect type for Classification model."
            )
        self.eval_params: ClassificationEvaluationParams  # For type hinting
        logger.info("ClassificationStabilityEvaluator initialized.")
        self.top_k_values_from_config = self.eval_params.top_k

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info(
            f"Starting classification stability evaluation for model type {self.model_type}..."
        )

        # The pre-processed `top_k_labels` and `top_k_scores` are useful for Top-1 consistency and confidence std.
        # However, for Jaccard, we need the raw `top{i}标签id` columns.
        required_cols = ["image_id", "loop", "top_k_labels", "top_k_scores"]
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            logger.error(
                f"Classification stability evaluation requires columns: {required_cols}. Missing: {missing_cols}"
            )
            return EvaluationResult(
                metrics={
                    "error": f"Missing columns for classification stability: {missing_cols}"
                }
            )

        if data_df.is_empty():
            logger.warning(
                "Input DataFrame is empty for classification stability evaluation."
            )
            return EvaluationResult(
                metrics={"warning": "Empty input data for classification stability."}
            )

        # Ensure 'loop' is integer
        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        # Store per-image stability results
        image_stability_results = []

        for image_id, group_df in data_df.group_by("image_id", maintain_order=False):
            if group_df.is_empty():
                continue

            num_loops = group_df["loop"].n_unique()
            if num_loops < 2:  # Need at least 2 loops to compare for stability
                image_stability_results.append(
                    {
                        "image_id": image_id,
                        "top_1_consistency_rate": (
                            1.0 if num_loops == 1 else None
                        ),  # Consistent if only one observation
                        "mean_top_k_jaccard": {
                            k: (1.0 if num_loops == 1 else None)
                            for k in self.top_k_values_from_config
                        },
                        "top_1_confidence_std": 0.0 if num_loops == 1 else None,
                        "num_loops_for_image": num_loops,
                    }
                )
                continue

            # Sort by loop to ensure consistent order if needed
            group_df = group_df.sort("loop")

            # --- Top-1 Consistency ---
            # This part correctly uses the pre-processed `top_k_labels` column.
            top_1_labels_per_loop = group_df.select(
                pl.col("top_k_labels").list.get(0).alias("top_1_label")
            )["top_1_label"]
            if top_1_labels_per_loop.null_count() == len(
                top_1_labels_per_loop
            ):  # All nulls
                top_1_consistency = None
            elif top_1_labels_per_loop.n_unique() == 1:
                top_1_consistency = 1.0
            else:
                most_frequent_label_count = (
                    top_1_labels_per_loop.drop_nulls().mode().len()
                    and top_1_labels_per_loop.drop_nulls()
                    .value_counts()
                    .sort(by="counts", descending=True)["counts"][0]
                    or 0
                )
                top_1_consistency = (
                    most_frequent_label_count / num_loops if num_loops > 0 else 0.0
                )

            # --- Top-K Jaccard Similarity (Corrected Logic) ---
            mean_jaccard_per_k: Dict[int, Optional[float]] = {}
            id_pattern = (
                self.config.data_loader.field_mapping.classification.top_k_id_pattern
            )

            for k_val in self.top_k_values_from_config:
                # Step 1: Dynamically generate the required column names for the Top-K set
                top_k_set_cols = [id_pattern.format(k=i) for i in range(1, k_val + 1)]

                # Step 2: Validate that all required columns for this K value exist in the original DataFrame
                missing_k_cols = [
                    col for col in top_k_set_cols if col not in data_df.columns
                ]
                if missing_k_cols:
                    logger.warning(
                        f"For image_id '{image_id}', cannot calculate Jaccard similarity for K={k_val} "
                        f"because required source columns are missing: {missing_k_cols}. "
                        f"Please ensure your CSV contains columns for all ranks up to K (e.g., top1, top2, ..., top{k_val})."
                    )
                    mean_jaccard_per_k[k_val] = None
                    continue

                # Step 3: Extract the Top-K sets for each loop
                top_k_sets_per_loop: Dict[int, Set[Any]] = {}
                for loop_id, loop_df in group_df.group_by("loop", maintain_order=True):
                    if loop_df.is_empty():
                        continue
                    row_dict = loop_df.row(0, named=True)
                    top_k_set = {
                        row_dict.get(col)
                        for col in top_k_set_cols
                        if row_dict.get(col) is not None
                    }
                    top_k_sets_per_loop[loop_id] = top_k_set

                # Step 4: Calculate Jaccard similarity stability
                loop_ids = sorted(top_k_sets_per_loop.keys())
                if (
                    len(loop_ids) < 2
                ):  # Should not happen due to outer check, but for safety
                    mean_jaccard_per_k[k_val] = 1.0 if len(loop_ids) == 1 else None
                    continue

                jaccard_scores = []
                first_loop_set = top_k_sets_per_loop[loop_ids[0]]
                for i in range(1, len(loop_ids)):
                    other_loop_set = top_k_sets_per_loop[loop_ids[i]]
                    jaccard_scores.append(
                        jaccard_similarity(first_loop_set, other_loop_set)
                    )

                mean_jaccard_per_k[k_val] = (
                    np.mean(jaccard_scores) if jaccard_scores else None
                )

            # --- Top-1 Confidence Fluctuation ---
            # This part correctly uses the pre-processed `top_k_scores` column.
            top_1_scores_per_loop = group_df.select(
                pl.col("top_k_scores").list.get(0).alias("top_1_score")
            )["top_1_score"]
            if top_1_scores_per_loop.drop_nulls().len() > 1:
                top_1_confidence_std = top_1_scores_per_loop.std()
            elif top_1_scores_per_loop.drop_nulls().len() == 1:
                top_1_confidence_std = 0.0
            else:
                top_1_confidence_std = None

            image_stability_results.append(
                {
                    "image_id": image_id,
                    "top_1_consistency_rate": top_1_consistency,
                    "mean_top_k_jaccard": mean_jaccard_per_k,  # This is a dict itself
                    "top_1_confidence_std": top_1_confidence_std,
                    "num_loops_for_image": num_loops,
                }
            )

        if not image_stability_results:
            logger.warning(
                "No per-image classification stability metrics could be calculated."
            )
            return EvaluationResult(
                metrics={
                    "warning": "No per-image classification stability metrics calculated."
                }
            )

        # Aggregate per-image metrics
        stability_stats_df = pl.DataFrame(image_stability_results)

        final_metrics: Dict[str, Any] = {
            "cls_stab_mean_top_1_consistency_rate": stability_stats_df[
                "top_1_consistency_rate"
            ].mean(),
            "cls_stab_mean_top_1_confidence_std": stability_stats_df[
                "top_1_confidence_std"
            ].mean(),
        }

        # Aggregate mean_top_k_jaccard (which is a dict column)
        for k_val in self.top_k_values_from_config:
            all_jaccards_for_k = [
                row["mean_top_k_jaccard"].get(k_val)
                for row in image_stability_results
                if row["mean_top_k_jaccard"]
                and row["mean_top_k_jaccard"].get(k_val) is not None
            ]
            if all_jaccards_for_k:
                final_metrics[f"cls_stab_mean_jaccard_top_{k_val}"] = np.mean(
                    all_jaccards_for_k
                )
            else:
                final_metrics[f"cls_stab_mean_jaccard_top_{k_val}"] = None

        final_metrics_cleaned = {
            k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
            for k, v in final_metrics.items()
        }

        self._calculate_stability_score(final_metrics_cleaned, stability_stats_df)

        logger.info("Classification stability evaluation completed.")
        return EvaluationResult(
            metrics=final_metrics_cleaned,
            plots={},
            extra_data={"classification_stability_details_df": stability_stats_df},
        )

    def _calculate_stability_score(
        self, metrics: Dict[str, Any], details_df: pl.DataFrame
    ):
        """Calculates overall classification stability score."""
        if not self.eval_params or not self.eval_params.scoring_weights:
            logger.warning(
                "Classification stability scoring weights not configured. Skipping score calculation."
            )
            return

        weights = self.eval_params.scoring_weights

        # Sub-score for Prediction Consistency (e.g., Top-1 Consistency Rate)
        # Higher is better (0-1 range).
        score_pred_consistency = normalize_metric_to_score(
            metrics.get("cls_stab_mean_top_1_consistency_rate"),
            is_0_1_rate_lower_better=False,
        )
        if score_pred_consistency is None:
            score_pred_consistency = 0.0
        # Could also incorporate Jaccard scores here if desired, with more weights.

        # Sub-score for Confidence Reliability/Stability (e.g., mean Top-1 Confidence Std Dev)
        # Lower std dev is better. Define good/bad thresholds.
        # Example: good_conf_std = 0.05, bad_conf_std = 0.2
        score_conf_reliability = normalize_metric_to_score(
            metrics.get("cls_stab_mean_top_1_confidence_std"),
            good_threshold=0.05,  # Example good value for std dev of confidence
            bad_threshold=0.2,  # Example bad value
            lower_is_better=True,
        )
        if score_conf_reliability is None:
            score_conf_reliability = 0.0
        # ECE would be another metric for confidence reliability if available.

        s_stability_classification = score_pred_consistency * weights.get(
            "prediction_consistency", 0.0
        ) + score_conf_reliability * weights.get("confidence_reliability", 0.0)

        total_weight = sum(
            weights.get(k, 0.0)
            for k in ["prediction_consistency", "confidence_reliability"]
        )
        if total_weight > 1e-6 and abs(total_weight - 1.0) > 1e-6:
            logger.warning(
                f"Classification stability weights ({weights}) do not sum to 1. Normalizing score."
            )
            s_stability_classification = s_stability_classification / total_weight
        s_stability_classification = max(0.0, min(100.0, s_stability_classification))

        metrics["cls_stab_score_prediction_consistency"] = score_pred_consistency
        metrics["cls_stab_score_confidence_reliability"] = score_conf_reliability
        metrics["cls_stab_score_overall"] = (
            s_stability_classification  # This is S_stability for classification
        )
        logger.info(
            f"Classification Stability Scores: PredictionCons={score_pred_consistency:.2f}, ConfidenceRel={score_conf_reliability:.2f}, Overall={s_stability_classification:.2f}"
        )


if __name__ == "__main__":
    StabilityEvaluatorFactory.register_evaluator(
        "classification", ClassificationStabilityEvaluator
    )

    from pathlib import Path

    # Config now needs to point to columns for each rank, e.g., pred_label_top1, pred_label_top2, etc.
    dummy_config_content = """
project_info: {project_name: "ClsStab Test", model_type: "classification"}
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    # These are now less relevant for the corrected Jaccard logic but still used for other metrics
    classification:
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"
evaluation_params:
  classification:
    top_k: [1, 3] # We will test for Top-1 and Top-3 consistency/jaccard
report_settings: {output_dir: "./test_cls_stab_output"}
"""
    dummy_config_path = Path("dummy_cls_stab_config.yaml")
    with open(dummy_config_path, "w") as f:
        f.write(dummy_config_content)
    Path("./test_cls_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_config_path.read_text()))

    evaluator = ClassificationStabilityEvaluator(config)

    # The test DataFrame now needs the raw columns that the new logic expects.
    # DataLoader will still produce `top_k_labels` and `top_k_scores` which are used for other metrics.
    test_data_df = pl.DataFrame(
        {
            "image_id": ["img1", "img1", "img2", "img2"],
            "loop": [1, 2, 1, 2],
            # Raw columns as they would appear in the CSV
            "pred_label_top1": ["cat", "cat", "bird", "bird"],
            "pred_label_top2": ["dog", "fox", "fish", "fish"],
            "pred_label_top3": ["cow", "cat", "frog", "frog"],
            # Pre-processed columns as if DataLoader ran (for other metrics)
            "top_k_labels": [
                ["cat", "cow"],
                ["cat", "cat"],
                ["bird", "frog"],
                ["bird", "frog"],
            ],  # Based on top_k=[1,3]
            "top_k_scores": [[0.9, 0.6], [0.88, 0.8], [0.95, 0.5], [0.96, 0.49]],
        }
    )

    logger.info("--- Testing Corrected ClassificationStabilityEvaluator ---")
    results = evaluator.evaluate(test_data_df.clone())

    print("\nClassification Stability Metrics:")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")

    # --- Assertions for Correctness ---
    # Top-1 consistency for img1: [cat, cat] -> 1.0. For img2: [bird, bird] -> 1.0. Mean = 1.0
    assert abs(results.metrics["cls_stab_mean_top_1_consistency_rate"] - 1.0) < 1e-6

    # Jaccard for K=3 for img1:
    # Loop 1 set: {'cat', 'dog', 'cow'}
    # Loop 2 set: {'cat', 'fox', 'cat'} -> {'cat', 'fox'}
    # Intersection: {'cat'} (1), Union: {'cat', 'dog', 'cow', 'fox'} (4). Jaccard = 1/4 = 0.25
    # Jaccard for K=3 for img2:
    # Loop 1 set: {'bird', 'fish', 'frog'}
    # Loop 2 set: {'bird', 'fish', 'frog'}
    # Jaccard = 1.0
    # Mean Jaccard@3 = (0.25 + 1.0) / 2 = 0.625
    assert "cls_stab_mean_jaccard_top_3" in results.metrics
    assert abs(results.metrics["cls_stab_mean_jaccard_top_3"] - 0.625) < 1e-6

    # Jaccard for K=1 for img1:
    # L1: {'cat'}, L2: {'cat'}. J=1.0
    # Jaccard for K=1 for img2:
    # L1: {'bird'}, L2: {'bird'}. J=1.0
    # Mean Jaccard@1 = (1.0 + 1.0) / 2 = 1.0
    assert "cls_stab_mean_jaccard_top_1" in results.metrics
    assert abs(results.metrics["cls_stab_mean_jaccard_top_1"] - 1.0) < 1e-6

    assert "cls_stab_score_overall" in results.metrics

    logger.info("ClassificationStabilityEvaluator test completed successfully.")
    if dummy_config_path.exists():
        dummy_config_path.unlink()
    import shutil

    if Path("./test_cls_stab_output").exists():
        shutil.rmtree("./test_cls_stab_output")
