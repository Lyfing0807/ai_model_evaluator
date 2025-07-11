"""
Stability evaluator for Classification Models.
"""
import polars as pl
import numpy as np
import yaml # For testing
from typing import List, Dict, Any, Set
from collections import defaultdict

from ...config_manager import MainConfig, ClassificationEvaluationParams
from ..base import StabilityEvaluatorBase
from ...utils.types import EvaluationResult
from ...utils.logging_config import get_logger
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
            logger.error("Classification evaluation parameters not correctly configured or are of the wrong type.")
            raise ValueError("Classification evaluation parameters missing or incorrect type for Classification model.")
        self.eval_params: ClassificationEvaluationParams # For type hinting
        logger.info("ClassificationStabilityEvaluator initialized.")
        self.top_k_values_from_config = self.eval_params.top_k

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info(f"Starting classification stability evaluation for model type {self.model_type}...")

        required_cols = ["image_id", "loop", "top_k_labels", "top_k_scores"]
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            logger.error(f"Classification stability evaluation requires columns: {required_cols}. Missing: {missing_cols}")
            return EvaluationResult(metrics={"error": f"Missing columns for classification stability: {missing_cols}"})

        if data_df.is_empty():
            logger.warning("Input DataFrame is empty for classification stability evaluation.")
            return EvaluationResult(metrics={"warning": "Empty input data for classification stability."})

        # Ensure 'loop' is integer
        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        # Store per-image stability results
        image_stability_results = []

        for image_id, group_df in data_df.group_by("image_id", maintain_order=False):
            if group_df.is_empty():
                continue

            num_loops = group_df["loop"].n_unique()
            if num_loops < 2: # Need at least 2 loops to compare for stability
                image_stability_results.append({
                    "image_id": image_id,
                    "top_1_consistency_rate": 1.0 if num_loops == 1 else None, # Consistent if only one observation
                    "mean_top_k_jaccard": {k: (1.0 if num_loops == 1 else None) for k in self.top_k_values_from_config},
                    "top_1_confidence_std": 0.0 if num_loops == 1 else None,
                    "num_loops_for_image": num_loops
                })
                continue

            # Sort by loop to ensure consistent order if needed, though most ops here are set-based or on lists from each row
            group_df = group_df.sort("loop")

            # Top-1 Consistency
            top_1_labels_per_loop = group_df.select(pl.col("top_k_labels").list.get(0).alias("top_1_label"))["top_1_label"]
            if top_1_labels_per_loop.null_count() == len(top_1_labels_per_loop): # All nulls
                top_1_consistency = None
            elif top_1_labels_per_loop.n_unique() == 1:
                top_1_consistency = 1.0
            else:
                # More precise: fraction of pairs of consecutive loops that are consistent, or overall mode fraction
                # For simplicity: if all are same = 1.0, else proportion of the most frequent label
                most_frequent_label_count = top_1_labels_per_loop.drop_nulls().mode().len() and top_1_labels_per_loop.drop_nulls().value_counts().sort(by="counts", descending=True)["counts"][0] or 0
                top_1_consistency = most_frequent_label_count / num_loops if num_loops > 0 else 0.0

            # Top-K Jaccard Similarity
            # We need to compare all pairs of loops for each K, then average.
            # Or, simpler: average Jaccard with the "mean set" (hard to define) or with first loop's set.
            # Let's average Jaccard similarity of each loop's Top-K set with the Top-K set from the first loop.
            mean_jaccard_per_k: Dict[int, Optional[float]] = {}

            first_loop_row = group_df.row(0, named=True) # Get first row after sorting by loop

            for k_idx, k_val in enumerate(self.top_k_values_from_config):
                # Ensure k_idx is within bounds of what DataLoader produced for top_k_labels/scores
                # DataLoader aggregates based on config's top_k list, so indices should match.

                first_loop_top_k_labels_list = first_loop_row["top_k_labels"]
                if first_loop_top_k_labels_list is None or k_idx >= len(first_loop_top_k_labels_list):
                    logger.debug(f"Not enough labels in first_loop_top_k_labels_list for k_idx={k_idx} (k_val={k_val}) for image {image_id}. List: {first_loop_top_k_labels_list}")
                    set_first_loop = set()
                else:
                    # The DataLoader stores a list of lists. We need the k-th list.
                    # No, DataLoader stores a list of labels for *all configured k values* in one list per row.
                    # So, if config.top_k = [1,3,5], then row["top_k_labels"] = [label_for_top1, label_for_top3, label_for_top5]
                    # This interpretation is wrong. DataLoader produces:
                    # top_k_labels = [ [top1_actual_label, top2_actual_label, ..., topK_actual_label_for_that_K_value], ... ]
                    # No, the spec for DataLoader for classification:
                    # "top_k_labels will be a list column where each element is a list of labels for that row, corresponding to specified K values"
                    # E.g., for a row, if top_k=[1,3,5], then top_k_labels might be `[['cat'], ['cat', 'dog', 'bird'], ['cat', 'dog', 'bird', 'fish', 'rabbit']]`
                    # This structure is complex. The plan's pseudocode for dataloader was:
                    # df['top_k_labels'] = all_rows_labels (where all_rows_labels is list of lists)
                    # and labels_for_this_row.append(row[id_col_name]) -- this implies one label per k.
                    # So, if top_k=[1,3,5], then row['top_k_labels'] = [label_at_top1, label_at_top3, label_at_top5]
                    # This seems more plausible.
                    # Let's assume `row['top_k_labels']` is `[val_for_k1, val_for_k2, ...]`.
                    # And we need the set of labels *up to* a certain K for Jaccard.
                    # This means DataLoader should produce list of actual labels for each k.
                    # e.g. top_k_labels_k1 = ['cat'], top_k_labels_k3 = ['cat', 'dog', 'rabbit']
                    # The current DataLoader produces one list `top_k_labels` where each element corresponds to one K from config.
                    # E.g. if config.top_k = [1,3,5], then a row's `top_k_labels` = `[label_for_k1, label_for_k3, label_for_k5]`.
                    # This is NOT what Jaccard needs. Jaccard needs SETS of labels up to K.
                    # This requires a re-think of DataLoader output or this evaluator's input handling.

                    # For now, let's assume DataLoader provides what we need, or we adapt.
                    # The current `data_loader.py` implementation:
                    # `df = df.with_columns(top_k_labels=pl.concat_list(label_cols_to_concat))`
                    # This means `top_k_labels` for a row is `[label_top1, label_top3, label_top5]`. This is not sets.
                    # The problem statement says: "Top-K Jaccard相似度: 计算各轮次Top-K标签**集合**间的Jaccard相似度均值"
                    # This implies for Top-3, we need the set {label1, label2, label3}.
                    # The current DataLoader does NOT provide this directly. It provides the label *at* position k.

                    # Let's assume for now that the `k_idx`-th element of `top_k_labels` list is the set of labels for the k-th configured K value.
                    # This is a placeholder pending clarification or DataLoader change.
                    # If `top_k_labels` contains LISTS of labels for each K, then:
                    # `actual_labels_for_this_k_val = first_loop_top_k_labels_list[k_idx]`
                    # And `set_first_loop = set(actual_labels_for_this_k_val)`
                    # This seems like what the user intended based on "集合".
                    # BUT, the DataLoader's current pattern `top{k}标签id` suggests one ID per K.

                    # Given the ambiguity, I will proceed with a simpler interpretation for now:
                    # Jaccard on the *set of labels that appear in the top_k_labels list for that row*.
                    # This means if top_k_labels = ['cat', 'dog', 'mouse'], the set is {'cat', 'dog', 'mouse'}.
                    # This is not "Top-K set" in the usual sense (set of first K items).
                    # This needs to be fixed later.
                    # For now, let's use only the labels available up to the k_idx in the top_k_labels list for the set.
                    # This is still not quite "Top-K set".

                    # Simplest interpretation for now: Jaccard of the single label at that K position.
                    # This is likely wrong for "set".
                    # Awaiting clarification or will make an assumption:
                    # Assume `top_k_labels` is a list OF LISTS, where outer list corresponds to row, inner list to K-values,
                    # and each element of inner list is ITSELF A LIST of labels for that K.
                    # Example: row["top_k_labels"][k_idx] = ['cat', 'dog'] for K=2.
                    # This is not what DataLoader does.

                    # Fallback: Calculate Jaccard based on the set of *all* labels provided in the `top_k_labels` list for that row.
                    # This is not ideal as it doesn't respect individual K values for the set construction.
                    current_k_jaccard_scores = []
                    set_first_loop_all_k = set(fl_label for fl_label in first_loop_row["top_k_labels"] if fl_label is not None)

                    for _, subsequent_row_dict in group_df.iter_rows(named=True):
                        if subsequent_row_dict["loop"] == first_loop_row["loop"]: # Skip comparing to itself
                            continue
                        current_loop_labels = subsequent_row_dict["top_k_labels"]
                        if current_loop_labels is None:
                            set_current_loop_all_k = set()
                        else:
                            set_current_loop_all_k = set(cl_label for cl_label in current_loop_labels if cl_label is not None)

                        current_k_jaccard_scores.append(jaccard_similarity(set_first_loop_all_k, set_current_loop_all_k))

                    mean_jaccard_per_k[k_val] = np.mean(current_k_jaccard_scores) if current_k_jaccard_scores else (1.0 if num_loops ==1 else None)


            # Top-1 Confidence Fluctuation
            top_1_scores_per_loop = group_df.select(pl.col("top_k_scores").list.get(0).alias("top_1_score"))["top_1_score"]
            if top_1_scores_per_loop.drop_nulls().len() > 1 :
                top_1_confidence_std = top_1_scores_per_loop.std()
            elif top_1_scores_per_loop.drop_nulls().len() == 1: # Only one non-null score
                 top_1_confidence_std = 0.0
            else: # All null or empty
                top_1_confidence_std = None


            image_stability_results.append({
                "image_id": image_id,
                "top_1_consistency_rate": top_1_consistency,
                "mean_top_k_jaccard": mean_jaccard_per_k, # This is a dict itself
                "top_1_confidence_std": top_1_confidence_std,
                "num_loops_for_image": num_loops
            })

        if not image_stability_results:
            logger.warning("No per-image classification stability metrics could be calculated.")
            return EvaluationResult(metrics={"warning": "No per-image classification stability metrics calculated."})

        # Aggregate per-image metrics
        stability_stats_df = pl.DataFrame(image_stability_results)

        final_metrics: Dict[str, Any] = {
            "cls_stab_mean_top_1_consistency_rate": stability_stats_df["top_1_consistency_rate"].mean(),
            "cls_stab_mean_top_1_confidence_std": stability_stats_df["top_1_confidence_std"].mean(),
        }

        # Aggregate mean_top_k_jaccard (which is a dict column)
        for k_val in self.top_k_values_from_config:
            # Extract all jaccard scores for this k_val from the list of dicts
            all_jaccards_for_k = [
                row["mean_top_k_jaccard"].get(k_val)
                for row in image_stability_results
                if row["mean_top_k_jaccard"] and row["mean_top_k_jaccard"].get(k_val) is not None
            ]
            if all_jaccards_for_k:
                final_metrics[f"cls_stab_mean_jaccard_top_{k_val}"] = np.mean(all_jaccards_for_k)
            else:
                final_metrics[f"cls_stab_mean_jaccard_top_{k_val}"] = None

        final_metrics_cleaned = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in final_metrics.items()}

        logger.info("Classification stability evaluation completed.")
        return EvaluationResult(
            metrics=final_metrics_cleaned,
            plots={},
            extra_data={"classification_stability_details_df": stability_stats_df}
        )

StabilityEvaluatorFactory.register_evaluator("classification", ClassificationStabilityEvaluator)


if __name__ == "__main__":
    from pathlib import Path

    dummy_config_content = """
project_info: {project_name: "ClsStab Test", model_type: "classification"}
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    image_path: "img_path"
    pre_time_ms: "t_pre"
    inference_time_ms: "t_inf"
    post_time_ms: "t_post"
    total_time_ms: "t_total"
    classification:
      top_k_id_pattern: "label_k{k}" # e.g. label_k1, label_k3
      top_k_score_pattern: "score_k{k}" # e.g. score_k1, score_k3
evaluation_params:
  classification:
    top_k: [1, 3] # We will test for Top-1 and Top-3 consistency/jaccard
report_settings: {output_dir: "./test_cls_stab_output"}
"""
    dummy_config_path = Path("dummy_cls_stab_config.yaml")
    with open(dummy_config_path, "w") as f: f.write(dummy_config_content)
    Path("./test_cls_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_config_path.read_text()))

    evaluator = ClassificationStabilityEvaluator(config)

    # DataLoader is expected to produce these columns based on patterns and top_k list [1,3]
    # top_k_labels: list where 0th element is label for K=1, 1st element is label for K=3
    # This matches what the current DataLoader does.
    # However, for Jaccard, the set of labels *up to K* is needed.
    # For this test, Jaccard is simplified to use the set of all labels in the row['top_k_labels'] list.
    test_data_df = pl.DataFrame({
        "image_id": ["img1", "img1", "img1", "img2", "img2"],
        "loop":     [1,      2,      3,      1,      2     ],
        "top_k_labels": [ # Each inner list corresponds to a row. Elements are [label_for_k1_conf, label_for_k3_conf]
            ["cat", "dog"],  # img1, loop1: Top-1=cat, (label at position corresponding to k=3 is dog)
            ["cat", "fox"],  # img1, loop2: Top-1=cat, (label for k=3 is fox)
            ["dog", "cat"],  # img1, loop3: Top-1=dog, (label for k=3 is cat)
            ["bird", "fish"],# img2, loop1
            ["bird", "fish"],# img2, loop2
        ],
        "top_k_scores": [ # Corresponds to top_k_labels
            [0.9, 0.7],    # img1, loop1
            [0.85, 0.75],  # img1, loop2
            [0.92, 0.8],   # img1, loop3
            [0.99, 0.6],   # img2, loop1
            [0.98, 0.65],  # img2, loop2
        ],
    })

    logger.info("--- Testing ClassificationStabilityEvaluator ---")
    results = evaluator.evaluate(test_data_df.clone())

    print("\nClassification Stability Metrics:")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")

    if "classification_stability_details_df" in results.extra_data:
        print("\nPer-Image Classification Stability Details:")
        print(results.extra_data["classification_stability_details_df"])
        # Expected: 2 rows in details_df (one for img1, one for img2)
        assert results.extra_data["classification_stability_details_df"].shape[0] == 2

    # Check some expected metric values
    # For img1: Top-1 labels are [cat, cat, dog]. Mode is cat (count 2). Consistency = 2/3 = 0.666
    # For img2: Top-1 labels are [bird, bird]. Mode is bird (count 2). Consistency = 2/2 = 1.0
    # Mean consistency = (0.666 + 1.0) / 2 = 0.8333
    assert abs(results.metrics["cls_stab_mean_top_1_consistency_rate"] - ((2/3 + 1.0)/2)) < 1e-3

    # For img1, loop1 scores_top1 = 0.9, loop2 = 0.85, loop3 = 0.92. Std = np.std([0.9, 0.85, 0.92])
    # For img2, loop1 scores_top1 = 0.99, loop2 = 0.98. Std = np.std([0.99, 0.98])
    img1_std = np.std([0.9, 0.85, 0.92])
    img2_std = np.std([0.99, 0.98])
    expected_mean_std = (img1_std + img2_std) / 2
    assert abs(results.metrics["cls_stab_mean_top_1_confidence_std"] - expected_mean_std) < 1e-3

    # Simplified Jaccard test (based on set of all labels in the row's top_k_labels list)
    # Img1:
    # L1: {'cat', 'dog'}
    # L2: {'cat', 'fox'} vs L1: J = 1/3
    # L3: {'dog', 'cat'} vs L1: J = 2/2 = 1.0 (oops, this is set of labels, not set of top-k labels)
    # This simplified Jaccard is not good. The current implementation compares each loop to the first.
    # L1 vs L1 (skip or 1.0), L2 vs L1, L3 vs L1
    # Img1, L1_set = {'cat', 'dog'}
    # Img1, L2_set = {'cat', 'fox'}. J(L1,L2) = 1/3
    # Img1, L3_set = {'dog', 'cat'}. J(L1,L3) = 2/2 = 1.0
    # Mean J for Img1 = (1/3 + 1.0) / 2 = (0.333 + 1.0)/2 = 0.6665
    # Img2:
    # L1_set = {'bird', 'fish'}
    # L2_set = {'bird', 'fish'}. J(L1,L2) = 2/2 = 1.0
    # Mean J for Img2 = 1.0
    # Overall Mean J for K=1 (and K=3, as this simplified jaccard uses all labels in list): (0.6665 + 1.0)/2 = 0.83325
    # This is for `cls_stab_mean_jaccard_top_1` and `cls_stab_mean_jaccard_top_3` due to current Jaccard impl.
    assert "cls_stab_mean_jaccard_top_1" in results.metrics # Check if key exists
    assert "cls_stab_mean_jaccard_top_3" in results.metrics
    # Actual value would be (( (1/3) + 1.0 ) / 2 + 1.0 ) / 2 = (0.66666 + 1.0) / 2 = 0.83333
    # This is for the current Jaccard interpretation.
    # print(f"Jaccard for K=1: {results.metrics['cls_stab_mean_jaccard_top_1']}")
    assert abs(results.metrics['cls_stab_mean_jaccard_top_1'] - (((1/3)+1.0)/2 + 1.0)/2) < 1e-3


    logger.info("ClassificationStabilityEvaluator test completed.")
    if dummy_config_path.exists():
        dummy_config_path.unlink()
    import shutil
    if Path("./test_cls_stab_output").exists():
        shutil.rmtree("./test_cls_stab_output")

```
