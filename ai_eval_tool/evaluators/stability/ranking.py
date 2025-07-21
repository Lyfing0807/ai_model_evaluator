"""
Stability evaluator for Ranking and Recommendation Models.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import polars as pl
import yaml  # For testing
from sklearn.metrics import ndcg_score

from ...config_manager import MainConfig, RankingEvaluationParams
from ...utils.logging_config import get_logger
from ...utils.scoring_utils import normalize_metric_to_score  # Assuming this path
from ...utils.types import EvaluationResult
from ..base import StabilityEvaluatorBase
from .factory import StabilityEvaluatorFactory

logger = get_logger(__name__)


def recall_at_k(y_true_list: List[Any], y_pred_list: List[Any], k: int) -> float:
    """Computes Recall@k."""
    if not y_true_list:  # No relevant items
        return (
            1.0 if not y_pred_list else 0.0
        )  # Or 0.0 always if no true items? Define behavior.
        # Common definition: if no true items, recall is undefined or 1 if no preds either.
        # Let's assume if no true items, it's perfect if no predictions, else 0.

    y_true_set = set(y_true_list)
    if (
        not y_true_set
    ):  # Handles case where y_true_list might contain non-hashable or is effectively empty
        return 1.0 if not y_pred_list else 0.0

    y_pred_at_k = set(y_pred_list[:k])

    hits = len(y_true_set.intersection(y_pred_at_k))
    return hits / len(y_true_set) if len(y_true_set) > 0 else 0.0


def precision_at_k(y_true_list: List[Any], y_pred_list: List[Any], k: int) -> float:
    """Computes Precision@k."""
    if k == 0:
        return 0.0
    y_true_set = set(y_true_list)
    y_pred_at_k = set(y_pred_list[:k])  # Consider only top K predictions
    if not y_pred_at_k:  # No predictions made up to K
        return 0.0  # Or 1.0 if no true items either? Usually 0 if no preds.

    hits = len(y_true_set.intersection(y_pred_at_k))
    return hits / k  # Hits divided by K (number of predictions considered)


def jaccard_similarity_at_k(list1: List[Any], list2: List[Any], k: int) -> float:
    """Computes Jaccard similarity for top-K elements of two lists."""
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


class RankingStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        if not isinstance(self.eval_params, RankingEvaluationParams):
            raise ValueError("Ranking evaluation parameters missing or incorrect type.")
        self.eval_params: RankingEvaluationParams
        self.k_values: List[int] = sorted(
            list(set(self.eval_params.k_values))
        )  # Ensure unique and sorted
        logger.info(
            f"RankingStabilityEvaluator initialized with K values: {self.k_values}."
        )

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info("Starting ranking stability evaluation...")
        required_cols = [
            "query_id",
            "loop",
            "internal_item_id_pred_list",
            "internal_item_id_gt_list",
        ]
        if not all(col in data_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data_df.columns]
            return EvaluationResult(
                metrics={"error": f"Missing required columns for Ranking: {missing}"}
            )

        if data_df.is_empty():
            return EvaluationResult(
                metrics={"warning": "Empty input data for Ranking eval."}
            )

        # Store per-query, per-loop, per-k metrics
        # List of dicts: {'query_id': q, 'loop': l, 'k': k_val, 'ndcg': score, 'recall': score}
        all_query_loop_metrics = []

        # Store per-query stability metrics (std dev of metrics, avg jaccard)
        # List of dicts: {'query_id': q, 'ndcg_at_5_std': val, 'recall_at_5_std': val, 'jaccard_at_5_mean': val}
        query_stability_metrics = []

        for query_id_val, query_group_df in data_df.group_by(
            "query_id", maintain_order=False
        ):
            num_loops = query_group_df["loop"].n_unique()

            # Store metrics for this query across all loops to calculate stability later
            # Dict: metric_name_at_k -> [val_loop1, val_loop2, ...]
            metrics_across_loops_for_query: Dict[str, List[float]] = defaultdict(list)
            # Store top-K item sets for this query across all loops for Jaccard stability
            # Dict: k_val -> {loop_id: set_of_top_k_items}
            top_k_sets_across_loops: Dict[int, Dict[Any, Set[Any]]] = defaultdict(
                lambda: defaultdict(set)
            )

            for loop_id_val, loop_group_df in query_group_df.group_by(
                "loop", maintain_order=True
            ):  # Usually 1 row per query-loop
                if loop_group_df.is_empty():
                    continue
                row = loop_group_df.row(
                    0, named=True
                )  # Assuming one entry per query-loop pair

                pred_items: List[str] = row.get("internal_item_id_pred_list") or []
                gt_items: List[str] = row.get("internal_item_id_gt_list") or []
                # Scores might be None if not provided in CSV
                pred_scores: Optional[List[float]] = row.get("internal_score_pred_list")

                if (
                    not pred_items and not gt_items
                ):  # Skip if both are empty for this query-loop
                    logger.debug(
                        f"Skipping query {query_id_val}, loop {loop_id_val} due to empty pred and gt lists."
                    )
                    continue

                for k in self.k_values:
                    # NDCG@k
                    # sklearn.ndcg_score needs true relevance scores (binary 0/1 or graded) for y_true
                    # and predicted scores for y_score.
                    # y_true: relevance scores for *all* items in a common universe, or just for predicted items.
                    # For NDCG, typically y_true is relevance of predicted items.
                    # We need to map gt_items to relevance scores for pred_items.

                    # Create relevance array for predicted items
                    relevance_scores_for_pred = [
                        1.0 if item in set(gt_items) else 0.0 for item in pred_items[:k]
                    ]
                    true_relevance_ideal = [1.0] * len(set(gt_items)) + [0.0] * (
                        k - len(set(gt_items))
                    )  # Ideal ranking
                    true_relevance_ideal = true_relevance_ideal[:k]

                    # If pred_scores are available and match length of pred_items, use them. Otherwise, use rank as score.
                    # sklearn's ndcg_score expects y_score for items in the order they appear.
                    # Here, pred_items are already ranked. So y_score can be their original scores, or decreasing ranks.
                    y_score_for_ndcg = np.array(
                        (
                            pred_scores[:k]
                            if pred_scores and len(pred_scores) >= k
                            else np.arange(k, 0, -1)
                        ),
                        dtype=float,
                    )

                    current_ndcg = 0.0
                    if (
                        len(relevance_scores_for_pred) > 0
                        and len(y_score_for_ndcg) > 0
                        and len(relevance_scores_for_pred) == len(y_score_for_ndcg)
                    ):
                        # Ensure y_true and y_score are 2D for sklearn
                        current_ndcg = ndcg_score(
                            np.asarray([relevance_scores_for_pred]),
                            np.asarray([y_score_for_ndcg]),
                            k=k,
                        )
                    elif (
                        not relevance_scores_for_pred and not gt_items
                    ):  # No relevant items and no predictions, perfect score
                        current_ndcg = 1.0

                    current_recall = recall_at_k(gt_items, pred_items, k)

                    all_query_loop_metrics.append(
                        {
                            "query_id": query_id_val,
                            "loop": loop_id_val,
                            "k": k,
                            "ndcg": current_ndcg,
                            "recall": current_recall,
                        }
                    )
                    metrics_across_loops_for_query[f"ndcg_at_{k}"].append(current_ndcg)
                    metrics_across_loops_for_query[f"recall_at_{k}"].append(
                        current_recall
                    )
                    top_k_sets_across_loops[k][loop_id_val] = set(pred_items[:k])

            # Calculate stability for this query_id
            if num_loops >= 2:
                query_stab_entry: Dict[str, Any] = {
                    "query_id": query_id_val,
                    "num_loops": num_loops,
                }
                for k in self.k_values:
                    ndcg_list = metrics_across_loops_for_query.get(f"ndcg_at_{k}", [])
                    recall_list = metrics_across_loops_for_query.get(
                        f"recall_at_{k}", []
                    )
                    query_stab_entry[f"ndcg_at_{k}_std"] = (
                        np.std(ndcg_list) if len(ndcg_list) > 1 else 0.0
                    )
                    query_stab_entry[f"recall_at_{k}_std"] = (
                        np.std(recall_list) if len(recall_list) > 1 else 0.0
                    )

                    # Jaccard stability for Top-K sets
                    jaccard_scores_for_k = []
                    loop_ids_for_k = list(top_k_sets_across_loops[k].keys())
                    if len(loop_ids_for_k) >= 2:
                        # Compare first loop's set with all subsequent loops' sets for this K
                        first_loop_set = top_k_sets_across_loops[k][loop_ids_for_k[0]]
                        for i in range(1, len(loop_ids_for_k)):
                            other_loop_set = top_k_sets_across_loops[k][
                                loop_ids_for_k[i]
                            ]
                            jaccard_scores_for_k.append(
                                jaccard_similarity_at_k(
                                    list(first_loop_set), list(other_loop_set), k
                                )
                            )  # k is effectively list length here
                    query_stab_entry[f"jaccard_at_{k}_mean_vs_first_loop"] = (
                        np.mean(jaccard_scores_for_k)
                        if jaccard_scores_for_k
                        else (1.0 if num_loops == 1 else None)
                    )
                query_stability_metrics.append(query_stab_entry)

        # Aggregate overall metrics
        final_metrics: Dict[str, Any] = {}
        extra_data: Dict[str, Any] = {}
        if all_query_loop_metrics:
            loop_metrics_df = pl.DataFrame(all_query_loop_metrics)
            for k in self.k_values:
                final_metrics[f"rnk_mean_ndcg_at_{k}"] = loop_metrics_df.filter(
                    pl.col("k") == k
                )["ndcg"].mean()
                final_metrics[f"rnk_mean_recall_at_{k}"] = loop_metrics_df.filter(
                    pl.col("k") == k
                )["recall"].mean()

        if query_stability_metrics:
            query_stability_df = pl.DataFrame(query_stability_metrics)
            for k in self.k_values:
                final_metrics[f"rnk_stab_mean_ndcg_at_{k}_std"] = query_stability_df[
                    f"ndcg_at_{k}_std"
                ].mean()
                final_metrics[f"rnk_stab_mean_recall_at_{k}_std"] = query_stability_df[
                    f"recall_at_{k}_std"
                ].mean()
                final_metrics[f"rnk_stab_mean_jaccard_at_{k}"] = query_stability_df[
                    f"jaccard_at_{k}_mean_vs_first_loop"
                ].mean()
            extra_data["ranking_stability_details_per_query_df"] = query_stability_df

        cleaned_metrics = {
            k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
            for k, v in final_metrics.items()
        }
        self._calculate_stability_score(
            cleaned_metrics,
            query_stability_df if query_stability_metrics else pl.DataFrame(),
        )

        return EvaluationResult(
            metrics=cleaned_metrics, plots={}, extra_data=extra_data
        )

    def _calculate_stability_score(
        self, metrics: Dict[str, Any], details_df: pl.DataFrame
    ):
        if not self.eval_params or not self.eval_params.scoring_weights:
            logger.warning("Ranking stability scoring weights not configured.")
            return
        weights = self.eval_params.scoring_weights

        # Example: Score based on average stability of NDCG@10 and Jaccard@10
        # Lower std is better for metric stability. Higher Jaccard is better.
        # Assume k=10 is one of the self.k_values and metrics exist
        k_for_scoring = 10
        if (
            10 not in self.k_values and self.k_values
        ):  # Fallback to largest K if 10 not present
            k_for_scoring = self.k_values[-1]

        score_ndcg_stability = normalize_metric_to_score(
            metrics.get(f"rnk_stab_mean_ndcg_at_{k_for_scoring}_std"),
            good_threshold=0.05,
            bad_threshold=0.2,
            lower_is_better=True,  # Example thresholds for std dev
        )
        if score_ndcg_stability is None:
            score_ndcg_stability = 0.0

        score_recall_stability = normalize_metric_to_score(
            metrics.get(f"rnk_stab_mean_recall_at_{k_for_scoring}_std"),
            good_threshold=0.05,
            bad_threshold=0.2,
            lower_is_better=True,
        )
        if score_recall_stability is None:
            score_recall_stability = 0.0

        score_jaccard = normalize_metric_to_score(
            metrics.get(f"rnk_stab_mean_jaccard_at_{k_for_scoring}"),
            is_0_1_rate_lower_better=False,  # Jaccard is 0-1, higher is better
        )
        if score_jaccard is None:
            score_jaccard = 0.0

        s_stability_ranking = (
            score_ndcg_stability * weights.get("ndcg_stability", 0.0)
            + score_recall_stability * weights.get("recall_stability", 0.0)
            + score_jaccard * weights.get("top_k_set_jaccard", 0.0)
        )
        total_w = sum(
            weights.get(k, 0.0)
            for k in ["ndcg_stability", "recall_stability", "top_k_set_jaccard"]
        )
        if total_w > 1e-6 and abs(total_w - 1.0) > 1e-6:
            s_stability_ranking /= total_w

        metrics[f"rnk_stab_score_ndcg_at_{k_for_scoring}_stability"] = (
            score_ndcg_stability
        )
        metrics[f"rnk_stab_score_recall_at_{k_for_scoring}_stability"] = (
            score_recall_stability
        )
        metrics[f"rnk_stab_score_jaccard_at_{k_for_scoring}"] = score_jaccard
        metrics["rnk_stab_score_overall"] = max(0.0, min(100.0, s_stability_ranking))
        logger.info(
            f"Ranking Stability Score (Overall): {metrics['rnk_stab_score_overall']:.2f}"
        )


StabilityEvaluatorFactory.register_evaluator("ranking", RankingStabilityEvaluator)

if __name__ == "__main__":
    from pathlib import Path

    dummy_cfg_content = """
project_info: {project_name: "RankingStab Test", model_type: "ranking"}
data_loader:
  field_mapping:
    loop: loop_col
    query_id: query_col
    ranking:
      item_id_pred_list: preds
      item_id_gt_list: gts
      list_delimiter: ","
evaluation_params:
  ranking:
    k_values: [2, 3]
    scoring_weights: {ndcg_stability: 0.5, top_k_set_jaccard: 0.5} # Recall stability weight will be 0
report_settings: {output_dir: "./test_rank_stab_output"}
"""
    dummy_path = Path("dummy_rank_stab_config.yaml")
    with open(dummy_path, "w") as f:
        f.write(dummy_cfg_content)
    Path("./test_rank_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_path.read_text()))
    evaluator = RankingStabilityEvaluator(config)

    test_data = {
        "loop_col": [1, 1, 2, 2, 1, 1, 2, 2],
        "query_col": ["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2"],
        # For each (query,loop) pair, preds and gts are lists of item IDs
        # Row 0: q1, L1; Row 1: q1, L1 (duplicate, assume dataloader handles or evaluator sees one row per query-loop)
        # Let's assume one row per query-loop for simplicity of test data here.
        "preds": [  # item_id_pred_list
            "a,b,c,d,e",  # q1, L1
            "a,c,b,e,d",  # q1, L2
            "x,y,z,w,v",  # q2, L1
            "x,z,y,v,w",  # q2, L2
        ],
        "gts": [  # item_id_gt_list
            "a,c,f",  # q1 GT
            "a,c,f",  # q1 GT
            "x,w,u",  # q2 GT
            "x,w,u",  # q2 GT
        ],
    }
    # Create a df where each row is unique for (query_id, loop_id)
    unique_test_data = {
        "loop_col": [1, 2, 1, 2],
        "query_col": ["q1", "q1", "q2", "q2"],
        "internal_item_id_pred_list": [  # Already list<str> as DataLoader would produce
            ["a", "b", "c", "d", "e"],
            ["a", "c", "b", "e", "d"],
            ["x", "y", "z", "w", "v"],
            ["x", "z", "y", "v", "w"],
        ],
        "internal_item_id_gt_list": [
            ["a", "c", "f"],
            ["a", "c", "f"],
            ["x", "w", "u"],
            ["x", "w", "u"],
        ],
    }
    test_df = pl.DataFrame(unique_test_data)

    logger.info("--- Testing RankingStabilityEvaluator ---")
    results = evaluator.evaluate(test_df.clone())
    print(
        "\nRanking Stability Metrics:",
        {k: v for k, v in results.metrics.items() if v is not None},
    )

    assert "rnk_mean_ndcg_at_2" in results.metrics
    assert "rnk_stab_mean_ndcg_at_2_std" in results.metrics
    assert "rnk_stab_mean_jaccard_at_2" in results.metrics
    assert "rnk_stab_score_overall" in results.metrics
    assert results.metrics["rnk_stab_score_overall"] is not None

    if "ranking_stability_details_per_query_df" in results.extra_data:
        print(
            "\nDetails per query:",
            results.extra_data["ranking_stability_details_per_query_df"],
        )
        assert (
            results.extra_data["ranking_stability_details_per_query_df"].shape[0] == 2
        )  # 2 queries

    if dummy_path.exists():
        dummy_path.unlink()
    import shutil

    if Path("./test_rank_stab_output").exists():
        shutil.rmtree("./test_rank_stab_output")
