"""
Stability evaluator for Detection Models.
"""
import polars as pl
import numpy as np
import yaml # For loading config in __main__
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

from ...config_manager import MainConfig
from ..base import StabilityEvaluatorBase
from ...utils.types import EvaluationResult
from ...utils.logging_config import get_logger
from .factory import StabilityEvaluatorFactory


logger = get_logger(__name__)

# Helper function for IoU calculation
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) for two bounding boxes.
    Boxes are expected in [x_min, y_min, x_max, y_max] format.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

class DetectionStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        # Type guard for eval_params
        if not self.eval_params or not hasattr(self.eval_params, 'iou_threshold'):
            logger.error("Detection evaluation parameters (e.g., iou_threshold) not correctly configured.")
            raise ValueError("Detection evaluation parameters missing or of incorrect type.")
        self.iou_threshold: float = self.eval_params.iou_threshold # type: ignore
        logger.info(f"DetectionStabilityEvaluator initialized with IoU threshold: {self.iou_threshold}.")

    def _track_objects_for_image(self, image_df: pl.DataFrame) -> pl.DataFrame:
        """
        Performs object tracking across loops for a single image.
        Adds a 'tracked_object_id' column to the DataFrame.
        """
        if image_df.is_empty() or "loop" not in image_df.columns or "internal_bbox" not in image_df.columns:
            logger.warning(f"Skipping tracking for image due to missing columns or empty data. Image ID: {image_df['image_id'][0] if not image_df.is_empty() else 'Unknown'}")
            return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        # Sort by loop to process sequentially
        image_df = image_df.sort("loop")

        tracked_detections_list = []

        detections_by_loop: Dict[Any, List[Dict[str, Any]]] = defaultdict(list) # Key is loop_id
        # Store original indices by adding a temporary unique ID column if not present
        # Polars iter_rows(named=True) does not give index, so add one
        image_df_with_orig_idx = image_df.with_row_count("_original_idx_temp")

        for row in image_df_with_orig_idx.iter_rows(named=True):
            detections_by_loop[row["loop"]].append(dict(row)) # Convert to dict

        loop_ids = sorted(detections_by_loop.keys())
        if not loop_ids:
            return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        current_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 0

        first_loop_id = loop_ids[0]
        for det_data in detections_by_loop[first_loop_id]:
            current_tracks[next_track_id] = det_data
            tracked_detections_list.append({**det_data, "tracked_object_id": next_track_id})
            next_track_id += 1

        for loop_idx in range(1, len(loop_ids)):
            current_loop_id = loop_ids[loop_idx]

            detections_in_current_loop = detections_by_loop[current_loop_id]
            active_tracks_from_prev_loop_list = list(current_tracks.values())

            new_current_tracks_for_next_iter: Dict[int, Dict[str, Any]] = {}

            if not active_tracks_from_prev_loop_list or not detections_in_current_loop:
                for det_data in detections_in_current_loop: # All are new tracks
                    new_track_id_for_det = next_track_id
                    new_current_tracks_for_next_iter[new_track_id_for_det] = det_data
                    tracked_detections_list.append({**det_data, "tracked_object_id": new_track_id_for_det})
                    next_track_id += 1
                current_tracks = new_current_tracks_for_next_iter
                continue

            num_prev_dets = len(active_tracks_from_prev_loop_list)
            num_curr_dets = len(detections_in_current_loop)
            cost_matrix = np.full((num_prev_dets, num_curr_dets), 1.0)

            for i, prev_det_data in enumerate(active_tracks_from_prev_loop_list):
                for j, curr_det_data in enumerate(detections_in_current_loop):
                    iou = calculate_iou(prev_det_data["internal_bbox"], curr_det_data["internal_bbox"])
                    if iou > 0:
                        cost_matrix[i, j] = 1.0 - iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_current_det_indices = set()

            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                iou = 1.0 - cost
                if iou >= self.iou_threshold:
                    # Matched: continue track
                    prev_det_data_matched = active_tracks_from_prev_loop_list[r]
                    curr_det_data_matched = detections_in_current_loop[c]
                    # The track_id comes from the prev_det_data_matched's *original* track_id
                    # This requires current_tracks keys to be the actual track_ids

                    # Find the track_id for prev_det_data_matched
                    # This is inefficient; current_tracks should map track_id -> det_data
                    original_track_id = -1
                    for tid, track_data_val in current_tracks.items(): # current_tracks holds data from previous loop
                        if track_data_val["_original_idx_temp"] == prev_det_data_matched["_original_idx_temp"]:
                            original_track_id = tid
                            break
                    if original_track_id == -1: # Should not happen if current_tracks is built correctly
                        logger.error("Could not find original track ID for a matched previous detection.")
                        continue

                    tracked_detections_list.append({**curr_det_data_matched, "tracked_object_id": original_track_id})
                    new_current_tracks_for_next_iter[original_track_id] = curr_det_data_matched
                    matched_current_det_indices.add(c)

            for c_idx, curr_det_data_unmatched in enumerate(detections_in_current_loop):
                if c_idx not in matched_current_det_indices: # New track
                    new_id = next_track_id
                    tracked_detections_list.append({**curr_det_data_unmatched, "tracked_object_id": new_id})
                    new_current_tracks_for_next_iter[new_id] = curr_det_data_unmatched
                    next_track_id += 1

            current_tracks = new_current_tracks_for_next_iter

        if not tracked_detections_list:
             return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        # Create a DF from tracked_detections_list and merge back to original image_df
        # This ensures all original rows get a tracked_object_id (even if None for some)
        # The `_original_idx_temp` is the key.

        # Create a mapping from _original_idx_temp to tracked_object_id
        idx_to_track_id_map = {
            td["_original_idx_temp"]: td["tracked_object_id"] for td in tracked_detections_list
        }

        # Apply this map to the original image_df_with_orig_idx
        final_tracked_ids = [idx_to_track_id_map.get(row["_original_idx_temp"]) for row in image_df_with_orig_idx.iter_rows(named=True)]

        return image_df.with_columns(pl.Series("tracked_object_id", final_tracked_ids, dtype=pl.Int64))


    def _calculate_average_box(self, group_df: pl.DataFrame) -> Optional[List[float]]:
        """Calculates the average box [x_min, y_min, x_max, y_max] for a tracked object."""
        if group_df.is_empty() or "internal_bbox" not in group_df.columns:
            return None

        bboxes_list = group_df["internal_bbox"].drop_nulls().to_list()
        if not bboxes_list: return None # Handle if all bboxes were null for this group

        bboxes = np.array(bboxes_list)

        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        avg_cx, avg_cy, avg_w, avg_h = np.mean(cx), np.mean(cy), np.mean(w), np.mean(h)

        avg_x_min = avg_cx - avg_w / 2.0
        avg_y_min = avg_cy - avg_h / 2.0
        avg_x_max = avg_cx + avg_w / 2.0
        avg_y_max = avg_cy + avg_h / 2.0

        return [avg_x_min, avg_y_min, avg_x_max, avg_y_max]

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info(f"Starting detection stability evaluation for model type {self.model_type}...")

        required_cols = ["image_id", "loop", "internal_bbox", "category_id", "score"]
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            logger.error(f"Detection stability evaluation requires columns: {required_cols}. Missing: {missing_cols}")
            return EvaluationResult(metrics={"error": f"Missing columns for detection stability: {missing_cols}"})

        if data_df.is_empty():
            logger.warning("Input DataFrame is empty for detection stability evaluation.")
            return EvaluationResult(metrics={"warning": "Empty input data for detection stability."})

        logger.info("Performing object tracking across loops for each image...")

        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        # Using group_by().apply() can be slow with Python UDFs on large number of groups.
        # If performance becomes an issue, this part might need optimization (e.g. custom Rust plugin or more vectorized Polars ops if feasible).
        # For now, clarity of Python UDF for tracking logic is prioritized.
        try:
            # Polars apply function needs to return a DataFrame.
            # The self._track_objects_for_image already returns a DataFrame.
            tracked_df = data_df.group_by("image_id", maintain_order=True).apply(self._track_objects_for_image)
        except Exception as e:
            logger.error(f"Error during object tracking: {e}", exc_info=True)
            return EvaluationResult(metrics={"error": "Object tracking failed."})

        logger.info("Object tracking completed.")

        tracked_df = tracked_df.filter(pl.col("tracked_object_id").is_not_null())
        if tracked_df.is_empty():
            logger.warning("No objects successfully tracked across loops.")
            return EvaluationResult(metrics={"warning": "No objects were successfully tracked."})

        object_stability_metrics = []
        # total_loops_per_image needed for appearance_consistency
        # This should be calculated from the original data_df before any filtering by tracking
        total_loops_per_image = data_df.group_by("image_id").agg(pl.col("loop").n_unique().alias("total_loops_for_image"))

        for (image_id, tracked_id), group in tracked_df.group_by(["image_id", "tracked_object_id"], maintain_order=False):
            if group.is_empty(): continue

            num_loops_appeared = group["loop"].n_unique()

            current_image_info = total_loops_per_image.filter(pl.col("image_id") == image_id)
            if current_image_info.is_empty():
                logger.warning(f"Could not find total_loops_for_image for image_id {image_id}. Skipping object {tracked_id}.")
                continue
            total_img_loops = current_image_info["total_loops_for_image"][0]

            appearance_consistency = num_loops_appeared / total_img_loops if total_img_loops > 0 else 0.0

            avg_box = self._calculate_average_box(group)
            iou_consistencies = []
            center_drifts = []
            size_jitters_area = []

            if avg_box:
                avg_center = np.array([(avg_box[0] + avg_box[2]) / 2, (avg_box[1] + avg_box[3]) / 2])
                avg_area = (avg_box[2] - avg_box[0]) * (avg_box[3] - avg_box[1])

                for det_box_list in group["internal_bbox"].drop_nulls().to_list(): # Ensure no null bboxes
                    det_box = np.array(det_box_list)
                    iou_consistencies.append(calculate_iou(det_box.tolist(), avg_box))

                    det_center = np.array([(det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2])
                    center_drifts.append(np.linalg.norm(det_center - avg_center))

                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    if avg_area > 1e-9:
                        size_jitters_area.append(abs(det_area - avg_area) / avg_area)

            mean_iou_consistency = np.mean(iou_consistencies) if iou_consistencies else None
            mean_center_drift_px = np.mean(center_drifts) if center_drifts else None
            mean_size_jitter_ratio = np.mean(size_jitters_area) if size_jitters_area else None

            confidence_std = group["score"].std() # Polars std handles nulls by ignoring them

            num_category_switches = 0
            if num_loops_appeared > 1:
                sorted_by_loop = group.sort("loop")
                categories = sorted_by_loop["category_id"]
                for i in range(len(categories) - 1):
                    if categories[i] != categories[i+1] and categories[i] is not None and categories[i+1] is not None: # Handle potential nulls
                        num_category_switches += 1
            category_switch_rate = num_category_switches / (num_loops_appeared - 1) if num_loops_appeared > 1 else 0.0

            object_stability_metrics.append({
                "image_id": image_id, "tracked_object_id": tracked_id,
                "appearance_consistency": appearance_consistency,
                "mean_iou_consistency": mean_iou_consistency,
                "mean_center_drift_px": mean_center_drift_px,
                "mean_size_jitter_ratio": mean_size_jitter_ratio,
                "confidence_std": confidence_std,
                "category_switch_rate": category_switch_rate,
                "num_loops_appeared": num_loops_appeared,
            })

        if not object_stability_metrics:
            logger.warning("No per-object stability metrics could be calculated.")
            return EvaluationResult(metrics={"warning": "No per-object stability metrics calculated."})

        stability_stats_df = pl.DataFrame(object_stability_metrics)

        final_metrics = {
            f"det_stab_mean_{col}": stability_stats_df[col].mean()
            for col in ["appearance_consistency", "mean_iou_consistency", "mean_center_drift_px",
                        "mean_size_jitter_ratio", "confidence_std", "category_switch_rate"]
            if stability_stats_df[col].null_count() < len(stability_stats_df) # Ensure there's some non-null data to mean
        }
        final_metrics["det_stab_num_total_tracked_instances"] = float(len(stability_stats_df))
        final_metrics["det_stab_num_unique_objects_tracked_multiple_loops"] = float(stability_stats_df.filter(pl.col("num_loops_appeared") > 1)["tracked_object_id"].n_unique())


        final_metrics_cleaned = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in final_metrics.items()}

        self._calculate_stability_score(cleaned_metrics, stats_df)

        logger.info("Detection stability evaluation completed.")
        return EvaluationResult(
            metrics=cleaned_metrics,
            plots={},
            extra_data={"object_stability_details_df": stats_df, "full_tracked_df_debug": tracked_df} # Keep full_tracked_df for now
        )

    def _calculate_stability_score(self, metrics: Dict[str, Any], details_df: pl.DataFrame):
        """Calculates overall detection stability score."""
        if not self.eval_params or not self.eval_params.scoring_weights: # type: ignore
            logger.warning("Detection stability scoring weights not configured. Skipping score calculation.")
            return

        weights = self.eval_params.scoring_weights # type: ignore

        # Sub-score for Existence Stability (based on mean appearance consistency)
        # Higher is better, already 0-1 range.
        score_existence = normalize_metric_to_score(
            metrics.get("det_stab_mean_appearance_consistency"),
            is_0_1_rate_lower_better=False
        )
        if score_existence is None: score_existence = 0.0

        # Sub-score for Position Stability (e.g. combination of IoU, drift, size jitter)
        # For simplicity, let's primarily use mean_iou_consistency. Higher is better (0-1).
        # And mean_center_drift_px (lower is better, need a target or good/bad thresholds)
        # For now, let's just use IoU consistency for position score.
        score_iou_consistency = normalize_metric_to_score(
            metrics.get("det_stab_mean_iou_consistency"),
            is_0_1_rate_lower_better=False
        )
        if score_iou_consistency is None: score_iou_consistency = 0.0
        # In a more complex model, you might normalize drift and jitter and combine them.
        # For now, we'll make position_stability primarily based on IoU consistency.
        score_position = score_iou_consistency


        # Sub-score for Confidence Stability (based on mean confidence_std). Lower is better.
        # Need to define what's a "good" or "bad" std dev for confidence.
        # Example: good_conf_std = 0.05, bad_conf_std = 0.2
        score_confidence = normalize_metric_to_score(
            metrics.get("det_stab_mean_confidence_std"),
            good_threshold=0.05, # Example good value
            bad_threshold=0.2,   # Example bad value
            lower_is_better=True
        )
        if score_confidence is None: score_confidence = 0.0

        # Sub-score for Category Stability (based on mean category_switch_rate). Lower is better (0-1).
        score_category = normalize_metric_to_score(
            metrics.get("det_stab_mean_category_switch_rate"),
            is_0_1_rate_lower_better=True # (1 - rate) * 100
        )
        if score_category is None: score_category = 0.0

        s_stability_detection = (
            score_existence * weights.get("existence_stability", 0.0) +
            score_position * weights.get("position_stability", 0.0) +
            score_confidence * weights.get("confidence_stability", 0.0) +
            score_category * weights.get("category_stability", 0.0)
        )

        total_weight = sum(weights.get(k,0.0) for k in ["existence_stability", "position_stability", "confidence_stability", "category_stability"])
        if total_weight > 1e-6 and abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Detection stability weights ({weights}) do not sum to 1. Normalizing score.")
            s_stability_detection = s_stability_detection / total_weight
        s_stability_detection = max(0.0, min(100.0, s_stability_detection))

        metrics["det_stab_score_existence"] = score_existence
        metrics["det_stab_score_position"] = score_position # Primarily IoU based for now
        metrics["det_stab_score_confidence"] = score_confidence
        metrics["det_stab_score_category"] = score_category
        metrics["det_stab_score_overall"] = s_stability_detection # This is S_stability for detection
        logger.info(f"Detection Stability Scores: Existence={score_existence:.2f}, Position={score_position:.2f}, Confidence={score_confidence:.2f}, Category={score_category:.2f}, Overall={s_stability_detection:.2f}")


StabilityEvaluatorFactory.register_evaluator("detection", DetectionStabilityEvaluator)


if __name__ == "__main__":
    from pathlib import Path

    dummy_config_content = """
project_info: {project_name: "DetStab Test", model_type: "detection"}
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    image_path: "image_path"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: {category_id: "category_id", score: "score", bbox: ["x_min", "y_min", "x_max", "y_max"]}
evaluation_params:
  detection: {iou_threshold: 0.3, bbox_format: "xyxy"}
report_settings: {output_dir: "./test_det_stab_output"} # Ensure output_dir is a valid path for Pydantic
"""
    dummy_config_path = Path("dummy_det_stab_config.yaml")
    with open(dummy_config_path, "w") as f: f.write(dummy_config_content)

    # Create output dir for Pydantic validation if it doesn't exist
    Path("./test_det_stab_output").mkdir(exist_ok=True)

    config = MainConfig.model_validate(yaml.safe_load(dummy_config_path.read_text()))

    evaluator = DetectionStabilityEvaluator(config)

    test_data_df = pl.DataFrame({
        "image_id": ["img1", "img1", "img1", "img1", "img2", "img2", "img3", "img3"],
        "loop":     [1,      1,      2,      2,      1,      2,      1,      1], # img3 has only one loop
        "internal_bbox": [ # Format [x_min, y_min, x_max, y_max]
            [10, 10, 20, 20], [50, 50, 60, 60], [12, 12, 22, 22], [80, 80, 90, 90],
            [30, 30, 40, 40], [33, 33, 43, 43], [5,5,10,10], [15,15,20,20]
        ],
        "category_id": [0, 1, 0, 0, 2, 2, 0, 1],
        "score":       [0.9, 0.8, 0.85, 0.7, 0.95, 0.92, 0.99, 0.88],
    })

    logger.info("--- Testing DetectionStabilityEvaluator ---")
    results = evaluator.evaluate(test_data_df.clone())

    print("\nDetection Stability Metrics:")
    for k, v in results.metrics.items():
        print(f"  {k}: {v}")

    if "object_stability_details_df" in results.extra_data:
        print("\nObject Stability Details:")
        print(results.extra_data["object_stability_details_df"])
        # Expected unique (image_id, tracked_object_id) pairs:
        # img1: objA (track0), objB (track1), objC (track2) -> 3
        # img2: objD (track3) -> 1
        # img3: objE (track4), objF (track5) -> 2 (objects in single loop still get tracked_ids)
        # Total = 6
        assert results.extra_data["object_stability_details_df"].shape[0] == 6
        assert results.metrics.get("det_stab_num_total_tracked_instances") == 6.0
        assert results.metrics.get("det_stab_num_unique_objects_tracked_multiple_loops") == 2.0
    assert "det_stab_score_overall" in results.metrics # Check if score was calculated
    assert results.metrics["det_stab_score_overall"] >= 0 and results.metrics["det_stab_score_overall"] <= 100


    logger.info("DetectionStabilityEvaluator test completed.")
    if dummy_config_path.exists():
        dummy_config_path.unlink()
    import shutil
    if Path("./test_det_stab_output").exists():
        shutil.rmtree("./test_det_stab_output")
```
