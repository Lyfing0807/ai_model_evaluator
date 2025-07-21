"""
Stability evaluator for Rotated Detection Models.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import yaml  # For testing
from scipy.optimize import linear_sum_assignment
from shapely.affinity import rotate as rotate_polygon
from shapely.affinity import translate as translate_polygon
from shapely.affinity import rotate as rotate_polygon, translate as translate_polygon
from shapely.geometry import Polygon  # Using Shapely for RIoU

from ...config_manager import MainConfig, RotatedDetectionEvaluationParams
from ...utils.logging_config import get_logger
from ...utils.scoring_utils import normalize_metric_to_score
from ...utils.types import EvaluationResult
from ..base import StabilityEvaluatorBase
from .factory import StabilityEvaluatorFactory

logger = get_logger(__name__)

# --- Geometry Helper Functions for RIoU using Shapely ---


def rbox_to_shapely_polygon(rbox: List[float]) -> Polygon:
    """Converts rbox [cx, cy, w, h, angle_degrees] to a Shapely Polygon."""
    # cx, cy, w, h, angle_degrees = rbox # Not needed if using shapely's rotate and translate

    # Create a rectangle centered at the origin
    # Shapely's box is (minx, miny, maxx, maxy)
    # For a box centered at origin:
    cx, cy, w, h, angle_degrees = rbox
    half_w, half_h = w / 2.0, h / 2.0

    # Corners relative to center (0,0) before rotation
    # (x,y) order: e.g. bottom-left, top-left, top-right, bottom-right for Polygon constructor
    # Or, more simply, use shapely.geometry.box and then affinity.rotate/translate

    # Create axis-aligned box at origin
    origin_box = Polygon(
        [(-half_w, -half_h), (-half_w, half_h), (half_w, half_h), (half_w, -half_h)]
    )

    # Rotate (around origin, as it's currently centered there)
    # Shapely's rotate takes angle in degrees, default origin is 'center' of polygon's bounding box
    rotated_box = rotate_polygon(origin_box, angle_degrees, origin="center")

    # Translate to the correct center
    # shapely.affinity.translate
    final_polygon = translate_polygon(rotated_box, xoff=cx, yoff=cy)
    return final_polygon


def calculate_riou_shapely(
    rbox1_params: List[float], rbox2_params: List[float]
) -> float:
    """Calculates Rotated IoU using Shapely library."""
    try:
        poly1 = rbox_to_shapely_polygon(rbox1_params)
        poly2 = rbox_to_shapely_polygon(rbox2_params)

        if not poly1.is_valid or not poly2.is_valid:
            logger.warning("Invalid polygon generated for RIoU calculation.")
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area

        if union_area == 0:
            return (
                0.0 if intersection_area == 0 else 1.0
            )  # Or handle as error if union is 0 but intersection isn't
        return intersection_area / union_area
    except Exception as e:
        logger.error(
            f"Error in calculate_riou_shapely: {e}. Rbox1: {rbox1_params}, Rbox2: {rbox2_params}",
            exc_info=True,
        )
        return 0.0


class RotatedDetectionStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        if not isinstance(self.eval_params, RotatedDetectionEvaluationParams):
            logger.error(
                "Rotated detection evaluation parameters not correctly configured."
            )
            raise ValueError(
                "Rotated detection evaluation parameters missing or incorrect type."
            )
        self.eval_params: RotatedDetectionEvaluationParams
        self.riou_threshold: float = self.eval_params.riou_threshold
        logger.info(
            f"RotatedDetectionStabilityEvaluator initialized with RIoU threshold: {self.riou_threshold} using Shapely."
        )

    def _track_objects_for_image(self, image_df: pl.DataFrame) -> pl.DataFrame:
        """Performs object tracking across loops for a single image using RIoU (Shapely)."""
        if (
            image_df.is_empty()
            or "loop" not in image_df.columns
            or "internal_rbbox" not in image_df.columns
        ):
            return image_df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("tracked_object_id")
            )

        image_df = image_df.sort("loop")
        image_df_with_orig_idx = image_df.with_row_count("_original_idx_temp")

        detections_by_loop: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in image_df_with_orig_idx.iter_rows(named=True):
            detections_by_loop[row["loop"]].append(dict(row))

        loop_ids = sorted(detections_by_loop.keys())
        if not loop_ids:
            return image_df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("tracked_object_id")
            )

        tracked_detections_list = []
        current_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 0

        for det_data in detections_by_loop[loop_ids[0]]:
            current_tracks[next_track_id] = det_data
            tracked_detections_list.append(
                {**det_data, "tracked_object_id": next_track_id}
            )
            next_track_id += 1

        for loop_idx in range(1, len(loop_ids)):
            detections_in_current_loop = detections_by_loop[loop_ids[loop_idx]]
            active_tracks_list = list(current_tracks.values())
            new_current_tracks_for_next_iter: Dict[int, Dict[str, Any]] = {}

            if not active_tracks_list or not detections_in_current_loop:
                for det_data in detections_in_current_loop:
                    new_id = next_track_id
                    new_current_tracks_for_next_iter[new_id] = det_data
                    tracked_detections_list.append(
                        {**det_data, "tracked_object_id": new_id}
                    )
                    next_track_id += 1
                current_tracks = new_current_tracks_for_next_iter
                continue

            cost_matrix = np.full(
                (len(active_tracks_list), len(detections_in_current_loop)), 1.0
            )
            for i, prev_det in enumerate(active_tracks_list):
                for j, curr_det in enumerate(detections_in_current_loop):
                    # Ensure internal_rbbox is not None before passing
                    if prev_det.get("internal_rbbox") and curr_det.get(
                        "internal_rbbox"
                    ):
                        riou = calculate_riou_shapely(
                            prev_det["internal_rbbox"], curr_det["internal_rbbox"]
                        )
                        if riou > 1e-6:
                            cost_matrix[i, j] = 1.0 - riou
                    else:  # Handle None rbbox case, assign max cost
                        cost_matrix[i, j] = 1.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_current_indices = set()
            for r, c in zip(row_ind, col_ind):
                if (1.0 - cost_matrix[r, c]) >= self.riou_threshold:
                    prev_det_matched = active_tracks_list[r]
                    curr_det_matched = detections_in_current_loop[c]
                    original_track_id = -1
                    for tid, track_data_val in current_tracks.items():
                        if (
                            track_data_val["_original_idx_temp"]
                            == prev_det_matched["_original_idx_temp"]
                        ):
                            original_track_id = tid
                            break
                    if original_track_id != -1:
                        tracked_detections_list.append(
                            {**curr_det_matched, "tracked_object_id": original_track_id}
                        )
                        new_current_tracks_for_next_iter[original_track_id] = (
                            curr_det_matched
                        )
                        matched_current_indices.add(c)

            for c_idx, curr_det_unmatched in enumerate(detections_in_current_loop):
                if c_idx not in matched_current_indices:
                    new_id = next_track_id
                    tracked_detections_list.append(
                        {**curr_det_unmatched, "tracked_object_id": new_id}
                    )
                    new_current_tracks_for_next_iter[new_id] = curr_det_unmatched
                    next_track_id += 1
            current_tracks = new_current_tracks_for_next_iter

        if not tracked_detections_list:
            return image_df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("tracked_object_id")
            )

        idx_to_track_id_map = {
            td["_original_idx_temp"]: td["tracked_object_id"]
            for td in tracked_detections_list
        }
        final_tracked_ids = [
            idx_to_track_id_map.get(row["_original_idx_temp"])
            for row in image_df_with_orig_idx.iter_rows(named=True)
        ]
        return image_df.with_columns(
            pl.Series("tracked_object_id", final_tracked_ids, dtype=pl.Int64)
        )

    def _calculate_average_rbox(self, group_df: pl.DataFrame) -> Optional[List[float]]:
        if group_df.is_empty() or "internal_rbbox" not in group_df.columns:
            return None
        rboxes_list = group_df["internal_rbbox"].drop_nulls().to_list()
        if not rboxes_list:
            return None
        rboxes_np = np.array(rboxes_list)
        avg_params = np.mean(rboxes_np, axis=0)
        avg_params[4] = avg_params[4] % 180.0
        return avg_params.tolist()

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info(
            "Starting rotated detection stability evaluation (using Shapely RIoU)..."
        )
        required_cols = ["image_id", "loop", "internal_rbbox", "category_id", "score"]
        if not all(col in data_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data_df.columns]
            return EvaluationResult(
                metrics={"error": f"Missing required columns for RDet: {missing}"}
            )

        if data_df.is_empty():
            return EvaluationResult(metrics={"warning": "Empty input for RDet eval."})
        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        try:
            tracked_df = data_df.group_by("image_id", maintain_order=True).apply(
                self._track_objects_for_image
            )
        except Exception as e:
            logger.error(f"Error during RDet object tracking: {e}", exc_info=True)
            return EvaluationResult(metrics={"error": "RDet object tracking failed."})

        tracked_df = tracked_df.filter(pl.col("tracked_object_id").is_not_null())
        if tracked_df.is_empty():
            return EvaluationResult(metrics={"warning": "No RDet objects tracked."})

        object_metrics_list = []
        total_loops_img = data_df.group_by("image_id").agg(
            pl.col("loop").n_unique().alias("total_loops")
        )

        for (img_id, trk_id), grp in tracked_df.group_by(
            ["image_id", "tracked_object_id"], maintain_order=False
        ):
            if grp.is_empty():
                continue
            n_loops = grp["loop"].n_unique()
            img_total_loops_series = total_loops_img.filter(
                pl.col("image_id") == img_id
            )["total_loops"]
            if img_total_loops_series.is_empty():
                continue  # Should not happen if img_id exists
            img_total_loops = img_total_loops_series[0]
            app_cons = n_loops / img_total_loops if img_total_loops > 0 else 0.0

            avg_rbox = self._calculate_average_rbox(grp)
            riou_cons, ang_devs = [], []
            if avg_rbox:
                for rbox_list in grp["internal_rbbox"].drop_nulls().to_list():
                    if rbox_list:  # Ensure list is not None
                        riou_cons.append(calculate_riou_shapely(rbox_list, avg_rbox))
                        angle_diff = abs(rbox_list[4] - avg_rbox[4])
                        ang_devs.append(min(angle_diff, 180 - angle_diff))

            centers_x = grp.select(pl.col("internal_rbbox").list.get(0).alias("cx"))[
                "cx"
            ].drop_nulls()
            centers_y = grp.select(pl.col("internal_rbbox").list.get(1).alias("cy"))[
                "cy"
            ].drop_nulls()
            widths = grp.select(pl.col("internal_rbbox").list.get(2).alias("w"))[
                "w"
            ].drop_nulls()
            heights = grp.select(pl.col("internal_rbbox").list.get(3).alias("h"))[
                "h"
            ].drop_nulls()

            mean_cx, mean_cy = centers_x.mean() if centers_x.len() > 0 else None, (
                centers_y.mean() if centers_y.len() > 0 else None
            )
            center_drifts_px = None
            if mean_cx is not None and mean_cy is not None and centers_x.len() > 0:
                center_drifts = (
                    (centers_x - mean_cx) ** 2 + (centers_y - mean_cy) ** 2
                ).sqrt()
                center_drifts_px = center_drifts.mean()

            areas = widths * heights
            mean_area = areas.mean() if areas.len() > 0 else None
            size_jitters_ratio = None
            if mean_area is not None and mean_area > 1e-9 and areas.len() > 0:
                size_jitters_area = (areas - mean_area).abs() / mean_area
                size_jitters_ratio = size_jitters_area.mean()

            object_metrics_list.append(
                {
                    "image_id": img_id,
                    "tracked_object_id": trk_id,
                    "appearance_consistency": app_cons,
                    "mean_riou_consistency": np.mean(riou_cons) if riou_cons else None,
                    "angle_std_dev": np.std(ang_devs) if ang_devs else None,
                    "mean_center_drift_px": center_drifts_px,
                    "mean_size_jitter_ratio": size_jitters_ratio,
                    "confidence_std": grp["score"].std(),
                    "category_switch_rate": (
                        (grp["category_id"].n_unique() - 1) / (n_loops - 1)
                        if n_loops > 1 and grp["category_id"].n_unique() > 1
                        else 0.0
                    ),
                    "num_loops_appeared": n_loops,
                }
            )

        if not object_metrics_list:
            return EvaluationResult(
                metrics={"warning": "No RDet object metrics calculated."}
            )
        stats_df = pl.DataFrame(object_metrics_list)

        final_metrics = {
            f"rdt_stab_mean_{col.replace('_placeholder','')}": stats_df[col].mean()
            for col in [
                "appearance_consistency",
                "mean_riou_consistency",
                "angle_std_dev",
                "mean_center_drift_px",
                "mean_size_jitter_ratio",
                "confidence_std",
                "category_switch_rate",
            ]
            if col in stats_df.columns and stats_df[col].drop_nulls().len() > 0
        }
        final_metrics["rdt_stab_num_total_tracked"] = float(len(stats_df))
        final_metrics["rdt_stab_num_tracked_multi_loop"] = float(
            stats_df.filter(pl.col("num_loops_appeared") > 1).shape[0]
        )

        cleaned_metrics = {
            k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
            for k, v in final_metrics.items()
        }

        self._calculate_stability_score(cleaned_metrics, stats_df)

        return EvaluationResult(
            metrics=cleaned_metrics, extra_data={"rdet_stability_details_df": stats_df}
        )

    def _calculate_stability_score(
        self, metrics: Dict[str, Any], details_df: pl.DataFrame
    ):
        """Calculates overall rotated detection stability score."""
        if not self.eval_params or not self.eval_params.scoring_weights:
            logger.warning(
                "Rotated detection stability scoring weights not configured. Skipping score calculation."
            )
            return

        weights = self.eval_params.scoring_weights

        # RIoU Consistency Score (higher is better, 0-1 range)
        score_riou_consistency = normalize_metric_to_score(
            metrics.get("rdt_stab_mean_riou_consistency"),
            is_0_1_rate_lower_better=False,
        )
        if score_riou_consistency is None:
            score_riou_consistency = 0.0

        # Angle Stability Score (based on mean angle_std_dev, lower is better)
        # Example: good_angle_std = 5 degrees, bad_angle_std = 30 degrees
        score_angle_stability = normalize_metric_to_score(
            metrics.get("rdt_stab_mean_angle_std_dev"),
            good_threshold=5.0,  # degrees
            bad_threshold=30.0,  # degrees
            lower_is_better=True,
        )
        if score_angle_stability is None:
            score_angle_stability = 0.0

        # Existence Stability Score (from appearance consistency, higher is better, 0-1)
        score_existence = normalize_metric_to_score(
            metrics.get("rdt_stab_mean_appearance_consistency"),
            is_0_1_rate_lower_better=False,
        )
        if score_existence is None:
            score_existence = 0.0

        # Other scores (position, confidence, category) can be adapted from standard detection if needed
        # For now, focus on RIoU, angle, and existence as per default weights in config.

        s_stability_rdet = (
            score_riou_consistency * weights.get("riou_consistency", 0.0)
            + score_angle_stability * weights.get("angle_stability", 0.0)
            + score_existence * weights.get("existence_stability", 0.0)
            # Add other components if their scores are calculated and weighted
        )

        current_weights_sum = sum(
            weights.get(k, 0.0)
            for k in ["riou_consistency", "angle_stability", "existence_stability"]
        )
        if current_weights_sum > 1e-6 and abs(current_weights_sum - 1.0) > 1e-6:
            # This normalization assumes all defined weights in the config should sum to 1.
            # If only a subset is used for scoring, this might not be desired.
            # For now, we normalize based on the sum of weights *used* in the calculation above.
            logger.warning(
                f"Rotated detection stability weights used ({ {k:weights.get(k) for k in ['riou_consistency', 'angle_stability', 'existence_stability']} }) do not sum to 1. Normalizing score based on used weights."
            )
            s_stability_rdet = (
                s_stability_rdet / current_weights_sum
                if current_weights_sum > 1e-9
                else 0.0
            )

        s_stability_rdet = max(0.0, min(100.0, s_stability_rdet))

        metrics["rdt_stab_score_riou_consistency"] = score_riou_consistency
        metrics["rdt_stab_score_angle_stability"] = score_angle_stability
        metrics["rdt_stab_score_existence_stability"] = score_existence
        metrics["rdt_stab_score_overall"] = s_stability_rdet
        logger.info(
            f"Rotated Detection Stability Scores: RIoUCons={score_riou_consistency:.2f}, AngleStab={score_angle_stability:.2f}, Existence={score_existence:.2f}, Overall={s_stability_rdet:.2f}"
        )


if __name__ == "__main__":
    StabilityEvaluatorFactory.register_evaluator(
        "rotated_detection", RotatedDetectionStabilityEvaluator
    )

    from pathlib import Path

    dummy_config_content = """
project_info: {project_name: "RDetStab Test Shapely", model_type: "rotated_detection"}
data_loader:
  field_mapping: {loop: lp, image_id: id, image_path: pth, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4,
                  rotated_detection: {category_id: cat, score: scr, rbbox: [cx,cy,w,h,a]}}
evaluation_params:
  rotated_detection: {riou_threshold: 0.1}
report_settings: {output_dir: "./test_rdet_stab_shapely_output"}
"""
    dummy_cfg_path = Path("dummy_rdet_stab_shapely_config.yaml")
    with open(dummy_cfg_path, "w") as f:
        f.write(dummy_config_content)
    Path("./test_rdet_stab_shapely_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_cfg_path.read_text()))
    evaluator = RotatedDetectionStabilityEvaluator(config)

    test_df = pl.DataFrame(
        {
            "id": ["img1"] * 2 + ["img2"] * 2,
            "lp": [1, 2, 1, 2],
            "internal_rbbox": [  # cx,cy,w,h,angle
                [50, 50, 20, 10, 0],
                [50, 50, 20, 10, 5],  # img1, objA, slightly rotated
                [100, 100, 30, 15, 45],
                [100, 100, 30, 15, 45],  # img2, objB, identical
            ],
            "cat": [0, 0, 1, 1],
            "scr": [0.9, 0.88, 0.95, 0.96],
        }
    )
    logger.info("--- Testing RotatedDetectionStabilityEvaluator with Shapely RIoU ---")
    results = evaluator.evaluate(test_df.clone())
    print("\nRotated Detection Metrics (Shapely):", results.metrics)
    if "rdet_stability_details_df" in results.extra_data:
        print("\nDetails:", results.extra_data["rdet_stability_details_df"])
        assert results.extra_data["rdet_stability_details_df"].shape[0] == 2
    assert results.metrics.get("rdt_stab_num_total_tracked") == 2.0
    assert results.metrics.get("rdt_stab_mean_riou_consistency") is not None
    assert results.metrics.get("rdt_stab_mean_riou_consistency") > 0.5
    assert "rdt_stab_score_overall" in results.metrics
    assert (
        results.metrics["rdt_stab_score_overall"] >= 0
        and results.metrics["rdt_stab_score_overall"] <= 100
    )

    if dummy_cfg_path.exists():
        dummy_cfg_path.unlink()
    import shutil

    if Path("./test_rdet_stab_shapely_output").exists():
        shutil.rmtree("./test_rdet_stab_shapely_output")
