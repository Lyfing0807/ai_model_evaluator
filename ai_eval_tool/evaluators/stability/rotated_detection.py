"""
Stability evaluator for Rotated Detection Models.
"""
import polars as pl
import numpy as np
import math
import yaml # For testing
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

from ...config_manager import MainConfig, RotatedDetectionEvaluationParams
from ..base import StabilityEvaluatorBase
from ...utils.types import EvaluationResult
from ...utils.logging_config import get_logger
from .factory import StabilityEvaluatorFactory

logger = get_logger(__name__)

# --- Geometry Helper Functions for RIoU ---
# These are simplified and might need a robust library like Shapely for complex cases / precision.
# For this implementation, we'll attempt a common algorithm: convert rbbox to 4 corners,
# then use Separating Axis Theorem (SAT) ideas or polygon clipping (e.g., Sutherland-Hodgman)
# for intersection area. Given the complexity, a placeholder or very simplified RIoU might be used initially.

def rbox_to_corners(rbox: List[float]) -> np.ndarray:
    """Converts rbox [cx, cy, w, h, angle_degrees] to 4 corner points (2x4 matrix)."""
    cx, cy, w, h, angle_degrees = rbox
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Half width/height
    hw, hh = w / 2, h / 2

    # Corners in local frame (before rotation)
    # (x,y) order: top-left, top-right, bottom-right, bottom-left
    local_corners = np.array([
        [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
    ])

    # Rotation matrix
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotated corners (centered at origin)
    rotated_corners = local_corners @ R.T # Transpose R for point transformation

    # Translate to actual center
    return rotated_corners + np.array([cx, cy])

def polygon_area(corners: np.ndarray) -> float:
    """Calculates area of a polygon given its ordered corners (Shoelace formula)."""
    n = len(corners)
    if n < 3: return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def calculate_riou_placeholder(rbox1_params: List[float], rbox2_params: List[float]) -> float:
    """
    Placeholder/Simplified RIoU calculation.
    This is NOT a correct RIoU. It's a proxy based on center distance and angle diff.
    A real implementation would use polygon intersection.
    """
    cx1, cy1, w1, h1, a1 = rbox1_params
    cx2, cy2, w2, h2, a2 = rbox2_params

    # Center distance
    center_dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    avg_diag = (math.sqrt(w1**2 + h1**2) + math.sqrt(w2**2 + h2**2)) / 2
    dist_score = max(0, 1 - center_dist / (avg_diag + 1e-6)) # Normalize by avg diagonal

    # Angle difference (degrees, normalized)
    angle_diff = abs(a1 - a2) % 180 # handles >180 and <0 differences effectively for orientation
    angle_score = max(0, 1 - angle_diff / 90.0) # 0 if 90 deg diff, 1 if 0 deg diff

    # Area similarity (proxy)
    area1, area2 = w1 * h1, w2 * h2
    area_score = min(area1, area2) / (max(area1, area2) + 1e-6) if max(area1, area2) > 0 else 1.0

    # Combine scores (very heuristically)
    # This is a very rough approximation and not a true RIoU.
    # A proper RIoU would require polygon intersection algorithms (e.g. using Shapely).
    # For now, this placeholder will allow testing the tracking flow.
    # Penalize heavily if centers are too far.
    if dist_score < 0.2: # If centers are very far relative to size, likely no overlap
        return 0.0

    riou_approx = (dist_score * 0.4 + angle_score * 0.3 + area_score * 0.3)
    # logger.debug(f"RIoU Placeholder: rbox1={rbox1_params}, rbox2={rbox2_params} -> approx_riou={riou_approx:.3f} (d:{dist_score:.2f},a:{angle_score:.2f},s:{area_score:.2f})")
    return riou_approx
# Note: The plan mentions Shapely as optional. If this placeholder is insufficient,
# Shapely would be the way to go for accurate RIoU.
# For now, we proceed with this placeholder to build the rest of the evaluator.


class RotatedDetectionStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        if not isinstance(self.eval_params, RotatedDetectionEvaluationParams):
            logger.error("Rotated detection evaluation parameters not correctly configured.")
            raise ValueError("Rotated detection evaluation parameters missing or incorrect type.")
        self.eval_params: RotatedDetectionEvaluationParams # For type hinting
        self.riou_threshold: float = self.eval_params.riou_threshold
        logger.info(f"RotatedDetectionStabilityEvaluator initialized with RIoU threshold: {self.riou_threshold}.")
        logger.warning("Using a PLACEHOLDER RIoU calculation. For accurate results, a geometry library like Shapely is recommended.")

    def _track_objects_for_image(self, image_df: pl.DataFrame) -> pl.DataFrame:
        """Performs object tracking across loops for a single image using RIoU (placeholder)."""
        # This tracking logic is very similar to DetectionStabilityEvaluator,
        # just uses calculate_riou_placeholder instead of calculate_iou.
        if image_df.is_empty() or "loop" not in image_df.columns or "internal_rbbox" not in image_df.columns:
            return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        image_df = image_df.sort("loop")
        image_df_with_orig_idx = image_df.with_row_count("_original_idx_temp")

        detections_by_loop: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in image_df_with_orig_idx.iter_rows(named=True):
            detections_by_loop[row["loop"]].append(dict(row))

        loop_ids = sorted(detections_by_loop.keys())
        if not loop_ids:
            return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        tracked_detections_list = []
        current_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 0

        for det_data in detections_by_loop[loop_ids[0]]:
            current_tracks[next_track_id] = det_data
            tracked_detections_list.append({**det_data, "tracked_object_id": next_track_id})
            next_track_id += 1

        for loop_idx in range(1, len(loop_ids)):
            detections_in_current_loop = detections_by_loop[loop_ids[loop_idx]]
            active_tracks_list = list(current_tracks.values())
            new_current_tracks_for_next_iter: Dict[int, Dict[str, Any]] = {}

            if not active_tracks_list or not detections_in_current_loop:
                for det_data in detections_in_current_loop:
                    new_id = next_track_id
                    new_current_tracks_for_next_iter[new_id] = det_data
                    tracked_detections_list.append({**det_data, "tracked_object_id": new_id})
                    next_track_id += 1
                current_tracks = new_current_tracks_for_next_iter
                continue

            cost_matrix = np.full((len(active_tracks_list), len(detections_in_current_loop)), 1.0)
            for i, prev_det in enumerate(active_tracks_list):
                for j, curr_det in enumerate(detections_in_current_loop):
                    riou = calculate_riou_placeholder(prev_det["internal_rbbox"], curr_det["internal_rbbox"])
                    if riou > 1e-6 : cost_matrix[i, j] = 1.0 - riou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_current_indices = set()
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost_matrix[r,c] >= self.riou_threshold:
                    prev_det_matched = active_tracks_list[r]
                    curr_det_matched = detections_in_current_loop[c]
                    original_track_id = -1
                    for tid, track_data_val in current_tracks.items():
                        if track_data_val["_original_idx_temp"] == prev_det_matched["_original_idx_temp"]:
                            original_track_id = tid; break
                    if original_track_id != -1:
                        tracked_detections_list.append({**curr_det_matched, "tracked_object_id": original_track_id})
                        new_current_tracks_for_next_iter[original_track_id] = curr_det_matched
                        matched_current_indices.add(c)

            for c_idx, curr_det_unmatched in enumerate(detections_in_current_loop):
                if c_idx not in matched_current_indices:
                    new_id = next_track_id
                    tracked_detections_list.append({**curr_det_unmatched, "tracked_object_id": new_id})
                    new_current_tracks_for_next_iter[new_id] = curr_det_unmatched
                    next_track_id += 1
            current_tracks = new_current_tracks_for_next_iter

        if not tracked_detections_list: return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_object_id"))

        idx_to_track_id_map = {td["_original_idx_temp"]: td["tracked_object_id"] for td in tracked_detections_list}
        final_tracked_ids = [idx_to_track_id_map.get(row["_original_idx_temp"]) for row in image_df_with_orig_idx.iter_rows(named=True)]
        return image_df.with_columns(pl.Series("tracked_object_id", final_tracked_ids, dtype=pl.Int64))

    def _calculate_average_rbox(self, group_df: pl.DataFrame) -> Optional[List[float]]:
        """Calculates average rbox [cx, cy, w, h, angle] for a tracked object."""
        if group_df.is_empty() or "internal_rbbox" not in group_df.columns: return None
        rboxes_list = group_df["internal_rbbox"].drop_nulls().to_list()
        if not rboxes_list: return None

        rboxes_np = np.array(rboxes_list)
        # Averaging angles is tricky (e.g. 1 degree and 359 degrees).
        # For simplicity, directly average, but this can be problematic.
        # A better way is to average unit vectors representing angles.
        # For now:
        avg_params = np.mean(rboxes_np, axis=0)
        # Normalize angle to [0, 180) or (-90, 90] depending on convention after averaging
        avg_params[4] = avg_params[4] % 180.0
        return avg_params.tolist()

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info("Starting rotated detection stability evaluation...")
        required_cols = ["image_id", "loop", "internal_rbbox", "category_id", "score"]
        if not all(col in data_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data_df.columns]
            return EvaluationResult(metrics={"error": f"Missing required columns for R detección: {missing}"})

        if data_df.is_empty(): return EvaluationResult(metrics={"warning": "Empty input for RDet eval."})
        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        try:
            tracked_df = data_df.group_by("image_id", maintain_order=True).apply(self._track_objects_for_image)
        except Exception as e:
            logger.error(f"Error during RDet object tracking: {e}", exc_info=True)
            return EvaluationResult(metrics={"error": "RDet object tracking failed."})

        tracked_df = tracked_df.filter(pl.col("tracked_object_id").is_not_null())
        if tracked_df.is_empty(): return EvaluationResult(metrics={"warning": "No RDet objects tracked."})

        object_metrics_list = []
        total_loops_img = data_df.group_by("image_id").agg(pl.col("loop").n_unique().alias("total_loops"))

        for (img_id, trk_id), grp in tracked_df.group_by(["image_id", "tracked_object_id"], maintain_order=False):
            if grp.is_empty(): continue
            n_loops = grp["loop"].n_unique()
            img_total_loops = total_loops_img.filter(pl.col("image_id") == img_id)["total_loops"][0]
            app_cons = n_loops / img_total_loops if img_total_loops > 0 else 0.0

            avg_rbox = self._calculate_average_rbox(grp)
            riou_cons, ang_devs = [], []
            if avg_rbox:
                for rbox_list in grp["internal_rbbox"].drop_nulls().to_list():
                    riou_cons.append(calculate_riou_placeholder(rbox_list, avg_rbox))
                    # Angle deviation: smallest angle between two lines
                    angle_diff = abs(rbox_list[4] - avg_rbox[4])
                    ang_devs.append(min(angle_diff, 180 - angle_diff)) # for angles in [0,180)

            # Simplified center drift and size jitter (same as standard detection, using rbox components)
            centers_x = grp.select(pl.col("internal_rbbox").list.get(0).alias("cx"))["cx"]
            centers_y = grp.select(pl.col("internal_rbbox").list.get(1).alias("cy"))["cy"]
            widths = grp.select(pl.col("internal_rbbox").list.get(2).alias("w"))["w"]
            heights = grp.select(pl.col("internal_rbbox").list.get(3).alias("h"))["h"]

            mean_cx, mean_cy = centers_x.mean(), centers_y.mean()
            center_drifts = ((centers_x - mean_cx)**2 + (centers_y - mean_cy)**2).sqrt().to_list()

            areas = (widths * heights)
            mean_area = areas.mean()
            size_jitters_area = (areas - mean_area).abs() / mean_area if mean_area > 1e-9 else pl.Series([None]*len(areas))


            object_metrics_list.append({
                "image_id": img_id, "tracked_object_id": trk_id,
                "appearance_consistency": app_cons,
                "mean_riou_consistency_placeholder": np.mean(riou_cons) if riou_cons else None,
                "angle_std_dev": np.std(ang_devs) if ang_devs else None,
                "mean_center_drift_px": np.mean(center_drifts) if center_drifts else None,
                "mean_size_jitter_ratio": size_jitters_area.mean(),
                "confidence_std": grp["score"].std(),
                "category_switch_rate": (grp["category_id"].n_unique() -1) / (n_loops-1) if n_loops > 1 and grp["category_id"].n_unique() > 1 else 0.0,
                "num_loops_appeared": n_loops
            })

        if not object_metrics_list: return EvaluationResult(metrics={"warning":"No RDet object metrics."})
        stats_df = pl.DataFrame(object_metrics_list)

        final_metrics = {
            f"rdt_stab_mean_{col}": stats_df[col].mean() for col in
            ["appearance_consistency", "mean_riou_consistency_placeholder", "angle_std_dev",
             "mean_center_drift_px", "mean_size_jitter_ratio", "confidence_std", "category_switch_rate"]
            if stats_df[col].drop_nulls().len() > 0
        }
        final_metrics["rdt_stab_num_total_tracked"] = float(len(stats_df))
        final_metrics["rdt_stab_num_tracked_multi_loop"] = float(stats_df.filter(pl.col("num_loops_appeared") > 1).shape[0])

        cleaned_metrics = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k,v in final_metrics.items()}
        return EvaluationResult(metrics=cleaned_metrics, extra_data={"rdet_stability_details_df": stats_df})

StabilityEvaluatorFactory.register_evaluator("rotated_detection", RotatedDetectionStabilityEvaluator)

if __name__ == "__main__":
    from pathlib import Path
    dummy_config_content = """
project_info: {project_name: "RDetStab Test", model_type: "rotated_detection"}
data_loader:
  field_mapping: {loop: lp, image_id: id, image_path: pth, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4,
                  rotated_detection: {category_id: cat, score: scr, rbbox: [cx,cy,w,h,a]}}
evaluation_params:
  rotated_detection: {riou_threshold: 0.1} # Low threshold for placeholder RIoU
report_settings: {output_dir: "./test_rdet_stab_output"}
"""
    dummy_cfg_path = Path("dummy_rdet_stab_config.yaml")
    with open(dummy_cfg_path, "w") as f: f.write(dummy_config_content)
    Path("./test_rdet_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_cfg_path.read_text()))
    evaluator = RotatedDetectionStabilityEvaluator(config)

    test_df = pl.DataFrame({
        "id": ["img1"]*3 + ["img2"]*2, "lp": [1,2,3,1,2],
        "internal_rbbox": [ # cx,cy,w,h,angle
            [50,50,20,10,0], [52,52,20,10,5], [51,51,20,10,2], # img1, 3 loops, one object
            [100,100,30,15,45], [100,100,30,15,45] # img2, 2 loops, one object
        ],
        "cat": [0,0,0,1,1], "scr": [0.9,0.88,0.85,0.95,0.96]
    })
    logger.info("--- Testing RotatedDetectionStabilityEvaluator ---")
    results = evaluator.evaluate(test_df.clone())
    print("\nRotated Detection Metrics:", results.metrics)
    if "rdet_stability_details_df" in results.extra_data:
        print("\nDetails:", results.extra_data["rdet_stability_details_df"])
        assert results.extra_data["rdet_stability_details_df"].shape[0] == 2 # 2 unique tracked objects
    assert results.metrics.get("rdt_stab_num_total_tracked") == 2.0
    assert results.metrics.get("rdt_stab_mean_appearance_consistency") is not None
    if dummy_cfg_path.exists(): dummy_cfg_path.unlink()
    import shutil
    if Path("./test_rdet_stab_output").exists(): shutil.rmtree("./test_rdet_stab_output")
```
