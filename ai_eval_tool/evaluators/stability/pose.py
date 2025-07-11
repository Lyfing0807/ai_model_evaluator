"""
Stability evaluator for Pose Estimation Models.
"""
import polars as pl
import numpy as np
import math
import yaml # For testing
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from collections import defaultdict

from ...config_manager import MainConfig, PoseEvaluationParams
from ..base import StabilityEvaluatorBase
from ...utils.types import EvaluationResult
from ...utils.logging_config import get_logger
from .factory import StabilityEvaluatorFactory
from .detection import calculate_iou # Reuse IoU from detection for person matching

logger = get_logger(__name__)

# --- Keypoint Definitions (Example: COCO 17 keypoints) ---
# This should ideally be loaded from a config or be more extensible.
COCO_LAYOUT = {
    "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ],
    "skeleton": [ # Pairs of keypoint names or indices
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        # Connections to nose for head (example)
        ("nose", "left_eye"),("nose", "right_eye"),("left_eye","left_ear"),("right_eye","right_ear"),
        ("left_shoulder","nose"),("right_shoulder","nose") # Simplified neck
    ]
}
# TODO: Add MPII and other layouts if specified in config.

class PoseStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        if not isinstance(self.eval_params, PoseEvaluationParams):
            raise ValueError("Pose estimation evaluation parameters missing or incorrect type.")
        self.eval_params: PoseEvaluationParams
        self.person_match_iou_threshold: float = self.eval_params.match_iou_threshold

        # Load keypoint layout
        self.keypoint_names: List[str] = []
        self.skeleton_indices: List[Tuple[int, int]] = [] # Store as indices
        self._load_keypoint_layout(self.eval_params.keypoint_layout)

        logger.info(f"PoseStabilityEvaluator initialized with layout '{self.eval_params.keypoint_layout}' ({len(self.keypoint_names)} kpts).")

    def _load_keypoint_layout(self, layout_name: str):
        layout_map = {"coco_17": COCO_LAYOUT} # Extend with more layouts
        if layout_name not in layout_map:
            raise ValueError(f"Unsupported keypoint_layout: {layout_name}. Supported: {list(layout_map.keys())}")

        layout_data = layout_map[layout_name]
        self.keypoint_names = layout_data["keypoints"]

        # Convert skeleton from names to indices
        name_to_idx = {name: i for i, name in enumerate(self.keypoint_names)}
        self.skeleton_indices = []
        for kp_name1, kp_name2 in layout_data["skeleton"]:
            if kp_name1 in name_to_idx and kp_name2 in name_to_idx:
                self.skeleton_indices.append((name_to_idx[kp_name1], name_to_idx[kp_name2]))
            else:
                logger.warning(f"Skipping skeleton link: '{kp_name1}' or '{kp_name2}' not in keypoint names for layout '{layout_name}'.")


    def _track_persons_for_image(self, image_df: pl.DataFrame) -> pl.DataFrame:
        """Tracks persons across loops for a single image using IoU on 'internal_person_bbox'."""
        # This is nearly identical to _track_objects_for_image in DetectionStabilityEvaluator
        # but uses 'internal_person_bbox' and self.person_match_iou_threshold.
        if image_df.is_empty() or "loop" not in image_df.columns or "internal_person_bbox" not in image_df.columns:
            return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_person_id"))

        image_df = image_df.sort("loop")
        image_df_with_orig_idx = image_df.with_row_count("_original_idx_temp")

        detections_by_loop: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in image_df_with_orig_idx.iter_rows(named=True):
            detections_by_loop[row["loop"]].append(dict(row))

        loop_ids = sorted(detections_by_loop.keys())
        if not loop_ids: return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_person_id"))

        tracked_detections_list = []
        current_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 0

        for det_data in detections_by_loop[loop_ids[0]]:
            current_tracks[next_track_id] = det_data
            tracked_detections_list.append({**det_data, "tracked_person_id": next_track_id})
            next_track_id += 1

        for loop_idx in range(1, len(loop_ids)):
            detections_in_current_loop = detections_by_loop[loop_ids[loop_idx]]
            active_tracks_list = list(current_tracks.values())
            new_current_tracks_for_next_iter: Dict[int, Dict[str, Any]] = {}

            if not active_tracks_list or not detections_in_current_loop: # Simplified handling
                for det_data in detections_in_current_loop:
                    new_id = next_track_id
                    new_current_tracks_for_next_iter[new_id] = det_data
                    tracked_detections_list.append({**det_data, "tracked_person_id": new_id})
                    next_track_id += 1
                current_tracks = new_current_tracks_for_next_iter
                continue

            cost_matrix = np.full((len(active_tracks_list), len(detections_in_current_loop)), 1.0)
            for i, prev_det in enumerate(active_tracks_list):
                for j, curr_det in enumerate(detections_in_current_loop):
                    iou = calculate_iou(prev_det["internal_person_bbox"], curr_det["internal_person_bbox"])
                    if iou > 1e-6 : cost_matrix[i, j] = 1.0 - iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_current_indices = set()
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost_matrix[r,c] >= self.person_match_iou_threshold:
                    prev_det_matched = active_tracks_list[r]
                    curr_det_matched = detections_in_current_loop[c]
                    original_track_id = -1 # Find original track_id from current_tracks keys
                    for tid, track_data_val in current_tracks.items():
                        if track_data_val["_original_idx_temp"] == prev_det_matched["_original_idx_temp"]:
                            original_track_id = tid; break
                    if original_track_id != -1:
                        tracked_detections_list.append({**curr_det_matched, "tracked_person_id": original_track_id})
                        new_current_tracks_for_next_iter[original_track_id] = curr_det_matched
                        matched_current_indices.add(c)

            for c_idx, curr_det_unmatched in enumerate(detections_in_current_loop):
                if c_idx not in matched_current_indices:
                    new_id = next_track_id
                    tracked_detections_list.append({**curr_det_unmatched, "tracked_person_id": new_id})
                    new_current_tracks_for_next_iter[new_id] = curr_det_unmatched
                    next_track_id += 1
            current_tracks = new_current_tracks_for_next_iter

        if not tracked_detections_list: return image_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("tracked_person_id"))

        idx_to_track_id_map = {td["_original_idx_temp"]: td["tracked_person_id"] for td in tracked_detections_list}
        final_tracked_ids = [idx_to_track_id_map.get(row["_original_idx_temp"]) for row in image_df_with_orig_idx.iter_rows(named=True)]
        return image_df.with_columns(pl.Series("tracked_person_id", final_tracked_ids, dtype=pl.Int64))

    def _calculate_oks(self, kps1: np.ndarray, kps2: np.ndarray, scale_area: float, oks_sigmas: Union[float, np.ndarray]) -> float:
        """Calculates Object Keypoint Similarity (OKS). kps are [N,3] arrays (x,y,vis/conf)."""
        if kps1.shape[0] != kps2.shape[0] or kps1.shape[0] != len(self.keypoint_names): return 0.0

        # Use keypoint confidence/visibility from kps1 as reference for active keypoints
        # Or, more commonly, use visibility from ground truth if kps2 is GT. Here kps2 is avg pose.
        # For pose stability, we consider a keypoint "active" if it's consistently visible across loops for kps1.
        # Let's assume kps1[:, 2] and kps2[:, 2] are confidence scores, and we use a threshold.
        # For simplicity here, we'll use kps1's visibility, or assume visibility if conf > 0.1

        # Distances (d_i^2)
        dx = kps1[:, 0] - kps2[:, 0]
        dy = kps1[:, 1] - kps2[:, 1]
        d_sq = dx**2 + dy**2

        # Visibility/Confidence v_i (use from kps1, the "predicted" pose)
        # For stability, v_i might be 1 if a keypoint is considered "reliably detected" in kps1.
        # Here, let's use the provided confidence values directly.
        v = kps1[:, 2]

        # Sigmas k_i (per-keypoint constants)
        if isinstance(oks_sigmas, float):
            k_sq = np.full(len(self.keypoint_names), oks_sigmas**2)
        elif isinstance(oks_sigmas, list) and len(oks_sigmas) == len(self.keypoint_names):
            k_sq = np.array(oks_sigmas)**2
        else: # Fallback if oks_sigmas is misconfigured
            logger.warning(f"OKS sigmas misconfigured ({oks_sigmas}), using default 0.5 for all keypoints.")
            k_sq = np.full(len(self.keypoint_names), 0.5**2)

        # OKS calculation
        # e = d_sq / (2 * scale_area * k_sq + 1e-9) # scale_area is person bbox area
        # The original COCO OKS uses s=sqrt(area) for scale, and k_i are sigmas.
        # OKS = sum( exp(-d_i^2 / (2 * s^2 * k_i^2)) * delta(v_i>0) ) / sum(delta(v_i>0))
        # Here, s is sqrt(scale_area)
        s_sq = scale_area # scale_area is person_bbox area, so s = sqrt(person_bbox_area)

        if s_sq < 1e-6 : # Avoid division by zero if area is tiny
            return 0.0

        # Numerator: exp(-d_i^2 / (2 * s_sq * k_i^2)) where v_i > threshold
        # Denominator: count of v_i > threshold
        # Let's use a visibility threshold, e.g. 0.1
        vis_thresh = 0.1

        oks_terms = np.exp(-d_sq / (2 * s_sq * k_sq + 1e-9))

        numerator = np.sum(oks_terms[v > vis_thresh])
        denominator = np.sum(v > vis_thresh)

        return numerator / denominator if denominator > 0 else 0.0


    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info("Starting pose stability evaluation...")
        required_cols = ["image_id", "loop", "internal_person_bbox", "person_score", "internal_keypoints"]
        if not all(col in data_df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data_df.columns]
            return EvaluationResult(metrics={"error": f"Missing required columns for Pose: {missing}"})

        if data_df.is_empty(): return EvaluationResult(metrics={"warning": "Empty input for Pose eval."})
        data_df = data_df.with_columns(pl.col("loop").cast(pl.Int64, strict=False))

        try:
            tracked_df = data_df.group_by("image_id", maintain_order=True).apply(self._track_persons_for_image)
        except Exception as e:
            logger.error(f"Error during Pose person tracking: {e}", exc_info=True)
            return EvaluationResult(metrics={"error": "Pose person tracking failed."})

        tracked_df = tracked_df.filter(pl.col("tracked_person_id").is_not_null())
        if tracked_df.is_empty(): return EvaluationResult(metrics={"warning": "No persons tracked for Pose."})

        all_person_stability_metrics = []
        total_loops_img = data_df.group_by("image_id").agg(pl.col("loop").n_unique().alias("total_loops"))

        for (img_id, person_trk_id), person_grp_df in tracked_df.group_by(["image_id", "tracked_person_id"], maintain_order=False):
            if person_grp_df.is_empty(): continue

            n_loops_person = person_grp_df["loop"].n_unique()
            img_total_loops = total_loops_img.filter(pl.col("image_id") == img_id)["total_loops"][0]
            person_app_cons = n_loops_person / img_total_loops if img_total_loops > 0 else 0.0

            # Keypoint analysis for this tracked person
            # `internal_keypoints` is a list of lists: [[x,y,c], [x,y,c], ...]
            # Convert to numpy array for easier processing: [num_loops, num_keypoints, 3]

            all_kps_for_person = [] # List of [num_keypoints, 3] arrays
            person_bbox_areas = [] # For OKS scale

            for row_idx in range(len(person_grp_df)):
                kps_list_of_lists = person_grp_df[row_idx, "internal_keypoints"]
                if kps_list_of_lists and isinstance(kps_list_of_lists, list) and len(kps_list_of_lists) == len(self.keypoint_names):
                    all_kps_for_person.append(np.array(kps_list_of_lists))
                else: # Mismatch or empty, append array of NaNs or handle as missing frame
                    logger.debug(f"Keypoint data issue for person {person_trk_id}, image {img_id}, loop {person_grp_df[row_idx, 'loop']}. Kps: {kps_list_of_lists}")
                    all_kps_for_person.append(np.full((len(self.keypoint_names), 3), np.nan))

                # Person bbox area for OKS scale
                pbbox = person_grp_df[row_idx, "internal_person_bbox"] # [xmin,ymin,xmax,ymax]
                if pbbox and len(pbbox) == 4:
                    person_bbox_areas.append( (pbbox[2]-pbbox[0]) * (pbbox[3]-pbbox[1]) )
                else:
                    person_bbox_areas.append(np.nan) # Default scale if bbox missing


            if not all_kps_for_person: continue

            kps_tensor = np.stack(all_kps_for_person, axis=0) # Shape: [num_frames_person_present, num_keypoints, 3]
            person_bbox_areas_np = np.array(person_bbox_areas)

            # 1. Keypoint Visibility Consistency (per keypoint type)
            # Calculate based on confidence threshold, e.g. > 0.1
            vis_thresh = 0.1
            kpt_visibility_conf = kps_tensor[:, :, 2] # [frames, kpts_conf]
            kpt_is_visible = kpt_visibility_conf > vis_thresh

            # Mean visibility for each keypoint type across frames where person was seen
            mean_kpt_visibility = np.mean(kpt_is_visible, axis=0) # [num_keypoints]
            avg_visibility_all_kpts = np.mean(mean_kpt_visibility) # Single scalar

            # 2. Keypoint Position Drift (for visible keypoints)
            # Calculate mean position for each keypoint, then drift from mean
            mean_kpt_positions = np.nanmean(np.where(kpt_is_visible[:,:,np.newaxis], kps_tensor[:,:,:2], np.nan), axis=0) # [kpts, 2 (x,y)]

            drifts_per_kpt_per_frame = np.linalg.norm(kps_tensor[:,:,:2] - mean_kpt_positions[np.newaxis,:,:], axis=2) # [frames, kpts]
            # Average drift for each keypoint type, only over frames where it was visible
            mean_drift_per_kpt_type = np.nanmean(np.where(kpt_is_visible, drifts_per_kpt_per_frame, np.nan), axis=0) # [kpts]
            avg_drift_all_kpts = np.nanmean(mean_drift_per_kpt_type) # Single scalar

            # 3. OKS Consistency
            # Calculate average pose (using mean_kpt_positions and mean_kpt_visibility as its confidence)
            avg_pose_kps = np.zeros((len(self.keypoint_names), 3))
            avg_pose_kps[:, :2] = mean_kpt_positions
            avg_pose_kps[:, 2] = mean_kpt_visibility # Use mean visibility as confidence of avg pose keypoint

            oks_scores = []
            for i in range(kps_tensor.shape[0]): # For each frame the person is present
                current_kps_frame = kps_tensor[i,:,:]
                current_bbox_area = person_bbox_areas_np[i]
                if np.isnan(current_bbox_area) or current_bbox_area < 1e-6: continue # Skip if no valid scale

                oks = self._calculate_oks(current_kps_frame, avg_pose_kps, current_bbox_area, self.eval_params.oks_sigma)
                oks_scores.append(oks)
            mean_oks_consistency = np.mean(oks_scores) if oks_scores else None

            all_person_stability_metrics.append({
                "image_id": img_id, "tracked_person_id": person_trk_id,
                "person_appearance_consistency": person_app_cons,
                "mean_keypoint_visibility": avg_visibility_all_kpts,
                "mean_keypoint_position_drift_px": avg_drift_all_kpts,
                "mean_oks_consistency_with_avg_pose": mean_oks_consistency,
                "num_loops_person_appeared": n_loops_person
            })

        if not all_person_stability_metrics: return EvaluationResult(metrics={"warning":"No Pose person metrics."})
        stats_df = pl.DataFrame(all_person_stability_metrics)

        final_metrics = {
            f"pos_stab_mean_{col}": stats_df[col].mean() for col in
            ["person_appearance_consistency", "mean_keypoint_visibility",
             "mean_keypoint_position_drift_px", "mean_oks_consistency_with_avg_pose"]
            if stats_df[col].drop_nulls().len() > 0
        }
        final_metrics["pos_stab_num_total_tracked_persons"] = float(len(stats_df))
        final_metrics["pos_stab_num_persons_tracked_multi_loop"] = float(stats_df.filter(pl.col("num_loops_person_appeared") > 1).shape[0])

        cleaned = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k,v in final_metrics.items()}
        return EvaluationResult(metrics=cleaned, extra_data={"pose_stability_details_df": stats_df})

StabilityEvaluatorFactory.register_evaluator("pose", PoseStabilityEvaluator)

if __name__ == "__main__":
    from pathlib import Path
    dummy_cfg_content = """
project_info: {project_name: "PoseStab Test", model_type: "pose"}
data_loader:
  field_mapping: {loop: lp, image_id: id, image_path: pth, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4,
                  pose: {person_bbox: [px,py,pw,ph], person_score: ps, keypoints: kps}}
evaluation_params:
  pose: {match_iou_threshold: 0.3, keypoint_layout: "coco_17", oks_sigma: 0.5}
report_settings: {output_dir: "./test_pose_stab_output"}
"""
    dummy_path = Path("dummy_pose_stab_config.yaml")
    with open(dummy_path, "w") as f: f.write(dummy_cfg_content)
    Path("./test_pose_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_path.read_text()))
    evaluator = PoseStabilityEvaluator(config)

    num_coco_kpts = 17
    kps_str1_l1 = ";".join([f"{10+i},{10+i},{0.9}" for i in range(num_coco_kpts)])
    kps_str1_l2 = ";".join([f"{12+i},{12+i},{0.8}" for i in range(num_coco_kpts)]) # Shifted
    kps_str2_l1 = ";".join([f"{100+i},{100+i},{0.7}" for i in range(num_coco_kpts)])

    test_df = pl.DataFrame({
        "id": ["img1"]*2 + ["img2"], "lp": [1,2,1],
        "internal_person_bbox": [[10,10,20,20], [11,11,20,20], [50,50,30,30]], #xywh from dataloader
        "person_score": [0.9,0.88,0.95],
        "internal_keypoints": [ # This is what DataLoader produces: List[List[float]]
            [[10+i,10+i,0.9] for i in range(num_coco_kpts)], # Person1, Loop1
            [[12+i,12+i,0.8] for i in range(num_coco_kpts)], # Person1, Loop2
            [[100+i,100+i,0.7] for i in range(num_coco_kpts)] # Person2, Loop1
        ]
    })
    logger.info("--- Testing PoseStabilityEvaluator ---")
    results = evaluator.evaluate(test_df.clone())
    print("\nPose Stability Metrics:", results.metrics)
    if "pose_stability_details_df" in results.extra_data:
        print("\nDetails:", results.extra_data["pose_stability_details_df"])
        assert results.extra_data["pose_stability_details_df"].shape[0] == 2 # 2 tracked persons
    assert results.metrics.get("pos_stab_num_total_tracked_persons") == 2.0
    assert results.metrics.get("pos_stab_mean_keypoint_visibility") is not None
    assert results.metrics.get("pos_stab_mean_oks_consistency_with_avg_pose") is not None

    if dummy_path.exists(): dummy_path.unlink()
    import shutil
    if Path("./test_pose_stab_output").exists(): shutil.rmtree("./test_pose_stab_output")

```
