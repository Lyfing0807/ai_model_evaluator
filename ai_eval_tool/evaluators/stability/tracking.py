"""
Stability evaluator for Object Tracking Models.
"""
import polars as pl
import numpy as np
import motmetrics as mm # For MOTA, MOTP, IDF1 etc.
from typing import List, Dict, Any, Optional
import yaml # For testing

from ...config_manager import MainConfig, TrackingEvaluationParams
from ..base import StabilityEvaluatorBase
from ...utils.types import EvaluationResult
from ...utils.logging_config import get_logger
from .factory import StabilityEvaluatorFactory
from .detection import calculate_iou # Can reuse for bbox matching if needed outside motmetrics

logger = get_logger(__name__)

class TrackingStabilityEvaluator(StabilityEvaluatorBase):
    def __init__(self, config: MainConfig):
        super().__init__(config)
        if not isinstance(self.eval_params, TrackingEvaluationParams):
            raise ValueError("Tracking evaluation parameters missing or incorrect type.")
        self.eval_params: TrackingEvaluationParams
        logger.info("TrackingStabilityEvaluator initialized.")

    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        logger.info("Starting object tracking stability evaluation...")

        # Required columns for MOT metrics if GT is present
        # DataLoader should have standardized these to:
        # frame_id, object_id_pred, internal_bbox_pred
        # object_id_gt, internal_bbox_gt (if available)
        # score_pred, category_id_pred, category_id_gt, visibility_gt, ignored_gt (optional)

        required_pred_cols = ["frame_id", "object_id_pred", "internal_bbox_pred"]
        has_gt = "object_id_gt" in data_df.columns and "internal_bbox_gt" in data_df.columns

        if not all(col in data_df.columns for col in required_pred_cols):
            missing = [c for c in required_pred_cols if c not in data_df.columns]
            return EvaluationResult(metrics={"error": f"Missing required prediction columns for Tracking: {missing}"})

        if data_df.is_empty():
            return EvaluationResult(metrics={"warning": "Empty input data for Tracking evaluation."})

        # Ensure frame_id is suitable for iteration (e.g., integer)
        # data_df = data_df.with_columns(pl.col("frame_id").cast(pl.Int64, strict=False).alias("frame_id_int"))
        # Using existing frame_id, assuming it's sortable.

        metrics: Dict[str, Any] = {}
        extra_data: Dict[str, Any] = {}

        # --- MOT Metrics Calculation (if GT is available) ---
        if has_gt:
            logger.info("Ground truth columns found. Calculating MOT metrics (MOTA, MOTP, IDF1, etc.).")
            acc = mm.MOTAccumulator(auto_id=True) # auto_id for frames

            # Iterate over each frame and update accumulator
            # Group by frame, then by loop (if multiple runs of the same video)
            # For now, assume one loop for MOT metrics, or average MOT metrics over loops.
            # Let's process loop by loop, then average. If no 'loop', assume one.

            loops = data_df["loop"].unique().to_list() if "loop" in data_df.columns else [1] # Default to one loop
            loop_mot_metrics_summary = []

            for loop_val in loops:
                loop_df = data_df.filter(pl.col("loop") == loop_val) if "loop" in data_df.columns else data_df
                if loop_df.is_empty(): continue

                acc.reset() # Reset accumulator for each loop run

                for frame_id_val, frame_df in loop_df.group_by("frame_id", maintain_order=True):
                    gt_ids = frame_df.get_column("object_id_gt").drop_nulls().to_list()
                    pred_ids = frame_df.get_column("object_id_pred").drop_nulls().to_list()

                    gt_bboxes = frame_df.filter(pl.col("object_id_gt").is_not_null())["internal_bbox_gt"].to_list()
                    pred_bboxes = frame_df.filter(pl.col("object_id_pred").is_not_null())["internal_bbox_pred"].to_list()

                    if not pred_ids: # No predictions in this frame
                        acc.update(gt_ids, [], [], frameid=frame_id_val) # All GTs are misses
                        continue
                    if not gt_ids: # No GTs in this frame
                         acc.update([], pred_ids, [[] for _ in pred_ids], frameid=frame_id_val) # All preds are FPs
                         continue

                    # Calculate distance matrix (1 - IoU)
                    # motmetrics expects distances, lower is better. Higher IoU means lower distance.
                    distances = np.full((len(gt_bboxes), len(pred_bboxes)), np.nan)
                    for i, gt_box in enumerate(gt_bboxes):
                        for j, pred_box in enumerate(pred_bboxes):
                            if gt_box and pred_box: # Ensure boxes are not None
                                iou = calculate_iou(gt_box, pred_box)
                                distances[i,j] = 1.0 - iou

                    acc.update(
                        gt_ids,
                        pred_ids,
                        distances,
                        frameid=frame_id_val
                    )

                mh = mm.metrics.create()
                summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=f'mot_summary_loop_{loop_val}')
                loop_mot_metrics_summary.append(summary)

            if loop_mot_metrics_summary:
                # Aggregate metrics from all loops (e.g., average)
                # For simplicity, taking the first loop's summary or averaging specific scalar metrics
                # This part needs careful thought on how to represent stability of MOT metrics.
                # For now, just report metrics for the first loop if multiple, or average simple ones.

                # Example: Average MOTA, MOTP, IDF1 if they exist across loops
                avg_mota = np.nanmean([s['mota'].iloc[0] for s in loop_mot_metrics_summary if 'mota' in s.columns and not s['mota'].empty])
                avg_motp = np.nanmean([s['motp'].iloc[0] for s in loop_mot_metrics_summary if 'motp' in s.columns and not s['motp'].empty])
                avg_idf1 = np.nanmean([s['idf1'].iloc[0] for s in loop_mot_metrics_summary if 'idf1' in s.columns and not s['idf1'].empty])
                # ... add others like num_switches, num_false_positives, num_misses etc.

                metrics["trk_mota"] = float(avg_mota) if not np.isnan(avg_mota) else None
                metrics["trk_motp"] = float(avg_motp) if not np.isnan(avg_motp) else None
                metrics["trk_idf1"] = float(avg_idf1) if not np.isnan(avg_idf1) else None

                # Stability of MOTA (e.g. std dev of MOTA across loops)
                if len(loop_mot_metrics_summary) > 1:
                    motas = [s['mota'].iloc[0] for s in loop_mot_metrics_summary if 'mota' in s.columns and not s['mota'].empty]
                    if len(motas) > 1:
                        metrics["trk_mota_std_across_loops"] = np.std(motas)

                extra_data["motmetrics_summary_per_loop"] = loop_mot_metrics_summary # Store detailed pandas summaries
                logger.info(f"MOT Metrics (avg across {len(loops)} loops): MOTA={metrics.get('trk_mota')}, MOTP={metrics.get('trk_motp')}, IDF1={metrics.get('trk_idf1')}")
            else:
                logger.warning("No MOT metrics could be computed (e.g. no valid loops or frames).")

        else: # No Ground Truth
            logger.warning("No ground truth track IDs (object_id_gt) or bboxes (internal_bbox_gt) found. Skipping MOT metrics.")
            # Implement stability metrics based on predictions only (e.g., track fragmentation, bbox consistency per track)
            # This part is TBD as per plan focusing on MOTA/MOTP first.
            metrics["trk_info"] = "Ground truth not available for MOT metrics."


        # --- Placeholder for other stability metrics (e.g. track fragmentation, bbox consistency) ---
        # These would typically be calculated per predicted track ID across frames.
        # Example:
        # For each unique object_id_pred:
        #   - Calculate number of frames it appears in (longevity)
        #   - Calculate std dev of its bbox width/height/area over time
        #   - If multiple loops: how consistently does this track_id appear for the same physical object?
        # This part needs further design based on what "stability" means without GT.

        self._calculate_stability_score(metrics, data_df) # Pass data_df for now, may need details_df later

        logger.info("Tracking stability evaluation completed.")
        return EvaluationResult(metrics=metrics, plots={}, extra_data=extra_data)

    def _calculate_stability_score(self, metrics: Dict[str, Any], details_df: pl.DataFrame):
        if not self.eval_params or not self.eval_params.scoring_weights:
            logger.warning("Tracking stability scoring weights not configured.")
            return
        weights = self.eval_params.scoring_weights

        score_mota_stability = normalize_metric_to_score(
            metrics.get("trk_mota"), # MOTA is typically 0-1 (or 0-100 if already scaled)
            is_0_1_rate_lower_better=False # Higher MOTA is better
        )
        if score_mota_stability is None: score_mota_stability = 0.0

        # Placeholder for other scores like track fragmentation, bbox consistency
        score_track_fragmentation = 50.0 # Dummy value
        score_bbox_consistency = 50.0  # Dummy value

        s_stability_tracking = (
            score_mota_stability * weights.get("mota_stability", 0.0) +
            score_track_fragmentation * weights.get("track_fragmentation", 0.0) +
            score_bbox_consistency * weights.get("bbox_consistency_per_track", 0.0)
        )
        # Normalize if weights don't sum to 1
        total_w = sum(weights.get(k,0.0) for k in weights)
        if total_w > 1e-6 and abs(total_w - 1.0) > 1e-6:
            s_stability_tracking /= total_w

        metrics["trk_stab_score_mota"] = score_mota_stability
        metrics["trk_stab_score_fragmentation_placeholder"] = score_track_fragmentation
        metrics["trk_stab_score_bbox_consistency_placeholder"] = score_bbox_consistency
        metrics["trk_stab_score_overall"] = max(0.0, min(100.0, s_stability_tracking))
        logger.info(f"Tracking Stability Score (Overall): {metrics['trk_stab_score_overall']:.2f}")


StabilityEvaluatorFactory.register_evaluator("tracking", TrackingStabilityEvaluator)

if __name__ == "__main__":
    from pathlib import Path
    from ai_eval_tool.utils.scoring_utils import normalize_metric_to_score # For direct import if needed

    dummy_cfg_content = """
project_info: {project_name: "TrackingStab Test", model_type: "tracking"}
data_loader:
  field_mapping:
    loop: loop_col
    frame_id: frame_col # Matches common field in config
    image_id: img_id_col # Generic image id, might be same as frame_id or different
    tracking:
      object_id_pred: pred_tid
      bbox_pred: [px,py,pw,ph]
      object_id_gt: gt_tid
      bbox_gt: [gx,gy,gw,gh]
evaluation_params:
  tracking: {mota_iou_threshold: 0.5, bbox_pred_format: xywh, bbox_gt_format: xywh}
report_settings: {output_dir: "./test_track_stab_output"}
"""
    dummy_path = Path("dummy_track_stab_config.yaml")
    with open(dummy_path, "w") as f: f.write(dummy_cfg_content)
    Path("./test_track_stab_output").mkdir(exist_ok=True)
    config = MainConfig.model_validate(yaml.safe_load(dummy_path.read_text()))
    evaluator = TrackingStabilityEvaluator(config)

    # Sample data: 2 frames, 2 GT objects, 2 Pred objects
    # Frame 1: GT0 matches P0, GT1 is missed. P1 is FP.
    # Frame 2: GT0 matches P0 (ID switch from P0 to P0), GT1 matches P1 (correctly continued).
    test_data = {
        "loop_col": [1,1,1,  1,1,1], # loop
        "frame_col": [1,1,1,  2,2,2], # frame_id
        "img_id_col": ["f1","f1","f1", "f2","f2","f2"], # image_id
        # Predictions
        "pred_tid":    [10, 11, None, 10, None, 11], # object_id_pred
        "px":          [10, 70, None, 12, None, 72], # bbox_pred x
        "py":          [10, 10, None, 12, None, 12], # bbox_pred y
        "pw":          [20, 20, None, 20, None, 20], # bbox_pred w
        "ph":          [20, 20, None, 20, None, 20], # bbox_pred h
        # Ground Truth
        "gt_tid":      [1,  2,  None, 1,  2, None],   # object_id_gt
        "gx":          [11, 40, None, 13, 73, None], # bbox_gt x
        "gy":          [11, 11, None, 13, 13, None], # bbox_gt y
        "gw":          [20, 20, None, 20, 20, None], # bbox_gt w
        "gh":          [20, 20, None, 20, 20, None], # bbox_gt h
    }
    test_df = pl.DataFrame(test_data).drop_nulls() # Drop rows where all GT/Pred specific are null if any

    logger.info("--- Testing TrackingStabilityEvaluator ---")
    results = evaluator.evaluate(test_df.clone())
    print("\nTracking Stability Metrics:", results.metrics)

    assert "trk_mota" in results.metrics
    assert "trk_motp" in results.metrics
    assert "trk_idf1" in results.metrics
    assert results.metrics["trk_mota"] is not None # Should be calculable

    if dummy_path.exists(): dummy_path.unlink()
    import shutil
    if Path("./test_track_stab_output").exists(): shutil.rmtree("./test_track_stab_output")

```
