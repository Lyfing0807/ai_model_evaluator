"""Tests for rotated detection stability evaluator."""

import numpy as np
import polars as pl
import pytest

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.rotated_detection import (
    RotatedDetectionStabilityEvaluator,
    calculate_riou,
    normalize_angle,
)
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def rotated_detection_config(test_data_dir) -> MainConfig:
    """Create a rotated detection config for testing."""
    config_data = {
        "project_info": {
            "project_name": "Test Rotated Detection Project",
            "model_type": "rotated_detection",
            "run_id": "test_rot_det_run_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop_id",
                "image_id": "img_name",
                "image_path": "path",
                "pre_time_ms": "t_pre",
                "inference_time_ms": "t_inf",
                "post_time_ms": "t_post",
                "total_time_ms": "t_total",
                "rotated_detection": {
                    "category_id": "det_cat",
                    "score": "det_score",
                    "rotated_bbox": ["cx", "cy", "w", "h", "angle"],
                },
            }
        },
        "evaluation_params": {
            "rotated_detection": {
                "riou_threshold": 0.5,
                "angle_threshold": 15.0,
                "bbox_format": "cxcywha",
            }
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_rotated_detection")},
    }
    return MainConfig(**config_data)


@pytest.fixture
def rotated_detection_stability_evaluator(
    rotated_detection_config: MainConfig,
) -> RotatedDetectionStabilityEvaluator:
    """Create rotated detection stability evaluator instance."""
    return RotatedDetectionStabilityEvaluator(rotated_detection_config)


def test_normalize_angle():
    """Test angle normalization to [-180, 180] range."""
    assert normalize_angle(0) == 0
    assert normalize_angle(180) == 180
    assert normalize_angle(-180) == -180
    assert normalize_angle(270) == -90
    assert normalize_angle(-270) == 90
    assert normalize_angle(360) == 0
    assert normalize_angle(450) == 90


def test_calculate_riou():
    """Test Rotated IoU calculation."""
    # Test with identical rotated boxes
    bbox1 = (100, 100, 50, 30, 0)  # cx, cy, w, h, angle
    bbox2 = (100, 100, 50, 30, 0)
    riou = calculate_riou(bbox1, bbox2)
    assert riou == pytest.approx(1.0, abs=1e-6)
    
    # Test with non-overlapping boxes
    bbox3 = (200, 200, 50, 30, 0)
    riou_no_overlap = calculate_riou(bbox1, bbox3)
    assert riou_no_overlap == pytest.approx(0.0, abs=1e-6)
    
    # Test with partially overlapping boxes
    bbox4 = (110, 110, 50, 30, 0)  # Slightly shifted
    riou_partial = calculate_riou(bbox1, bbox4)
    assert 0 < riou_partial < 1.0
    
    # Test with rotated boxes
    bbox5 = (100, 100, 50, 30, 45)  # Same position, rotated 45 degrees
    riou_rotated = calculate_riou(bbox1, bbox5)
    assert 0 < riou_rotated < 1.0


def test_rotated_detection_stability_evaluator_basic(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test basic rotated detection stability evaluation."""
    # Create test data with rotated bounding boxes
    test_data = {
        "image_id": ["img1", "img1", "img1", "img2", "img2", "img2"],
        "loop": [1, 1, 2, 1, 1, 2],
        "det_cat": [0, 1, 0, 0, 1, 0],
        "det_score": [0.9, 0.8, 0.85, 0.95, 0.75, 0.92],
        "cx": [100, 200, 102, 150, 250, 148],
        "cy": [100, 150, 98, 200, 180, 202],
        "w": [50, 40, 48, 60, 35, 58],
        "h": [30, 25, 32, 40, 20, 38],
        "angle": [0, 45, 5, 30, 50, 28],
    }
    
    input_df = pl.DataFrame(test_data)
    result = rotated_detection_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    assert "rot_det_stab_mean_riou" in result.metrics
    assert "rot_det_stab_mean_angle_diff" in result.metrics
    assert "rot_det_stab_mean_confidence_std" in result.metrics
    assert "rot_det_stab_detection_consistency_rate" in result.metrics
    assert "rotated_detection_stability_details_df" in result.extra_data
    
    # Check that metrics are in reasonable ranges
    assert 0 <= result.metrics["rot_det_stab_mean_riou"] <= 1.0
    assert 0 <= result.metrics["rot_det_stab_mean_angle_diff"] <= 180
    assert result.metrics["rot_det_stab_mean_confidence_std"] >= 0
    assert 0 <= result.metrics["rot_det_stab_detection_consistency_rate"] <= 1.0


def test_rotated_detection_stability_single_loop(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test rotated detection stability with single loop."""
    test_data = {
        "image_id": ["img1"],
        "loop": [1],
        "det_cat": [0],
        "det_score": [0.9],
        "cx": [100],
        "cy": [100],
        "w": [50],
        "h": [30],
        "angle": [0],
    }
    
    input_df = pl.DataFrame(test_data)
    result = rotated_detection_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    # With single loop, metrics should indicate perfect stability
    assert result.metrics["rot_det_stab_mean_riou"] == 1.0
    assert result.metrics["rot_det_stab_mean_angle_diff"] == 0.0
    assert result.metrics["rot_det_stab_mean_confidence_std"] == 0.0
    assert result.metrics["rot_det_stab_detection_consistency_rate"] == 1.0


def test_rotated_detection_stability_empty_input(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test rotated detection stability with empty input."""
    empty_df = pl.DataFrame(
        {
            "image_id": [],
            "loop": [],
            "det_cat": [],
            "det_score": [],
            "cx": [],
            "cy": [],
            "w": [],
            "h": [],
            "angle": [],
        },
        schema={
            "image_id": pl.Utf8,
            "loop": pl.Int64,
            "det_cat": pl.Int64,
            "det_score": pl.Float64,
            "cx": pl.Float64,
            "cy": pl.Float64,
            "w": pl.Float64,
            "h": pl.Float64,
            "angle": pl.Float64,
        },
    )
    
    result = rotated_detection_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]


def test_rotated_detection_stability_missing_columns(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test rotated detection stability with missing required columns."""
    incomplete_df = pl.DataFrame({
        "image_id": ["img1"],
        "loop": [1],
        "det_cat": [0],
        # Missing rotated bbox columns
    })
    
    result = rotated_detection_stability_evaluator.evaluate(incomplete_df)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]


def test_rotated_detection_angle_consistency(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test angle consistency calculation with various angle differences."""
    # Test data with different angle variations
    test_data = {
        "image_id": ["img1", "img1", "img2", "img2", "img3", "img3"],
        "loop": [1, 2, 1, 2, 1, 2],
        "det_cat": [0, 0, 0, 0, 0, 0],
        "det_score": [0.9, 0.85, 0.95, 0.92, 0.88, 0.83],
        "cx": [100, 100, 200, 200, 300, 300],
        "cy": [100, 100, 200, 200, 300, 300],
        "w": [50, 50, 60, 60, 40, 40],
        "h": [30, 30, 35, 35, 25, 25],
        "angle": [0, 5, 45, 50, 90, 95],  # Small angle differences
    }
    
    input_df = pl.DataFrame(test_data)
    result = rotated_detection_stability_evaluator.evaluate(input_df)
    
    details_df = result.extra_data["rotated_detection_stability_details_df"]
    
    # Check that angle differences are calculated correctly
    assert "mean_angle_diff" in details_df.columns
    
    # All angle differences should be small (5 degrees)
    for i in range(len(details_df)):
        angle_diff = details_df[i, "mean_angle_diff"]
        assert angle_diff == pytest.approx(5.0, abs=1.0)


def test_rotated_detection_riou_threshold_filtering(
    rotated_detection_stability_evaluator: RotatedDetectionStabilityEvaluator,
):
    """Test that detections below RIoU threshold are handled correctly."""
    # Create test data where some detections have low overlap
    test_data = {
        "image_id": ["img1", "img1"],
        "loop": [1, 2],
        "det_cat": [0, 0],
        "det_score": [0.9, 0.85],
        "cx": [100, 200],  # Far apart centers
        "cy": [100, 200],
        "w": [50, 50],
        "h": [30, 30],
        "angle": [0, 0],
    }
    
    input_df = pl.DataFrame(test_data)
    result = rotated_detection_stability_evaluator.evaluate(input_df)
    
    # Should still produce results, but with low RIoU values
    assert isinstance(result, EvaluationResult)
    assert result.metrics["rot_det_stab_mean_riou"] < 0.5  # Should be low due to poor overlap