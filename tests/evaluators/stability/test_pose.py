"""Tests for pose estimation stability evaluator."""

import numpy as np
import polars as pl
import pytest

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.pose import (
    PoseStabilityEvaluator,
    calculate_oks,
    euclidean_distance,
    COCO_KEYPOINT_LAYOUT,
)
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def pose_config(test_data_dir) -> MainConfig:
    """Create a pose estimation config for testing."""
    config_data = {
        "project_info": {
            "project_name": "Test Pose Project",
            "model_type": "pose",
            "run_id": "test_pose_run_001",
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
                "pose": {
                    "keypoints": "keypoints",
                    "bbox": ["x", "y", "w", "h"],
                    "score": "pose_score",
                },
            }
        },
        "evaluation_params": {
            "pose": {
                "keypoint_layout": "coco",
                "oks_threshold": 0.5,
                "bbox_format": "xywh",
            }
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_pose")},
    }
    return MainConfig(**config_data)


@pytest.fixture
def pose_stability_evaluator(pose_config: MainConfig) -> PoseStabilityEvaluator:
    """Create pose stability evaluator instance."""
    return PoseStabilityEvaluator(pose_config)


def test_euclidean_distance():
    """Test euclidean distance calculation."""
    point1 = (0, 0)
    point2 = (3, 4)
    assert euclidean_distance(point1, point2) == 5.0
    
    # Test with same points
    assert euclidean_distance(point1, point1) == 0.0
    
    # Test with negative coordinates
    point3 = (-1, -1)
    point4 = (2, 3)
    expected = np.sqrt((2 - (-1))**2 + (3 - (-1))**2)
    assert euclidean_distance(point3, point4) == pytest.approx(expected)


def test_calculate_oks():
    """Test Object Keypoint Similarity (OKS) calculation."""
    # Create simple test keypoints (17 keypoints for COCO format)
    keypoints1 = [(i, i) for i in range(17)]  # (0,0), (1,1), ..., (16,16)
    keypoints2 = [(i+1, i+1) for i in range(17)]  # (1,1), (2,2), ..., (17,17)
    
    # Test with identical keypoints
    identical_keypoints = [(i, i) for i in range(17)]
    oks_identical = calculate_oks(identical_keypoints, identical_keypoints, area=100)
    assert oks_identical == 1.0
    
    # Test with different keypoints
    oks_different = calculate_oks(keypoints1, keypoints2, area=100)
    assert 0 <= oks_different <= 1.0
    
    # Test with zero area (should handle gracefully)
    oks_zero_area = calculate_oks(keypoints1, keypoints2, area=0)
    assert oks_zero_area == 0.0


def test_pose_stability_evaluator_basic(pose_stability_evaluator: PoseStabilityEvaluator):
    """Test basic pose stability evaluation."""
    # Create test data with pose keypoints
    test_data = {
        "image_id": ["img1", "img1", "img2", "img2"],
        "loop": [1, 2, 1, 2],
        "keypoints": [
            [(10, 10), (20, 20), (30, 30)] + [(0, 0)] * 14,  # 17 keypoints total
            [(11, 11), (21, 21), (31, 31)] + [(0, 0)] * 14,  # Slightly moved
            [(50, 50), (60, 60), (70, 70)] + [(0, 0)] * 14,
            [(51, 51), (61, 61), (71, 71)] + [(0, 0)] * 14,
        ],
        "x": [100, 100, 200, 200],
        "y": [100, 100, 200, 200],
        "w": [50, 50, 50, 50],
        "h": [50, 50, 50, 50],
        "pose_score": [0.9, 0.85, 0.95, 0.92],
    }
    
    input_df = pl.DataFrame(test_data)
    result = pose_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    assert "pose_stab_mean_oks" in result.metrics
    assert "pose_stab_mean_keypoint_displacement" in result.metrics
    assert "pose_stab_mean_confidence_std" in result.metrics
    assert "pose_stability_details_df" in result.extra_data
    
    # Check that metrics are reasonable
    assert 0 <= result.metrics["pose_stab_mean_oks"] <= 1.0
    assert result.metrics["pose_stab_mean_keypoint_displacement"] >= 0
    assert result.metrics["pose_stab_mean_confidence_std"] >= 0


def test_pose_stability_single_loop(pose_stability_evaluator: PoseStabilityEvaluator):
    """Test pose stability with single loop (should handle gracefully)."""
    test_data = {
        "image_id": ["img1"],
        "loop": [1],
        "keypoints": [[(10, 10), (20, 20)] + [(0, 0)] * 15],
        "x": [100],
        "y": [100],
        "w": [50],
        "h": [50],
        "pose_score": [0.9],
    }
    
    input_df = pl.DataFrame(test_data)
    result = pose_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    # With single loop, OKS should be 1.0, displacement should be 0
    assert result.metrics["pose_stab_mean_oks"] == 1.0
    assert result.metrics["pose_stab_mean_keypoint_displacement"] == 0.0
    assert result.metrics["pose_stab_mean_confidence_std"] == 0.0


def test_pose_stability_empty_input(pose_stability_evaluator: PoseStabilityEvaluator):
    """Test pose stability with empty input."""
    empty_df = pl.DataFrame(
        {
            "image_id": [],
            "loop": [],
            "keypoints": [],
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "pose_score": [],
        },
        schema={
            "image_id": pl.Utf8,
            "loop": pl.Int64,
            "keypoints": pl.List(pl.List(pl.Float64)),
            "x": pl.Float64,
            "y": pl.Float64,
            "w": pl.Float64,
            "h": pl.Float64,
            "pose_score": pl.Float64,
        },
    )
    
    result = pose_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]


def test_pose_stability_missing_columns(pose_stability_evaluator: PoseStabilityEvaluator):
    """Test pose stability with missing required columns."""
    incomplete_df = pl.DataFrame({
        "image_id": ["img1"],
        "loop": [1],
        # Missing keypoints and other required columns
    })
    
    result = pose_stability_evaluator.evaluate(incomplete_df)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]


def test_coco_keypoint_layout():
    """Test COCO keypoint layout constants."""
    assert len(COCO_KEYPOINT_LAYOUT["keypoints"]) == 17
    assert len(COCO_KEYPOINT_LAYOUT["skeleton"]) > 0
    assert len(COCO_KEYPOINT_LAYOUT["sigmas"]) == 17
    
    # Check that all skeleton connections reference valid keypoint indices
    for connection in COCO_KEYPOINT_LAYOUT["skeleton"]:
        keypoint1, keypoint2 = connection
        assert keypoint1 in COCO_KEYPOINT_LAYOUT["keypoints"]
        assert keypoint2 in COCO_KEYPOINT_LAYOUT["keypoints"]