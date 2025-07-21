"""Tests for tracking stability evaluator."""

import polars as pl
import pytest

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.tracking import (
    TrackingStabilityEvaluator,
    calculate_track_consistency,
    calculate_id_switches,
)
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def tracking_config(test_data_dir) -> MainConfig:
    """Create a tracking config for testing."""
    config_data = {
        "project_info": {
            "project_name": "Test Tracking Project",
            "model_type": "tracking",
            "run_id": "test_tracking_run_001",
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
                "tracking": {
                    "track_id": "track_id",
                    "category_id": "det_cat",
                    "score": "det_score",
                    "bbox": ["x", "y", "w", "h"],
                    "frame_id": "frame_id",
                },
            }
        },
        "evaluation_params": {
            "tracking": {
                "iou_threshold": 0.5,
                "bbox_format": "xywh",
                "min_track_length": 3,
            }
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_tracking")},
    }
    return MainConfig(**config_data)


@pytest.fixture
def tracking_stability_evaluator(tracking_config: MainConfig) -> TrackingStabilityEvaluator:
    """Create tracking stability evaluator instance."""
    return TrackingStabilityEvaluator(tracking_config)


def test_calculate_track_consistency():
    """Test track consistency calculation."""
    # Test with consistent tracks (same track IDs across loops)
    tracks_loop1 = [1, 2, 3, 4]
    tracks_loop2 = [1, 2, 3, 4]
    consistency = calculate_track_consistency(tracks_loop1, tracks_loop2)
    assert consistency == 1.0
    
    # Test with completely different tracks
    tracks_loop3 = [5, 6, 7, 8]
    consistency_diff = calculate_track_consistency(tracks_loop1, tracks_loop3)
    assert consistency_diff == 0.0
    
    # Test with partially overlapping tracks
    tracks_loop4 = [1, 2, 5, 6]
    consistency_partial = calculate_track_consistency(tracks_loop1, tracks_loop4)
    assert 0 < consistency_partial < 1.0
    
    # Test with empty tracks
    consistency_empty = calculate_track_consistency([], [])
    assert consistency_empty == 1.0  # Both empty should be consistent


def test_calculate_id_switches():
    """Test ID switch calculation."""
    # Test with no ID switches (consistent tracking)
    track_sequence1 = [
        {"frame": 1, "track_id": 1, "bbox": [10, 10, 20, 20]},
        {"frame": 2, "track_id": 1, "bbox": [12, 12, 20, 20]},
        {"frame": 3, "track_id": 1, "bbox": [14, 14, 20, 20]},
    ]
    switches1 = calculate_id_switches(track_sequence1)
    assert switches1 == 0
    
    # Test with ID switches
    track_sequence2 = [
        {"frame": 1, "track_id": 1, "bbox": [10, 10, 20, 20]},
        {"frame": 2, "track_id": 2, "bbox": [12, 12, 20, 20]},  # ID switch
        {"frame": 3, "track_id": 2, "bbox": [14, 14, 20, 20]},
    ]
    switches2 = calculate_id_switches(track_sequence2)
    assert switches2 == 1
    
    # Test with multiple ID switches
    track_sequence3 = [
        {"frame": 1, "track_id": 1, "bbox": [10, 10, 20, 20]},
        {"frame": 2, "track_id": 2, "bbox": [12, 12, 20, 20]},  # Switch 1
        {"frame": 3, "track_id": 3, "bbox": [14, 14, 20, 20]},  # Switch 2
    ]
    switches3 = calculate_id_switches(track_sequence3)
    assert switches3 == 2


def test_tracking_stability_evaluator_basic(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test basic tracking stability evaluation."""
    # Create test data with tracking results across multiple loops and frames
    test_data = {
        "image_id": ["seq1", "seq1", "seq1", "seq1", "seq1", "seq1"],
        "loop": [1, 1, 1, 2, 2, 2],
        "frame_id": [1, 2, 3, 1, 2, 3],
        "track_id": [1, 1, 1, 1, 1, 2],  # ID switch in loop 2, frame 3
        "det_cat": [0, 0, 0, 0, 0, 0],
        "det_score": [0.9, 0.85, 0.8, 0.88, 0.83, 0.78],
        "x": [100, 110, 120, 102, 112, 118],
        "y": [100, 105, 110, 98, 103, 108],
        "w": [50, 50, 50, 48, 48, 48],
        "h": [30, 30, 30, 32, 32, 32],
    }
    
    input_df = pl.DataFrame(test_data)
    result = tracking_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    assert "track_stab_mean_track_consistency" in result.metrics
    assert "track_stab_mean_id_switches_per_sequence" in result.metrics
    assert "track_stab_mean_track_length_std" in result.metrics
    assert "track_stab_mean_confidence_std" in result.metrics
    assert "tracking_stability_details_df" in result.extra_data
    
    # Check that metrics are in reasonable ranges
    assert 0 <= result.metrics["track_stab_mean_track_consistency"] <= 1.0
    assert result.metrics["track_stab_mean_id_switches_per_sequence"] >= 0
    assert result.metrics["track_stab_mean_track_length_std"] >= 0
    assert result.metrics["track_stab_mean_confidence_std"] >= 0


def test_tracking_stability_single_loop(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test tracking stability with single loop."""
    test_data = {
        "image_id": ["seq1"],
        "loop": [1],
        "frame_id": [1],
        "track_id": [1],
        "det_cat": [0],
        "det_score": [0.9],
        "x": [100],
        "y": [100],
        "w": [50],
        "h": [30],
    }
    
    input_df = pl.DataFrame(test_data)
    result = tracking_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    # With single loop, consistency should be perfect
    assert result.metrics["track_stab_mean_track_consistency"] == 1.0
    assert result.metrics["track_stab_mean_id_switches_per_sequence"] == 0.0
    assert result.metrics["track_stab_mean_track_length_std"] == 0.0
    assert result.metrics["track_stab_mean_confidence_std"] == 0.0


def test_tracking_stability_empty_input(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test tracking stability with empty input."""
    empty_df = pl.DataFrame(
        {
            "image_id": [],
            "loop": [],
            "frame_id": [],
            "track_id": [],
            "det_cat": [],
            "det_score": [],
            "x": [],
            "y": [],
            "w": [],
            "h": [],
        },
        schema={
            "image_id": pl.Utf8,
            "loop": pl.Int64,
            "frame_id": pl.Int64,
            "track_id": pl.Int64,
            "det_cat": pl.Int64,
            "det_score": pl.Float64,
            "x": pl.Float64,
            "y": pl.Float64,
            "w": pl.Float64,
            "h": pl.Float64,
        },
    )
    
    result = tracking_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]


def test_tracking_stability_missing_columns(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test tracking stability with missing required columns."""
    incomplete_df = pl.DataFrame({
        "image_id": ["seq1"],
        "loop": [1],
        "frame_id": [1],
        # Missing track_id and other required columns
    })
    
    result = tracking_stability_evaluator.evaluate(incomplete_df)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]


def test_tracking_stability_multiple_sequences(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test tracking stability with multiple sequences."""
    # Create test data with multiple sequences
    test_data = {
        "image_id": ["seq1", "seq1", "seq2", "seq2", "seq1", "seq1", "seq2", "seq2"],
        "loop": [1, 1, 1, 1, 2, 2, 2, 2],
        "frame_id": [1, 2, 1, 2, 1, 2, 1, 2],
        "track_id": [1, 1, 2, 2, 1, 1, 2, 3],  # ID switch in seq2, loop 2
        "det_cat": [0, 0, 1, 1, 0, 0, 1, 1],
        "det_score": [0.9, 0.85, 0.95, 0.9, 0.88, 0.83, 0.92, 0.87],
        "x": [100, 110, 200, 210, 102, 112, 198, 205],
        "y": [100, 105, 150, 155, 98, 103, 148, 152],
        "w": [50, 50, 40, 40, 48, 48, 42, 42],
        "h": [30, 30, 25, 25, 32, 32, 27, 27],
    }
    
    input_df = pl.DataFrame(test_data)
    result = tracking_stability_evaluator.evaluate(input_df)
    
    details_df = result.extra_data["tracking_stability_details_df"]
    
    # Should have details for both sequences
    assert len(details_df) == 2
    assert "seq1" in details_df["image_id"].to_list()
    assert "seq2" in details_df["image_id"].to_list()
    
    # Check that ID switches are detected for seq2
    seq2_details = details_df.filter(pl.col("image_id") == "seq2")
    assert seq2_details[0, "mean_id_switches_per_loop"] > 0


def test_tracking_stability_track_length_consistency(tracking_stability_evaluator: TrackingStabilityEvaluator):
    """Test track length consistency across loops."""
    # Create test data where track lengths vary across loops
    test_data = {
        "image_id": ["seq1"] * 8,
        "loop": [1, 1, 1, 1, 2, 2, 2, 2],
        "frame_id": [1, 2, 3, 4, 1, 2, 3, 4],
        "track_id": [1, 1, 1, 1, 1, 1, 2, 2],  # Track 1 full in loop 1, partial in loop 2
        "det_cat": [0] * 8,
        "det_score": [0.9, 0.85, 0.8, 0.75, 0.88, 0.83, 0.78, 0.73],
        "x": [100, 110, 120, 130, 102, 112, 140, 150],
        "y": [100, 105, 110, 115, 98, 103, 125, 130],
        "w": [50] * 8,
        "h": [30] * 8,
    }
    
    input_df = pl.DataFrame(test_data)
    result = tracking_stability_evaluator.evaluate(input_df)
    
    # Should detect variation in track lengths
    assert result.metrics["track_stab_mean_track_length_std"] > 0