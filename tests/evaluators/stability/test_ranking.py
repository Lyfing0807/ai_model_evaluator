"""Tests for ranking stability evaluator."""

import polars as pl
import pytest

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.ranking import (
    RankingStabilityEvaluator,
    kendall_tau_distance,
    spearman_rank_correlation,
)
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def ranking_config(test_data_dir) -> MainConfig:
    """Create a ranking config for testing."""
    config_data = {
        "project_info": {
            "project_name": "Test Ranking Project",
            "model_type": "ranking",
            "run_id": "test_ranking_run_001",
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
                "ranking": {
                    "item_ids": "item_ids",
                    "scores": "scores",
                    "ranks": "ranks",
                },
            }
        },
        "evaluation_params": {
            "ranking": {
                "top_k": [5, 10],
                "correlation_method": "spearman",
            }
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_ranking")},
    }
    return MainConfig(**config_data)


@pytest.fixture
def ranking_stability_evaluator(ranking_config: MainConfig) -> RankingStabilityEvaluator:
    """Create ranking stability evaluator instance."""
    return RankingStabilityEvaluator(ranking_config)


def test_kendall_tau_distance():
    """Test Kendall tau distance calculation."""
    # Test with identical rankings
    rank1 = [1, 2, 3, 4, 5]
    rank2 = [1, 2, 3, 4, 5]
    assert kendall_tau_distance(rank1, rank2) == 0.0
    
    # Test with completely reversed rankings
    rank3 = [5, 4, 3, 2, 1]
    distance = kendall_tau_distance(rank1, rank3)
    assert distance == 1.0  # Maximum distance
    
    # Test with partially different rankings
    rank4 = [1, 3, 2, 4, 5]  # Swapped positions 2 and 3
    distance_partial = kendall_tau_distance(rank1, rank4)
    assert 0 < distance_partial < 1.0


def test_spearman_rank_correlation():
    """Test Spearman rank correlation calculation."""
    # Test with identical rankings
    rank1 = [1, 2, 3, 4, 5]
    rank2 = [1, 2, 3, 4, 5]
    assert spearman_rank_correlation(rank1, rank2) == pytest.approx(1.0)
    
    # Test with completely reversed rankings
    rank3 = [5, 4, 3, 2, 1]
    correlation = spearman_rank_correlation(rank1, rank3)
    assert correlation == pytest.approx(-1.0)
    
    # Test with uncorrelated rankings
    rank4 = [3, 1, 4, 5, 2]
    correlation_uncorr = spearman_rank_correlation(rank1, rank4)
    assert -1.0 <= correlation_uncorr <= 1.0


def test_ranking_stability_evaluator_basic(ranking_stability_evaluator: RankingStabilityEvaluator):
    """Test basic ranking stability evaluation."""
    # Create test data with ranking results
    test_data = {
        "image_id": ["query1", "query1", "query2", "query2"],
        "loop": [1, 2, 1, 2],
        "item_ids": [
            ["item1", "item2", "item3", "item4", "item5"],
            ["item1", "item3", "item2", "item4", "item5"],  # Slightly different order
            ["itemA", "itemB", "itemC", "itemD", "itemE"],
            ["itemA", "itemB", "itemD", "itemC", "itemE"],  # Slightly different order
        ],
        "scores": [
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.85, 0.75, 0.78, 0.58, 0.48],
            [0.95, 0.85, 0.75, 0.65, 0.55],
            [0.92, 0.82, 0.68, 0.72, 0.52],
        ],
        "ranks": [
            [1, 2, 3, 4, 5],
            [1, 3, 2, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 4, 3, 5],
        ],
    }
    
    input_df = pl.DataFrame(test_data)
    result = ranking_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    assert "rank_stab_mean_kendall_tau" in result.metrics
    assert "rank_stab_mean_spearman_correlation" in result.metrics
    assert "rank_stab_mean_score_std" in result.metrics
    assert "ranking_stability_details_df" in result.extra_data
    
    # Check that metrics are in reasonable ranges
    assert 0 <= result.metrics["rank_stab_mean_kendall_tau"] <= 1.0
    assert -1.0 <= result.metrics["rank_stab_mean_spearman_correlation"] <= 1.0
    assert result.metrics["rank_stab_mean_score_std"] >= 0


def test_ranking_stability_single_loop(ranking_stability_evaluator: RankingStabilityEvaluator):
    """Test ranking stability with single loop."""
    test_data = {
        "image_id": ["query1"],
        "loop": [1],
        "item_ids": [["item1", "item2", "item3"]],
        "scores": [[0.9, 0.8, 0.7]],
        "ranks": [[1, 2, 3]],
    }
    
    input_df = pl.DataFrame(test_data)
    result = ranking_stability_evaluator.evaluate(input_df)
    
    assert isinstance(result, EvaluationResult)
    # With single loop, correlation should be perfect, distance should be 0
    assert result.metrics["rank_stab_mean_kendall_tau"] == 0.0
    assert result.metrics["rank_stab_mean_spearman_correlation"] == 1.0
    assert result.metrics["rank_stab_mean_score_std"] == 0.0


def test_ranking_stability_empty_input(ranking_stability_evaluator: RankingStabilityEvaluator):
    """Test ranking stability with empty input."""
    empty_df = pl.DataFrame(
        {
            "image_id": [],
            "loop": [],
            "item_ids": [],
            "scores": [],
            "ranks": [],
        },
        schema={
            "image_id": pl.Utf8,
            "loop": pl.Int64,
            "item_ids": pl.List(pl.Utf8),
            "scores": pl.List(pl.Float64),
            "ranks": pl.List(pl.Int64),
        },
    )
    
    result = ranking_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]


def test_ranking_stability_missing_columns(ranking_stability_evaluator: RankingStabilityEvaluator):
    """Test ranking stability with missing required columns."""
    incomplete_df = pl.DataFrame({
        "image_id": ["query1"],
        "loop": [1],
        # Missing ranking-specific columns
    })
    
    result = ranking_stability_evaluator.evaluate(incomplete_df)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]


def test_ranking_stability_top_k_metrics(ranking_stability_evaluator: RankingStabilityEvaluator):
    """Test top-k specific metrics."""
    # Create test data where top-k results differ
    test_data = {
        "image_id": ["query1", "query1"],
        "loop": [1, 2],
        "item_ids": [
            ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"],
            ["item1", "item3", "item2", "item5", "item4", "item6", "item7", "item8", "item9", "item10"],
        ],
        "scores": [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            [0.85, 0.75, 0.78, 0.55, 0.58, 0.38, 0.28, 0.18, 0.08, 0.03],
        ],
        "ranks": [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 3, 2, 5, 4, 6, 7, 8, 9, 10],
        ],
    }
    
    input_df = pl.DataFrame(test_data)
    result = ranking_stability_evaluator.evaluate(input_df)
    
    # Should have top-k specific metrics based on config (top_k: [5, 10])
    details_df = result.extra_data["ranking_stability_details_df"]
    assert "top_k_kendall_tau" in details_df.columns
    assert "top_k_spearman_correlation" in details_df.columns
    
    # Check that top-k metrics are computed for each k value
    top_k_tau = details_df[0, "top_k_kendall_tau"]
    top_k_spearman = details_df[0, "top_k_spearman_correlation"]
    
    assert isinstance(top_k_tau, dict)
    assert isinstance(top_k_spearman, dict)
    assert 5 in top_k_tau
    assert 10 in top_k_tau
    assert 5 in top_k_spearman
    assert 10 in top_k_spearman