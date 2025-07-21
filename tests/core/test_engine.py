import json  # For engine serialization tests
from pathlib import Path

import polars as pl
import pytest
import yaml

from ai_eval_tool.config_manager import MainConfig, load_config
from ai_eval_tool.engine import EvaluationEngine

# Import dummy evaluators that might be registered in factory's __main__ or conftest
from ai_eval_tool.evaluators.stability.factory import (
    DummyClassificationEvaluator,
    DummyDetectionEvaluator,
    StabilityEvaluatorFactory,
)
from ai_eval_tool.utils.types import EvaluationResult


def assert_frame_equal(df1: pl.DataFrame, df2: pl.DataFrame):
    """Helper function to assert that two Polars DataFrames are equal."""
    assert df1.frame_equal(df2), f"DataFrames are not equal.\nDF1:\n{df1}\nDF2:\n{df2}"


# Fixture to ensure dummy evaluators are registered if not already by module import
@pytest.fixture(autouse=True)  # Autouse to ensure it runs for all tests in this module
def register_dummy_evaluators_for_engine_tests():
    # This is a bit of a workaround if factory doesn't always have dummies registered
    # when tests run. Ideally, factory manages its state or tests explicitly register.
    if "detection" not in StabilityEvaluatorFactory._REGISTRY:
        StabilityEvaluatorFactory.register_evaluator(
            "detection", DummyDetectionEvaluator
        )
    if "classification" not in StabilityEvaluatorFactory._REGISTRY:
        StabilityEvaluatorFactory.register_evaluator(
            "classification", DummyClassificationEvaluator
        )


@pytest.fixture
def engine_detection(detection_config: MainConfig) -> EvaluationEngine:
    return EvaluationEngine(detection_config)


@pytest.fixture
def engine_classification(classification_config: MainConfig) -> EvaluationEngine:
    return EvaluationEngine(classification_config)


@pytest.fixture
def sample_input_df() -> pl.DataFrame:
    # A generic DataFrame that can be used by performance and dummy stability evaluators
    return pl.DataFrame(
        {
            "loop": [1, 1, 2, 1],
            "image_id": ["img1", "img2", "img1", "img1"],
            "pre_time_ms": [10.0, 12.0, 11.0, 10.0],
            "inference_time_ms": [100.0, 110.0, 105.0, 100.0],
            "post_time_ms": [5.0, 6.0, 5.0, 5.0],
            "total_time_ms": [115.0, 128.0, 121.0, 115.0],
            # Dummy fields for specific evaluators if they check for columns
            "category_id": [0, 1, 0, 0],
            "score": [0.9, 0.8, 0.85, 0.9],
            "internal_bbox": [[[1, 1, 1, 1]]] * 4,  # Placeholder list of lists
            "top_k_labels": [[["l1"]]] * 4,
            "top_k_scores": [[[0.9]]] * 4,  # Placeholder
        }
    )


def test_engine_run_detection_model(
    engine_detection: EvaluationEngine, sample_input_df: pl.DataFrame
):
    results = engine_detection.run(sample_input_df)

    assert isinstance(results, EvaluationResult)
    assert "perf_mean_total_time_ms" in results.metrics  # From PerformanceEvaluator
    assert "detection_stability" in results.metrics  # From DummyDetectionEvaluator

    # Check if deduplicated_perf_df_for_charts is in extra_data (added by PerformanceEvaluator)
    assert "deduplicated_perf_df_for_charts" in results.extra_data
    assert isinstance(
        results.extra_data["deduplicated_perf_df_for_charts"], pl.DataFrame
    )


def test_engine_run_classification_model(
    engine_classification: EvaluationEngine, sample_input_df: pl.DataFrame
):
    results = engine_classification.run(sample_input_df)

    assert isinstance(results, EvaluationResult)
    assert "perf_mean_total_time_ms" in results.metrics
    assert (
        "classification_stability" in results.metrics
    )  # From DummyClassificationEvaluator


def test_engine_unsupported_model_type(
    test_data_dir: Path, sample_input_df: pl.DataFrame
):
    config_content = {
        "project_info": {
            "project_name": "Unsupported Test",
            "model_type": "super_pose_estimator",
        },
        "data_loader": {"field_mapping": {"loop": "loop"}},  # Minimal
        "report_settings": {"output_dir": str(test_data_dir / "reports_unsupported")},
    }
    config_file = test_data_dir / "unsupported_model_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_unsupported").mkdir(parents=True, exist_ok=True)

    config = load_config(config_file)
    engine = EvaluationEngine(config)

    results = engine.run(sample_input_df)  # Should still run performance part
    assert "perf_mean_total_time_ms" in results.metrics
    # No stability metrics for unsupported type, and a warning should have been logged by engine
    assert "classification_stability" not in results.metrics
    assert "detection_stability" not in results.metrics
    # Check logs or add a specific warning metric if engine is modified to do so


def test_engine_serialization_deserialization(
    engine_detection: EvaluationEngine,
    sample_input_df: pl.DataFrame,
    temp_cache_dir: Path,
):
    original_results = engine_detection.run(sample_input_df)

    # Add a DataFrame to extra_data for thorough testing of Parquet part
    original_results.extra_data["test_df_for_cache"] = pl.DataFrame(
        {"colA": [10, 20], "colB": ["x", "y"]}
    )

    run_id = engine_detection.config.project_info.run_id or "test_cache_run"

    engine_detection.serialize_results(original_results, temp_cache_dir, run_id)

    # Check if files were created
    run_cache_path = temp_cache_dir / run_id
    assert (run_cache_path / "results.json").exists()
    assert (
        run_cache_path / "extra_data_deduplicated_perf_df_for_charts.parquet"
    ).exists()
    assert (run_cache_path / "extra_data_test_df_for_cache.parquet").exists()

    deserialized_results = engine_detection.deserialize_results(temp_cache_dir, run_id)

    assert isinstance(deserialized_results, EvaluationResult)
    assert original_results.metrics == deserialized_results.metrics

    # Compare DataFrames in extra_data
    original_df_perf = original_results.extra_data["deduplicated_perf_df_for_charts"]
    deserialized_df_perf = deserialized_results.extra_data[
        "deduplicated_perf_df_for_charts"
    ]
    assert_frame_equal(original_df_perf, deserialized_df_perf)

    original_df_test = original_results.extra_data["test_df_for_cache"]
    deserialized_df_test = deserialized_results.extra_data["test_df_for_cache"]
    assert_frame_equal(original_df_test, deserialized_df_test)


def test_engine_deserialize_non_existent_cache(
    engine_detection: EvaluationEngine, temp_cache_dir: Path
):
    with pytest.raises(FileNotFoundError):
        engine_detection.deserialize_results(temp_cache_dir, "non_existent_run_id")


def test_engine_deserialize_corrupted_json(
    engine_detection: EvaluationEngine, temp_cache_dir: Path
):
    run_id = "corrupted_json_run"
    run_cache_path = temp_cache_dir / run_id
    run_cache_path.mkdir(parents=True, exist_ok=True)
    with open(run_cache_path / "results.json", "w") as f:
        f.write("{this_is_not_valid_json,,,")

    with pytest.raises(json.JSONDecodeError):  # Or whatever error json.load throws
        engine_detection.deserialize_results(temp_cache_dir, run_id)
