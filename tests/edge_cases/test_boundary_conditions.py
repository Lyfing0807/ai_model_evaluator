"""
Edge case and boundary condition tests for the AI evaluation tool.
Tests various error conditions, malformed inputs, and edge cases.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from ai_eval_tool.config_manager import MainConfig, load_config
from ai_eval_tool.data_loader import DataLoader
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.evaluators.stability.classification import (
    ClassificationStabilityEvaluator,
)
from ai_eval_tool.evaluators.stability.detection import DetectionStabilityEvaluator
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for edge case tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataLoaderEdgeCases:
    """Test edge cases for data loading."""

    def test_empty_csv_file(self, temp_workspace):
        """Test loading an empty CSV file."""
        empty_csv = temp_workspace / "empty.csv"
        empty_csv.write_text("")

        config_content = """
project_info:
  project_name: "Empty CSV Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
evaluation_params:
  detection:
    iou_threshold: 0.5
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)

        with pytest.raises(Exception):  # Should raise an exception for empty CSV
            data_loader.load_data(empty_csv)

    def test_csv_with_only_headers(self, temp_workspace):
        """Test CSV file with only headers, no data rows."""
        headers_only_csv = temp_workspace / "headers_only.csv"
        headers_only_csv.write_text("loop_id,img_name,x,y,w,h\n")

        config_content = """
project_info:
  project_name: "Headers Only Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)
        df = data_loader.load_data(headers_only_csv)

        assert df.shape[0] == 0  # Should have 0 rows
        assert len(df.columns) > 0  # But should have columns

    def test_csv_with_missing_columns(self, temp_workspace):
        """Test CSV file missing required columns."""
        incomplete_csv = temp_workspace / "incomplete.csv"
        incomplete_csv.write_text("loop_id,img_name\n1,img1.jpg\n")

        config_content = """
project_info:
  project_name: "Missing Columns Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]  # These columns don't exist
      category_id: "cat_id"
      score: "score"
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)

        with pytest.raises(ValueError, match="Missing bbox columns"):
            data_loader.load_data(incomplete_csv)

    def test_csv_with_malformed_data(self, temp_workspace):
        """Test CSV with malformed numeric data."""
        malformed_csv = temp_workspace / "malformed.csv"
        malformed_csv.write_text(
            """loop_id,img_name,x,y,w,h,score
1,img1.jpg,abc,def,50,50,0.9
2,img2.jpg,100,100,ghi,jkl,invalid_score
"""
        )

        config_content = """
project_info:
  project_name: "Malformed Data Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
      score: "score"
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)

        # Should handle malformed data gracefully by casting with strict=False
        df = data_loader.load_data(malformed_csv)
        assert df.shape[0] == 2  # Should still load the rows
        # Non-numeric values should become null
        assert df["x"].null_count() > 0 or df["w"].null_count() > 0

    def test_csv_with_null_values(self, temp_workspace):
        """Test CSV with null/missing values."""
        null_csv = temp_workspace / "null_values.csv"
        null_csv.write_text(
            """loop_id,img_name,x,y,w,h,score
1,img1.jpg,100,,50,50,0.9
2,,150,150,60,40,
3,img3.jpg,200,200,,30,0.8
"""
        )

        config_content = """
project_info:
  project_name: "Null Values Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
      score: "score"
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)
        df = data_loader.load_data(null_csv)

        assert df.shape[0] == 3
        # Should handle null values gracefully
        assert df["y"].null_count() >= 1
        assert df["w"].null_count() >= 1

    def test_nonexistent_csv_file(self, temp_workspace):
        """Test loading a non-existent CSV file."""
        config_content = """
project_info:
  project_name: "Nonexistent File Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  detection:
    iou_threshold: 0.5
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)

        nonexistent_file = temp_workspace / "does_not_exist.csv"
        with pytest.raises(Exception):
            data_loader.load_data(nonexistent_file)


class TestConfigurationEdgeCases:
    """Test edge cases for configuration loading."""

    def test_invalid_yaml_syntax(self, temp_workspace):
        """Test loading config with invalid YAML syntax."""
        invalid_yaml = temp_workspace / "invalid.yaml"
        invalid_yaml.write_text(
            """
project_info:
  project_name: "Invalid YAML Test"
  model_type: detection  # Missing quotes
data_loader:
  field_mapping:
    loop: loop_id
    - invalid_list_item  # Invalid syntax
"""
        )

        with pytest.raises(Exception):
            load_config(invalid_yaml)

    def test_missing_required_fields(self, temp_workspace):
        """Test config missing required fields."""
        incomplete_config = temp_workspace / "incomplete.yaml"
        incomplete_config.write_text(
            """
project_info:
  project_name: "Incomplete Config Test"
  # Missing model_type
data_loader:
  field_mapping: {}
# Missing evaluation_params and report_settings
"""
        )

        with pytest.raises(Exception):
            load_config(incomplete_config)

    def test_invalid_model_type(self, temp_workspace):
        """Test config with invalid model type."""
        invalid_model_config = temp_workspace / "invalid_model.yaml"
        invalid_model_config.write_text(
            """
project_info:
  project_name: "Invalid Model Type Test"
  model_type: "unsupported_model_type"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params: {}
report_settings:
  output_dir: "output"
"""
        )

        config = load_config(invalid_model_config)
        # Should load but warn about unsupported model type
        assert config.project_info.model_type == "unsupported_model_type"


class TestEvaluatorEdgeCases:
    """Test edge cases for evaluators."""

    def test_detection_evaluator_empty_data(self):
        """Test detection evaluator with empty data."""
        config = Mock(spec=MainConfig)
        config.evaluation_params = Mock()
        config.evaluation_params.detection = Mock()
        config.evaluation_params.detection.iou_threshold = 0.5

        evaluator = DetectionStabilityEvaluator(config)

        empty_df = pl.DataFrame(
            {
                "image_id": [],
                "loop": [],
                "internal_bbox": [],
                "category_id": [],
                "score": [],
            },
            schema={
                "image_id": pl.Utf8,
                "loop": pl.Int64,
                "internal_bbox": pl.List(pl.Float64),
                "category_id": pl.Int64,
                "score": pl.Float64,
            },
        )

        result = evaluator.evaluate(empty_df)
        assert "warning" in result.metrics
        assert "Empty input data" in result.metrics["warning"]

    def test_detection_evaluator_single_image_single_loop(self):
        """Test detection evaluator with only one image and one loop."""
        config = Mock(spec=MainConfig)
        config.evaluation_params = Mock()
        config.evaluation_params.detection = Mock()
        config.evaluation_params.detection.iou_threshold = 0.5

        evaluator = DetectionStabilityEvaluator(config)

        single_df = pl.DataFrame(
            {
                "image_id": ["img1"],
                "loop": [1],
                "internal_bbox": [[[100.0, 100.0, 150.0, 150.0]]],
                "category_id": [0],
                "score": [0.9],
            }
        )

        result = evaluator.evaluate(single_df)
        # Should handle single loop gracefully
        assert "det_stab_mean_iou_consistency" in result.metrics
        assert (
            result.metrics["det_stab_mean_iou_consistency"] == 1.0
        )  # Perfect consistency with single loop

    def test_classification_evaluator_malformed_top_k_data(self):
        """Test classification evaluator with malformed top-k data."""
        config = Mock(spec=MainConfig)
        config.evaluation_params = Mock()
        config.evaluation_params.classification = Mock()
        config.evaluation_params.classification.top_k = [1, 3]

        evaluator = ClassificationStabilityEvaluator(config)

        # Test with mismatched list lengths
        malformed_df = pl.DataFrame(
            {
                "image_id": ["img1", "img1"],
                "loop": [1, 2],
                "top_k_labels": [["cat", "dog"], ["bird"]],  # Different lengths
                "top_k_scores": [[0.9, 0.8], [0.95, 0.85]],  # Different lengths
            }
        )

        result = evaluator.evaluate(malformed_df)
        # Should handle gracefully, possibly with warnings
        assert "cls_stab_mean_top_1_consistency_rate" in result.metrics

    def test_evaluator_with_null_bboxes(self):
        """Test detection evaluator with null bounding boxes."""
        config = Mock(spec=MainConfig)
        config.evaluation_params = Mock()
        config.evaluation_params.detection = Mock()
        config.evaluation_params.detection.iou_threshold = 0.5

        evaluator = DetectionStabilityEvaluator(config)

        null_bbox_df = pl.DataFrame(
            {
                "image_id": ["img1", "img1"],
                "loop": [1, 2],
                "internal_bbox": [
                    [[100.0, 100.0, 150.0, 150.0]],
                    None,
                ],  # One null bbox
                "category_id": [0, 0],
                "score": [0.9, 0.8],
            }
        )

        result = evaluator.evaluate(null_bbox_df)
        # Should handle null bboxes gracefully
        assert "det_stab_mean_iou_consistency" in result.metrics


class TestMemoryAndPerformanceEdgeCases:
    """Test memory and performance edge cases."""

    def test_very_large_dataset_simulation(self, temp_workspace):
        """Test handling of very large dataset (simulated)."""
        # Create a moderately large CSV for testing
        large_csv = temp_workspace / "large_dataset.csv"

        # Generate 10,000 rows of data
        rows = ["loop_id,img_name,x,y,w,h,score"]
        for i in range(10000):
            loop_id = (i % 5) + 1
            img_id = f"img_{i % 1000}.jpg"
            x, y, w, h = 100 + (i % 200), 100 + (i % 200), 50, 50
            score = 0.8 + (i % 20) * 0.01
            rows.append(f"{loop_id},{img_id},{x},{y},{w},{h},{score}")

        large_csv.write_text("\n".join(rows))

        config_content = """
project_info:
  project_name: "Large Dataset Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
      score: "score"
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)

        # Should handle large dataset without crashing
        df = data_loader.load_data(large_csv)
        assert df.shape[0] == 10000
        assert "internal_bbox" in df.columns

    def test_memory_optimization_with_extreme_values(self, temp_workspace):
        """Test memory optimization with extreme numeric values."""
        extreme_csv = temp_workspace / "extreme_values.csv"
        extreme_csv.write_text(
            """loop_id,img_name,x,y,w,h,score
1,img1.jpg,999999999,999999999,50,50,0.9
2,img2.jpg,-999999999,-999999999,60,40,0.8
3,img3.jpg,0,0,1,1,0.1
"""
        )

        config_content = """
project_info:
  project_name: "Extreme Values Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
      score: "score"
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        data_loader = DataLoader(config)
        df = data_loader.load_data(extreme_csv)

        # Should handle extreme values without overflow
        assert df.shape[0] == 3
        assert "internal_bbox" in df.columns


class TestEngineEdgeCases:
    """Test edge cases for the evaluation engine."""

    def test_engine_with_corrupted_cache(self, temp_workspace):
        """Test engine behavior with corrupted cache files."""
        config_content = """
project_info:
  project_name: "Corrupted Cache Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  detection:
    iou_threshold: 0.5
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        engine = EvaluationEngine(config)

        # Create corrupted cache files
        cache_dir = temp_workspace / "cache" / "test_run"
        cache_dir.mkdir(parents=True)

        # Write corrupted JSON
        (cache_dir / "metrics.json").write_text("invalid json content {")
        (cache_dir / "extra_data.pkl").write_bytes(b"corrupted pickle data")

        with pytest.raises(Exception):
            engine.deserialize_results(temp_workspace / "cache", "test_run")

    def test_engine_with_missing_evaluator(self, temp_workspace):
        """Test engine behavior with unsupported model type."""
        config_content = """
project_info:
  project_name: "Unsupported Model Test"
  model_type: "unsupported_model"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
evaluation_params: {}
report_settings:
  output_dir: "output"
"""
        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        engine = EvaluationEngine(config)

        # Create minimal test data
        test_df = pl.DataFrame({"image_id": ["img1"], "loop": [1]})

        # Should handle unsupported model type gracefully
        result = engine.run(test_df)
        assert (
            "perf_mean_fps" in result.metrics
        )  # Performance metrics should still work


class TestReportingEdgeCases:
    """Test edge cases for report generation."""

    def test_report_generation_with_no_data(self, temp_workspace):
        """Test report generation with empty results."""
        from ai_eval_tool.reporting.generator import ReportGenerator

        config_content = f"""
project_info:
  project_name: "Empty Results Test"
  model_type: "detection"
data_loader:
  field_mapping: {temp_workspace}
evaluation_params: {temp_workspace}
report_settings:
  output_dir: "{temp_workspace}"
  formats: ["markdown"]
"""

        config_path = temp_workspace / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        # Create empty evaluation result
        empty_result = EvaluationResult(metrics={}, extra_data={})

        report_generator = ReportGenerator(config)

        # Should handle empty results gracefully
        try:
            report_generator.generate(empty_result, "empty_test")
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # Should not crash, but if it does, it should be a handled exception
            assert "empty" in str(e).lower() or "no data" in str(e).lower()


@pytest.mark.parametrize("invalid_input", [None, "", "not_a_path", 123, [], {}])
def test_invalid_file_paths(invalid_input, temp_workspace):
    """Test various invalid file path inputs."""
    config_content = """
project_info:
  project_name: "Invalid Path Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  detection:
    iou_threshold: 0.5
report_settings:
  output_dir: "output"
"""
    config_path = temp_workspace / "config.yaml"
    config_path.write_text(config_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)

    with pytest.raises(Exception):
        data_loader.load_data(invalid_input)


def test_concurrent_access_simulation(temp_workspace):
    """Test simulation of concurrent access to the same files."""
    # This is a simplified test since true concurrency testing is complex
    csv_content = """loop_id,img_name,x,y,w,h
1,img1.jpg,100,100,50,50
2,img2.jpg,150,150,60,40
"""
    csv_path = temp_workspace / "concurrent_test.csv"
    csv_path.write_text(csv_content)

    config_content = """
project_info:
  project_name: "Concurrent Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    detection:
      bbox: ["x", "y", "w", "h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings:
  output_dir: "output"
"""
    config_path = temp_workspace / "config.yaml"
    config_path.write_text(config_content)

    config = load_config(config_path)

    # Simulate multiple data loaders accessing the same file
    data_loader1 = DataLoader(config)
    data_loader2 = DataLoader(config)

    df1 = data_loader1.load_data(csv_path)
    df2 = data_loader2.load_data(csv_path)

    # Both should succeed and produce identical results
    assert df1.shape == df2.shape
    assert df1.columns == df2.columns
