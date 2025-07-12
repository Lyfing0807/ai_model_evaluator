"""
End-to-end integration tests for the AI evaluation tool.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import polars as pl

from ai_eval_tool.config_manager import load_config
from ai_eval_tool.data_loader import DataLoader
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.reporting.generator import ReportGenerator


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for integration tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_detection_data(temp_workspace):
    """Create sample detection data CSV."""
    data = {
        "loop_id": [1, 1, 2, 2] * 3,
        "img_name": ["img1.jpg", "img2.jpg"] * 6,
        "rel_path": ["imgs/img1.jpg", "imgs/img2.jpg"] * 6,
        "t_pre": [10, 12] * 6,
        "t_inf": [20, 22] * 6,
        "t_post": [5, 6] * 6,
        "t_total": [35, 40] * 6,
        "det_cat": [0, 1] * 6,
        "det_score": [0.9, 0.8, 0.88, 0.82] * 3,
        "x": [100, 150] * 6,
        "y": [100, 150] * 6,
        "w": [50, 60] * 6,
        "h": [50, 40] * 6,
    }
    
    df = pl.DataFrame(data)
    csv_path = temp_workspace / "detection_data.csv"
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture
def detection_config(temp_workspace):
    """Create detection model configuration."""
    config_content = f"""
project_info:
  project_name: "Integration Test Detection"
  model_type: "detection"
  run_id: "integration-test-001"

data_loader:
  image_base_dir: "{temp_workspace}"
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    image_path: "rel_path"
    pre_time_ms: "t_pre"
    inference_time_ms: "t_inf"
    post_time_ms: "t_post"
    total_time_ms: "t_total"
    detection:
      category_id: "det_cat"
      score: "det_score"
      bbox: ["x", "y", "w", "h"]

evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"

ai_insights:
  enabled: false

report_settings:
  output_dir: "{temp_workspace / 'output'}"
  formats: ["markdown"]
  include_charts: true
"""
    
    config_path = temp_workspace / "config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestEndToEndDetection:
    """End-to-end tests for detection model evaluation."""
    
    def test_full_detection_pipeline(self, detection_config, sample_detection_data, temp_workspace):
        """Test the complete detection evaluation pipeline."""
        # Load configuration
        config = load_config(detection_config)
        assert config.project_info.model_type == "detection"
        
        # Load data
        data_loader = DataLoader(config)
        data_df = data_loader.load_data(sample_detection_data)
        
        # Verify data loading
        assert data_df.shape[0] > 0
        assert "internal_bbox" in data_df.columns
        assert "category_id" in data_df.columns
        assert "score" in data_df.columns
        
        # Run evaluation
        engine = EvaluationEngine(config)
        results = engine.run(data_df)
        
        # Verify evaluation results
        assert "perf_mean_fps" in results.metrics
        assert "perf_mean_inference_time_ms" in results.metrics
        assert "det_stab_mean_iou_consistency" in results.metrics
        assert "det_stab_mean_bbox_drift" in results.metrics
        
        # Verify performance metrics are reasonable
        assert results.metrics["perf_mean_fps"] > 0
        assert results.metrics["perf_mean_inference_time_ms"] > 0
        
        # Verify stability metrics are in expected range
        assert 0 <= results.metrics["det_stab_mean_iou_consistency"] <= 1
        assert results.metrics["det_stab_mean_bbox_drift"] >= 0
        
        # Generate report
        report_generator = ReportGenerator(config)
        report_generator.generate(results, config.project_info.run_id)
        
        # Verify report files exist
        output_dir = Path(config.report_settings.output_dir)
        assert (output_dir / "report.md").exists()
        
        # Verify report content
        report_content = (output_dir / "report.md").read_text()
        assert "Integration Test Detection" in report_content
        assert "Performance Metrics" in report_content
        assert "Stability Analysis" in report_content
    
    def test_memory_optimization_large_dataset(self, detection_config, temp_workspace):
        """Test memory optimization with a larger dataset."""
        # Create a larger dataset
        large_data = {
            "loop_id": list(range(1, 6)) * 1000,  # 5000 rows
            "img_name": [f"img{i}.jpg" for i in range(1000)] * 5,
            "rel_path": [f"imgs/img{i}.jpg" for i in range(1000)] * 5,
            "t_pre": [10 + i % 5 for i in range(5000)],
            "t_inf": [20 + i % 10 for i in range(5000)],
            "t_post": [5 + i % 3 for i in range(5000)],
            "t_total": [35 + i % 15 for i in range(5000)],
            "det_cat": [i % 3 for i in range(5000)],
            "det_score": [0.8 + (i % 20) * 0.01 for i in range(5000)],
            "x": [100 + i % 200 for i in range(5000)],
            "y": [100 + i % 200 for i in range(5000)],
            "w": [50 + i % 50 for i in range(5000)],
            "h": [50 + i % 50 for i in range(5000)],
        }
        
        df = pl.DataFrame(large_data)
        large_csv_path = temp_workspace / "large_detection_data.csv"
        df.write_csv(large_csv_path)
        
        # Load and process large dataset
        config = load_config(detection_config)
        data_loader = DataLoader(config)
        data_df = data_loader.load_data(large_csv_path)
        
        # Verify data was loaded and optimized
        assert data_df.shape[0] == 5000
        assert "internal_bbox" in data_df.columns
        
        # Run evaluation on large dataset
        engine = EvaluationEngine(config)
        results = engine.run(data_df)
        
        # Verify results are still valid
        assert "perf_mean_fps" in results.metrics
        assert "det_stab_mean_iou_consistency" in results.metrics
        assert results.metrics["perf_mean_fps"] > 0


class TestCacheAndSerialization:
    """Test caching and serialization functionality."""
    
    def test_serialize_deserialize_results(self, detection_config, sample_detection_data, temp_workspace):
        """Test serialization and deserialization of evaluation results."""
        config = load_config(detection_config)
        data_loader = DataLoader(config)
        data_df = data_loader.load_data(sample_detection_data)
        
        engine = EvaluationEngine(config)
        original_results = engine.run(data_df)
        
        # Serialize results
        cache_path = temp_workspace / "cache"
        run_id = "test-serialize-001"
        engine.serialize_results(original_results, cache_path, run_id)
        
        # Verify cache files exist
        cache_dir = cache_path / run_id
        assert cache_dir.exists()
        assert (cache_dir / "metrics.json").exists()
        assert (cache_dir / "extra_data.pkl").exists()
        
        # Deserialize results
        deserialized_results = engine.deserialize_results(cache_path, run_id)
        
        # Verify deserialized results match original
        assert deserialized_results.metrics == original_results.metrics
        assert len(deserialized_results.extra_data) == len(original_results.extra_data)
        
        # Verify DataFrames are preserved correctly
        for key in original_results.extra_data:
            if isinstance(original_results.extra_data[key], pl.DataFrame):
                original_df = original_results.extra_data[key]
                deserialized_df = deserialized_results.extra_data[key]
                assert original_df.shape == deserialized_df.shape
                assert original_df.columns == deserialized_df.columns


@pytest.fixture
def classification_data(temp_workspace):
    """Create sample classification data."""
    data = {
        "loop_id": [1, 1, 2, 2] * 2,
        "img_name": ["imgA.jpg", "imgB.jpg"] * 4,
        "rel_path": ["imgs/imgA.jpg", "imgs/imgB.jpg"] * 4,
        "t_pre": [8, 9] * 4,
        "t_inf": [15, 16] * 4,
        "t_post": [3, 4] * 4,
        "t_total": [26, 29] * 4,
        "class_top_1_id": ["cat", "bird", "cat", "bird"] * 2,
        "class_top_1_score": [0.9, 0.95, 0.88, 0.93] * 2,
        "class_top_3_id": ["dog", "fish", "fox", "fish"] * 2,
        "class_top_3_score": [0.8, 0.85, 0.75, 0.82] * 2,
    }
    
    df = pl.DataFrame(data)
    csv_path = temp_workspace / "classification_data.csv"
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture
def classification_config(temp_workspace):
    """Create classification model configuration."""
    config_content = f"""
project_info:
  project_name: "Integration Test Classification"
  model_type: "classification"
  run_id: "integration-test-cls-001"

data_loader:
  image_base_dir: "{temp_workspace}"
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    image_path: "rel_path"
    pre_time_ms: "t_pre"
    inference_time_ms: "t_inf"
    post_time_ms: "t_post"
    total_time_ms: "t_total"
    classification:
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"

evaluation_params:
  classification:
    top_k: [1, 3]

ai_insights:
  enabled: false

report_settings:
  output_dir: "{temp_workspace / 'output'}"
  formats: ["markdown"]
  include_charts: true
"""
    
    config_path = temp_workspace / "cls_config.yaml"
    config_path.write_text(config_content)
    return config_path


class TestEndToEndClassification:
    """End-to-end tests for classification model evaluation."""
    
    def test_full_classification_pipeline(self, classification_config, classification_data, temp_workspace):
        """Test the complete classification evaluation pipeline."""
        config = load_config(classification_config)
        data_loader = DataLoader(config)
        data_df = data_loader.load_data(classification_data)
        
        # Verify classification-specific data processing
        assert "top_k_labels" in data_df.columns
        assert "top_k_scores" in data_df.columns
        
        # Run evaluation
        engine = EvaluationEngine(config)
        results = engine.run(data_df)
        
        # Verify classification-specific metrics
        assert "cls_stab_mean_top_1_consistency_rate" in results.metrics
        assert "cls_stab_mean_jaccard_top_1" in results.metrics
        assert "cls_stab_mean_jaccard_top_3" in results.metrics
        
        # Generate report
        report_generator = ReportGenerator(config)
        report_generator.generate(results, config.project_info.run_id)
        
        # Verify report
        output_dir = Path(config.report_settings.output_dir)
        assert (output_dir / "report.md").exists()
        
        report_content = (output_dir / "report.md").read_text()
        assert "Classification" in report_content
        assert "Top-K Consistency" in report_content