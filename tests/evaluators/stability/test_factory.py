"""Tests for stability evaluator factory."""

import pytest
import polars as pl

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.factory import (
    StabilityEvaluatorFactory,
    DummyDetectionEvaluator,
    DummyClassificationEvaluator,
)
from ai_eval_tool.evaluators.base import StabilityEvaluatorBase
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture
def dummy_config(test_data_dir) -> MainConfig:
    """Create a dummy config for testing factory."""
    config_data = {
        "project_info": {
            "project_name": "Factory Test Project",
            "model_type": "dummy_detection",
            "run_id": "factory_test_run_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop_id",
                "image_id": "img_name",
            }
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_factory")},
    }
    return MainConfig(**config_data)


class TestEvaluator(StabilityEvaluatorBase):
    """Test evaluator for factory testing."""
    
    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        return EvaluationResult(metrics={"test_metric": 1.0})


def test_factory_register_evaluator():
    """Test registering a new evaluator."""
    # Register a test evaluator
    StabilityEvaluatorFactory.register_evaluator("test_model", TestEvaluator)
    
    # Check that it's in the registry
    assert "test_model" in StabilityEvaluatorFactory._REGISTRY
    assert StabilityEvaluatorFactory._REGISTRY["test_model"] == TestEvaluator


def test_factory_get_evaluator(dummy_config: MainConfig):
    """Test getting an evaluator from factory."""
    # Ensure dummy evaluators are registered
    StabilityEvaluatorFactory.register_evaluator("dummy_detection", DummyDetectionEvaluator)
    
    # Get evaluator
    evaluator = StabilityEvaluatorFactory.get_evaluator("dummy_detection", dummy_config)
    
    assert isinstance(evaluator, DummyDetectionEvaluator)
    assert evaluator.config == dummy_config


def test_factory_get_unsupported_evaluator(dummy_config: MainConfig):
    """Test getting an unsupported evaluator raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        StabilityEvaluatorFactory.get_evaluator("unsupported_model_type", dummy_config)
    
    assert "Unsupported model type" in str(exc_info.value)
    assert "unsupported_model_type" in str(exc_info.value)


def test_factory_overwrite_evaluator():
    """Test overwriting an existing evaluator registration."""
    # Register initial evaluator
    StabilityEvaluatorFactory.register_evaluator("overwrite_test", TestEvaluator)
    assert StabilityEvaluatorFactory._REGISTRY["overwrite_test"] == TestEvaluator
    
    # Register a different evaluator with same key
    StabilityEvaluatorFactory.register_evaluator("overwrite_test", DummyDetectionEvaluator)
    assert StabilityEvaluatorFactory._REGISTRY["overwrite_test"] == DummyDetectionEvaluator


def test_dummy_detection_evaluator(dummy_config: MainConfig):
    """Test dummy detection evaluator functionality."""
    evaluator = DummyDetectionEvaluator(dummy_config)
    
    # Create dummy data
    test_df = pl.DataFrame({
        "image_id": ["img1", "img2"],
        "loop": [1, 1],
    })
    
    result = evaluator.evaluate(test_df)
    
    assert isinstance(result, EvaluationResult)
    assert "detection_stability" in result.metrics
    assert result.metrics["detection_stability"] == 1.0


def test_dummy_classification_evaluator(dummy_config: MainConfig):
    """Test dummy classification evaluator functionality."""
    # Update config for classification
    dummy_config.project_info.model_type = "dummy_classification"
    evaluator = DummyClassificationEvaluator(dummy_config)
    
    # Create dummy data
    test_df = pl.DataFrame({
        "image_id": ["img1", "img2"],
        "loop": [1, 1],
    })
    
    result = evaluator.evaluate(test_df)
    
    assert isinstance(result, EvaluationResult)
    assert "classification_stability" in result.metrics
    assert result.metrics["classification_stability"] == 1.0


def test_factory_registry_state():
    """Test that factory registry maintains state correctly."""
    initial_count = len(StabilityEvaluatorFactory._REGISTRY)
    
    # Register new evaluator
    StabilityEvaluatorFactory.register_evaluator("registry_test", TestEvaluator)
    
    # Check count increased
    assert len(StabilityEvaluatorFactory._REGISTRY) == initial_count + 1
    
    # Check specific evaluator exists
    assert "registry_test" in StabilityEvaluatorFactory._REGISTRY


def test_factory_available_types():
    """Test getting available evaluator types."""
    # Ensure some evaluators are registered
    StabilityEvaluatorFactory.register_evaluator("test_type_1", TestEvaluator)
    StabilityEvaluatorFactory.register_evaluator("test_type_2", DummyDetectionEvaluator)
    
    available_types = list(StabilityEvaluatorFactory._REGISTRY.keys())
    
    assert "test_type_1" in available_types
    assert "test_type_2" in available_types
    assert len(available_types) >= 2