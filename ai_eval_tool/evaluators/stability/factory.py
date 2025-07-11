"""
Factory for creating model-specific stability evaluators.
"""
from typing import Type, Dict
from ...config_manager import MainConfig
from ..base import StabilityEvaluatorBase # StabilityEvaluatorBase from ..base
# Import concrete evaluators here. They will be registered.
# For now, these will be placeholders.
# from .detection import DetectionStabilityEvaluator
# from .classification import ClassificationStabilityEvaluator
# from .rotated_detection import RotatedDetectionStabilityEvaluator
# from .pose import PoseStabilityEvaluator
from ...utils.logging_config import get_logger

logger = get_logger(__name__)

class StabilityEvaluatorFactory:
    """
    Factory class to instantiate the appropriate stability evaluator based on model type.
    """
    # Registry to hold model_type -> evaluator_class mapping
    _REGISTRY: Dict[str, Type[StabilityEvaluatorBase]] = {}

    @classmethod
    def register_evaluator(cls, model_type: str, evaluator_class: Type[StabilityEvaluatorBase]):
        """
        Registers a stability evaluator class for a given model type.
        To be called by specific evaluator modules (e.g., in their __init__.py or at module level).
        Alternatively, populate _REGISTRY directly below after imports.
        """
        if model_type in cls._REGISTRY:
            logger.warning(
                f"Evaluator for model type '{model_type}' already registered. "
                f"Overwriting {cls._REGISTRY[model_type].__name__} with {evaluator_class.__name__}."
            )
        cls._REGISTRY[model_type] = evaluator_class
        logger.debug(f"Registered stability evaluator '{evaluator_class.__name__}' for model type '{model_type}'.")

    @classmethod
    def get_evaluator(cls, model_type: str, config: MainConfig) -> StabilityEvaluatorBase:
        """
        Returns an instance of the stability evaluator for the given model type.

        Args:
            model_type: The type of the model (e.g., "detection", "classification").
            config: The main configuration object.

        Returns:
            An instance of a StabilityEvaluatorBase subclass.

        Raises:
            ValueError: If no evaluator is registered for the given model type.
        """
        evaluator_class = cls._REGISTRY.get(model_type)
        if not evaluator_class:
            logger.error(f"No stability evaluator registered for model type: {model_type}")
            raise ValueError(f"Unsupported model type for stability evaluation: {model_type}. "
                             f"Available types: {list(cls._REGISTRY.keys())}")

        logger.info(f"Creating stability evaluator '{evaluator_class.__name__}' for model type '{model_type}'.")
        return evaluator_class(config)

# Placeholder: Actual evaluators will be imported and registered later.
# Example of how registration would look if done here:
# from .detection import DetectionStabilityEvaluator # Assuming this exists
# StabilityEvaluatorFactory.register_evaluator("detection", DetectionStabilityEvaluator)

# For now, let's create dummy evaluators for testing the factory mechanism
if __name__ == "__main__" or "pytest" in __import__("sys").modules: # Avoid side effects during normal import
    from ....ai_eval_tool.utils.types import EvaluationResult # Adjust path for test context if needed
    import polars as pl

    class DummyDetectionEvaluator(StabilityEvaluatorBase):
        def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
            logger.info(f"DummyDetectionEvaluator evaluating for model type {self.model_type}...")
            return EvaluationResult(metrics={"detection_stability": 1.0})

    class DummyClassificationEvaluator(StabilityEvaluatorBase):
        def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
            logger.info(f"DummyClassificationEvaluator evaluating for model type {self.model_type}...")
            return EvaluationResult(metrics={"classification_stability": 1.0})

    StabilityEvaluatorFactory.register_evaluator("detection", DummyDetectionEvaluator)
    StabilityEvaluatorFactory.register_evaluator("classification", DummyClassificationEvaluator)


if __name__ == "__main__":
    from pathlib import Path
    from ....ai_eval_tool.config_manager import load_config # Adjust path for test context

    # Create dummy configs for testing
    dummy_det_config_content = """
project_info:
  project_name: "Factory Test Detection"
  model_type: "detection"
data_loader: {field_mapping: {loop: l, image_id: i, image_path: p, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4, detection: {category_id: c, score: s, bbox: [x,y,w,h]}}}
evaluation_params: {detection: {iou_threshold: 0.5, bbox_format: xywh}}
report_settings: {}
"""
    dummy_cls_config_content = """
project_info:
  project_name: "Factory Test Classification"
  model_type: "classification"
data_loader: {field_mapping: {loop: l, image_id: i, image_path: p, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4, classification: {top_k_id_pattern: "id{k}", top_k_score_pattern: "s{k}"}}}
evaluation_params: {classification: {top_k: [1]}}
report_settings: {}
"""
    dummy_det_config_path = Path("dummy_factory_det_config.yaml")
    dummy_cls_config_path = Path("dummy_factory_cls_config.yaml")

    with open(dummy_det_config_path, "w") as f: f.write(dummy_det_config_content)
    with open(dummy_cls_config_path, "w") as f: f.write(dummy_cls_config_content)

    config_det = load_config(dummy_det_config_path)
    config_cls = load_config(dummy_cls_config_path)

    logger.info("--- Testing StabilityEvaluatorFactory ---")
    try:
        # Test getting detection evaluator
        det_evaluator = StabilityEvaluatorFactory.get_evaluator("detection", config_det)
        assert isinstance(det_evaluator, DummyDetectionEvaluator)
        logger.info(f"Successfully got evaluator: {type(det_evaluator).__name__}")
        # det_results = det_evaluator.evaluate(pl.DataFrame()) # Dummy DF
        # assert "detection_stability" in det_results.metrics

        # Test getting classification evaluator
        cls_evaluator = StabilityEvaluatorFactory.get_evaluator("classification", config_cls)
        assert isinstance(cls_evaluator, DummyClassificationEvaluator)
        logger.info(f"Successfully got evaluator: {type(cls_evaluator).__name__}")
        # cls_results = cls_evaluator.evaluate(pl.DataFrame()) # Dummy DF
        # assert "classification_stability" in cls_results.metrics

        # Test getting an unsupported evaluator
        try:
            StabilityEvaluatorFactory.get_evaluator("pose", config_det) # Assuming pose is not registered yet
        except ValueError as e:
            logger.info(f"Correctly failed for unsupported model type 'pose': {e}")
            assert "Unsupported model type" in str(e)

    except Exception as e:
        logger.error(f"Error during StabilityEvaluatorFactory test: {e}", exc_info=True)
    finally:
        if dummy_det_config_path.exists(): dummy_det_config_path.unlink()
        if dummy_cls_config_path.exists(): dummy_cls_config_path.unlink()
        logger.info("StabilityEvaluatorFactory test completed.")

```
