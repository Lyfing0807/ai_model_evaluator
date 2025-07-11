"""
Base classes for all evaluators.
"""
from abc import ABC, abstractmethod
import polars as pl
from ..config_manager import MainConfig
from ..utils.types import EvaluationResult # Assuming EvaluationResult is defined in utils.types
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class EvaluatorBase(ABC):
    """
    Abstract base class for all evaluators.
    """
    def __init__(self, config: MainConfig):
        self.config = config
        self.model_type = config.project_info.model_type
        # Common evaluation parameters could be accessed here if defined at a higher level in config
        logger.debug(f"EvaluatorBase initialized for model type: {self.model_type}")

    @abstractmethod
    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        """
        Performs evaluation on the given data.

        Args:
            data_df: The input DataFrame, typically preprocessed by DataLoader.

        Returns:
            An EvaluationResult object containing metrics, plots, and any extra data.
        """
        pass

class StabilityEvaluatorBase(EvaluatorBase):
    """
    Abstract base class for model-specific stability evaluators.
    It inherits from EvaluatorBase and can add more specific common methods
    for stability evaluators if needed in the future.
    """
    def __init__(self, config: MainConfig):
        super().__init__(config)
        # Access model-specific evaluation parameters
        self.eval_params = getattr(config.evaluation_params, self.model_type, None)
        if not self.eval_params:
            logger.warning(
                f"No specific evaluation parameters found for model type '{self.model_type}' "
                f"in StabilityEvaluatorBase initialization."
            )
        logger.debug(f"StabilityEvaluatorBase initialized for model type: {self.model_type}")

    @abstractmethod
    def evaluate(self, data_df: pl.DataFrame) -> EvaluationResult:
        """
        Performs stability evaluation on the given data.

        Args:
            data_df: The input DataFrame.

        Returns:
            An EvaluationResult object.
        """
        pass
```
