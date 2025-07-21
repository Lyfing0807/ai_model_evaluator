"""
Utilities Package.

This package contains common utility functions and helper classes used across
the AI Model Evaluator tool, such as logging configuration and custom type definitions.
"""

from .logging_config import get_logger, setup_logging
from .types import EvaluationResult

__all__ = [
    "setup_logging",
    "get_logger",
    "EvaluationResult",
]
