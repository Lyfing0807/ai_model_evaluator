"""
集成测试：Engine -> EvaluatorFactory -> Evaluators
根据 TODO 要求添加此测试文件。
"""

import pytest
from pathlib import Path
import polars as pl
import yaml

from ai_eval_tool.config_manager import load_config, MainConfig
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.evaluators.stability.factory import StabilityEvaluatorFactory
from ai_eval_tool.evaluators.stability.detection import DetectionStabilityEvaluator
from ai_eval_tool.evaluators.stability.classification import ClassificationStabilityEvaluator
from ai_eval_tool.utils.types import EvaluationResult


def create_test_config(tmp_path, model_type="detection"):
    """创建测试配置文件。"""
    
    config_content = {
        "project_info": {
            "project_name": f"Engine Factory Test {model_type.title()}",
            "model_type": model_type,
            "run_id": f"engine_factory_test_{model_type}_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop",
                "image_id": "image_id",
                "pre_time_ms": "pre_time_ms",
                "inference_time_ms": "inference_time_ms",
                "post_time_ms": "post_time_ms",
                "total_time_ms": "total_time_ms",
            }
        },
        "report_settings": {
            "output_dir": str(tmp_path / f"reports_{model_type}")
        },
    }
    
    # 根据模型类型添加特定配置
    if model_type == "detection":
        config_content["data_loader"]["field_mapping"]["detection"] = {
            "category_id": "category_id",
            "score": "score",
            "bbox": ["x1", "y1", "x2", "y2"],
        }
        config_content["evaluation_params"] = {
            "detection": {
                "iou_threshold": 0.5,
                "bbox_format": "xyxy",
            }
        }
    elif model_type == "classification":
        config_content["data_loader"]["field_mapping"]["classification"] = {
            "top_k_id_pattern": "pred_label_top{k}",
            "top_k_score_pattern": "pred_score_top{k}",
        }
        config_content["evaluation_params"] = {
            "classification": {
                "top_k": [1, 3],
            }
        }
    
    # 写入配置文件
    config_file = tmp_path / f"config_{model_type}.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    
    # 创建输出目录
    (tmp_path / f"reports_{model_type}").mkdir(parents=True, exist_ok=True)
    
    return config_file


def test_engine_factory_detection_evaluator_integration(tmp_path):
    """
    验证EvaluationEngine能否根据config.yaml中的model_type，通过StabilityEvaluatorFactory
    正确地实例化并调用对应的稳定性评估器（检测）。
    根据 TODO 要求添加此测试。
    """
    # 确保检测评估器已注册
    StabilityEvaluatorFactory.register_evaluator("detection", DetectionStabilityEvaluator)
    
    # 创建检测配置
    config_file = create_test_config(tmp_path, "detection")
    config = load_config(config_file)
    
    # 创建测试数据
    test_data = pl.DataFrame({
        "loop": [1, 1, 2, 2],
        "image_id": ["img1", "img1", "img1", "img1"],
        "pre_time_ms": [10.0, 12.0, 11.0, 13.0],
        "inference_time_ms": [100.0, 110.0, 105.0, 115.0],
        "post_time_ms": [5.0, 6.0, 5.5, 6.5],
        "total_time_ms": [115.0, 128.0, 121.5, 134.5],
        "internal_bbox": [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [12, 12, 22, 22],  # 与第一个目标相近
            [32, 32, 42, 42],  # 与第二个目标相近
        ],
        "category_id": [0, 1, 0, 1],
        "score": [0.9, 0.8, 0.85, 0.75],
    })
    
    # 创建并运行引擎
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 验证结果
    assert isinstance(results, EvaluationResult)
    
    # 验证性能评估器的结果
    assert "perf_mean_total_time_ms" in results.metrics
    
    # 验证稳定性评估器被正确调用（检测特定指标）
    # 注意：这里可能会有警告，因为数据量较小
    if "warning" not in results.metrics:
        # 如果没有警告，应该有检测稳定性指标
        detection_metrics = [k for k in results.metrics.keys() if k.startswith("det_stab_")]
        assert len(detection_metrics) > 0, "Detection stability metrics should be present"
    
    # 验证没有因工厂模式或评估器实例化导致的错误
    assert "error" not in results.metrics


def test_engine_factory_classification_evaluator_integration(tmp_path):
    """
    验证EvaluationEngine能否根据config.yaml中的model_type，通过StabilityEvaluatorFactory
    正确地实例化并调用对应的稳定性评估器（分类）。
    根据 TODO 要求添加此测试。
    """
    # 确保分类评估器已注册
    StabilityEvaluatorFactory.register_evaluator("classification", ClassificationStabilityEvaluator)
    
    # 创建分类配置
    config_file = create_test_config(tmp_path, "classification")
    config = load_config(config_file)
    
    # 创建测试数据
    test_data = pl.DataFrame({
        "loop": [1, 1, 2, 2],
        "image_id": ["img1", "img2", "img1", "img2"],
        "pre_time_ms": [8.0, 9.0, 8.5, 9.5],
        "inference_time_ms": [80.0, 90.0, 85.0, 95.0],
        "post_time_ms": [4.0, 5.0, 4.5, 5.5],
        "total_time_ms": [92.0, 104.0, 98.0, 110.0],
        "top_k_labels": [
            ["cat", "dog", "bird"],
            ["dog", "cat", "fish"],
            ["cat", "bird", "dog"],  # img1 的第二次循环
            ["dog", "fish", "cat"],  # img2 的第二次循环
        ],
        "top_k_scores": [
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
            [0.88, 0.78, 0.68],
            [0.83, 0.73, 0.63],
        ],
    })
    
    # 创建并运行引擎
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 验证结果
    assert isinstance(results, EvaluationResult)
    
    # 验证性能评估器的结果
    assert "perf_mean_total_time_ms" in results.metrics
    
    # 验证稳定性评估器被正确调用（分类特定指标）
    if "warning" not in results.metrics:
        # 如果没有警告，应该有分类稳定性指标
        classification_metrics = [k for k in results.metrics.keys() if k.startswith("cls_stab_")]
        assert len(classification_metrics) > 0, "Classification stability metrics should be present"
    
    # 验证没有因工厂模式或评估器实例化导致的错误
    assert "error" not in results.metrics


def test_engine_unsupported_model_type_handling(tmp_path):
    """
    测试引擎对不支持的模型类型的处理。
    根据 TODO 要求添加此测试。
    """
    # 创建不支持的模型类型配置
    config_content = {
        "project_info": {
            "project_name": "Unsupported Model Test",
            "model_type": "unsupported_model_type",
            "run_id": "unsupported_test_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop",
                "image_id": "image_id",
                "pre_time_ms": "pre_time_ms",
                "inference_time_ms": "inference_time_ms",
                "post_time_ms": "post_time_ms",
                "total_time_ms": "total_time_ms",
            }
        },
        "report_settings": {
            "output_dir": str(tmp_path / "reports_unsupported")
        },
    }
    
    config_file = tmp_path / "config_unsupported.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    
    (tmp_path / "reports_unsupported").mkdir(parents=True, exist_ok=True)
    
    config = load_config(config_file)
    
    # 创建简单的测试数据
    test_data = pl.DataFrame({
        "loop": [1, 2],
        "image_id": ["img1", "img1"],
        "pre_time_ms": [10.0, 11.0],
        "inference_time_ms": [100.0, 105.0],
        "post_time_ms": [5.0, 5.5],
        "total_time_ms": [115.0, 121.5],
    })
    
    # 创建并运行引擎
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 验证结果
    assert isinstance(results, EvaluationResult)
    
    # 应该仍然有性能指标（性能评估器不依赖模型类型）
    assert "perf_mean_total_time_ms" in results.metrics
    
    # 不应该有稳定性指标（因为没有对应的评估器）
    stability_metrics = [k for k in results.metrics.keys() if "stab_" in k]
    assert len(stability_metrics) == 0, "Should not have stability metrics for unsupported model type"


def test_factory_registry_state_consistency(tmp_path):
    """
    测试工厂注册表状态的一致性。
    根据 TODO 要求添加此测试。
    """
    # 记录初始状态
    initial_registry = StabilityEvaluatorFactory._REGISTRY.copy()
    
    # 注册测试评估器
    StabilityEvaluatorFactory.register_evaluator("detection", DetectionStabilityEvaluator)
    StabilityEvaluatorFactory.register_evaluator("classification", ClassificationStabilityEvaluator)
    
    # 验证注册成功
    assert "detection" in StabilityEvaluatorFactory._REGISTRY
    assert "classification" in StabilityEvaluatorFactory._REGISTRY
    
    # 测试通过工厂获取评估器
    config_file = create_test_config(tmp_path, "detection")
    config = load_config(config_file)
    
    detection_evaluator = StabilityEvaluatorFactory.get_evaluator("detection", config)
    assert isinstance(detection_evaluator, DetectionStabilityEvaluator)
    
    # 测试不存在的评估器
    with pytest.raises(ValueError, match="Unsupported model type"):
        StabilityEvaluatorFactory.get_evaluator("nonexistent_type", config)
    
    # 验证注册表状态没有被意外修改
    assert "detection" in StabilityEvaluatorFactory._REGISTRY
    assert "classification" in StabilityEvaluatorFactory._REGISTRY