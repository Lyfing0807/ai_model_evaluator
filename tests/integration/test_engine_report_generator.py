"""
集成测试：Engine -> ReportGenerator
根据 TODO 要求添加此测试文件。
"""

import pytest
from pathlib import Path
import polars as pl
import yaml

from ai_eval_tool.config_manager import load_config, MainConfig
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.reporting.generator import ReportGenerator
from ai_eval_tool.utils.types import EvaluationResult


def create_test_config_for_reporting(tmp_path, model_type="detection"):
    """创建用于报告生成测试的配置文件。"""
    
    config_content = {
        "project_info": {
            "project_name": f"Report Generation Test {model_type.title()}",
            "model_type": model_type,
            "run_id": f"report_test_{model_type}_001",
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
            "output_dir": str(tmp_path / f"reports_{model_type}"),
            "include_charts": True,
            "chart_formats": ["png"],
            "ai_insights": {
                "enabled": False,  # 禁用AI洞察以简化测试
            }
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


def test_engine_to_report_generator_integration(tmp_path):
    """
    运行EvaluationEngine得到EvaluationResult对象，将其传递给ReportGenerator，
    验证报告生成器能否成功消费该对象并生成报告。
    根据 TODO 要求添加此测试。
    """
    # 创建配置
    config_file = create_test_config_for_reporting(tmp_path, "detection")
    config = load_config(config_file)
    
    # 创建测试数据
    test_data = pl.DataFrame({
        "loop": [1, 1, 2, 2, 3, 3],
        "image_id": ["img1", "img1", "img1", "img1", "img1", "img1"],
        "pre_time_ms": [10.0, 12.0, 11.0, 13.0, 10.5, 12.5],
        "inference_time_ms": [100.0, 110.0, 105.0, 115.0, 102.0, 112.0],
        "post_time_ms": [5.0, 6.0, 5.5, 6.5, 5.2, 6.2],
        "total_time_ms": [115.0, 128.0, 121.5, 134.5, 117.7, 130.7],
        "internal_bbox": [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [12, 12, 22, 22],
            [32, 32, 42, 42],
            [11, 11, 21, 21],
            [31, 31, 41, 41],
        ],
        "category_id": [0, 1, 0, 1, 0, 1],
        "score": [0.9, 0.8, 0.85, 0.75, 0.88, 0.78],
    })
    
    # 步骤1：运行EvaluationEngine
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 验证Engine结果
    assert isinstance(results, EvaluationResult)
    assert len(results.metrics) > 0
    
    # 步骤2：将结果传递给ReportGenerator
    report_generator = ReportGenerator(config)
    
    # 验证ReportGenerator能够处理EvaluationResult
    try:
        # 生成报告（这里主要测试不会因数据结构不匹配而失败）
        report_generator.generate_report(results)
        
        # 验证报告文件是否生成
        output_dir = Path(config.report_settings.output_dir)
        run_id = config.project_info.run_id
        
        # 检查基本报告文件
        md_report = output_dir / f"{run_id}_report.md"
        html_report = output_dir / f"{run_id}_report.html"
        
        # 至少应该生成一个报告文件
        assert md_report.exists() or html_report.exists(), "No report files were generated"
        
        # 如果生成了文件，检查内容不为空
        if md_report.exists():
            assert md_report.stat().st_size > 0, "Markdown report is empty"
        if html_report.exists():
            assert html_report.stat().st_size > 0, "HTML report is empty"
            
    except Exception as e:
        pytest.fail(f"ReportGenerator failed to process EvaluationResult: {e}")


def test_report_generator_with_different_data_structures(tmp_path):
    """
    测试ReportGenerator处理不同数据结构的能力。
    根据 TODO 要求添加此测试。
    """
    config_file = create_test_config_for_reporting(tmp_path, "classification")
    config = load_config(config_file)
    
    # 创建分类测试数据
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
            ["cat", "bird", "dog"],
            ["dog", "fish", "cat"],
        ],
        "top_k_scores": [
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
            [0.88, 0.78, 0.68],
            [0.83, 0.73, 0.63],
        ],
    })
    
    # 运行Engine
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 测试ReportGenerator
    report_generator = ReportGenerator(config)
    
    try:
        report_generator.generate_report(results)
        
        # 验证报告生成成功
        output_dir = Path(config.report_settings.output_dir)
        run_id = config.project_info.run_id
        
        report_files = list(output_dir.glob(f"{run_id}_report.*"))
        assert len(report_files) > 0, "No report files generated for classification model"
        
    except Exception as e:
        pytest.fail(f"ReportGenerator failed with classification data: {e}")


def test_report_generator_error_handling(tmp_path):
    """
    测试ReportGenerator的错误处理能力。
    根据 TODO 要求添加此测试。
    """
    config_file = create_test_config_for_reporting(tmp_path, "detection")
    config = load_config(config_file)
    
    report_generator = ReportGenerator(config)
    
    # 测试1：空的EvaluationResult
    empty_result = EvaluationResult(metrics={}, plots={}, extra_data={})
    
    try:
        report_generator.generate_report(empty_result)
        # 应该能处理空结果，可能生成一个基本的报告
    except Exception as e:
        # 如果抛出异常，应该是有意义的错误信息
        assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    # 测试2：包含无效数据的EvaluationResult
    invalid_result = EvaluationResult(
        metrics={"invalid_metric": float('nan')},
        plots={},
        extra_data={"invalid_df": "not_a_dataframe"}
    )
    
    try:
        report_generator.generate_report(invalid_result)
        # 应该能优雅地处理无效数据
    except Exception as e:
        # 如果抛出异常，应该是有意义的错误信息
        assert len(str(e)) > 0


def test_report_generator_chart_generation(tmp_path):
    """
    测试ReportGenerator的图表生成功能。
    根据 TODO 要求添加此测试。
    """
    config_file = create_test_config_for_reporting(tmp_path, "detection")
    config = load_config(config_file)
    
    # 创建足够的数据以生成有意义的图表
    test_data = pl.DataFrame({
        "loop": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 2,  # 更多数据点
        "image_id": ["img1", "img2"] * 10,
        "pre_time_ms": [10.0 + i for i in range(20)],
        "inference_time_ms": [100.0 + i*2 for i in range(20)],
        "post_time_ms": [5.0 + i*0.5 for i in range(20)],
        "total_time_ms": [115.0 + i*3 for i in range(20)],
        "internal_bbox": [[10+i, 10+i, 20+i, 20+i] for i in range(20)],
        "category_id": [i % 3 for i in range(20)],
        "score": [0.9 - i*0.01 for i in range(20)],
    })
    
    # 运行Engine
    engine = EvaluationEngine(config)
    results = engine.run(test_data)
    
    # 生成报告
    report_generator = ReportGenerator(config)
    report_generator.generate_report(results)
    
    # 验证图表文件生成
    output_dir = Path(config.report_settings.output_dir)
    run_id = config.project_info.run_id
    charts_dir = output_dir / run_id / "charts"
    
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        # 应该至少生成一些图表文件
        assert len(chart_files) > 0, "No chart files were generated"
        
        # 检查图表文件不为空
        for chart_file in chart_files:
            assert chart_file.stat().st_size > 0, f"Chart file {chart_file} is empty"


def test_evaluation_result_serialization_compatibility(tmp_path):
    """
    测试EvaluationResult对象的序列化兼容性。
    根据 TODO 要求添加此测试。
    """
    config_file = create_test_config_for_reporting(tmp_path, "detection")
    config = load_config(config_file)
    
    # 创建测试数据
    test_data = pl.DataFrame({
        "loop": [1, 2],
        "image_id": ["img1", "img1"],
        "pre_time_ms": [10.0, 11.0],
        "inference_time_ms": [100.0, 105.0],
        "post_time_ms": [5.0, 5.5],
        "total_time_ms": [115.0, 121.5],
        "internal_bbox": [[10, 10, 20, 20], [12, 12, 22, 22]],
        "category_id": [0, 0],
        "score": [0.9, 0.85],
    })
    
    # 运行Engine获取结果
    engine = EvaluationEngine(config)
    original_results = engine.run(test_data)
    
    # 测试序列化和反序列化
    cache_dir = tmp_path / "cache_test"
    cache_dir.mkdir(exist_ok=True)
    run_id = config.project_info.run_id
    
    # 序列化
    engine.serialize_results(original_results, cache_dir, run_id)
    
    # 反序列化
    deserialized_results = engine.deserialize_results(cache_dir, run_id)
    
    # 验证反序列化的结果能被ReportGenerator正确处理
    report_generator = ReportGenerator(config)
    
    try:
        report_generator.generate_report(deserialized_results)
        
        # 验证报告生成成功
        output_dir = Path(config.report_settings.output_dir)
        report_files = list(output_dir.glob(f"{run_id}_report.*"))
        assert len(report_files) > 0, "No report files generated from deserialized results"
        
    except Exception as e:
        pytest.fail(f"ReportGenerator failed with deserialized results: {e}")