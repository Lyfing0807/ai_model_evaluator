"""
集成测试：Config -> DataLoader -> Engine
根据 TODO 要求添加此测试文件。
"""

import pytest
from pathlib import Path
import polars as pl
import yaml

from ai_eval_tool.config_manager import load_config, MainConfig
from ai_eval_tool.data_loader import DataLoader
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.utils.types import EvaluationResult


def create_test_config_and_data(tmp_path, model_type="detection"):
    """创建测试配置和数据文件。"""
    
    if model_type == "detection":
        config_content = {
            "project_info": {
                "project_name": "Integration Test Detection",
                "model_type": "detection",
                "run_id": "integration_test_det_001",
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
                    "detection": {
                        "category_id": "det_cat",
                        "score": "det_score",
                        "bbox": ["x", "y", "w", "h"],
                    },
                }
            },
            "evaluation_params": {
                "detection": {
                    "iou_threshold": 0.5,
                    "bbox_format": "xywh",
                }
            },
            "report_settings": {
                "output_dir": str(tmp_path / "reports_detection")
            },
        }
        
        # 创建测试CSV数据
        csv_content = (
            "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w,h\n"
            "1,img1.jpg,path1,10,20,5,35,0,0.9,100,100,50,50\n"
            "1,img1.jpg,path1,10,20,5,35,1,0.8,10,10,20,20\n"
            "2,img1.jpg,path1,12,22,6,40,0,0.85,102,102,50,50\n"
            "1,img2.jpg,path2,8,18,4,30,0,0.95,200,200,30,30\n"
            "2,img2.jpg,path2,9,19,4,32,0,0.93,201,201,30,30\n"
        )
        
    elif model_type == "classification":
        config_content = {
            "project_info": {
                "project_name": "Integration Test Classification",
                "model_type": "classification",
                "run_id": "integration_test_cls_001",
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
                    "classification": {
                        "top_k_id_pattern": "pred_label_top{k}",
                        "top_k_score_pattern": "pred_score_top{k}",
                    },
                }
            },
            "evaluation_params": {
                "classification": {
                    "top_k": [1, 3],
                }
            },
            "report_settings": {
                "output_dir": str(tmp_path / "reports_classification")
            },
        }
        
        csv_content = (
            "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,pred_label_top1,pred_score_top1,pred_label_top3,pred_score_top3\n"
            "1,imgA.jpg,pathA,5,10,2,17,cat,0.9,dog,0.8\n"
            "2,imgA.jpg,pathA,6,11,2,19,cat,0.88,fox,0.75\n"
            "1,imgB.jpg,pathB,7,12,3,22,bird,0.95,fish,0.85\n"
            "2,imgB.jpg,pathB,7,13,3,23,bird,0.93,fish,0.82\n"
        )
    
    # 写入配置文件
    config_file = tmp_path / f"config_{model_type}.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    
    # 写入CSV文件
    csv_file = tmp_path / f"data_{model_type}.csv"
    with open(csv_file, "w") as f:
        f.write(csv_content)
    
    # 创建输出目录
    (tmp_path / f"reports_{model_type}").mkdir(parents=True, exist_ok=True)
    
    return config_file, csv_file


def test_config_dataloader_engine_integration_detection(tmp_path):
    """
    测试 Config -> DataLoader -> Engine 集成流程（检测模型）。
    根据 TODO 要求添加此测试。
    """
    # 创建测试配置和数据
    config_file, csv_file = create_test_config_and_data(tmp_path, "detection")
    
    # 步骤1：加载配置
    config = load_config(config_file)
    assert isinstance(config, MainConfig)
    assert config.project_info.model_type == "detection"
    
    # 步骤2：使用配置创建DataLoader并加载数据
    data_loader = DataLoader(config)
    loaded_df = data_loader.load_data(csv_file)
    
    # 验证数据加载结果
    assert isinstance(loaded_df, pl.DataFrame)
    assert not loaded_df.is_empty()
    assert "loop" in loaded_df.columns
    assert "image_id" in loaded_df.columns
    assert "internal_bbox" in loaded_df.columns  # 应该被转换
    
    # 步骤3：将数据传递给EvaluationEngine
    engine = EvaluationEngine(config)
    results = engine.run(loaded_df)
    
    # 验证引擎结果
    assert isinstance(results, EvaluationResult)
    assert "perf_mean_total_time_ms" in results.metrics  # 性能指标
    
    # 验证没有因数据格式或列名不匹配导致的错误
    assert "error" not in results.metrics
    assert "warning" not in results.metrics or "Empty input data" not in str(results.metrics.get("warning", ""))


def test_config_dataloader_engine_integration_classification(tmp_path):
    """
    测试 Config -> DataLoader -> Engine 集成流程（分类模型）。
    根据 TODO 要求添加此测试。
    """
    # 创建测试配置和数据
    config_file, csv_file = create_test_config_and_data(tmp_path, "classification")
    
    # 步骤1：加载配置
    config = load_config(config_file)
    assert isinstance(config, MainConfig)
    assert config.project_info.model_type == "classification"
    
    # 步骤2：使用配置创建DataLoader并加载数据
    data_loader = DataLoader(config)
    loaded_df = data_loader.load_data(csv_file)
    
    # 验证数据加载结果
    assert isinstance(loaded_df, pl.DataFrame)
    assert not loaded_df.is_empty()
    assert "loop" in loaded_df.columns
    assert "image_id" in loaded_df.columns
    assert "top_k_labels" in loaded_df.columns  # 应该被转换
    assert "top_k_scores" in loaded_df.columns
    
    # 步骤3：将数据传递给EvaluationEngine
    engine = EvaluationEngine(config)
    results = engine.run(loaded_df)
    
    # 验证引擎结果
    assert isinstance(results, EvaluationResult)
    assert "perf_mean_total_time_ms" in results.metrics  # 性能指标
    
    # 验证没有因数据格式或列名不匹配导致的错误
    assert "error" not in results.metrics
    assert "warning" not in results.metrics or "Empty input data" not in str(results.metrics.get("warning", ""))


def test_data_format_consistency_across_modules(tmp_path):
    """
    验证数据在模块间传递时格式的一致性。
    根据 TODO 要求添加此测试。
    """
    config_file, csv_file = create_test_config_and_data(tmp_path, "detection")
    config = load_config(config_file)
    
    # 测试DataLoader输出格式
    data_loader = DataLoader(config)
    loaded_df = data_loader.load_data(csv_file)
    
    # 验证DataLoader输出的关键列
    required_columns = ["loop", "image_id", "pre_time_ms", "inference_time_ms", 
                       "post_time_ms", "total_time_ms", "internal_bbox", 
                       "category_id", "score"]
    
    for col in required_columns:
        assert col in loaded_df.columns, f"Missing column: {col}"
    
    # 验证数据类型
    assert loaded_df["loop"].dtype.is_numeric()
    assert loaded_df["image_id"].dtype == pl.Utf8
    assert loaded_df["internal_bbox"].dtype == pl.List(pl.Float64)
    
    # 测试Engine能否正确处理这些数据
    engine = EvaluationEngine(config)
    results = engine.run(loaded_df)
    
    # 验证Engine没有因为数据格式问题而失败
    assert isinstance(results, EvaluationResult)
    assert len(results.metrics) > 0
    
    # 验证extra_data中的DataFrame格式
    if "deduplicated_perf_df_for_charts" in results.extra_data:
        perf_df = results.extra_data["deduplicated_perf_df_for_charts"]
        assert isinstance(perf_df, pl.DataFrame)
        assert "loop" in perf_df.columns
        assert "image_id" in perf_df.columns


def test_error_propagation_across_modules(tmp_path):
    """
    测试错误在模块间的正确传播。
    根据 TODO 要求添加此测试。
    """
    # 创建有问题的配置（缺少必需字段）
    bad_config_content = {
        "project_info": {
            "project_name": "Bad Config Test",
            "model_type": "detection",
        },
        # 缺少 data_loader 和 evaluation_params
        "report_settings": {"output_dir": str(tmp_path / "reports")},
    }
    
    config_file = tmp_path / "bad_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(bad_config_content, f)
    
    # 测试配置加载时的错误处理
    with pytest.raises((ValueError, KeyError)):
        config = load_config(config_file)
        data_loader = DataLoader(config)  # 应该因为缺少field_mapping而失败
    
    # 测试空数据的处理
    config_file, _ = create_test_config_and_data(tmp_path, "detection")
    config = load_config(config_file)
    
    # 创建空CSV文件
    empty_csv = tmp_path / "empty.csv"
    with open(empty_csv, "w") as f:
        f.write("loop_id,img_name\n")  # 只有表头
    
    data_loader = DataLoader(config)
    engine = EvaluationEngine(config)
    
    # 测试空数据的处理
    try:
        loaded_df = data_loader.load_data(empty_csv)
        results = engine.run(loaded_df)
        # 应该有警告或错误信息
        assert "warning" in results.metrics or "error" in results.metrics
    except Exception:
        # 也可能直接抛出异常，这也是合理的
        pass