import pytest
from pathlib import Path
import polars as pl
import numpy as np
import yaml

from ai_eval_tool.config_manager import load_config
from ai_eval_tool.evaluators.performance import PerformanceEvaluator


# Helper function to create a dummy config file for testing
def create_dummy_config(tmp_path, config_content, filename="config.yaml"):
    config_path = tmp_path / filename
    config_path.write_text(config_content)
    return config_path


# --- Test PerformanceEvaluator ---
def test_performance_evaluation_deduplication(tmp_path):
    """
    测试 PerformanceEvaluator 的数据去重逻辑。
    """
    config_content = """
project_info:
  project_name: "PerfEval Deduplication Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    # Create a DataFrame with duplicate (loop, image_id) entries
    data = {
        "loop": [1, 1, 1, 2, 2],
        "image_id": ["img1", "img1", "img2", "img1", "img2"],
        "pre_time_ms": [10, 10, 12, 11, 13],
        "inference_time_ms": [100, 100, 120, 110, 130],
        "post_time_ms": [5, 5, 6, 5, 7],
        "total_time_ms": [115, 115, 138, 126, 150],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)

    # Assert that the deduplicated DataFrame has the expected number of rows
    # Original df has 5 rows. After unique by (loop, image_id):
    # (1, img1), (1, img2), (2, img1), (2, img2) -> 4 unique inference events
    deduplicated_df = results.extra_data["deduplicated_perf_df_for_charts"]
    assert deduplicated_df.shape[0] == 4

    # Assert that the calculated mean is based on the deduplicated data
    expected_mean = np.mean([115, 138, 126, 150])  # (115+138+126+150)/4 = 132.25
    assert results.metrics["perf_mean_total_time_ms"] == pytest.approx(expected_mean)


def test_performance_metrics_accuracy(tmp_path):
    """
    测试 PerformanceEvaluator 计算的各项指标的准确性。
    """
    config_content = """
project_info:
  project_name: "PerfEval Metrics Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection:
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
  performance:
    target_latency_ms: 100
    cv_target_threshold: 0.1
    component_weights:
      throughput: 0.5
      latency_stability: 0.5
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1, 1, 1, 1, 1],
        "image_id": ["img1", "img2", "img3", "img4", "img5"],
        "pre_time_ms": [10, 11, 12, 13, 14],
        "inference_time_ms": [50, 55, 60, 65, 70],
        "post_time_ms": [5, 6, 7, 8, 9],
        "total_time_ms": [65, 72, 79, 86, 93],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    metrics = results.metrics

    # Expected values (calculated manually)
    total_times = np.array([65, 72, 79, 86, 93])
    expected_mean = np.mean(total_times)
    expected_median = np.median(total_times)
    expected_std = np.std(total_times)
    expected_min = np.min(total_times)
    expected_max = np.max(total_times)
    expected_p90 = np.quantile(total_times, 0.90, interpolation='linear')
    expected_p95 = np.quantile(total_times, 0.95, interpolation='linear')
    expected_p99 = np.quantile(total_times, 0.99, interpolation='linear')

    assert metrics["perf_mean_total_time_ms"] == pytest.approx(expected_mean)
    assert metrics["perf_median_total_time_ms"] == pytest.approx(expected_median)
    assert metrics["perf_std_total_time_ms"] == pytest.approx(expected_std)
    assert metrics["perf_min_total_time_ms"] == pytest.approx(expected_min)
    assert metrics["perf_max_total_time_ms"] == pytest.approx(expected_max)
    assert metrics["perf_p90_total_time_ms"] == pytest.approx(expected_p90)
    assert metrics["perf_p95_total_time_ms"] == pytest.approx(expected_p95)
    assert metrics["perf_p99_total_time_ms"] == pytest.approx(expected_p99)

    assert metrics["perf_avg_fps"] == pytest.approx(1000.0 / expected_mean)
    assert metrics["perf_cv_total_time_ms"] == pytest.approx(expected_std / expected_mean)

    # Jitter
    expected_jitter_diffs = np.abs(np.diff(total_times))
    expected_jitter_mean = np.mean(expected_jitter_diffs)
    assert metrics["perf_jitter_ms"] == pytest.approx(expected_jitter_mean)

    # Worst-case amplification
    assert metrics["perf_worst_case_amplification"] == pytest.approx(expected_max / expected_median)

    # High latency rate
    threshold = expected_mean + 3 * expected_std
    high_latency_count = np.sum(total_times > threshold)
    expected_high_latency_rate = high_latency_count / len(total_times)
    assert metrics["perf_high_latency_rate"] == pytest.approx(expected_high_latency_rate)

    # Time Composition Analysis
    pre_times = np.array([10, 11, 12, 13, 14])
    inference_times = np.array([50, 55, 60, 65, 70])
    post_times = np.array([5, 6, 7, 8, 9])

    assert metrics["perf_mean_pre_time_ms"] == pytest.approx(np.mean(pre_times))
    assert metrics["perf_mean_inference_time_ms"] == pytest.approx(np.mean(inference_times))
    assert metrics["perf_mean_post_time_ms"] == pytest.approx(np.mean(post_times))

    assert metrics["perf_percentage_pre_time_ms"] == pytest.approx(np.mean(pre_times) / expected_mean * 100)
    assert metrics["perf_percentage_inference_time_ms"] == pytest.approx(np.mean(inference_times) / expected_mean * 100)
    assert metrics["perf_percentage_post_time_ms"] == pytest.approx(np.mean(post_times) / expected_mean * 100)

    # Performance Score
    target_latency = 100
    cv_target = 0.1
    # Manual calculation for score_throughput (value=p95_latency, target=target_latency, lower_is_better=True)
    # p95_latency = 92.0
    # score_throughput = (target_latency / p95_latency) * 100 if p95_latency > 0 else 0
    # score_throughput = (100 / 92.0) * 100 = 108.6956... capped at 100
    assert metrics["perf_score_throughput"] == pytest.approx(100.0)

    # Manual calculation for score_latency_stability (value=cv_latency, good_threshold=cv_target, bad_threshold=cv_target*5, lower_is_better=True)
    # cv_latency = 0.1597...
    # good_threshold = 0.1
    # bad_threshold = 0.5
    # score_latency_stability = 100 * (1 - (cv_latency - good_threshold) / (bad_threshold - good_threshold))
    # score_latency_stability = 100 * (1 - (0.1597 - 0.1) / (0.5 - 0.1)) = 100 * (1 - 0.0597 / 0.4) = 100 * (1 - 0.14925) = 100 * 0.85075 = 85.075
    assert metrics["perf_score_latency_stability"] == pytest.approx(85.075, rel=1e-3)

    # Overall score
    expected_overall_score = 0.5 * 100.0 + 0.5 * 85.075
    assert metrics["perf_score_overall"] == pytest.approx(expected_overall_score)


def test_performance_empty_dataframe(tmp_path):
    """
    测试 PerformanceEvaluator 处理空 DataFrame 的情况。
    """
    config_content = """
project_info:
  project_name: "PerfEval Empty DF Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    empty_df = pl.DataFrame(
        {
            "loop": [],
            "image_id": [],
            "pre_time_ms": [],
            "inference_time_ms": [],
            "post_time_ms": [],
            "total_time_ms": [],
        },
        schema={
            "loop": pl.Int64,
            "image_id": pl.Utf8,
            "pre_time_ms": pl.Float64,
            "inference_time_ms": pl.Float64,
            "post_time_ms": pl.Float64,
            "total_time_ms": pl.Float64,
        },
    )

    results = evaluator.evaluate(empty_df)
    assert "warning" in results.metrics
    assert results.metrics["warning"] == "Input DataFrame is empty for performance evaluation"


def test_performance_all_null_total_time_ms(tmp_path):
    """
    测试 PerformanceEvaluator 处理 total_time_ms 全为 null 的情况。
    """
    config_content = """
project_info:
  project_name: "PerfEval Null Total Time Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1, 2],
        "image_id": ["img1", "img2"],
        "pre_time_ms": [10, 12],
        "inference_time_ms": [100, 120],
        "post_time_ms": [5, 6],
        "total_time_ms": [None, None],
    }
    df = pl.DataFrame(data).with_columns(pl.col("total_time_ms").cast(pl.Float64))

    results = evaluator.evaluate(df)
    assert "warning" in results.metrics
    assert results.metrics["warning"] == "'total_time_ms' column is empty or all nulls."


def test_performance_missing_required_columns(tmp_path):
    """
    测试 PerformanceEvaluator 处理缺少必需列的情况。
    """
    config_content = """
project_info:
  project_name: "PerfEval Missing Cols Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1, 2],
        "image_id": ["img1", "img2"],
        "pre_time_ms": [10, 12],
        "inference_time_ms": [100, 120],
        "post_time_ms": [5, 6],
        # "total_time_ms": [115, 138], # Missing total_time_ms
    }
    df = pl.DataFrame(data)

    with pytest.raises(ValueError, match=r"Missing required columns for performance evaluation: \['total_time_ms'\]"):
        evaluator.evaluate(df)


def test_performance_non_numeric_time_columns(tmp_path):
    """
    测试 PerformanceEvaluator 处理时间列非数字的情况。
    """
    config_content = """
project_info:
  project_name: "PerfEval Non-Numeric Time Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1, 2],
        "image_id": ["img1", "img2"],
        "pre_time_ms": ["10", "12"], # String type
        "inference_time_ms": [100, 120],
        "post_time_ms": [5, 6],
        "total_time_ms": [115, 138],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    # Assert that conversion happened and metrics are calculated
    assert results.metrics["perf_mean_pre_time_ms"] == pytest.approx(11.0)


def test_performance_score_calculation(tmp_path):
    """
    测试 PerformanceEvaluator 的性能分数计算。
    """
    config_content = """
project_info:
  project_name: "PerfEval Score Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
  performance:
    target_latency_ms: 100
    cv_target_threshold: 0.1
    component_weights:
      throughput: 0.6
      latency_stability: 0.4
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1, 1, 1, 1, 1],
        "image_id": ["img1", "img2", "img3", "img4", "img5"],
        "pre_time_ms": [10, 11, 12, 13, 14],
        "inference_time_ms": [50, 55, 60, 65, 70],
        "post_time_ms": [5, 6, 7, 8, 9],
        "total_time_ms": [65, 72, 79, 86, 93],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    metrics = results.metrics

    # Expected values (calculated manually)
    # p95_latency = 92.0
    # cv_latency = 0.1597...
    # score_throughput = 100.0 (capped)
    # score_latency_stability = 85.075 (calculated in test_performance_metrics_accuracy)
    expected_overall_score = 0.6 * 100.0 + 0.4 * 85.075

    assert metrics["perf_score_throughput"] == pytest.approx(100.0)
    assert metrics["perf_score_latency_stability"] == pytest.approx(85.075, rel=1e-3)
    assert metrics["perf_score_overall"] == pytest.approx(expected_overall_score)


def test_performance_score_no_weights(tmp_path):
    """
    测试 PerformanceEvaluator 在没有配置 component_weights 时跳过分数计算。
    """
    config_content = """
project_info:
  project_name: "PerfEval No Weights Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
  performance: # No component_weights here
    target_latency_ms: 100
    cv_target_threshold: 0.1
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1],
        "image_id": ["img1"],
        "pre_time_ms": [10],
        "inference_time_ms": [50],
        "post_time_ms": [5],
        "total_time_ms": [65],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    metrics = results.metrics

    assert "perf_score_overall" not in metrics
    assert "perf_score_throughput" not in metrics
    assert "perf_score_latency_stability" not in metrics


def test_performance_score_no_performance_params(tmp_path):
    """
    测试 PerformanceEvaluator 在没有配置 evaluation_params.performance 时跳过分数计算。
    """
    config_content = """
project_info:
  project_name: "PerfEval No Perf Params Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    data = {
        "loop": [1],
        "image_id": ["img1"],
        "pre_time_ms": [10],
        "inference_time_ms": [50],
        "post_time_ms": [5],
        "total_time_ms": [65],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    metrics = results.metrics

    assert "perf_score_overall" not in metrics
    assert "perf_score_throughput" not in metrics
    assert "perf_score_latency_stability" not in metrics


def test_performance_evaluator_with_duplicates(tmp_path):
    """
    测试包含重复（loop, image_id）项的DataFrame，验证去重逻辑是否生效。
    根据 TODO 要求添加此测试。
    """
    config_content = """
project_info:
  project_name: "PerfEval Duplicates Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection:
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    # 创建包含重复项的数据
    data = {
        "loop": [1, 1, 1, 2, 2, 2],  # 重复的 loop + image_id 组合
        "image_id": ["img1", "img1", "img2", "img1", "img1", "img2"],  # 重复项
        "pre_time_ms": [10, 12, 15, 11, 13, 16],
        "inference_time_ms": [50, 52, 55, 51, 53, 56],
        "post_time_ms": [5, 6, 7, 5, 6, 7],
        "total_time_ms": [65, 70, 77, 67, 72, 79],
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    
    # 验证去重后的数据在 extra_data 中
    assert "deduplicated_perf_df_for_charts" in results.extra_data
    dedup_df = results.extra_data["deduplicated_perf_df_for_charts"]
    
    # 去重后应该只有4行：(1, img1), (1, img2), (2, img1), (2, img2)
    assert len(dedup_df) == 4
    
    # 验证去重逻辑：每个 (loop, image_id) 组合只保留一行
    unique_combinations = dedup_df.select(["loop", "image_id"]).unique()
    assert len(unique_combinations) == 4


def test_performance_metrics_manual_calculation(tmp_path):
    """
    使用可预测的小型DataFrame，手动计算并验证所有性能指标的准确性。
    根据 TODO 要求添加此测试。
    """
    config_content = """
project_info:
  project_name: "PerfEval Manual Calc Test"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop"
    image_id: "image_id"
    pre_time_ms: "pre_time_ms"
    inference_time_ms: "inference_time_ms"
    post_time_ms: "post_time_ms"
    total_time_ms: "total_time_ms"
    detection:
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)
    evaluator = PerformanceEvaluator(config)

    # 创建可预测的小型数据集
    data = {
        "loop": [1, 1, 2, 2],
        "image_id": ["img1", "img2", "img1", "img2"],
        "pre_time_ms": [10.0, 20.0, 15.0, 25.0],  # 平均值: 17.5
        "inference_time_ms": [100.0, 200.0, 150.0, 250.0],  # 平均值: 175.0
        "post_time_ms": [5.0, 10.0, 7.5, 12.5],  # 平均值: 8.75
        "total_time_ms": [115.0, 230.0, 172.5, 287.5],  # 平均值: 201.25
    }
    df = pl.DataFrame(data)

    results = evaluator.evaluate(df)
    metrics = results.metrics

    # 手动计算期望值
    pre_times = [10.0, 20.0, 15.0, 25.0]
    inf_times = [100.0, 200.0, 150.0, 250.0]
    post_times = [5.0, 10.0, 7.5, 12.5]
    total_times = [115.0, 230.0, 172.5, 287.5]

    expected_pre_mean = np.mean(pre_times)
    expected_inf_mean = np.mean(inf_times)
    expected_post_mean = np.mean(post_times)
    expected_total_mean = np.mean(total_times)
    
    expected_pre_p95 = np.percentile(pre_times, 95)
    expected_inf_p95 = np.percentile(inf_times, 95)
    expected_post_p95 = np.percentile(post_times, 95)
    expected_total_p95 = np.percentile(total_times, 95)
    
    expected_pre_cv = np.std(pre_times) / np.mean(pre_times) if np.mean(pre_times) > 0 else 0
    expected_inf_cv = np.std(inf_times) / np.mean(inf_times) if np.mean(inf_times) > 0 else 0
    expected_post_cv = np.std(post_times) / np.mean(post_times) if np.mean(post_times) > 0 else 0
    expected_total_cv = np.std(total_times) / np.mean(total_times) if np.mean(total_times) > 0 else 0

    # 验证计算结果
    assert metrics["perf_mean_pre_time_ms"] == pytest.approx(expected_pre_mean, abs=1e-6)
    assert metrics["perf_mean_inference_time_ms"] == pytest.approx(expected_inf_mean, abs=1e-6)
    assert metrics["perf_mean_post_time_ms"] == pytest.approx(expected_post_mean, abs=1e-6)
    assert metrics["perf_mean_total_time_ms"] == pytest.approx(expected_total_mean, abs=1e-6)
    
    assert metrics["perf_p95_pre_time_ms"] == pytest.approx(expected_pre_p95, abs=1e-6)
    assert metrics["perf_p95_inference_time_ms"] == pytest.approx(expected_inf_p95, abs=1e-6)
    assert metrics["perf_p95_post_time_ms"] == pytest.approx(expected_post_p95, abs=1e-6)
    assert metrics["perf_p95_total_time_ms"] == pytest.approx(expected_total_p95, abs=1e-6)
    
    assert metrics["perf_cv_pre_time_ms"] == pytest.approx(expected_pre_cv, abs=1e-6)
    assert metrics["perf_cv_inference_time_ms"] == pytest.approx(expected_inf_cv, abs=1e-6)
    assert metrics["perf_cv_post_time_ms"] == pytest.approx(expected_post_cv, abs=1e-6)
    assert metrics["perf_cv_total_time_ms"] == pytest.approx(expected_total_cv, abs=1e-6)