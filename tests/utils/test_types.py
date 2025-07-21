
import polars as pl
from ai_eval_tool.utils.types import EvaluationResult


def test_evaluation_result_default_values():
    """
    验证 EvaluationResult 数据类的默认值。
    """
    result = EvaluationResult()
    assert result.metrics == {}
    assert result.plots == {}
    assert result.extra_data == {}


def test_evaluation_result_with_metrics():
    """
    验证 EvaluationResult 能够正确存储指标数据。
    """
    metrics_data = {"accuracy": 0.95, "f1_score": 0.92}
    result = EvaluationResult(metrics=metrics_data)
    assert result.metrics == metrics_data
    assert result.plots == {}
    assert result.extra_data == {}


def test_evaluation_result_with_plots():
    """
    验证 EvaluationResult 能够正确存储图表数据。
    """
    plots_data = {"plot_1": "path/to/plot1.png", "plot_2": "html_string_for_plot2"}
    result = EvaluationResult(plots=plots_data)
    assert result.metrics == {}
    assert result.plots == plots_data
    assert result.extra_data == {}


def test_evaluation_result_with_extra_data():
    """
    验证 EvaluationResult 能够正确存储额外数据，包括 Polars DataFrame。
    """
    extra_data = {"sample_ids": [1, 2, 3], "config_name": "test_config"}
    result = EvaluationResult(extra_data=extra_data)
    assert result.metrics == {}
    assert result.plots == {}
    assert result.extra_data == extra_data

    # Test with Polars DataFrame
    df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    result_with_df = EvaluationResult(extra_data={"dataframe_key": df})
    assert result_with_df.extra_data["dataframe_key"].equals(df)


def test_evaluation_result_all_fields():
    """
    验证 EvaluationResult 能够正确存储所有字段。
    """
    metrics_data = {"precision": 0.88}
    plots_data = {"plot_summary": "summary.html"}
    extra_data = {"log_file": "run.log"}
    result = EvaluationResult(metrics=metrics_data, plots=plots_data, extra_data=extra_data)
    assert result.metrics == metrics_data
    assert result.plots == plots_data
    assert result.extra_data == extra_data
