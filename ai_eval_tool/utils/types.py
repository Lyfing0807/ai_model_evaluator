"""
Core data structures and type definitions used throughout the application.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union
import polars as pl
# For plot objects, it's tricky. Matplotlib Figure, Plotly Figure, or just paths/HTML strings.
# Using 'Any' for now, or can define a Union type if specific plot libraries are fixed.
PlotObject = Any # Union[matplotlib.figure.Figure, plotly.graph_objects.Figure, str, Path]

@dataclass
class EvaluationResult:
    """
    Standardized data structure for returning results from any evaluator.

    Attributes:
        metrics: A dictionary storing quantitative metrics.
                 Example: {"mean_latency_ms": 100.5, "iou_consistency": 0.85}
        plots: A dictionary storing generated plot objects or paths to saved plots.
               The value can be a Matplotlib Figure, a Plotly Figure, a path to an image file,
               or an HTML string for embedding.
               Example: {"latency_distribution_plot": <matplotlib.figure.Figure object>}
                        {"interactive_iou_plot": "<plotly_div_html_string>"}
        extra_data: A dictionary for any additional data that needs to be passed along,
                    such as DataFrames of abnormal samples, intermediate calculations, etc.
                    Example: {"slow_samples_df": <polars.DataFrame object>}
    """
    metrics: Dict[str, Union[float, int, str, bool, None, List[Any]]] = field(default_factory=dict)
    plots: Dict[str, PlotObject] = field(default_factory=dict) # Plot objects or their representations
    extra_data: Dict[str, Any] = field(default_factory=dict) # e.g., DataFrames of outliers


if __name__ == "__main__":
    # Example usage of EvaluationResult

    # Create an empty result
    empty_result = EvaluationResult()
    print(f"Empty result: {empty_result}")

    # Create a result with some metrics
    metrics_data = {"accuracy": 0.95, "f1_score": 0.92, "total_samples": 1000}
    result_with_metrics = EvaluationResult(metrics=metrics_data)
    print(f"\nResult with metrics: {result_with_metrics}")

    # Create a result with metrics and a placeholder for a plot path
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot([1,2,3], [1,4,9])
    # plot_path_placeholder = "path/to/my_plot.png" # In reality, this would be a Path object or similar

    result_with_plot_path = EvaluationResult(
        metrics={"recall": 0.88},
        plots={"recall_precision_curve": "path/to/recall_precision_curve.png"} # Placeholder path
        # plots={"my_actual_plot": fig} # If storing actual plot object
    )
    print(f"\nResult with plot path: {result_with_plot_path}")

    # Create a result with extra data (e.g., a Polars DataFrame)
    try:
        outlier_df = pl.DataFrame({
            "image_id": ["img_001.jpg", "img_005.jpg"],
            "latency_ms": [500.7, 610.2],
            "reason": ["high_complexity", "resource_contention"]
        })
        result_with_extra_data = EvaluationResult(
            extra_data={"outlier_samples": outlier_df}
        )
        print(f"\nResult with extra data (DataFrame):")
        print(f"  Metrics: {result_with_extra_data.metrics}")
        print(f"  Plots: {result_with_extra_data.plots}")
        if "outlier_samples" in result_with_extra_data.extra_data:
            print(f"  Outlier Samples DF Head:\n{result_with_extra_data.extra_data['outlier_samples'].head(1)}")

    except Exception as e:
        print(f"\nError creating Polars DataFrame for example (Polars might not be fully installed in this basic env): {e}")
        result_with_extra_data = EvaluationResult(
            extra_data={"outlier_samples_placeholder": "DataFrame would be here"}
        )
        print(f"\nResult with extra data (placeholder): {result_with_extra_data}")


    # Example of a more complete result
    complex_result = EvaluationResult(
        metrics={
            "mean_total_time_ms": 150.2,
            "p95_total_time_ms": 300.5,
            "average_fps": 6.65,
            "iou_consistency_mean": 0.78
        },
        plots={
            "latency_distribution": "reports/run123/charts/latency_dist.png",
            "iou_vs_time_interactive": "<div id='plotly_iou_time'>...</div>"
        },
        extra_data={
            "failed_tracking_ids": [101, 203, 505],
            "low_confidence_detections_df": "DataFrame object or path to CSV" # Placeholder
        }
    )
    print(f"\nComplex result example: {complex_result.metrics['average_fps']}")

```
