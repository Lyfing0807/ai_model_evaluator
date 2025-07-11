"""
Chart Generation: Functions to create various plots using Seaborn and Plotly.
"""
import polars as pl
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting, suitable for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

def set_seaborn_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana', 'Tahoma']
    plt.rcParams['axes.unicode_minus'] = False
set_seaborn_style()

# --- Matplotlib/Seaborn Chart Generators (for MD reports) ---

def generate_latency_distribution_plot_mpl(perf_df: pl.DataFrame, time_col: str = "total_time_ms", title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_pd = perf_df.select(pl.col(time_col).drop_nulls()).to_pandas()
    if data_pd.empty or data_pd[time_col].isnull().all():
        logger.warning(f"No valid data for Matplotlib latency distribution (column: {time_col}).")
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
        return fig
    sns.violinplot(y=data_pd[time_col], ax=ax, inner="quartile", color="skyblue", cut=0)
    ax.set_title(title or f"{time_col.replace('_', ' ').title()} Distribution", fontsize=15)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_xlabel(time_col.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()
    return fig

def generate_time_composition_plot_mpl(metrics: Dict[str, Any]) -> plt.Figure:
    # Implementation remains the same as previously provided
    components = {
        "Pre-processing": metrics.get("perf_mean_pre_time_ms"),
        "Inference": metrics.get("perf_mean_inference_time_ms"),
        "Post-processing": metrics.get("perf_mean_post_time_ms"),
    }
    valid_components = {k: v for k, v in components.items() if v is not None and isinstance(v, (int, float)) and v > 0}
    fig, ax = plt.subplots(figsize=(8, 6))
    if not valid_components:
        ax.text(0.5, 0.5, "No data for time composition", ha='center', va='center', fontsize=12)
        return fig
    labels = list(valid_components.keys())
    values = list(valid_components.values())
    bars = ax.bar(labels, values, color=sns.color_palette("pastel", len(labels)))
    ax.set_ylabel("Average Time (ms)", fontsize=12)
    ax.set_title("Average Time Composition", fontsize=15)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.01, f"{yval:.2f} ms", ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig

def generate_performance_timeline_plot_mpl(perf_df: pl.DataFrame, time_col: str = "total_time_ms", title: Optional[str] = None) -> plt.Figure:
    # Implementation remains the same
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = perf_df.with_row_count("_event_index") if "_event_index" not in perf_df.columns else perf_df
    data_pd = plot_df.select(["_event_index", pl.col(time_col).drop_nulls()]).to_pandas()
    if data_pd.empty or data_pd[time_col].isnull().all():
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
        return fig
    sns.scatterplot(x="_event_index", y=time_col, data=data_pd, ax=ax, alpha=0.6, s=30)
    ax.set_title(title or f"{time_col.replace('_', ' ').title()} Over Time", fontsize=15)
    ax.set_xlabel("Event Index (Sequence)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    plt.tight_layout()
    return fig

def generate_iou_consistency_distribution_plot_mpl(stability_details_df: pl.DataFrame, iou_col: str = "mean_iou_consistency", title: Optional[str] = None) -> plt.Figure:
    # Implementation remains the same
    fig, ax = plt.subplots(figsize=(10, 6))
    if stability_details_df.is_empty() or iou_col not in stability_details_df.columns:
        ax.text(0.5, 0.5, "No data for IoU consistency", ha='center', va='center', fontsize=12)
        return fig
    data_pd = stability_details_df.select(pl.col(iou_col).drop_nulls()).to_pandas()
    if data_pd.empty or data_pd[iou_col].isnull().all():
        ax.text(0.5, 0.5, "No valid data for IoU consistency", ha='center', va='center', fontsize=12)
        return fig
    sns.histplot(data_pd[iou_col], kde=True, ax=ax, bins=20, color="cornflowerblue")
    ax.set_title(title or "Distribution of Mean IoU Consistency", fontsize=15)
    ax.set_xlabel("Mean IoU Consistency", fontsize=12)
    ax.set_ylabel("Number of Tracked Objects", fontsize=12)
    plt.tight_layout()
    return fig

def generate_center_drift_distribution_plot_mpl(stability_details_df: pl.DataFrame, drift_col: str = "mean_center_drift_px", title: Optional[str] = None) -> plt.Figure:
    # Implementation remains the same
    fig, ax = plt.subplots(figsize=(10, 6))
    if stability_details_df.is_empty() or drift_col not in stability_details_df.columns:
        ax.text(0.5, 0.5, "No data for center drift", ha='center', va='center', fontsize=12)
        return fig
    data_pd = stability_details_df.select(pl.col(drift_col).drop_nulls()).to_pandas()
    if data_pd.empty or data_pd[drift_col].isnull().all():
        ax.text(0.5, 0.5, "No valid data for center drift", ha='center', va='center', fontsize=12)
        return fig
    sns.histplot(data_pd[drift_col], kde=True, ax=ax, bins=20, color="mediumseagreen")
    ax.set_title(title or "Distribution of Mean Center Point Drift", fontsize=15)
    ax.set_xlabel("Mean Center Drift (pixels)", fontsize=12)
    ax.set_ylabel("Number of Tracked Objects", fontsize=12)
    plt.tight_layout()
    return fig

# --- Plotly Chart Generators (for HTML reports) ---

def generate_latency_distribution_plotly(perf_df: pl.DataFrame, time_col: str = "total_time_ms", title: Optional[str] = None) -> go.Figure:
    # Implementation remains the same
    data_pd = perf_df.select(pl.col(time_col).drop_nulls()).to_pandas()
    if data_pd.empty or data_pd[time_col].isnull().all():
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = go.Figure()
    fig.add_trace(go.Violin(y=data_pd[time_col], name=time_col.replace('_', ' ').title(), box_visible=True, meanline_visible=True, points='all', jitter=0.3, pointpos=-1.8))
    fig.update_layout(title_text=title or f"{time_col.replace('_', ' ').title()} Distribution (Interactive)", yaxis_title="Time (ms)")
    return fig

def generate_time_composition_plotly(metrics: Dict[str, Any]) -> go.Figure:
    # Implementation remains the same
    components = {"Pre-processing": metrics.get("perf_mean_pre_time_ms"), "Inference": metrics.get("perf_mean_inference_time_ms"), "Post-processing": metrics.get("perf_mean_post_time_ms")}
    valid_components = {k: v for k, v in components.items() if v is not None and isinstance(v, (int, float)) and v > 0}
    if not valid_components:
        return go.Figure().add_annotation(text="No data for time composition", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.bar(x=list(valid_components.keys()), y=list(valid_components.values()), labels={'x': 'Processing Stage', 'y': 'Average Time (ms)'}, text_auto='.2f')
    fig.update_layout(title_text="Average Time Composition (Interactive)")
    fig.update_traces(textposition='outside')
    return fig

def generate_performance_timeline_plotly(perf_df: pl.DataFrame, time_col: str = "total_time_ms", title: Optional[str] = None) -> go.Figure:
    plot_df = perf_df.with_row_count("_event_index") if "_event_index" not in perf_df.columns else perf_df
    data = plot_df.select(["_event_index", pl.col(time_col).drop_nulls()])
    if data.is_empty() or data[time_col].null_count() == data.height:
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.scatter(data.to_pandas(), x="_event_index", y=time_col, title=title or f"{time_col.replace('_', ' ').title()} Over Time (Interactive)")
    fig.update_layout(xaxis_title="Event Index (Sequence)", yaxis_title="Time (ms)")
    return fig

def generate_iou_consistency_distribution_plotly(stability_details_df: pl.DataFrame, iou_col: str = "mean_iou_consistency", title: Optional[str] = None) -> go.Figure:
    if stability_details_df.is_empty() or iou_col not in stability_details_df.columns:
        return go.Figure().add_annotation(text="No data for IoU consistency", xref="paper", yref="paper", showarrow=False, font_size=12)
    data = stability_details_df.select(pl.col(iou_col).drop_nulls())
    if data.is_empty() or data[iou_col].null_count() == data.height:
        return go.Figure().add_annotation(text="No valid data for IoU consistency", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.histogram(data.to_pandas(), x=iou_col, marginal="box", title=title or "Distribution of Mean IoU Consistency (Interactive)", nbins=30)
    fig.update_layout(xaxis_title="Mean IoU Consistency", yaxis_title="Number of Tracked Objects")
    return fig

def generate_center_drift_distribution_plotly(stability_details_df: pl.DataFrame, drift_col: str = "mean_center_drift_px", title: Optional[str] = None) -> go.Figure:
    if stability_details_df.is_empty() or drift_col not in stability_details_df.columns:
        return go.Figure().add_annotation(text="No data for center drift", xref="paper", yref="paper", showarrow=False, font_size=12)
    data = stability_details_df.select(pl.col(drift_col).drop_nulls())
    if data.is_empty() or data[drift_col].null_count() == data.height:
        return go.Figure().add_annotation(text="No valid data for center drift", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.histogram(data.to_pandas(), x=drift_col, marginal="box", title=title or "Distribution of Mean Center Point Drift (Interactive)", nbins=30)
    fig.update_layout(xaxis_title="Mean Center Drift (pixels)", yaxis_title="Number of Tracked Objects")
    return fig

# --- Classification Stability Charts (Plotly) ---
def generate_top_1_consistency_distribution_plotly(cls_stability_df: pl.DataFrame, consistency_col: str = "top_1_consistency_rate", title: Optional[str] = None) -> go.Figure:
    if cls_stability_df.is_empty() or consistency_col not in cls_stability_df.columns:
        return go.Figure().add_annotation(text="No data for Top-1 Consistency", xref="paper", yref="paper", showarrow=False, font_size=12)
    data = cls_stability_df.select(pl.col(consistency_col).drop_nulls())
    if data.is_empty() or data[consistency_col].null_count() == data.height:
         return go.Figure().add_annotation(text="No valid data for Top-1 Consistency", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.histogram(data.to_pandas(), x=consistency_col, title=title or "Distribution of Top-1 Consistency Rate per Image (Interactive)", nbins=20, range_x=[0,1])
    fig.update_layout(xaxis_title="Top-1 Consistency Rate", yaxis_title="Number of Images")
    return fig

def generate_jaccard_similarity_trends_plotly(metrics: Dict[str, Any], k_values: List[int], title: Optional[str] = None) -> go.Figure:
    jaccard_data = []
    for k in k_values:
        metric_name = f"cls_stab_mean_jaccard_top_{k}"
        if metric_name in metrics and metrics[metric_name] is not None:
            jaccard_data.append({"K": f"Top-{k}", "Mean Jaccard Similarity": metrics[metric_name]})

    if not jaccard_data:
        return go.Figure().add_annotation(text="No Jaccard similarity data available", xref="paper", yref="paper", showarrow=False, font_size=12)

    fig = px.bar(
        jaccard_data, x="K", y="Mean Jaccard Similarity",
        title=title or "Mean Top-K Jaccard Similarity (Interactive)",
        text="Mean Jaccard Similarity" # Show values on bars
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0,1])
    return fig

def generate_confidence_std_distribution_plotly(cls_stability_df: pl.DataFrame, conf_std_col: str = "top_1_confidence_std", title: Optional[str] = None) -> go.Figure:
    if cls_stability_df.is_empty() or conf_std_col not in cls_stability_df.columns:
        return go.Figure().add_annotation(text="No data for Confidence Std Dev", xref="paper", yref="paper", showarrow=False, font_size=12)
    data = cls_stability_df.select(pl.col(conf_std_col).drop_nulls())
    if data.is_empty() or data[conf_std_col].null_count() == data.height:
        return go.Figure().add_annotation(text="No valid data for Confidence Std Dev", xref="paper", yref="paper", showarrow=False, font_size=12)
    fig = px.histogram(data.to_pandas(), x=conf_std_col, title=title or "Distribution of Top-1 Confidence Std Dev per Image (Interactive)", nbins=20)
    fig.update_layout(xaxis_title="Top-1 Confidence Std Dev", yaxis_title="Number of Images")
    return fig


# --- Main test block ---
if __name__ == "__main__":
    logger.info("--- Testing Chart Generation Functions (Matplotlib/Seaborn) ---")

    dummy_perf_data_dict = {
        "total_time_ms": [100, 110, 105, 120, 95, 100, 115, 130, 90, 108, 200, 210, None],
        "pre_time_ms": [10,10,10,10,10,10,10,10,10,10,10,10,10],
        "inference_time_ms": [80,90,85,100,75,80,95,110,70,88,180,190, None],
        "post_time_ms": [10,10,10,10,10,10,10,10,10,10,10,10, 10]
    }
    dummy_perf_df_main = pl.DataFrame(dummy_perf_data_dict)
    deduplicated_perf_df = dummy_perf_df_main.unique(subset=["total_time_ms"], keep="first")

    dummy_metrics_main = {
        "perf_mean_pre_time_ms": dummy_perf_df_main["pre_time_ms"].mean(),
        "perf_mean_inference_time_ms": dummy_perf_df_main["inference_time_ms"].mean(),
        "perf_mean_post_time_ms": dummy_perf_df_main["post_time_ms"].mean(),
        "cls_stab_mean_jaccard_top_1": 0.8,
        "cls_stab_mean_jaccard_top_3": 0.7,
        "cls_stab_mean_jaccard_top_5": 0.6,
    }

    dummy_det_stability_df = pl.DataFrame([
        {"mean_iou_consistency": 0.8, "mean_center_drift_px": 5.0},
        {"mean_iou_consistency": 0.9, "mean_center_drift_px": 2.0}
    ])
    dummy_cls_stability_df = pl.DataFrame([
        {"top_1_consistency_rate": 1.0, "top_1_confidence_std": 0.05},
        {"top_1_consistency_rate": 0.5, "top_1_confidence_std": 0.15}
    ])


    test_charts_dir = Path("./test_generated_charts_output")
    test_charts_dir.mkdir(exist_ok=True)

    def save_test_fig_mpl(fig: plt.Figure, name: str):
        try:
            path = test_charts_dir / f"{name}_mpl.png"; fig.savefig(path)
            logger.info(f"Saved MPL chart: {path}")
        except Exception as e: logger.error(f"Failed to save {name}_mpl.png: {e}")
        finally: plt.close(fig)

    def save_test_fig_plotly(fig: go.Figure, name: str):
        try:
            path = test_charts_dir / f"{name}_plotly.html"; fig.write_html(path)
            logger.info(f"Saved Plotly chart: {path}")
        except Exception as e: logger.error(f"Failed to save {name}_plotly.html: {e}")

    # Test performance charts
    save_test_fig_mpl(generate_latency_distribution_plot_mpl(deduplicated_perf_df), "latency_dist")
    save_test_fig_plotly(generate_latency_distribution_plotly(deduplicated_perf_df), "latency_dist")

    save_test_fig_mpl(generate_time_composition_plot_mpl(dummy_metrics_main), "time_comp")
    save_test_fig_plotly(generate_time_composition_plotly(dummy_metrics_main), "time_comp")

    save_test_fig_mpl(generate_performance_timeline_plot_mpl(deduplicated_perf_df), "perf_timeline")
    save_test_fig_plotly(generate_performance_timeline_plotly(deduplicated_perf_df), "perf_timeline")

    # Test detection stability charts
    save_test_fig_mpl(generate_iou_consistency_distribution_plot_mpl(dummy_det_stability_df), "iou_consistency_dist")
    save_test_fig_plotly(generate_iou_consistency_distribution_plotly(dummy_det_stability_df), "iou_consistency_dist")

    save_test_fig_mpl(generate_center_drift_distribution_plot_mpl(dummy_det_stability_df), "center_drift_dist")
    save_test_fig_plotly(generate_center_drift_distribution_plotly(dummy_det_stability_df), "center_drift_dist")

    # Test classification stability charts (Plotly only as per current additions)
    save_test_fig_plotly(generate_top_1_consistency_distribution_plotly(dummy_cls_stability_df), "cls_top1_consistency_dist")
    save_test_fig_plotly(generate_jaccard_similarity_trends_plotly(dummy_metrics_main, k_values=[1,3,5]), "cls_jaccard_trends")
    save_test_fig_plotly(generate_confidence_std_distribution_plotly(dummy_cls_stability_df), "cls_conf_std_dist")

    logger.info(f"Chart generation function tests completed. Check figs in {test_charts_dir.resolve()}")
```
