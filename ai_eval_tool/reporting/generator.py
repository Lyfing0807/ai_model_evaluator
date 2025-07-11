"""
Report Generator: Creates evaluation reports in Markdown and HTML formats.
"""
from pathlib import Path
from typing import Dict, List, Any
import polars as pl
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import yaml
from datetime import datetime

from ..config_manager import ReportSettings, MainConfig, ClassificationEvaluationParams
from ..utils.types import EvaluationResult
from ..utils.logging_config import get_logger
from . import charts as chart_generators

logger = get_logger(__name__)

class ReportGenerator:
    def __init__(self, config: MainConfig):
        self.main_config = config
        self.report_settings: ReportSettings = config.report_settings
        self.project_info = config.project_info

        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

        logger.info("ReportGenerator initialized.")
        self.report_settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_chart_mpl(self, fig: plt.Figure, chart_name: str, run_id: str, charts_subdir: Path) -> Path:
        charts_subdir.mkdir(parents=True, exist_ok=True)
        chart_filename = f"{chart_name}.png"
        chart_path_abs = charts_subdir / chart_filename
        try:
            fig.savefig(chart_path_abs)
            logger.info(f"Saved Matplotlib chart: {chart_path_abs}")
        except Exception as e:
            logger.error(f"Failed to save Matplotlib chart {chart_name} to {chart_path_abs}: {e}")
            plt.close(fig)
            return Path(f"ERROR_SAVING_{chart_name}.png")
        plt.close(fig)
        return Path(run_id) / "charts" / chart_filename

    def _get_plotly_div(self, fig_generator, *args, **kwargs) -> str:
        """Generates a Plotly figure and returns its HTML div string."""
        try:
            fig = fig_generator(*args, **kwargs)
            if fig is None: # Handle cases where chart gen returns None (e.g. no data)
                 return "<p><i>Chart could not be generated (likely no data).</i></p>"
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Failed to generate Plotly div for {fig_generator.__name__}: {e}", exc_info=True)
            return f"<p><i>Error generating Plotly chart: {fig_generator.__name__}.</i></p>"


    def _prepare_report_context(self, eval_results: EvaluationResult, run_id: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "project_info": self.project_info.model_dump(),
            "run_id": run_id,
            "metrics": eval_results.metrics,
            "config_yaml": yaml.dump(self.main_config.model_dump(exclude_none=True), indent=2, sort_keys=False),
            "model_type": self.project_info.model_type,
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "md_charts_paths": {},
            "plotly_charts": {}, # For HTML divs
            "extra_data_html": {},
            "extra_data_csv_paths": {},
        }

        run_output_dir = self.report_settings.output_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        charts_subdir = run_output_dir / "charts"
        extra_data_subdir = run_output_dir / "data"
        extra_data_subdir.mkdir(parents=True, exist_ok=True)

        # --- Generate and Save/Prepare Charts ---
        perf_df = eval_results.extra_data.get("deduplicated_perf_df_for_charts")

        if perf_df is not None and isinstance(perf_df, pl.DataFrame) and not perf_df.is_empty():
            # Matplotlib versions for MD
            fig_lat_mpl = chart_generators.generate_latency_distribution_plot_mpl(perf_df)
            context["md_charts_paths"]["latency_distribution"] = self._save_chart_mpl(fig_lat_mpl, "latency_distribution", run_id, charts_subdir)
            fig_perf_tl_mpl = chart_generators.generate_performance_timeline_plot_mpl(perf_df)
            context["md_charts_paths"]["performance_timeline"] = self._save_chart_mpl(fig_perf_tl_mpl, "performance_timeline", run_id, charts_subdir)

            # Plotly versions for HTML
            context["plotly_charts"]["latency_distribution"] = self._get_plotly_div(chart_generators.generate_latency_distribution_plotly, perf_df)
            context["plotly_charts"]["performance_timeline"] = self._get_plotly_div(chart_generators.generate_performance_timeline_plotly, perf_df)

        if eval_results.metrics:
            fig_tc_mpl = chart_generators.generate_time_composition_plot_mpl(eval_results.metrics)
            context["md_charts_paths"]["time_composition"] = self._save_chart_mpl(fig_tc_mpl, "time_composition", run_id, charts_subdir)
            context["plotly_charts"]["time_composition"] = self._get_plotly_div(chart_generators.generate_time_composition_plotly, eval_results.metrics)

        # Model-specific charts
        if self.project_info.model_type == "detection":
            obj_stab_df = eval_results.extra_data.get("object_stability_details_df")
            if obj_stab_df is not None and isinstance(obj_stab_df, pl.DataFrame) and not obj_stab_df.is_empty():
                fig_iou_mpl = chart_generators.generate_iou_consistency_distribution_plot_mpl(obj_stab_df)
                context["md_charts_paths"]["iou_consistency_distribution"] = self._save_chart_mpl(fig_iou_mpl, "iou_consistency_distribution", run_id, charts_subdir)
                fig_drift_mpl = chart_generators.generate_center_drift_distribution_plot_mpl(obj_stab_df)
                context["md_charts_paths"]["center_drift_distribution"] = self._save_chart_mpl(fig_drift_mpl, "center_drift_distribution", run_id, charts_subdir)

                context["plotly_charts"]["iou_consistency_distribution"] = self._get_plotly_div(chart_generators.generate_iou_consistency_distribution_plotly, obj_stab_df)
                context["plotly_charts"]["center_drift_distribution"] = self._get_plotly_div(chart_generators.generate_center_drift_distribution_plotly, obj_stab_df)

        elif self.project_info.model_type == "classification":
            cls_stab_df = eval_results.extra_data.get("classification_stability_details_df")
            cls_eval_params = self.main_config.evaluation_params.classification

            if cls_stab_df is not None and isinstance(cls_stab_df, pl.DataFrame) and not cls_stab_df.is_empty():
                context["plotly_charts"]["top_1_consistency_distribution"] = self._get_plotly_div(chart_generators.generate_top_1_consistency_distribution_plotly, cls_stab_df)
                context["plotly_charts"]["confidence_std_distribution"] = self._get_plotly_div(chart_generators.generate_confidence_std_distribution_plotly, cls_stab_df)

            if cls_eval_params and eval_results.metrics: # For Jaccard trends
                 k_vals = cls_eval_params.top_k
                 context["plotly_charts"]["jaccard_similarity_trends"] = self._get_plotly_div(chart_generators.generate_jaccard_similarity_trends_plotly, eval_results.metrics, k_vals)

        # --- Process Extra Data ---
        for key, data_item in eval_results.extra_data.items():
            if isinstance(data_item, pl.DataFrame):
                try:
                    if data_item.height > 0 :
                        csv_filename = f"{key}.csv"
                        csv_path_abs = extra_data_subdir / csv_filename
                        data_item.write_csv(csv_path_abs)
                        context["extra_data_csv_paths"][key] = Path(run_id) / "data" / csv_filename
                        df_html = data_item.head(15).to_pandas().to_html(classes="table table-striped table-sm", index=False, border=0, escape=True)
                        context["extra_data_html"][key] = df_html
                    else:
                        context["extra_data_html"][key] = "<p><i>DataFrame is empty.</i></p>"
                except Exception as e:
                    logger.error(f"Failed to process/convert DataFrame {key} for report: {e}")
                    context["extra_data_html"][key] = f"<p><i>Error processing DataFrame {key}.</i></p>"

        logger.debug("Report context prepared with chart paths and extra data.")
        return context

    def _generate_md_report(self, context: Dict[str, Any], run_id: str):
        report_path = self.report_settings.output_dir / f"{run_id}_report.md"
        logger.info(f"Generating Markdown report: {report_path}")

        content = f"# Evaluation Report: {context['project_info']['project_name']}\n\n"
        content += f"**Run ID:** `{run_id}`\n"
        content += f"**Model Type:** {context['model_type']}\n"
        content += f"**Model Version:** {context['project_info'].get('model_version', 'N/A')}\n"
        content += f"**Generated on:** {context['generation_timestamp']}\n\n"

        content += "## Metrics Summary\n\n"
        if context["metrics"]:
            perf_metrics = {k: v for k, v in context["metrics"].items() if k.startswith("perf_")}
            model_stab_prefix = f"{context['model_type'][:3]}_stab_" if context['model_type'] != "detection" else "det_stab_"
            stab_metrics = {k: v for k, v in context["metrics"].items() if k.startswith(model_stab_prefix)}
            other_metrics = {k:v for k,v in context["metrics"].items() if k not in perf_metrics and k not in stab_metrics}

            if perf_metrics:
                content += "### Performance Metrics\n"
                for key, value in perf_metrics.items():
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    if value is None: val_str = "N/A"
                    content += f"- **{key.replace('perf_', '', 1).replace('_', ' ').title()}**: `{val_str}`\n"
                content += "\n"

            if stab_metrics:
                content += f"### {context['model_type'].title()} Stability Metrics\n"
                for key, value in stab_metrics.items():
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    if value is None: val_str = "N/A"
                    content += f"- **{key.replace(model_stab_prefix, '', 1).replace('_', ' ').title()}**: `{val_str}`\n"
                content += "\n"

            if other_metrics:
                content += "### Other Metrics\n"
                for key, value in other_metrics.items():
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    if value is None: val_str = "N/A"
                    content += f"- **{key.replace('_', ' ').title()}**: `{val_str}`\n"
                content += "\n"
        else:
            content += "_No metrics were generated._\n\n"

        content += "## Charts\n\n"
        if context["md_charts_paths"]:
            for chart_name, chart_path in context["md_charts_paths"].items():
                md_relative_chart_path = chart_path
                content += f"### {chart_name.replace('_', ' ').title()}\n"
                content += f"![{chart_name.replace('_', ' ').title()}]({md_relative_chart_path})\n\n"
        else:
            content += "_No charts were generated for this report._\n\n"

        content += "## Additional Data\n\n"
        if context["extra_data_csv_paths"]:
            for key, csv_path in context["extra_data_csv_paths"].items():
                content += f"### {key.replace('_', ' ').title()}\n"
                content += f"Data saved to: [{csv_path.name}]({csv_path})\n\n"
        else:
            content += "_No additional data files were generated._\n\n"

        content += "\n## Configuration Used\n\n"
        content += "```yaml\n"
        content += context['config_yaml']
        content += "\n```\n"

        try:
            with open(report_path, "w", encoding="utf-8") as f: f.write(content)
            logger.info(f"Markdown report saved to {report_path}")
        except IOError as e:
            logger.error(f"Failed to write Markdown report {report_path}: {e}")

    def _generate_html_report(self, context: Dict[str, Any], run_id: str):
        report_path = self.report_settings.output_dir / f"{run_id}_report.html"
        logger.info(f"Generating HTML report: {report_path}")

        html_context = context.copy()
        # Static MPL charts for HTML report (can be primary if Plotly fails or for speed)
        html_context["static_charts_for_html"] = {}
        for name, chart_path_obj in context.get("md_charts_paths", {}).items():
            html_context["static_charts_for_html"][name] = {
                "title": name.replace('_', ' ').title(),
                "path": str(chart_path_obj)
            }
        # Plotly charts are already in context["plotly_charts"] as HTML divs

        try:
            template = self.jinja_env.get_template("report_template.html")
            html_content = template.render(html_context)
            with open(report_path, "w", encoding="utf-8") as f: f.write(html_content)
            logger.info(f"HTML report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report {report_path}: {e}", exc_info=True)

    def generate(self, eval_results: EvaluationResult, run_id: str):
        logger.info(f"Starting report generation for run ID: {run_id}.")
        report_data_context = self._prepare_report_context(eval_results, run_id)
        if "md" in self.report_settings.formats:
            self._generate_md_report(report_data_context, run_id)
        if "html" in self.report_settings.formats:
            self._generate_html_report(report_data_context, run_id)
        logger.info(f"Report generation completed. Reports saved in: {self.report_settings.output_dir.resolve()}")


if __name__ == "__main__":
    from ai_eval_tool.config_manager import load_config
    from ai_eval_tool.utils.logging_config import setup_logging
    setup_logging(level="DEBUG")

    dummy_config_content = """
project_info:
  project_name: "ReportGen Full Test"
  model_type: "classification" # Test with classification
  model_version: "v0.2-test"
data_loader:
  field_mapping: {loop: l, image_id: i, image_path: p, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4,
                  classification: {top_k_id_pattern: "id{k}", top_k_score_pattern: "sc{k}"}}
evaluation_params:
  classification: {top_k: [1, 3]}
report_settings:
  output_dir: "./test_full_report_output"
  formats: ["md", "html"]
"""
    dummy_config_path = Path("dummy_reportgen_full_config.yaml")
    with open(dummy_config_path, "w") as f: f.write(dummy_config_content)

    template_dir_for_test = Path(__file__).parent / "templates"
    if not (template_dir_for_test / "report_template.html").exists():
        # Create a minimal template if it's missing for the test
        (template_dir_for_test / "report_template.html").write_text(
            "<html><head><title>Report {{ project_info.project_name }}</title></head><body>"
            "<h1>Report for {{ run_id }}</h1>"
            "<h2>Metrics</h2><ul>{% for k,v in metrics.items() %}<li>{{k}}: {{v}}</li>{% endfor %}</ul>"
            "<h2>Static Charts</h2>{% for name, chart in static_charts_for_html.items() %}"
            "<h3>{{chart.title}}</h3><img src='{{chart.path}}' alt='{{chart.title}}' />{% endfor %}"
            "<h2>Plotly Charts</h2>{% for name, div_html in plotly_charts.items() %}"
            "<h3>{{name|replace('_',' ')|title}}</h3><div>{{div_html|safe}}</div>{% endfor %}"
            "<h2>Config</h2><pre>{{config_yaml}}</pre>"
            "</body></html>"
        )


    try:
        logger.info("--- Testing ReportGenerator (Full Flow with Plotly) ---")
        config = load_config(dummy_config_path)
        report_generator = ReportGenerator(config)

        dummy_perf_df = pl.DataFrame({"total_time_ms": [100, 110, 105, 120, 95], "_event_index": range(5)})
        dummy_cls_stab_df = pl.DataFrame({"top_1_consistency_rate": [1.0, 0.5], "top_1_confidence_std": [0.01, 0.05]})

        dummy_eval_results = EvaluationResult(
            metrics={
                "perf_mean_total_time_ms": 106.0, "perf_avg_fps": 9.43,
                "perf_mean_pre_time_ms": 10.0, "perf_mean_inference_time_ms": 80.0, "perf_mean_post_time_ms": 10.0,
                "cls_stab_mean_top_1_consistency_rate": 0.75,
                "cls_stab_mean_jaccard_top_1": 0.8, "cls_stab_mean_jaccard_top_3": 0.7,
                "cls_stab_mean_top_1_confidence_std": 0.03,
            },
            extra_data={
                "deduplicated_perf_df_for_charts": dummy_perf_df,
                "classification_stability_details_df": dummy_cls_stab_df,
                "perf_top_slow_samples_df": dummy_perf_df.sort("total_time_ms", descending=True).head(2)
            }
        )
        test_run_id = "test_plotly_run_" + datetime.now().strftime("%Y%m%d%H%M%S")

        report_generator.generate(dummy_eval_results, test_run_id)

        output_dir = config.report_settings.output_dir
        assert (output_dir / f"{test_run_id}_report.md").exists()
        assert (output_dir / f"{test_run_id}_report.html").exists()
        charts_output_dir = output_dir / test_run_id / "charts"
        assert charts_output_dir.exists()
        assert len(list(charts_output_dir.glob("*.png"))) > 0
        data_output_dir = output_dir / test_run_id / "data"
        assert data_output_dir.exists()
        assert (data_output_dir / "perf_top_slow_samples_df.csv").exists()

        logger.info(f"Test reports generated in: {output_dir.resolve()}")

    except Exception as e:
        logger.error(f"Error during ReportGenerator full test: {e}", exc_info=True)
        raise
    finally:
        if dummy_config_path.exists(): dummy_config_path.unlink()
        logger.info("ReportGenerator full test completed. Please check ./test_full_report_output")
