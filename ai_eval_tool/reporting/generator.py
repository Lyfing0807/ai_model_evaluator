"""
Report Generator: Creates evaluation reports in Markdown and HTML formats.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import polars as pl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from jinja2 import Environment, FileSystemLoader
import yaml
from datetime import datetime
import shutil

from ..config_manager import ReportSettings, MainConfig, ClassificationEvaluationParams
from ..utils.types import EvaluationResult
from ..utils.logging_config import get_logger
from . import charts as chart_generators
from .insights import AIInsightsGenerator # Import AIInsightsGenerator

logger = get_logger(__name__)

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 15)
except IOError:
    try: FONT = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        logger.warning("DejaVuSans.ttf or Arial.ttf not found. Using Pillow's default font.")
        FONT = ImageFont.load_default()

def _draw_bounding_box(draw: ImageDraw.ImageDraw, box: List[float], color: str = "red", width: int = 2, label: Optional[str] = None):
    xmin, ymin, xmax, ymax = box
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)
    if label:
        try: # textbbox can fail with some fonts / PIL versions if text is empty or very small
            text_bbox = draw.textbbox((xmin, ymin - 20 if ymin - 20 > 0 else ymin), label, font=FONT)
            draw.rectangle(text_bbox, fill=color)
            draw.text((xmin, ymin - 20 if ymin - 20 > 0 else ymin), label, fill="white", font=FONT)
        except Exception as e:
            logger.warning(f"Failed to draw text label '{label}' on image: {e}")
            draw.text((xmin, ymin -5), "LabelErr", fill="white", font=ImageFont.load_default()) # Fallback


class ReportGenerator:
    def __init__(self, config: MainConfig):
        self.main_config = config
        self.report_settings: ReportSettings = config.report_settings
        self.project_info = config.project_info
        self.image_base_dir: Optional[Path] = config.data_loader.image_base_dir

        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

        self.ai_insights_generator: Optional[AIInsightsGenerator] = None
        if self.report_settings.ai_insights.enabled:
            try:
                self.ai_insights_generator = AIInsightsGenerator(
                    insights_config=self.report_settings.ai_insights,
                    project_config=self.main_config
                )
                logger.info("AIInsightsGenerator initialized for ReportGenerator.")
            except Exception as e:
                logger.error(f"Failed to initialize AIInsightsGenerator: {e}", exc_info=True)

        logger.info("ReportGenerator initialized.")
        self.report_settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_chart_mpl(self, fig: plt.Figure, chart_name: str, run_id: str, charts_subdir: Path) -> Path:
        charts_subdir.mkdir(parents=True, exist_ok=True)
        chart_filename = f"{chart_name}.png"
        chart_path_abs = charts_subdir / chart_filename
        try: fig.savefig(chart_path_abs); logger.info(f"Saved Matplotlib chart: {chart_path_abs}")
        except Exception as e: logger.error(f"Failed to save Matplotlib chart {chart_name} to {chart_path_abs}: {e}"); plt.close(fig); return Path(f"ERROR_SAVING_{chart_name}.png")
        plt.close(fig)
        return Path(run_id) / "charts" / chart_filename

    def _get_plotly_div(self, fig_generator, *args, **kwargs) -> str:
        try:
            fig = fig_generator(*args, **kwargs)
            if fig is None: return "<p><i>Chart could not be generated (likely no data).</i></p>"
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        except Exception as e: logger.error(f"Failed to generate Plotly div for {fig_generator.__name__}: {e}", exc_info=True); return f"<p><i>Error generating Plotly chart: {fig_generator.__name__}.</i></p>"

    def _resolve_image_path(self, relative_path_str: Optional[str]) -> Optional[Path]:
        """
        Resolves image path with enhanced error handling and path normalization.
        
        Args:
            relative_path_str: Image path string from CSV data
            
        Returns:
            Absolute path to the image file if found, None otherwise
        """
        if not relative_path_str or not isinstance(relative_path_str, str):
            return None
            
        try:
            # Normalize path separators for cross-platform compatibility
            normalized_path_str = relative_path_str.replace('\\', '/').strip()
            relative_path = Path(normalized_path_str)
            
            # If already absolute and exists, return it
            if relative_path.is_absolute():
                return relative_path if relative_path.exists() and relative_path.is_file() else None
            
            # Try resolving relative to image_base_dir
            if self.image_base_dir:
                abs_path = self.image_base_dir / relative_path
                if abs_path.exists() and abs_path.is_file():
                    return abs_path.resolve()  # Resolve to canonical path
            
            # Try resolving relative to current working directory as fallback
            cwd_path = Path.cwd() / relative_path
            if cwd_path.exists() and cwd_path.is_file():
                logger.debug(f"Found image relative to CWD: {cwd_path}")
                return cwd_path.resolve()
                
            logger.debug(f"Could not resolve image path: {relative_path_str} (base_dir: {self.image_base_dir})")
            return None
            
        except Exception as e:
            logger.warning(f"Error resolving image path '{relative_path_str}': {e}")
            return None

    def _process_image_for_report(self, resolved_img_path: Path, row_dict: Dict[str, Any], 
                                 sample_index: int, run_id: str, annotated_img_subdir: Path, 
                                 original_path_str: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Processes an image for report display: creates thumbnail and optionally annotated version.
        
        Args:
            resolved_img_path: Absolute path to the source image
            row_dict: Row data containing metadata
            sample_index: Index of the sample for unique naming
            run_id: Report run ID
            annotated_img_subdir: Directory for processed images
            original_path_str: Original path string for filename generation
            
        Returns:
            Tuple of (thumbnail_rel_path, lightbox_rel_path) relative to report root
        """
        try:
            # Generate safe filename components
            img_name_slug = self._generate_safe_filename(original_path_str, sample_index)
            loop_id = row_dict.get('loop', row_dict.get('loop_id', 'L'))
            image_id_val = row_dict.get('image_id', row_dict.get('img_name', f'ID{sample_index}'))
            
            # Create base filename with sanitized components
            base_filename = f"{img_name_slug}_{loop_id}_{image_id_val}"
            base_filename = self._sanitize_filename(base_filename)
            
            # Validate and open source image
            if not self._validate_image_file(resolved_img_path):
                logger.warning(f"Invalid image file: {resolved_img_path}")
                return None, None
            
            # Create thumbnail
            thumbnail_rel_path = self._create_thumbnail(
                resolved_img_path, base_filename, annotated_img_subdir, run_id
            )
            
            # Create lightbox image (annotated or original copy)
            lightbox_rel_path = self._create_lightbox_image(
                resolved_img_path, base_filename, annotated_img_subdir, run_id, row_dict, original_path_str
            )
            
            return thumbnail_rel_path, lightbox_rel_path
            
        except Exception as e:
            logger.error(f"Error processing image {resolved_img_path} for report: {e}", exc_info=True)
            return None, None

    def _generate_safe_filename(self, original_path_str: Optional[str], sample_index: int) -> str:
        """Generate a safe filename component from the original path."""
        if original_path_str:
            try:
                return Path(original_path_str).stem.replace(" ", "_")[:50]  # Limit length
            except Exception:
                pass
        return f"unknown_img_{sample_index}"

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing problematic characters."""
        import re
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Limit total length
        return sanitized[:100]

    def _validate_image_file(self, img_path: Path) -> bool:
        """Validate that the file is a readable image."""
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            return True
        except Exception as e:
            logger.debug(f"Image validation failed for {img_path}: {e}")
            return False

    def _create_thumbnail(self, source_path: Path, base_filename: str, 
                         output_dir: Path, run_id: str) -> Optional[Path]:
        """Create a thumbnail image."""
        try:
            thumbnail_filename = f"thumb_{base_filename}.png"
            thumbnail_abs_path = output_dir / thumbnail_filename
            
            with Image.open(source_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                img.save(thumbnail_abs_path, 'PNG', optimize=True)
                
            return Path(run_id) / "charts" / "annotated_images" / thumbnail_filename
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {source_path}: {e}")
            return None

    def _create_lightbox_image(self, source_path: Path, base_filename: str, 
                              output_dir: Path, run_id: str, row_dict: Dict[str, Any], 
                              original_path_str: Optional[str]) -> Optional[Path]:
        """Create lightbox image (annotated or copied original)."""
        try:
            if (self.report_settings.display_images.draw_annotations and 
                self.project_info.model_type in ["detection", "pose", "rotated_detection"]):
                
                return self._create_annotated_image(
                    source_path, base_filename, output_dir, run_id, row_dict
                )
            else:
                return self._copy_original_image(
                    source_path, base_filename, output_dir, run_id, original_path_str
                )
                
        except Exception as e:
            logger.error(f"Failed to create lightbox image for {source_path}: {e}")
            return None

    def _create_annotated_image(self, source_path: Path, base_filename: str, 
                               output_dir: Path, run_id: str, row_dict: Dict[str, Any]) -> Optional[Path]:
        """Create an annotated version of the image."""
        try:
            annotated_filename = f"annotated_{base_filename}.png"
            annotated_abs_path = output_dir / annotated_filename
            
            with Image.open(source_path).convert("RGBA") as img:
                draw = ImageDraw.Draw(img)
                
                # Add model-specific annotations
                if self.project_info.model_type == "detection" and "internal_bbox" in row_dict:
                    bbox = row_dict["internal_bbox"]
                    if bbox and len(bbox) == 4:
                        label = f"Cat: {row_dict.get('category_id', 'N/A')}, Score: {row_dict.get('score', 0.0):.2f}"
                        _draw_bounding_box(draw, bbox, label=label)
                
                # TODO: Add annotations for pose and rotated_detection
                
                # Convert back to RGB for PNG saving
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                
                img.save(annotated_abs_path, 'PNG', optimize=True)
                
            return Path(run_id) / "charts" / "annotated_images" / annotated_filename
            
        except Exception as e:
            logger.error(f"Failed to create annotated image: {e}")
            return None

    def _copy_original_image(self, source_path: Path, base_filename: str, 
                            output_dir: Path, run_id: str, original_path_str: Optional[str]) -> Optional[Path]:
        """Copy the original image to the report directory."""
        try:
            # Preserve original extension if possible
            original_suffix = Path(original_path_str).suffix if original_path_str else '.png'
            if not original_suffix:
                original_suffix = source_path.suffix or '.png'
                
            copied_filename = f"orig_{base_filename}{original_suffix}"
            copied_abs_path = output_dir / copied_filename
            
            # Copy and optionally optimize
            if original_suffix.lower() in ['.png', '.jpg', '.jpeg']:
                # Optimize image while copying
                with Image.open(source_path) as img:
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(copied_abs_path, format='PNG' if original_suffix.lower() == '.png' else 'JPEG', 
                            optimize=True, quality=85)
            else:
                # Direct copy for other formats
                shutil.copy2(source_path, copied_abs_path)
                
            return Path(run_id) / "charts" / "annotated_images" / copied_filename
            
        except Exception as e:
            logger.error(f"Failed to copy original image: {e}")
            return None

    def _prepare_abnormal_samples_display_data(self, df: pl.DataFrame, category_name: str, sort_by_col: str, sort_ascending: bool, run_id: str, annotated_img_subdir: Path) -> List[Dict[str, Any]]:
        display_data = []
        if df.is_empty() or sort_by_col not in df.columns or "image_path" not in df.columns:
            logger.warning(f"Cannot prepare abnormal samples for '{category_name}': DataFrame empty or missing required columns (sort_by_col:'{sort_by_col}', 'image_path').")
            return display_data
        sorted_df = df.sort(sort_by_col, descending=not sort_ascending)
        samples_to_display = sorted_df.head(self.report_settings.display_images.max_per_category)

        for i, row_dict in enumerate(samples_to_display.to_dicts()):
            original_img_path_str = row_dict.get("image_path") # From config field_mapping
            resolved_original_img_path = self._resolve_image_path(original_img_path_str)

            sample_info: Dict[str, Any] = {"metrics": {k:v for k,v in row_dict.items() if k not in ["image_path", "internal_bbox", "internal_rbbox", "internal_keypoints"]}}
            sample_info["original_image_path_str"] = original_img_path_str

            thumbnail_rel_path, lightbox_rel_path = None, None
            if resolved_original_img_path:
                thumbnail_rel_path, lightbox_rel_path = self._process_image_for_report(
                    resolved_original_img_path, row_dict, i, run_id, annotated_img_subdir, original_img_path_str
                )
            else: logger.warning(f"Could not resolve image path '{original_img_path_str}' (base dir: {self.image_base_dir}).")

            sample_info["thumbnail_path"] = str(thumbnail_rel_path) if thumbnail_rel_path else None
            sample_info["lightbox_image_path"] = str(lightbox_rel_path) if lightbox_rel_path else (str(thumbnail_rel_path) if thumbnail_rel_path else None)
            display_data.append(sample_info)
        return display_data

    def _prepare_report_context(self, eval_results: EvaluationResult, run_id: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "project_info": self.project_info.model_dump(), "run_id": run_id, "metrics": eval_results.metrics,
            "config_yaml": yaml.dump(self.main_config.model_dump(exclude_none=True), indent=2, sort_keys=False),
            "model_type": self.project_info.model_type, "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "md_charts_paths": {}, "plotly_charts": {}, "extra_data_html": {}, "extra_data_csv_paths": {},
            "abnormal_image_samples_by_category": {}
        }
        run_output_dir = self.report_settings.output_dir / run_id
        charts_subdir = run_output_dir / "charts"
        extra_data_subdir = run_output_dir / "data"
        annotated_img_subdir = charts_subdir / "annotated_images"
        for d in [run_output_dir, charts_subdir, extra_data_subdir, annotated_img_subdir]: d.mkdir(parents=True, exist_ok=True)

        perf_df = eval_results.extra_data.get("deduplicated_perf_df_for_charts")
        if perf_df is not None and not perf_df.is_empty():
            # Latency Distribution
            context["md_charts_paths"]["latency_distribution"] = self._save_chart_mpl(chart_generators.generate_latency_distribution_plot_mpl(perf_df), "latency_distribution", run_id, charts_subdir)
            latency_plotly_div = self._get_plotly_div(chart_generators.generate_latency_distribution_plotly, perf_df)
            latency_ai_insight = None
            if self.ai_insights_generator:
                latency_summary = perf_df.select(pl.col("total_time_ms")).describe().to_dicts()
                latency_ai_insight = self.ai_insights_generator.get_insight_for_chart("Latency Distribution", latency_summary, f"Model: {self.project_info.model_name or 'N/A'}")
            context["plotly_charts"]["latency_distribution"] = {"html_div": latency_plotly_div, "ai_insight": latency_ai_insight}

            # Performance Timeline
            context["md_charts_paths"]["performance_timeline"] = self._save_chart_mpl(chart_generators.generate_performance_timeline_plot_mpl(perf_df), "performance_timeline", run_id, charts_subdir)
            context["plotly_charts"]["performance_timeline"] = {"html_div": self._get_plotly_div(chart_generators.generate_performance_timeline_plotly, perf_df)} # AI insight can be added later

        if eval_results.metrics: # Time Composition
            context["md_charts_paths"]["time_composition"] = self._save_chart_mpl(chart_generators.generate_time_composition_plot_mpl(eval_results.metrics), "time_composition", run_id, charts_subdir)
            context["plotly_charts"]["time_composition"] = {"html_div": self._get_plotly_div(chart_generators.generate_time_composition_plotly, eval_results.metrics)}

        # Model-specific charts
        model_type = self.project_info.model_type
        if model_type == "detection":
            obj_stab_df = eval_results.extra_data.get("object_stability_details_df")
            if obj_stab_df is not None and not obj_stab_df.is_empty():
                context["md_charts_paths"]["iou_consistency_distribution"] = self._save_chart_mpl(chart_generators.generate_iou_consistency_distribution_plot_mpl(obj_stab_df), "iou_consistency_distribution", run_id, charts_subdir)
                iou_plotly_div = self._get_plotly_div(chart_generators.generate_iou_consistency_distribution_plotly, obj_stab_df)
                iou_ai_insight = None
                if self.ai_insights_generator:
                    iou_summary = obj_stab_df.select(pl.col("mean_iou_consistency")).describe().to_dicts()
                    iou_ai_insight = self.ai_insights_generator.get_insight_for_chart("IoU Consistency Distribution", iou_summary, "Detection model IoU stability.")
                context["plotly_charts"]["iou_consistency_distribution"] = {"html_div": iou_plotly_div, "ai_insight": iou_ai_insight}

                context["md_charts_paths"]["center_drift_distribution"] = self._save_chart_mpl(chart_generators.generate_center_drift_distribution_plot_mpl(obj_stab_df), "center_drift_distribution", run_id, charts_subdir)
                context["plotly_charts"]["center_drift_distribution"] = {"html_div": self._get_plotly_div(chart_generators.generate_center_drift_distribution_plotly, obj_stab_df)}

        elif model_type == "classification":
            cls_stab_df = eval_results.extra_data.get("classification_stability_details_df")
            if cls_stab_df is not None and not cls_stab_df.is_empty():
                context["plotly_charts"]["top_1_consistency_distribution"] = {"html_div": self._get_plotly_div(chart_generators.generate_top_1_consistency_distribution_plotly, cls_stab_df)}
                context["plotly_charts"]["confidence_std_distribution"] = {"html_div": self._get_plotly_div(chart_generators.generate_confidence_std_distribution_plotly, cls_stab_df)}
            cls_eval_params = self.main_config.evaluation_params.classification
            if cls_eval_params and eval_results.metrics:
                 context["plotly_charts"]["jaccard_similarity_trends"] = {"html_div": self._get_plotly_div(chart_generators.generate_jaccard_similarity_trends_plotly, eval_results.metrics, cls_eval_params.top_k)}

        for key, data_item in eval_results.extra_data.items():
            if isinstance(data_item, pl.DataFrame):
                try:
                    if data_item.height > 0 :
                        csv_fn = f"{key}.csv"; csv_abs = extra_data_subdir / csv_fn; data_item.write_csv(csv_abs)
                        context["extra_data_csv_paths"][key] = Path(run_id) / "data" / csv_fn
                        context["extra_data_html"][key] = data_item.head(15).to_pandas().to_html(classes="table table-striped table-sm", index=False, border=0, escape=True)
                    else: context["extra_data_html"][key] = "<p><i>DataFrame is empty.</i></p>"
                except Exception as e: logger.error(f"Failed to process DataFrame {key}: {e}"); context["extra_data_html"][key] = f"<p><i>Error processing DataFrame {key}.</i></p>"

        if self.report_settings.display_images.enabled:
            slow_samples_df = eval_results.extra_data.get("perf_top_slow_samples_df")
            if slow_samples_df is not None and isinstance(slow_samples_df, pl.DataFrame):
                # Ensure 'image_path' column name matches what's in the DataFrame from PerformanceEvaluator
                # PerformanceEvaluator adds 'image_id', 'loop', 'total_time_ms', etc. It does NOT add 'image_path' by default.
                # This needs to be joined from the original deduplicated_perf_df_for_charts or passed through.
                # For now, assuming 'image_path' is present in slow_samples_df (will require fix in PerformanceEvaluator or here)
                # Let's assume 'image_path' is the original column name from mapping.
                img_path_col_name = self.main_config.data_loader.field_mapping.image_path
                if img_path_col_name not in slow_samples_df.columns and "image_id" in slow_samples_df.columns and perf_df is not None:
                    # Attempt to join image_path from the deduplicated perf_df
                    slow_samples_df = slow_samples_df.join(
                        perf_df.select(["image_id", "loop", img_path_col_name]).unique(subset=["image_id", "loop"]),
                        on=["image_id", "loop"],
                        how="left"
                    ).rename({img_path_col_name: "image_path"}) # Standardize to "image_path" for _prepare_abnormal_samples

                if "image_path" in slow_samples_df.columns:
                    context["abnormal_image_samples_by_category"]["Performance: Top Slow Samples"] = \
                        self._prepare_abnormal_samples_display_data(slow_samples_df, "PerfSlow", "total_time_ms", False, run_id, annotated_img_subdir)
                else:
                    logger.warning("'image_path' column not found in 'perf_top_slow_samples_df'. Cannot display abnormal images for performance.")

        logger.debug("Report context prepared.")
        return context

    def _generate_md_report(self, context: Dict[str, Any], run_id: str):
        report_path = self.report_settings.output_dir / f"{run_id}_report.md"
        logger.info(f"Generating Markdown report: {report_path}")
        content = f"# Evaluation Report: {context['project_info']['project_name']}\n\n"
        content += f"**Run ID:** `{run_id}`\n**Model Type:** {context['model_type']}\n"
        content += f"**Model Version:** {context['project_info'].get('model_version', 'N/A')}\n"
        content += f"**Generated on:** {context['generation_timestamp']}\n\n"

        score_metrics = {k: v for k, v in context.get("metrics", {}).items() if "score" in k.lower()}
        if score_metrics:
            content += "## Overall Scores\n\n"
            for score_key in ["total_score_overall", "perf_score_overall",
                              f"{context['model_type'][:3] if context['model_type'] != 'detection' else 'det'}_stab_score_overall"]:
                if score_key in score_metrics:
                    val = score_metrics[score_key]; val_s = f"{val:.2f}" if isinstance(val, float) else str(val)
                    title = score_key.replace('_score_overall','').replace('_', ' ').title()
                    if score_key == "total_score_overall": title += " (Perf & Stab)"
                    content += f"- **{title}**: **`{val_s}`**\n"
            content += "\n"

        content += "## Metrics Summary\n\n"
        if context["metrics"]:
            model_stab_prefix_short = context['model_type'][:3] + '_stab_'
            model_stab_prefix_det = 'det_stab_'

            perf_m = {k:v for k,v in context["metrics"].items() if k.startswith("perf_") and "score" not in k}
            stab_m = {k:v for k,v in context["metrics"].items() if (k.startswith(model_stab_prefix_short) or k.startswith(model_stab_prefix_det)) and "score" not in k}
            other_m = {k:v for k,v in context["metrics"].items() if not (k.startswith("perf_") or k.startswith(model_stab_prefix_short) or k.startswith(model_stab_prefix_det) or "score" in k.lower())}

            def format_metrics_md(title, metrics_dict, prefix_to_strip=""):
                s = f"### {title}\n"
                if not metrics_dict: return s + "_No metrics in this category._\n\n"
                for k, v in metrics_dict.items():
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v); val_str = "N/A" if v is None else val_str
                    disp_k = k.replace(prefix_to_strip, '', 1).replace('_', ' ').title()
                    s += f"- **{disp_k}**: `{val_str}`\n"
                return s + "\n"

            content += format_metrics_md("Performance Metrics", perf_m, "perf_")
            content += format_metrics_md(f"{context['model_type'].title()} Stability Metrics", stab_m, model_stab_prefix_short if context['model_type'] != "detection" else model_stab_prefix_det)
            if other_m: content += format_metrics_md("Other Metrics", other_m)
        else: content += "_No metrics were generated._\n\n"

        content += "## Charts\n\n"
        if context["md_charts_paths"]:
            for chart_name, chart_path in context["md_charts_paths"].items():
                content += f"### {chart_name.replace('_', ' ').title()}\n![{chart_name.replace('_', ' ').title()}]({chart_path})\n\n"
        else: content += "_No charts were generated._\n\n"

        content += "## Additional Data\n\n"
        if context["extra_data_csv_paths"]:
            for key, csv_path in context["extra_data_csv_paths"].items():
                content += f"### {key.replace('_', ' ').title()}\nData saved to: [{csv_path.name}]({csv_path})\n\n"
        else: content += "_No additional data files generated._\n\n"

        content += "\n## Configuration Used\n\n```yaml\n" + context['config_yaml'] + "\n```\n"
        try:
            with open(report_path, "w", encoding="utf-8") as f: f.write(content)
            logger.info(f"Markdown report saved to {report_path}")
        except IOError as e: logger.error(f"Failed to write MD report {report_path}: {e}")

    def _generate_html_report(self, context: Dict[str, Any], run_id: str):
        report_path = self.report_settings.output_dir / f"{run_id}_report.html"
        logger.info(f"Generating HTML report: {report_path}")
        html_context = context.copy()
        html_context["static_charts_for_html"] = {name: {"title": name.replace('_', ' ').title(), "path": str(path_obj)} for name, path_obj in context.get("md_charts_paths", {}).items()}
        # Plotly charts (divs and insights) are already in context["plotly_charts"]
        # Abnormal image samples are in context["abnormal_image_samples_by_category"]
        try:
            template = self.jinja_env.get_template("report_template.html")
            html_content = template.render(html_context)
            with open(report_path, "w", encoding="utf-8") as f: f.write(html_content)
            logger.info(f"HTML report saved to {report_path}")
        except Exception as e: logger.error(f"Failed to generate HTML report {report_path}: {e}", exc_info=True)

    def generate(self, eval_results: EvaluationResult, run_id: str):
        logger.info(f"Starting report generation for run ID: {run_id}.")
        report_data_context = self._prepare_report_context(eval_results, run_id)
        if "md" in self.report_settings.formats: self._generate_md_report(report_data_context, run_id)
        if "html" in self.report_settings.formats: self._generate_html_report(report_data_context, run_id)
        logger.info(f"Report generation completed. Reports saved in: {self.report_settings.output_dir.resolve()}")

if __name__ == "__main__":
    from ai_eval_tool.config_manager import load_config
    from ai_eval_tool.utils.logging_config import setup_logging
    setup_logging(level="DEBUG")

    dummy_config_content = """
project_info:
  project_name: "ReportGen Full Test"
  model_type: "detection"
  model_version: "v0.2-test"
data_loader:
  field_mapping: {loop: l, image_id: i, image_path: image_path_col, pre_time_ms: t1, inference_time_ms: t2, post_time_ms: t3, total_time_ms: t4,
                  detection: {category_id: cat_col, score: score_col, bbox: [x,y,w,h]}}
  image_base_dir: "./test_report_images"
evaluation_params:
  detection: {iou_threshold: 0.5, bbox_format: "xywh"}
report_settings:
  output_dir: "./test_full_report_output"
  formats: ["md", "html"]
  display_images: {enabled: true, max_per_category: 2, draw_annotations: true}
  ai_insights: {enabled: false} # Keep false for basic test to avoid API calls
"""
    dummy_config_path = Path("dummy_reportgen_full_config.yaml")
    with open(dummy_config_path, "w") as f: f.write(dummy_config_content)
    Path("./test_report_images").mkdir(exist_ok=True)
    try:
        img = Image.new('RGB', (200, 150), color = 'skyblue'); img_draw = ImageDraw.Draw(img)
        img_draw.text((10,10), "Sample1", fill="black", font=FONT); img.save(Path("./test_report_images/sample1.png"))
        img = Image.new('RGB', (180, 120), color = 'lightgreen'); img_draw = ImageDraw.Draw(img)
        img_draw.text((10,10), "Sample2", fill="black", font=FONT); img.save(Path("./test_report_images/sample2.png"))
    except Exception as e: logger.warning(f"Could not create dummy images: {e}")

    template_dir_for_test = Path(__file__).parent / "templates"
    if not (template_dir_for_test / "report_template.html").exists():
        (template_dir_for_test / "report_template.html").write_text("<html><body><h1>Test Report {{run_id}}</h1></body></html>")

    try:
        logger.info("--- Testing ReportGenerator (Full Flow, AI Insights OFF) ---")
        config = load_config(dummy_config_path)
        report_generator = ReportGenerator(config)

        dummy_perf_df = pl.DataFrame({"total_time_ms": [100,250], "_event_index": range(2), "loop":[1,1], "image_id":["s1","s2"], "image_path_col":["sample1.png", "sample2.png"], "internal_bbox":[[[10,10,50,50]],[[20,20,60,60]]], "cat_col":[0,1], "score_col":[0.9,0.8]})
        dummy_obj_stab_df = pl.DataFrame({"mean_iou_consistency": [0.8], "mean_center_drift_px": [5.0]})

        dummy_eval_results = EvaluationResult(
            metrics={"perf_mean_total_time_ms": 106.0, "det_stab_mean_iou_consistency": 0.81},
            extra_data={
                "deduplicated_perf_df_for_charts": dummy_perf_df, # Used for latency charts & AI insight
                "object_stability_details_df": dummy_obj_stab_df, # Used for IoU charts & AI insight
                "perf_top_slow_samples_df": dummy_perf_df.sort("total_time_ms", descending=True) # Has image_path and bbox
            }
        )
        test_run_id = "test_report_" + datetime.now().strftime("%Y%m%d%H%M%S")
        report_generator.generate(dummy_eval_results, test_run_id)

        output_dir = config.report_settings.output_dir
        run_out_dir = output_dir / test_run_id
        assert (output_dir / f"{test_run_id}_report.md").exists()
        assert (output_dir / f"{test_run_id}_report.html").exists()
        assert (run_out_dir / "charts" / "latency_distribution.png").exists()
        assert (run_out_dir / "charts" / "annotated_images" / "thumb_sample1_1_s1.png").exists() # Check one thumb
        logger.info(f"Test reports generated in: {run_out_dir.resolve()}")
    except Exception as e: logger.error(f"Error during ReportGenerator test: {e}", exc_info=True); raise
    finally:
        if dummy_config_path.exists(): dummy_config_path.unlink()
        if Path("./test_report_images").exists(): shutil.rmtree("./test_report_images")
        # if Path("./test_full_report_output").exists(): shutil.rmtree("./test_full_report_output") # Keep for inspection
        logger.info("ReportGenerator test completed.")

```
