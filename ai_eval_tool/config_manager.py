"""
Configuration Manager: Loads, validates, and provides access to evaluation
settings.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, DirectoryPath, Field, validator


class ProjectInfo(BaseModel):
    project_name: str = "AI Model Evaluation"
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Literal[
        "detection",
        "classification",
        "rotated_detection",
        "pose",
        "tracking",
        "ranking",
    ]  # Added ranking


class CommonFieldMapping(BaseModel):
    loop: str = "loop"
    image_id: Optional[str] = (
        "image_id"  # Made optional as ranking/tracking might not use it
    )
    image_path: Optional[str] = None
    frame_id: Optional[str] = "frame_id"
    query_id: Optional[str] = "query_id"  # Added for ranking/recsys

    pre_time_ms: Optional[str] = "pre_time_ms"
    inference_time_ms: Optional[str] = "inference_time_ms"
    post_time_ms: Optional[str] = "post_time_ms"
    total_time_ms: Optional[str] = "total_time_ms"


class DetectionFieldMapping(BaseModel):
    category_id: str = "category_id"
    score: str = "score"
    bbox: List[str] = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]


class ClassificationFieldMapping(BaseModel):
    top_k_id_pattern: str = "pred_label_top{k}"
    top_k_score_pattern: str = "pred_score_top{k}"


class RotatedDetectionFieldMapping(BaseModel):
    category_id: str = "category_id"
    score: str = "score"
    rbbox: List[str] = ["cx", "cy", "w", "h", "angle"]


class PoseFieldMapping(BaseModel):
    person_bbox: List[str] = ["person_x", "person_y", "person_w", "person_h"]
    person_score: str = "person_score"
    keypoints: str = "keypoints_str"


class TrackingFieldMapping(BaseModel):
    object_id_pred: str = "track_id_pred"
    bbox_pred: List[str] = ["x_pred", "y_pred", "w_pred", "h_pred"]
    category_id_pred: Optional[str] = "category_id_pred"
    score_pred: Optional[str] = "score_pred"
    object_id_gt: Optional[str] = "track_id_gt"
    bbox_gt: Optional[List[str]] = ["x_gt", "y_gt", "w_gt", "h_gt"]
    category_id_gt: Optional[str] = "category_id_gt"
    visibility_gt: Optional[str] = "visibility_gt"
    ignored_gt: Optional[str] = "ignored_gt"


class RankingFieldMapping(BaseModel):
    # query_id is in CommonFieldMapping
    item_id_pred_list: str = "item_id_pred_list"
    score_pred_list: Optional[str] = "score_pred_list"
    item_id_gt_list: str = "item_id_gt_list"
    list_delimiter: str = Field(",", description="Delimiter for list-like CSV fields.")


class FieldMapping(CommonFieldMapping):
    detection: Optional[DetectionFieldMapping] = None
    classification: Optional[ClassificationFieldMapping] = None
    rotated_detection: Optional[RotatedDetectionFieldMapping] = None
    pose: Optional[PoseFieldMapping] = None
    tracking: Optional[TrackingFieldMapping] = None
    ranking: Optional[RankingFieldMapping] = None  # Added ranking


class DataLoaderConfig(BaseModel):
    image_base_dir: Optional[DirectoryPath] = None
    field_mapping: FieldMapping


# --- Evaluation Parameter Models ---
class DetectionEvaluationParams(BaseModel):
    iou_threshold: float = Field(0.5, ge=0, le=1)
    z_score_threshold: float = Field(3.0, gt=0)
    bbox_format: Literal["xywh", "xyxy"] = "xywh"
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "existence_stability": 0.3,
            "position_stability": 0.4,
            "confidence_stability": 0.15,
            "category_stability": 0.15,
        }
    )


class ClassificationEvaluationParams(BaseModel):
    top_k: List[int] = [1, 3, 5]
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "prediction_consistency": 0.6,
            "confidence_reliability": 0.4,
        }
    )

    @validator("top_k")
    def top_k_rules(cls, v):  # Shortened name
        if not v:
            raise ValueError(
                "top_k list cannot be empty.\n"
                "Please provide at least one positive integer value.\n"
                "Example: top_k: [1, 3, 5]"
            )
        if any(k <= 0 for k in v):
            invalid_values = [k for k in v if k <= 0]
            raise ValueError(
                f"All k values must be positive integers. Found invalid values: {invalid_values}\n"
                f"Please ensure all values in top_k are greater than 0.\n"
                f"Example: top_k: [1, 3, 5]"
            )
        if sorted(list(set(v))) != sorted(v):
            raise ValueError(
                f"top_k list must be sorted in ascending order and contain unique values.\n"
                f"Current list: {v}\n"
                f"Expected format: {sorted(list(set(v)))}\n"
                f"Please sort the values and remove duplicates."
            )
        return v


class RotatedDetectionEvaluationParams(BaseModel):
    riou_threshold: float = Field(0.5, ge=0, le=1)
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "riou_consistency": 0.4,
            "angle_stability": 0.3,
            "existence_stability": 0.3,
        }
    )


class PoseEvaluationParams(BaseModel):
    oks_sigma: Union[float, List[float]] = 0.5
    keypoint_layout: str = "coco_17"
    match_iou_threshold: float = Field(0.5, ge=0, le=1)
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "oks_consistency": 0.5,
            "kpt_visibility": 0.25,
            "kpt_drift": 0.25,
        }
    )


class TrackingEvaluationParams(BaseModel):
    mota_iou_threshold: float = Field(0.5, ge=0, le=1)
    bbox_pred_format: Literal["xywh", "xyxy"] = "xywh"
    bbox_gt_format: Optional[Literal["xywh", "xyxy"]] = "xywh"
    min_track_length_for_stability: int = Field(5, gt=0)
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "mota_stability": 0.4,
            "track_fragmentation": 0.3,
            "id_consistency": 0.3,
        }
    )


class RankingEvaluationParams(BaseModel):
    k_values: List[int] = Field(default_factory=lambda: [5, 10, 20])
    scoring_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "ndcg_stability": 0.4,
            "recall_stability": 0.3,
            "top_k_set_jaccard": 0.3,
        }
    )

    @validator("k_values")
    def k_values_rules(cls, v):  # Shortened name
        if not v:
            raise ValueError(
                "k_values list cannot be empty.\n"
                "Please provide at least one positive integer value for ranking evaluation.\n"
                "Example: k_values: [5, 10, 20]"
            )
        if any(k <= 0 for k in v):
            invalid_values = [k for k in v if k <= 0]
            raise ValueError(
                f"All k values must be positive integers. Found invalid values: {invalid_values}\n"
                f"Please ensure all values in k_values are greater than 0.\n"
                f"Example: k_values: [5, 10, 20]"
            )
        if sorted(list(set(v))) != sorted(v):
            raise ValueError(
                f"k_values list must be sorted in ascending order and contain unique values.\n"
                f"Current list: {v}\n"
                f"Expected format: {sorted(list(set(v)))}\n"
                f"Please sort the values and remove duplicates."
            )
        return v


class PerformanceEvaluationParams(BaseModel):
    target_latency_ms: Optional[float] = Field(50.0, gt=0)
    cv_target_threshold: Optional[float] = Field(0.1, ge=0, le=1)
    component_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"throughput": 0.6, "latency_stability": 0.4}
    )


class OverallScoringWeights(BaseModel):
    performance: float = Field(0.4, ge=0, le=1)
    stability: float = Field(0.6, ge=0, le=1)


class EvaluationParams(BaseModel):
    performance: Optional[PerformanceEvaluationParams] = Field(
        default_factory=PerformanceEvaluationParams
    )
    detection: Optional[DetectionEvaluationParams] = None
    classification: Optional[ClassificationEvaluationParams] = None
    rotated_detection: Optional[RotatedDetectionEvaluationParams] = None
    pose: Optional[PoseEvaluationParams] = None
    tracking: Optional[TrackingEvaluationParams] = None
    ranking: Optional[RankingEvaluationParams] = None  # Added ranking
    overall_scoring_weights: Optional[OverallScoringWeights] = Field(
        default_factory=OverallScoringWeights
    )


# --- Report Settings Models ---
class ReportDisplayImagesConfig(BaseModel):
    enabled: bool = True
    max_per_category: int = Field(10, ge=0)
    draw_annotations: bool = True


class AIInsightsConfig(BaseModel):
    enabled: bool = False
    model_name: Optional[str] = "gpt-3.5-turbo"
    api_endpoint: Optional[str] = "https://api.openai.com/v1/chat/completions"
    cache_responses: bool = True


class ReportSettings(BaseModel):
    output_dir: DirectoryPath = Field(default_factory=lambda: Path("./reports"))
    formats: List[Literal["md", "html"]] = ["md", "html"]
    display_images: ReportDisplayImagesConfig = Field(
        default_factory=ReportDisplayImagesConfig
    )
    ai_insights: AIInsightsConfig = Field(default_factory=AIInsightsConfig)

    @validator("output_dir", pre=True, always=True)
    def create_output_dir_if_not_exists(cls, v):
        path = Path(v) if v is not None else Path("./reports")
        path.mkdir(parents=True, exist_ok=True)
        return path


# --- Main Configuration Model ---
class MainConfig(BaseModel):
    project_info: ProjectInfo
    data_loader: DataLoaderConfig
    evaluation_params: EvaluationParams = Field(default_factory=EvaluationParams)
    report_settings: ReportSettings = Field(default_factory=ReportSettings)

    @validator("evaluation_params")
    def check_model_specific_params_exist(cls, v, values):
        if "project_info" not in values:
            return v
        model_type = values["project_info"].model_type
        # Performance params are always present due to default_factory
        if model_type != "performance" and not getattr(v, model_type, None):
            raise ValueError(
                f"Missing evaluation parameters for model type '{model_type}'.\n"
                f"Please add an 'evaluation_params.{model_type}' section to your configuration.\n"
                f"Example:\n"
                f"evaluation_params:\n"
                f"  {model_type}:\n"
                f"    # Add {model_type}-specific parameters here"
            )
        return v

    @validator("data_loader")
    def check_model_specific_field_mapping_exist(cls, v, values):
        if "project_info" not in values:
            return v
        model_type = values["project_info"].model_type
        # Common fields are in FieldMapping itself. Check model-specific part.
        if not getattr(v.field_mapping, model_type, None) and model_type not in [
            "performance_only"
        ]:  # Example
            raise ValueError(
                f"Missing field mapping for model type '{model_type}'.\n"
                f"Please add a 'data_loader.field_mapping.{model_type}' section to your configuration.\n"
                f"This section should map the column names in your CSV file to the expected field names.\n"
                f"Example:\n"
                f"data_loader:\n"
                f"  field_mapping:\n"
                f"    {model_type}:\n"
                f"      # Add {model_type}-specific field mappings here"
            )
        return v


def load_config(config_path: Union[str, Path]) -> MainConfig:
    """
    Loads and validates configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Validated MainConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    p = Path(config_path)

    # Check file existence with helpful message
    if not p.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure the file exists and the path is correct.\n"
            f"Expected file: {p.absolute()}"
        )

    # Check if it's actually a file
    if not p.is_file():
        raise ValueError(f"Path exists but is not a file: {config_path}")

    # Load and parse YAML
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML configuration file: {config_path}\n"
            f"YAML Error: {e}\n"
            f"Please check the YAML syntax and ensure proper indentation."
        )
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Failed to read configuration file due to encoding issues: {config_path}\n"
            f"Error: {e}\n"
            f"Please ensure the file is saved with UTF-8 encoding."
        )
    except Exception as e:
        raise ValueError(
            f"Unexpected error reading configuration file {config_path}: {e}"
        )

    # Check if YAML content is valid
    if raw_config is None:
        raise ValueError(
            f"Configuration file is empty or contains only comments: {config_path}\n"
            f"Please provide a valid YAML configuration."
        )

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML object (dictionary), got {type(raw_config).__name__}: {config_path}\n"
            f"Please ensure your configuration starts with key-value pairs, not a list or scalar value."
        )

    # Validate configuration with enhanced error messages
    try:
        return MainConfig(**raw_config)
    except Exception as e:
        # Enhanced error message for common configuration issues
        error_msg = f"Configuration validation failed for file: {config_path}\n"

        if "project_info" not in raw_config:
            error_msg += "Missing required section: 'project_info'\n"
            error_msg += "Please add a project_info section with at least 'project_name' and 'model_type'.\n"
        elif "model_type" not in raw_config.get("project_info", {}):
            error_msg += "Missing required field: 'project_info.model_type'\n"
            error_msg += "Please specify model_type as one of: detection, classification, rotated_detection, pose, tracking, ranking\n"

        if "data_loader" not in raw_config:
            error_msg += "Missing required section: 'data_loader'\n"
            error_msg += (
                "Please add a data_loader section with field_mapping configuration.\n"
            )

        # Add the original error for technical details
        error_msg += f"\nDetailed error: {str(e)}"

        raise ValueError(error_msg) from e


if __name__ == "__main__":
    # ... (existing __main__ block for testing detection and classification) ...

    # Example for Tracking config (add to dummy_config_content or separate)
    dummy_tracking_config = """
project_info:
  project_name: "Test Tracking Project"
  model_type: "tracking"
data_loader:
  image_base_dir: "./"
  field_mapping:
    loop: "loop_id"
    frame_id: "frame_num"
    image_id: "frame_num" # If frame_num is also the unique image identifier
    tracking:
      object_id_pred: "track_id"
      bbox_pred: ["px","py","pw","ph"]
      object_id_gt: "gt_track_id"
      bbox_gt: ["gx","gy","gw","gh"]
evaluation_params:
  tracking:
    mota_iou_threshold: 0.6
    bbox_pred_format: "xywh"
    bbox_gt_format: "xywh"
report_settings: {output_dir: "./test_reports_tracking"}
"""
    # Example for Ranking config
    dummy_ranking_config = """
project_info:
  project_name: "Test Ranking Project"
  model_type: "ranking"
data_loader:
  field_mapping:
    loop: "run_version"
    query_id: "user_query_id"
    ranking:
      item_id_pred_list: "predicted_items"
      item_id_gt_list: "relevant_items"
      list_delimiter: ","
evaluation_params:
  ranking:
    k_values: [5, 10]
report_settings: {output_dir: "./test_reports_ranking"}
"""
    # To test these, you'd write them to temp files and call load_config
    # e.g. Path("dummy_track.yaml").write_text(dummy_tracking_config)
    # cfg_track = load_config("dummy_track.yaml")
    # print(f"Tracking MOTA threshold: {cfg_track.evaluation_params.tracking.mota_iou_threshold}")
    pass
