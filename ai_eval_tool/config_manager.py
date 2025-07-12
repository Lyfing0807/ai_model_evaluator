"""
Configuration Manager: Loads, validates, and provides access to evaluation settings.
"""
from typing import List, Dict, Optional, Union, Literal
import yaml
from pydantic import BaseModel, Field, DirectoryPath, FilePath, validator
from pathlib import Path

class ProjectInfo(BaseModel):
    project_name: str = "AI Model Evaluation"
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Literal["detection", "classification", "rotated_detection", "pose", "tracking", "ranking"] # Added ranking

class CommonFieldMapping(BaseModel):
    loop: str = "loop"
    image_id: Optional[str] = "image_id" # Made optional as ranking/tracking might not use it
    image_path: Optional[str] = None
    frame_id: Optional[str] = "frame_id"
    query_id: Optional[str] = "query_id" # Added for ranking/recsys

    pre_time_ms: Optional[str] = "pre_time_ms"
    inference_time_ms: Optional[str] = "inference_time_ms"
    post_time_ms: Optional[str] = "post_time_ms"
    total_time_ms: Optional[str] = "total_time_ms"

class DetectionFieldMapping(BaseModel):
    category_id: str = "category_id"
    score: str = "score"
    bbox: List[str] = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]

class ClassificationFieldMapping(BaseModel):
    top_k_id_pattern: str = "top{k}_label_id"
    top_k_score_pattern: str = "top{k}_score"

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
    ranking: Optional[RankingFieldMapping] = None # Added ranking

class DataLoaderConfig(BaseModel):
    image_base_dir: Optional[DirectoryPath] = None
    field_mapping: FieldMapping

# --- Evaluation Parameter Models ---
class DetectionEvaluationParams(BaseModel):
    iou_threshold: float = Field(0.5, ge=0, le=1)
    z_score_threshold: float = Field(3.0, gt=0)
    bbox_format: Literal["xywh", "xyxy"] = "xywh"
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"existence_stability":0.3, "position_stability":0.4, "confidence_stability":0.15, "category_stability":0.15})

class ClassificationEvaluationParams(BaseModel):
    top_k: List[int] = [1, 3, 5]
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"prediction_consistency": 0.6, "confidence_reliability": 0.4})
    @validator('top_k')
    def top_k_rules(cls, v): # Shortened name
        if not v: raise ValueError("top_k list cannot be empty")
        if any(k <= 0 for k in v): raise ValueError("All k values must be positive")
        if sorted(list(set(v))) != sorted(v): raise ValueError("top_k list must be sorted and unique")
        return v

class RotatedDetectionEvaluationParams(BaseModel):
    riou_threshold: float = Field(0.5, ge=0, le=1)
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"riou_consistency": 0.4, "angle_stability": 0.3, "existence_stability": 0.3})

class PoseEvaluationParams(BaseModel):
    oks_sigma: Union[float, List[float]] = 0.5
    keypoint_layout: str = "coco_17"
    match_iou_threshold: float = Field(0.5, ge=0, le=1)
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"oks_consistency": 0.5, "kpt_visibility": 0.25, "kpt_drift": 0.25})

class TrackingEvaluationParams(BaseModel):
    mota_iou_threshold: float = Field(0.5, ge=0, le=1)
    bbox_pred_format: Literal["xywh", "xyxy"] = "xywh"
    bbox_gt_format: Optional[Literal["xywh", "xyxy"]] = "xywh"
    min_track_length_for_stability: int = Field(5, gt=0)
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"mota_stability": 0.4, "track_fragmentation": 0.3, "id_consistency": 0.3})

class RankingEvaluationParams(BaseModel):
    k_values: List[int] = Field(default_factory=lambda: [5, 10, 20])
    scoring_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"ndcg_stability": 0.4, "recall_stability": 0.3, "top_k_set_jaccard": 0.3})
    @validator('k_values')
    def k_values_rules(cls, v): # Shortened name
        if not v: raise ValueError("k_values list cannot be empty")
        if any(k <= 0 for k in v): raise ValueError("All k values must be positive")
        if sorted(list(set(v))) != sorted(v): raise ValueError("k_values list must be sorted and unique")
        return v

class PerformanceEvaluationParams(BaseModel):
    target_latency_ms: Optional[float] = Field(50.0, gt=0)
    cv_target_threshold: Optional[float] = Field(0.1, ge=0, le=1)
    component_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {"throughput": 0.6, "latency_stability": 0.4})

class OverallScoringWeights(BaseModel):
    performance: float = Field(0.4, ge=0, le=1)
    stability: float = Field(0.6, ge=0, le=1)

class EvaluationParams(BaseModel):
    performance: Optional[PerformanceEvaluationParams] = Field(default_factory=PerformanceEvaluationParams)
    detection: Optional[DetectionEvaluationParams] = None
    classification: Optional[ClassificationEvaluationParams] = None
    rotated_detection: Optional[RotatedDetectionEvaluationParams] = None
    pose: Optional[PoseEvaluationParams] = None
    tracking: Optional[TrackingEvaluationParams] = None
    ranking: Optional[RankingEvaluationParams] = None # Added ranking
    overall_scoring_weights: Optional[OverallScoringWeights] = Field(default_factory=OverallScoringWeights)

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
    display_images: ReportDisplayImagesConfig = Field(default_factory=ReportDisplayImagesConfig)
    ai_insights: AIInsightsConfig = Field(default_factory=AIInsightsConfig)
    @validator('output_dir', pre=True, always=True)
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

    @validator('evaluation_params')
    def check_model_specific_params_exist(cls, v, values):
        if 'project_info' not in values: return v
        model_type = values['project_info'].model_type
        # Performance params are always present due to default_factory
        if model_type != "performance" and not getattr(v, model_type, None) :
            raise ValueError(f"evaluation_params for model_type '{model_type}' are missing.")
        return v

    @validator('data_loader')
    def check_model_specific_field_mapping_exist(cls, v, values):
        if 'project_info' not in values: return v
        model_type = values['project_info'].model_type
        # Common fields are in FieldMapping itself. Check model-specific part.
        if not getattr(v.field_mapping, model_type, None) and model_type not in ["performance_only"]: # Example
             raise ValueError(f"data_loader.field_mapping for model_type '{model_type}' is missing.")
        return v

def load_config(config_path: Union[str, Path]) -> MainConfig:
    p = Path(config_path)
    if not p.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(p, 'r', encoding='utf-8') as f:
        try: raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e: raise yaml.YAMLError(f"Error parsing YAML: {e}")
    return MainConfig(**raw_config)

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
  model_type": "ranking"
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
```
