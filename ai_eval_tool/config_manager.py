"""
Configuration Manager: Loads, validates, and provides access to evaluation settings.
"""
from typing import List, Dict, Optional, Union, Literal
import yaml
from pydantic import BaseModel, Field, DirectoryPath, FilePath, validator

# Define Pydantic models to mirror the structure of config.yaml

class ProjectInfo(BaseModel):
    project_name: str = "AI Model Evaluation"
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Literal["detection", "classification", "rotated_detection", "pose"]

class CommonFieldMapping(BaseModel):
    loop: str = "loop"
    image_id: str = "image_id"
    image_path: str = "image_path"
    pre_time_ms: str = "pre_time_ms"
    inference_time_ms: str = "inference_time_ms"
    post_time_ms: str = "post_time_ms"
    total_time_ms: str = "total_time_ms"

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


class FieldMapping(CommonFieldMapping):
    detection: Optional[DetectionFieldMapping] = None
    classification: Optional[ClassificationFieldMapping] = None
    rotated_detection: Optional[RotatedDetectionFieldMapping] = None
    pose: Optional[PoseFieldMapping] = None

class DataLoaderConfig(BaseModel):
    # csv_file_path: FilePath # This should be passed as a CLI argument or discovered
    image_base_dir: Optional[DirectoryPath] = None
    field_mapping: FieldMapping

class DetectionEvaluationParams(BaseModel):
    iou_threshold: float = Field(0.5, ge=0, le=1)
    z_score_threshold: float = Field(3.0, gt=0)
    bbox_format: Literal["xywh", "xyxy"] = "xywh"
    # scoring_weights: Optional[Dict[str, float]] = None # For future use

class ClassificationEvaluationParams(BaseModel):
    top_k: List[int] = [1, 3, 5]

    @validator('top_k')
    def top_k_must_be_positive_and_sorted(cls, v):
        if not v:
            raise ValueError("top_k list cannot be empty")
        if any(k <= 0 for k in v):
            raise ValueError("All k values in top_k must be positive")
        if sorted(list(set(v))) != sorted(v):
            raise ValueError("top_k list must be sorted and contain unique values")
        return v

class RotatedDetectionEvaluationParams(BaseModel):
    riou_threshold: float = Field(0.5, ge=0, le=1)
    # bbox_format might be relevant here if rbbox isn't always cx,cy,w,h,a
    # For now, assume rbbox field mapping directly maps to these 5 components.

class PoseEvaluationParams(BaseModel):
    oks_sigma: Union[float, List[float]] = 0.5 # Could be a single value or list per keypoint type
    keypoint_layout: str = "coco_17" # E.g., "coco_17", "mpii_16"
    match_iou_threshold: float = Field(0.5, ge=0, le=1)


class EvaluationParams(BaseModel):
    detection: Optional[DetectionEvaluationParams] = None
    classification: Optional[ClassificationEvaluationParams] = None
    rotated_detection: Optional[RotatedDetectionEvaluationParams] = None
    pose: Optional[PoseEvaluationParams] = None

class ReportDisplayImagesConfig(BaseModel):
    enabled: bool = True
    max_per_category: int = Field(10, ge=0)
    draw_annotations: bool = True

class AIInsightsConfig(BaseModel):
    enabled: bool = False
    model_name: Optional[str] = "gpt-3.5-turbo"
    api_endpoint: Optional[str] = "https://api.openai.com/v1/chat/completions"
    # api_key is handled by environment variable
    cache_responses: bool = True

class ReportSettings(BaseModel):
    output_dir: DirectoryPath = Field(default_factory=lambda: Path("./reports"))
    formats: List[Literal["md", "html"]] = ["md", "html"]
    display_images: ReportDisplayImagesConfig = Field(default_factory=ReportDisplayImagesConfig)
    ai_insights: AIInsightsConfig = Field(default_factory=AIInsightsConfig)

    @validator('output_dir', pre=True, always=True)
    def create_output_dir_if_not_exists(cls, v):
        from pathlib import Path # Local import to avoid circularity if Path is used elsewhere at top level
        if v is None:
            v = Path("./reports")
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class MainConfig(BaseModel):
    project_info: ProjectInfo
    data_loader: DataLoaderConfig
    evaluation_params: EvaluationParams = Field(default_factory=EvaluationParams)
    report_settings: ReportSettings = Field(default_factory=ReportSettings)

    @validator('evaluation_params')
    def check_model_specific_params_exist(cls, v, values):
        if 'project_info' not in values:
            # This can happen if project_info itself fails validation earlier
            return v
        model_type = values['project_info'].model_type
        if not getattr(v, model_type, None):
            raise ValueError(f"evaluation_params for model_type '{model_type}' are missing.")
        return v

    @validator('data_loader')
    def check_model_specific_field_mapping_exist(cls, v, values):
        if 'project_info' not in values:
            return v
        model_type = values['project_info'].model_type
        if not getattr(v.field_mapping, model_type, None) and model_type not in ["performance_only"]: # Example if we had a type with no specific fields
             raise ValueError(f"data_loader.field_mapping for model_type '{model_type}' is missing.")
        return v


def load_config(config_path: Union[str, 'Path']) -> MainConfig:
    """
    Loads the YAML configuration file and validates it using Pydantic models.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A MainConfig object populated with settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
        pydantic.ValidationError: If the configuration data does not match the schema.
    """
    from pathlib import Path # Local import
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(p, 'r', encoding='utf-8') as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

    return MainConfig(**raw_config)

# Example usage (for testing this module directly)
if __name__ == "__main__":
    from pathlib import Path
    # Create a dummy config.yaml for testing
    dummy_config_content = """
project_info:
  project_name: "Test Detection Project"
  model_type: "detection"

data_loader:
  # csv_file_path: "dummy_data.csv" # Will fail if file doesn't exist, handled by CLI later
  image_base_dir: "./"
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    image_path: "rel_path"
    pre_time_ms: "t_pre"
    inference_time_ms: "t_inf"
    post_time_ms: "t_post"
    total_time_ms: "t_total"
    detection:
      category_id: "det_cat"
      score: "det_score"
      bbox: ["x", "y", "w", "h"]

evaluation_params:
  detection:
    iou_threshold: 0.6
    bbox_format: "xywh"

report_settings:
  output_dir: "./test_reports"
  formats: ["md"]
"""
    dummy_config_path = Path("dummy_config_test.yaml")
    with open(dummy_config_path, 'w') as f:
        f.write(dummy_config_content)

    try:
        print(f"Attempting to load config: {dummy_config_path.resolve()}")
        cfg = load_config(dummy_config_path)
        print("Configuration loaded successfully!")
        print(f"Project Name: {cfg.project_info.project_name}")
        print(f"Output Dir: {cfg.report_settings.output_dir}")
        if cfg.evaluation_params.detection:
            print(f"Detection IoU Threshold: {cfg.evaluation_params.detection.iou_threshold}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    finally:
        if dummy_config_path.exists():
            dummy_config_path.unlink()
        # Clean up created test_reports dir
        test_reports_dir = Path("./test_reports")
        if test_reports_dir.exists() and test_reports_dir.is_dir():
            import shutil
            shutil.rmtree(test_reports_dir)

    # Test classification config
    dummy_class_config_content = """
project_info:
  project_name: "Test Classification Project"
  model_type: "classification"

data_loader:
  image_base_dir: "./"
  field_mapping:
    loop: "loop_id"
    image_id: "img_name"
    image_path: "rel_path"
    pre_time_ms: "t_pre"
    inference_time_ms: "t_inf"
    post_time_ms: "t_post"
    total_time_ms: "t_total"
    classification:
      top_k_id_pattern: "class_top_{k}_id"
      top_k_score_pattern: "class_top_{k}_score"

evaluation_params:
  classification:
    top_k: [1, 5]

report_settings:
  output_dir: "./test_reports_class"
"""
    dummy_class_config_path = Path("dummy_config_class_test.yaml")
    with open(dummy_class_config_path, 'w') as f:
        f.write(dummy_class_config_content)

    try:
        print(f"Attempting to load classification config: {dummy_class_config_path.resolve()}")
        cfg_class = load_config(dummy_class_config_path)
        print("Classification configuration loaded successfully!")
        print(f"Top K: {cfg_class.evaluation_params.classification.top_k}")
    except Exception as e:
        print(f"Error loading classification configuration: {e}")
    finally:
        if dummy_class_config_path.exists():
            dummy_class_config_path.unlink()
        test_reports_class_dir = Path("./test_reports_class")
        if test_reports_class_dir.exists() and test_reports_class_dir.is_dir():
            import shutil
            shutil.rmtree(test_reports_class_dir)
```
