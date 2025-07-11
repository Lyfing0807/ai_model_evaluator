import pytest
from pathlib import Path
import yaml
from pydantic import ValidationError

from ai_eval_tool.config_manager import load_config, MainConfig, ProjectInfo, DataLoaderConfig, EvaluationParams, ReportSettings

# Test successful loading of a valid config (using fixture from conftest.py)
def test_load_valid_detection_config(detection_config: MainConfig):
    assert detection_config is not None
    assert detection_config.project_info.project_name == "Pytest Detection Project"
    assert detection_config.project_info.model_type == "detection"
    assert detection_config.data_loader.field_mapping.loop == "loop_id"
    assert detection_config.evaluation_params.detection is not None
    assert detection_config.evaluation_params.detection.iou_threshold == 0.5
    assert "reports_detection" in str(detection_config.report_settings.output_dir)

def test_load_valid_classification_config(classification_config: MainConfig):
    assert classification_config is not None
    assert classification_config.project_info.project_name == "Pytest Classification Project"
    assert classification_config.project_info.model_type == "classification"
    assert classification_config.data_loader.field_mapping.classification.top_k_id_pattern == "label_k{k}"
    assert classification_config.evaluation_params.classification is not None
    assert classification_config.evaluation_params.classification.top_k == [1, 3]
    assert "reports_classification" in str(classification_config.report_settings.output_dir)


def test_load_config_file_not_found(test_data_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_config(test_data_dir / "non_existent_config.yaml")

def test_load_invalid_yaml_format(test_data_dir: Path):
    invalid_yaml_file = test_data_dir / "invalid_format.yaml"
    with open(invalid_yaml_file, 'w') as f:
        f.write("project_info: {project_name: Test\nmodel_type: detection") # Malformed YAML

    with pytest.raises(yaml.YAMLError):
        load_config(invalid_yaml_file)

def test_pydantic_validation_missing_required_field(test_data_dir: Path):
    config_content = {
        # project_info is missing
        "data_loader": {
            "field_mapping": {"loop": "loop"} # Incomplete but valid for this part
        }
    }
    config_file = test_data_dir / "missing_project_info.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "project_info" in str(excinfo.value).lower()
    assert "field required" in str(excinfo.value).lower()


def test_pydantic_validation_incorrect_type(test_data_dir: Path):
    config_content = {
        "project_info": {"project_name": "Incorrect Type Test", "model_type": "detection"},
        "data_loader": {"field_mapping": {"loop": "loop_id"}}, # Valid up to here
        "evaluation_params": {
            "detection": {"iou_threshold": "not_a_float"} # Incorrect type
        },
         "report_settings": {"output_dir": str(test_data_dir / "reports_incorrect_type")}
    }
    config_file = test_data_dir / "incorrect_type.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_incorrect_type").mkdir(parents=True, exist_ok=True)


    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "iou_threshold" in str(excinfo.value)
    assert "Input should be a valid number" in str(excinfo.value) # Pydantic v2 error message

def test_pydantic_validation_invalid_enum_value(test_data_dir: Path):
    config_content = {
        "project_info": {"project_name": "Invalid Enum", "model_type": "super_detector"}, # Invalid model_type
        "data_loader": {"field_mapping": {"loop": "loop_id"}},
         "report_settings": {"output_dir": str(test_data_dir / "reports_invalid_enum")}
    }
    config_file = test_data_dir / "invalid_enum.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_invalid_enum").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "model_type" in str(excinfo.value)
    # Pydantic v2 error message for Literal might be like: "Input should be 'detection', 'classification', ..."
    assert "Input should be" in str(excinfo.value) and "detection" in str(excinfo.value)


def test_custom_validator_top_k(test_data_dir: Path):
    # Test invalid top_k: not sorted
    config_content = {
        "project_info": {"project_name": "TopK Test", "model_type": "classification"},
        "data_loader": {"field_mapping": {"classification": {"top_k_id_pattern": "id{k}"}}},
        "evaluation_params": {"classification": {"top_k": [3, 1, 5]}}, # Not sorted
        "report_settings": {"output_dir": str(test_data_dir / "reports_topk_invalid")}
    }
    config_file = test_data_dir / "invalid_topk.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_topk_invalid").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "top_k list must be sorted and contain unique values" in str(excinfo.value)

    # Test invalid top_k: contains non-positive
    config_content["evaluation_params"]["classification"]["top_k"] = [0, 1, 5]
    with open(config_file, 'w') as f: # Overwrite
        yaml.dump(config_content, f)
    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "All k values in top_k must be positive" in str(excinfo.value)

def test_custom_validator_model_specific_params_missing(test_data_dir: Path):
    config_content = {
        "project_info": {"project_name": "Missing Params", "model_type": "detection"},
        "data_loader": { # Provide valid detection field mapping to pass that check
            "field_mapping": {"detection": {"category_id": "cat", "score": "sc", "bbox": ["x","y","w","h"]}}
        },
        # evaluation_params.detection is missing
        "report_settings": {"output_dir": str(test_data_dir / "reports_missing_params")}
    }
    config_file = test_data_dir / "missing_params.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_missing_params").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "evaluation_params for model_type 'detection' are missing" in str(excinfo.value)

def test_custom_validator_model_specific_field_mapping_missing(test_data_dir: Path):
    config_content = {
        "project_info": {"project_name": "Missing Mapping", "model_type": "classification"},
        "data_loader": {
            "field_mapping": {} # classification field mapping is missing
        },
        "evaluation_params": {"classification": {"top_k": [1,3,5]}}, # Valid params
        "report_settings": {"output_dir": str(test_data_dir / "reports_missing_mapping")}
    }
    config_file = test_data_dir / "missing_mapping.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_missing_mapping").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValidationError) as excinfo:
        load_config(config_file)
    assert "data_loader.field_mapping for model_type 'classification' is missing" in str(excinfo.value)

def test_output_dir_creation(test_data_dir: Path):
    output_dir_name = "new_test_reports_dir"
    output_dir_path = test_data_dir / output_dir_name
    assert not output_dir_path.exists() # Ensure it doesn't exist before test

    config_content = {
        "project_info": {"project_name": "Dir Creation Test", "model_type": "detection"},
        "data_loader": {"field_mapping": {"detection": {"category_id":"c", "score":"s", "bbox":["x","y","w","h"]}}},
        "evaluation_params": {"detection": {"iou_threshold": 0.5, "bbox_format": "xywh"}},
        "report_settings": {"output_dir": str(output_dir_path)}
    }
    config_file = test_data_dir / "dir_creation_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    loaded_cfg = load_config(config_file)
    assert loaded_cfg.report_settings.output_dir == output_dir_path
    assert output_dir_path.exists()
    assert output_dir_path.is_dir()

```
