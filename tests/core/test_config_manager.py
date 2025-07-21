import pytest
from pathlib import Path
import yaml

from ai_eval_tool.config_manager import load_config, MainConfig

# Helper function to create a dummy config file for testing
def create_dummy_config(tmp_path, config_content, filename="config.yaml"):
    config_path = tmp_path / filename
    config_path.write_text(config_content)
    return config_path


# --- Test Successful Loading ---
def test_load_valid_config(tmp_path):
    """
    测试成功加载一个完整且合法的 config.yaml 文件。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "img_id"
    classification:
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"
evaluation_params:
  classification:
    top_k: [1, 5]
report_settings:
  output_dir: "./reports"
"""
    config_path = create_dummy_config(tmp_path, config_content)
    config = load_config(config_path)

    assert isinstance(config, MainConfig)
    assert config.project_info.project_name == "Test Project"
    assert config.project_info.model_type == "classification"
    assert config.data_loader.field_mapping.loop == "loop_id"
    assert config.evaluation_params.classification.top_k == [1, 5]
    assert config.report_settings.output_dir == Path("./reports")


# --- Test Pydantic Validation - Failure Scenarios ---
def test_load_config_missing_project_info(tmp_path):
    """
    测试加载缺少 project_info 字段的配置。
    """
    config_content = """
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  classification:
    top_k: [1]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="Missing required section: 'project_info'"):
        load_config(config_path)


def test_load_config_missing_model_type(tmp_path):
    """
    测试加载缺少 model_type 字段的配置。
    """
    config_content = """
project_info:
  project_name: "Test Project"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  classification:
    top_k: [1]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="Missing required field: 'project_info.model_type'"):
        load_config(config_path)


def test_load_config_invalid_field_type(tmp_path):
    """
    测试加载字段类型错误的配置 (iou_threshold 为字符串)。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    detection:
      bbox_pred: "bbox_pred"
      bbox_gt: "bbox_gt"
evaluation_params:
  detection:
    iou_threshold: "invalid_string"
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match=r"Input should be a valid number, unable to parse string as a number"):
        load_config(config_path)


def test_load_config_model_type_mismatch_evaluation_params(tmp_path):
    """
    测试 model_type 与 evaluation_params 不匹配的情况。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
detection:
  category_id: "cat_id"
evaluation_params:
  classification:
    top_k: [1]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="Missing evaluation parameters for model type 'detection'"):
        load_config(config_path)


def test_load_config_model_type_mismatch_field_mapping(tmp_path):
    """
    测试 model_type 与 field_mapping 不匹配的情况。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    classification:
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"
evaluation_params:
  detection:
    iou_threshold: 0.5
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="Missing field mapping for model type 'detection'"):
        load_config(config_path)


def test_load_config_top_k_invalid_values(tmp_path):
    """
    测试 top_k 列表包含非正整数。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  classification:
    top_k: [1, 0, 5]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="All k values must be positive integers"):
        load_config(config_path)


def test_load_config_top_k_unsorted_or_duplicates(tmp_path):
    """
    测试 top_k 列表未排序或包含重复值。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
evaluation_params:
  classification:
    top_k: [5, 1, 5]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="top_k list must be sorted in ascending order and contain unique values"):
        load_config(config_path)


def test_load_config_k_values_invalid_values(tmp_path):
    """
    测试 k_values 列表包含非正整数。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "ranking"
data_loader:
  field_mapping:
    loop: "loop_id"
    query_id: "query_id"
    ranking:
      item_id_pred_list: "pred_list"
      item_id_gt_list: "gt_list"
evaluation_params:
  ranking:
    k_values: [5, 0, 10]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="All k values must be positive integers"):
        load_config(config_path)


def test_load_config_k_values_unsorted_or_duplicates(tmp_path):
    """
    测试 k_values 列表未排序或包含重复值。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "ranking"
data_loader:
  field_mapping:
    loop: "loop_id"
    query_id: "query_id"
    ranking:
      item_id_pred_list: "pred_list"
      item_id_gt_list: "gt_list"
evaluation_params:
  ranking:
    k_values: [10, 5, 10]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="k_values list must be sorted in ascending order and contain unique values"):
        load_config(config_path)


def test_load_config_file_not_found():
    """
    测试配置文件不存在的情况。
    """
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("non_existent_config.yaml")


def test_load_config_path_is_not_file(tmp_path):
    """
    测试路径存在但不是文件的情况。
    """
    dir_path = tmp_path / "dummy_dir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="Path exists but is not a file"):
        load_config(dir_path)


def test_load_config_empty_file(tmp_path):
    """
    测试配置文件为空的情况。
    """
    config_path = create_dummy_config(tmp_path, "")
    with pytest.raises(ValueError, match="Configuration file is empty or contains only comments"):
        load_config(config_path)


def test_load_config_invalid_yaml_syntax(tmp_path):
    """
    测试 YAML 语法错误的情况。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
  - invalid_line
evaluation_params:
  classification:
    top_k: [1]
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(yaml.YAMLError, match="Failed to parse YAML configuration file"):
        load_config(config_path)


def test_load_config_root_not_dict(tmp_path):
    """
    测试 YAML 根节点不是字典的情况。
    """
    config_content = """
- item1
- item2
    """
    config_path = create_dummy_config(tmp_path, config_content)
    with pytest.raises(ValueError, match="Configuration file must contain a YAML object \(dictionary\)"):
        load_config(config_path)