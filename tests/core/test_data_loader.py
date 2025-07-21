import pytest
from pathlib import Path
import polars as pl
import yaml

from ai_eval_tool.config_manager import load_config
from ai_eval_tool.data_loader import DataLoader


# Helper function to create a dummy config file for testing
def create_dummy_config(tmp_path, config_content, filename="config.yaml"):
    config_path = tmp_path / filename
    config_path.write_text(config_content)
    return config_path


# Helper function to create a dummy CSV file for testing
def create_dummy_csv(tmp_path, csv_content, filename="data.csv"):
    csv_path = tmp_path / filename
    csv_path.write_text(csv_content)
    return csv_path


# --- Test Basic Loading and Common Column Renaming ---
def test_load_data_basic_and_common_rename(tmp_path):
    """
    测试基本数据加载和通用列重命名。
    """
    config_content = """
project_info:
  project_name: "Test Project"
  model_type: "detection" # Changed to detection for a more generic config
data_loader:
  field_mapping:
    loop: "my_loop_col"
    image_id: "my_image_id_col"
    pre_time_ms: "pre_ms"
    inference_time_ms: "inf_ms"
    post_time_ms: "post_ms"
    total_time_ms: "total_ms"
    detection: # Added minimal detection config to satisfy validation
      category_id: "dummy_cat"
      score: "dummy_score"
      bbox: ["dummy_x", "dummy_y", "dummy_w", "dummy_h"]
evaluation_params:
  detection: # Added minimal detection config to satisfy validation
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    csv_content = """my_loop_col,my_image_id_col,pre_ms,inf_ms,post_ms,total_ms,other_col,dummy_cat,dummy_score,dummy_x,dummy_y,dummy_w,dummy_h
1,img_001,10,20,5,35,data_a,1,0.9,10,10,10,10
2,img_002,12,22,6,40,data_b,2,0.8,20,20,20,20
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert df.shape == (2, 11)
    assert "loop" in df.columns
    assert "image_id" in df.columns
    assert "pre_time_ms" in df.columns
    assert "inference_time_ms" in df.columns
    assert "post_time_ms" in df.columns
    assert "total_time_ms" in df.columns
    assert "other_col" in df.columns
    assert df["loop"].to_list() == [1, 2]
    assert df["image_id"].to_list() == ["img_001", "img_002"]


# --- Test Model-Specific Field Processing ---

# Detection
def test_process_detection_fields_xywh(tmp_path):
    """
    测试 Detection 模型 xywh 格式 bbox 的处理。
    """
    config_content = """
project_info:
  project_name: "Test Detection"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    detection:
      category_id: "cat_id"
      score: "score_val"
      bbox: ["x_center", "y_center", "width", "height"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    csv_content = """loop_id,image_id,cat_id,score_val,x_center,y_center,width,height
1,img1,1,0.9,100,100,20,20
2,img2,2,0.8,50,50,10,10
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_bbox" in df.columns
    assert df["internal_bbox"].dtype == pl.List(pl.Float64)
    # Expected xyxy: [x_min, y_min, x_max, y_max]
    # For (100,100,20,20) -> [90,90,110,110]
    # For (50,50,10,10) -> [45,45,55,55]
    assert df["internal_bbox"].to_list() == [[90.0, 90.0, 110.0, 110.0], [45.0, 45.0, 55.0, 55.0]]
    assert "category_id" in df.columns
    assert "score" in df.columns


def test_process_detection_fields_xyxy(tmp_path):
    """
    测试 Detection 模型 xyxy 格式 bbox 的处理。
    """
    config_content = """
project_info:
  project_name: "Test Detection"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    detection:
      category_id: "cat_id"
      score: "score_val"
      bbox: ["x_min", "y_min", "x_max", "y_max"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xyxy"
report_settings: {}
"""
    csv_content = """loop_id,image_id,cat_id,score_val,x_min,y_min,x_max,y_max
1,img1,1,0.9,10,10,30,30
2,img2,2,0.8,40,40,60,60
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_bbox" in df.columns
    assert df["internal_bbox"].dtype == pl.List(pl.Float64)
    assert df["internal_bbox"].to_list() == [[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]]
    assert "category_id" in df.columns
    assert "score" in df.columns


# Classification
def test_process_classification_fields(tmp_path):
    """
    测试 Classification 模型 Top-K 字段的处理。
    """
    config_content = """
project_info:
  project_name: "Test Classification"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    classification:
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"
evaluation_params:
  classification:
    top_k: [1, 3]
report_settings: {}
"""
    csv_content = """loop_id,image_id,pred_label_top1,pred_score_top1,pred_label_top2,pred_score_top2,pred_label_top3,pred_score_top3
1,img1,cat,0.9,dog,0.08,bird,0.02
2,img2,dog,0.85,cat,0.1,bird,0.05
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "top_k_labels" in df.columns
    assert "top_k_scores" in df.columns
    assert df["top_k_labels"].dtype == pl.List(pl.Utf8)
    # Use a loop for pytest.approx with nested lists
    expected_scores = [[0.9, 0.02], [0.85, 0.05]]
    actual_scores = df["top_k_scores"].to_list()
    for i in range(len(expected_scores)):
        for j in range(len(expected_scores[i])):
            assert actual_scores[i][j] == pytest.approx(expected_scores[i][j])

    assert df["top_k_labels"].to_list() == [["cat", "bird"], ["dog", "bird"]]


# Rotated Detection
def test_process_rotated_detection_fields(tmp_path):
    """
    测试 Rotated Detection 模型 rbbox 字段的处理。
    """
    config_content = """
project_info:
  project_name: "Test Rotated Detection"
  model_type: "rotated_detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    rotated_detection:
      category_id: "r_cat_id"
      score: "r_score_val"
      rbbox: ["cx", "cy", "w", "h", "angle_deg"]
evaluation_params:
  rotated_detection:
    riou_threshold: 0.6
report_settings: {}
"""
    csv_content = """loop_id,image_id,r_cat_id,r_score_val,cx,cy,w,h,angle_deg
1,img1,10,0.95,100,100,20,30,45
2,img2,20,0.85,50,50,15,25,90
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_rbbox" in df.columns
    assert df["internal_rbbox"].dtype == pl.List(pl.Float64)
    assert df["internal_rbbox"].to_list() == [[100.0, 100.0, 20.0, 30.0, 45.0], [50.0, 50.0, 15.0, 25.0, 90.0]]
    assert "category_id" in df.columns
    assert "score" in df.columns


# Pose
def test_process_pose_fields(tmp_path):
    """
    测试 Pose 模型 keypoints 字段的处理。
    """
    config_content = """
project_info:
  project_name: "Test Pose"
  model_type: "pose"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    pose:
      person_bbox: ["pb_x", "pb_y", "pb_w", "pb_h"]
      person_score: "p_score"
      keypoints: "kpts_str"
evaluation_params:
  pose:
    oks_sigma: 0.5
report_settings: {}
"""
    csv_content = """loop_id,image_id,pb_x,pb_y,pb_w,pb_h,p_score,kpts_str
1,img1,10,10,50,100,0.99,"10.5,20.1,0.9;30.2,40.5,0.8"
2,img2,20,20,60,120,0.98,"5.0,15.0,0.7"
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_person_bbox" in df.columns
    assert df["internal_person_bbox"].dtype == pl.List(pl.Float64)
    # Expected xyxy: [x_min, y_min, x_max, y_max]
    # For (10,10,50,100) -> [ -15.0, -40.0, 35.0, 60.0]
    # For (20,20,60,120) -> [ -10.0, -40.0, 50.0, 80.0]
    assert df["internal_person_bbox"].to_list() == [[-15.0, -40.0, 35.0, 60.0], [-10.0, -40.0, 50.0, 80.0]]

    assert "internal_keypoints" in df.columns
    assert df["internal_keypoints"].dtype == pl.Object # Polars stores List[List[float]] as Object
    assert df["internal_keypoints"].to_list() == [[[10.5, 20.1, 0.9], [30.2, 40.5, 0.8]], [[5.0, 15.0, 0.7]]]
    assert "person_score" in df.columns


# Tracking
def test_process_tracking_fields(tmp_path):
    """
    测试 Tracking 模型 bbox_pred 和 bbox_gt 字段的处理。
    """
    config_content = """
project_info:
  project_name: "Test Tracking"
  model_type: "tracking"
data_loader:
  field_mapping:
    loop: "loop_id"
    frame_id: "frame_num"
    object_id_pred: "pred_obj_id"
    tracking:
      bbox_pred: ["px", "py", "pw", "ph"]
      object_id_gt: "gt_obj_id"
      bbox_gt: ["gx", "gy", "gw", "gh"]
evaluation_params:
  tracking:
    mota_iou_threshold: 0.7
    bbox_pred_format: "xywh"
    bbox_gt_format: "xywh"
report_settings: {}
"""
    csv_content = """loop_id,frame_num,pred_obj_id,px,py,pw,ph,gt_obj_id,gx,gy,gw,gh
1,1,101,100,100,20,20,1,101,101,20,20
2,2,101,102,102,20,20,1,103,103,20,20
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_bbox_pred" in df.columns
    assert "internal_bbox_gt" in df.columns
    assert df["internal_bbox_pred"].dtype == pl.List(pl.Float64)
    assert df["internal_bbox_gt"].dtype == pl.List(pl.Float64)

    # Expected xyxy for pred: [90,90,110,110], [92,92,112,112]
    # Expected xyxy for gt: [91,91,111,111], [93,93,113,113]
    assert df["internal_bbox_pred"].to_list() == [[90.0, 90.0, 110.0, 110.0], [92.0, 92.0, 112.0, 112.0]]
    assert df["internal_bbox_gt"].to_list() == [[91.0, 91.0, 111.0, 111.0], [93.0, 93.0, 113.0, 113.0]]
    assert "object_id_pred" in df.columns
    assert "object_id_gt" in df.columns
    assert "frame_id" in df.columns


# Ranking
def test_process_ranking_fields(tmp_path):
    """
    测试 Ranking 模型列表字段的处理。
    """
    config_content = """
project_info:
  project_name: "Test Ranking"
  model_type: "ranking"
data_loader:
  field_mapping:
    loop: "session_id"
    query_id: "user_query"
    ranking:
      item_id_pred_list: "predicted_items_str"
      score_pred_list: "predicted_scores_str"
      item_id_gt_list: "ground_truth_items_str"
      list_delimiter: ";"
evaluation_params:
  ranking:
    k_values: [5, 10]
report_settings: {}
"""
    csv_content = """session_id,user_query,predicted_items_str,predicted_scores_str,ground_truth_items_str
1,query_A,item1;item2;item3,0.9;0.8;0.7,item1;item3;item4
2,query_B,itemX;itemY,0.95;0.85,itemY;itemZ
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    df = data_loader.load_data(csv_path)

    assert "internal_item_id_pred_list" in df.columns
    assert "internal_score_pred_list" in df.columns
    assert "internal_item_id_gt_list" in df.columns

    assert df["internal_item_id_pred_list"].dtype == pl.List(pl.Utf8)
    assert df["internal_score_pred_list"].dtype == pl.List(pl.Float64)
    assert df["internal_item_id_gt_list"].dtype == pl.List(pl.Utf8)

    assert df["internal_item_id_pred_list"].to_list() == [["item1", "item2", "item3"], ["itemX", "itemY"]]
    assert df["internal_score_pred_list"].to_list() == [[0.9, 0.8, 0.7], [0.95, 0.85]]
    assert df["internal_item_id_gt_list"].to_list() == [["item1", "item3", "item4"], ["itemY", "itemZ"]]
    assert "query_id" in df.columns # Check if common field is renamed


# --- Test Robustness ---
def test_load_data_missing_columns_in_csv(tmp_path):
    """
    测试加载 CSV 文件时缺少部分配置中定义的列。
    """
    config_content = """
project_info:
  project_name: "Test Missing Cols"
  model_type: "detection"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    detection:
      category_id: "cat_id"
      score: "score_val"
      bbox: ["x_center", "y_center", "width", "height"] # 'height' will be missing
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    csv_content = """loop_id,image_id,cat_id,score_val,x_center,y_center,width
1,img1,1,0.9,100,100,20
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    with pytest.raises(ValueError, match=r"Model-specific field processing failed for detection: Missing bbox columns: .*height.*"):
        data_loader.load_data(csv_path)


def test_load_data_empty_csv(tmp_path):
    """
    测试加载空的 CSV 文件。
    """
    config_content = """
project_info:
  project_name: "Test Empty CSV"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    classification:
      top_k_id_pattern: "dummy_id"
      top_k_score_pattern: "dummy_score"
evaluation_params:
  classification:
    top_k: [1]
report_settings: {}
"""
    csv_content = """
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    with pytest.raises(ValueError, match=r"CSV file reading failed: empty CSV"):
        data_loader.load_data(csv_path)


def test_load_data_header_only_csv(tmp_path):
    """
    测试加载只有表头的 CSV 文件。
    """
    config_content = """
project_info:
  project_name: "Test Header Only CSV"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    classification:
      top_k_id_pattern: "dummy_id"
      top_k_score_pattern: "dummy_score"
evaluation_params:
  classification:
    top_k: [1]
report_settings: {}
"""
    csv_content = """loop_id,image_id
"""
    config_path = create_dummy_config(tmp_path, config_content)
    csv_path = create_dummy_csv(tmp_path, csv_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    with pytest.raises(ValueError, match=r"CSV file contains no data rows"):
        data_loader.load_data(csv_path)


def test_load_data_file_not_found():
    """
    测试 CSV 文件不存在的情况。
    """
    config_content = """
project_info:
  project_name: "Test File Not Found"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    classification:
      top_k_id_pattern: "dummy_id"
      top_k_score_pattern: "dummy_score"
evaluation_params:
  classification:
    top_k: [1]
report_settings: {}
"""
    config_path = create_dummy_config(Path("./"), config_content, filename="temp_config_for_not_found.yaml") # Use current dir for config
    
    # Ensure the config file is created for load_config to succeed
    with open(config_path, "w") as f:
        f.write(config_content)

    config = load_config(config_path)
    data_loader = DataLoader(config)
    with pytest.raises(FileNotFoundError, match="CSV file not found:"):
        data_loader.load_data("non_existent_data.csv")
    
    # Clean up the temporary config file
    if config_path.exists():
        config_path.unlink()


def test_load_data_path_is_not_file(tmp_path):
    """
    测试 CSV 路径存在但不是文件的情况。
    """
    config_content = """
project_info:
  project_name: "Test Path Not File"
  model_type: "classification"
data_loader:
  field_mapping:
    loop: "loop_id"
    image_id: "image_id"
    classification:
      top_k_id_pattern: "dummy_id"
      top_k_score_pattern: "dummy_score"
evaluation_params:
  classification:
    top_k: [1]
report_settings: {}
"""
    config_path = create_dummy_config(tmp_path, config_content)
    dummy_dir = tmp_path / "dummy_data_dir"
    dummy_dir.mkdir()

    config = load_config(config_path)
    data_loader = DataLoader(config)
    with pytest.raises(ValueError, match="Path is not a file:"):
        data_loader.load_data(dummy_dir)