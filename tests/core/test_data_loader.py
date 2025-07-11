import pytest
from pathlib import Path
import polars as pl
from polars.testing import assert_frame_equal

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.data_loader import DataLoader

def test_load_detection_data_xywh(detection_config: MainConfig, dummy_detection_csv_path: Path):
    """Test loading detection data with xywh format."""
    data_loader = DataLoader(detection_config) # detection_config uses xywh by default
    df = data_loader.load_data(dummy_detection_csv_path)

    assert isinstance(df, pl.DataFrame)
    assert "internal_bbox" in df.columns
    assert "category_id" in df.columns # Renamed from det_cat
    assert "score" in df.columns       # Renamed from det_score
    assert "loop_id" not in df.columns # Original name, should be renamed to loop
    assert "loop" in df.columns

    # Check a specific bbox conversion for xywh: [100,100,50,50] -> [75,75,125,125]
    # x_min = x_c - w/2 = 100 - 50/2 = 75
    # y_min = y_c - h/2 = 100 - 50/2 = 75
    # x_max = x_c + w/2 = 100 + 50/2 = 125
    # y_max = y_c + h/2 = 100 + 50/2 = 125
    first_row_bbox = df.filter(pl.col("img_name") == "img1.jpg", pl.col("loop") == 1)[0, "internal_bbox"]
    assert first_row_bbox == [75.0, 75.0, 125.0, 125.0]

    # Check that original bbox columns are still there if not renamed to internal_bbox
    assert "x" in df.columns

def test_load_detection_data_xyxy(detection_config: MainConfig, test_data_dir: Path):
    """Test loading detection data with xyxy format."""
    # Modify config for xyxy
    detection_config.evaluation_params.detection.bbox_format = "xyxy"
    # field_mapping.bbox should now represent x_min, y_min, x_max, y_max
    detection_config.data_loader.field_mapping.detection.bbox = ["x_min", "y_min", "x_max", "y_max"]

    # Create a new CSV for xyxy
    csv_content_xyxy = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x_min,y_min,x_max,y_max\n"
        "1,img1.jpg,path1,10,20,5,35,0,0.9,75,75,125,125\n"
    )
    csv_xyxy_path = test_data_dir / "dummy_detection_data_xyxy.csv"
    with open(csv_xyxy_path, 'w') as f:
        f.write(csv_content_xyxy)

    data_loader = DataLoader(detection_config)
    df = data_loader.load_data(csv_xyxy_path)

    assert "internal_bbox" in df.columns
    first_row_bbox = df[0, "internal_bbox"]
    assert first_row_bbox == [75.0, 75.0, 125.0, 125.0] # Should be direct mapping for xyxy


def test_load_classification_data(classification_config: MainConfig, dummy_classification_csv_path: Path):
    """Test loading classification data with Top-K patterns."""
    data_loader = DataLoader(classification_config)
    df = data_loader.load_data(dummy_classification_csv_path)

    assert isinstance(df, pl.DataFrame)
    assert "top_k_labels" in df.columns
    assert "top_k_scores" in df.columns
    assert "loop" in df.columns

    # Configured top_k is [1, 3]
    # For first row: label_k1=cat, score_k1=0.9, label_k3=dog, score_k3=0.8
    first_row_labels = df[0, "top_k_labels"]
    first_row_scores = df[0, "top_k_scores"]

    assert first_row_labels == ["cat", "dog"]
    assert first_row_scores == [0.9, 0.8]

def test_missing_column_error(detection_config: MainConfig, test_data_dir: Path):
    """Test error handling for missing columns."""
    # Create CSV missing a bbox column 'h'
    csv_content_missing = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w\n" # 'h' is missing
        "1,img1.jpg,path1,10,20,5,35,0,0.9,100,100,50\n"
    )
    csv_missing_path = test_data_dir / "missing_col_data.csv"
    with open(csv_missing_path, 'w') as f:
        f.write(csv_content_missing)

    data_loader = DataLoader(detection_config)
    with pytest.raises(ValueError, match="Missing bbox columns"):
        data_loader.load_data(csv_missing_path)

def test_missing_top_k_column_error(classification_config: MainConfig, test_data_dir: Path):
    """Test error for missing Top-K column."""
    csv_content_missing_topk = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,label_k1,score_k1\n" # label_k3 and score_k3 missing
        "1,imgA.jpg,pathA,5,10,2,17,cat,0.9\n"
    )
    csv_path = test_data_dir / "missing_topk_data.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content_missing_topk)

    data_loader = DataLoader(classification_config) # config expects top_k = [1,3]
    with pytest.raises(ValueError, match="Missing Top-K ID column: label_k3"):
        data_loader.load_data(csv_path)


def test_empty_csv_file(detection_config: MainConfig, test_data_dir: Path):
    """Test loading an empty CSV file."""
    empty_csv_path = test_data_dir / "empty_data.csv"
    # Create a CSV with only headers
    header = "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w,h\n"
    with open(empty_csv_path, 'w') as f:
        f.write(header)

    data_loader = DataLoader(detection_config)
    df = data_loader.load_data(empty_csv_path)
    assert df.is_empty()

def test_numeric_conversion_for_bbox(detection_config: MainConfig, test_data_dir: Path):
    """Test that bbox columns are converted to numeric if they are strings."""
    csv_content_str_bbox = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w,h\n"
        "1,img1.jpg,path1,10,20,5,35,0,0.9,\"100\",\"100\",\"50\",\"50\"\n" # Bbox values as strings
    )
    csv_path = test_data_dir / "str_bbox_data.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content_str_bbox)

    data_loader = DataLoader(detection_config)
    df = data_loader.load_data(csv_path)

    assert "internal_bbox" in df.columns
    first_row_bbox = df[0, "internal_bbox"]
    # Check if conversion happened correctly: [100,100,50,50] (xywh) -> [75,75,125,125] (xyxy)
    assert first_row_bbox == [75.0, 75.0, 125.0, 125.0]
    assert df["internal_bbox"].dtype == pl.List(pl.Float64)

# TODO: Add tests for rotated_detection and pose data loading when their evaluators are more fleshed out.
# For now, their parsing logic is in DataLoader, so basic tests could be added.

def test_load_pose_data_basic(test_data_dir: Path):
    config_content = {
        "project_info": {"project_name": "Pose Test", "model_type": "pose"},
        "data_loader": {
            "field_mapping": {
                "loop": "loop", "image_id": "image_id", "image_path": "path",
                "pre_time_ms": "t_pre", "inference_time_ms": "t_inf",
                "post_time_ms": "t_post", "total_time_ms": "t_total",
                "pose": {
                    "person_bbox": ["px", "py", "pw", "ph"],
                    "person_score": "pscore",
                    "keypoints": "kps_str"
                }
            }
        },
        "evaluation_params": {"pose": {"oks_sigma": 0.5, "keypoint_layout": "coco_17", "match_iou_threshold": 0.5}},
        "report_settings": {"output_dir": str(test_data_dir / "reports_pose")}
    }
    pose_config_file = test_data_dir / "dummy_pose_config.yaml"
    with open(pose_config_file, 'w') as f: yaml.dump(config_content, f)
    (test_data_dir / "reports_pose").mkdir(parents=True, exist_ok=True)
    config = load_config(pose_config_file)

    csv_content = (
        "loop,image_id,path,t_pre,t_inf,t_post,t_total,px,py,pw,ph,pscore,kps_str\n"
        "1,imgP1,p,1,1,1,3,10,10,5,5,0.9,\"1.0,2.0,0.8;3.0,4.0,0.7\"\n"
        "1,imgP1,p,1,1,1,3,20,20,6,6,0.8,\"5.0,6.0,0.9\"\n" # Second person in same image/loop
    )
    pose_csv_file = test_data_dir / "dummy_pose_data.csv"
    with open(pose_csv_file, 'w') as f: f.write(csv_content)

    data_loader = DataLoader(config)
    df = data_loader.load_data(pose_csv_file)

    assert "internal_person_bbox" in df.columns
    assert "internal_keypoints" in df.columns
    assert df[0, "internal_keypoints"] == [[1.0,2.0,0.8],[3.0,4.0,0.7]]
    assert df[1, "internal_keypoints"] == [[5.0,6.0,0.9]]
    # Check person_bbox conversion (xywh to xyxy)
    # For first row: px=10,py=10,pw=5,ph=5 -> xmin=7.5,ymin=7.5,xmax=12.5,ymax=12.5
    assert df[0, "internal_person_bbox"] == [7.5, 7.5, 12.5, 12.5]

```
