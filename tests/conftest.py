from pathlib import Path

import polars as pl
import pytest
import yaml

from ai_eval_tool.config_manager import MainConfig, load_config
from ai_eval_tool.utils.types import EvaluationResult


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Creates a temporary directory for test input data and configs."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def dummy_detection_config_path(test_data_dir: Path) -> Path:
    config_content = {
        "project_info": {
            "project_name": "Pytest Detection Project",
            "model_type": "detection",
            "run_id": "pytest_det_run_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop_id",
                "image_id": "img_name",
                "image_path": "path",
                "pre_time_ms": "t_pre",
                "inference_time_ms": "t_inf",
                "post_time_ms": "t_post",
                "total_time_ms": "t_total",
                "detection": {
                    "category_id": "det_cat",
                    "score": "det_score",
                    "bbox": ["x", "y", "w", "h"],
                },
            }
        },
        "evaluation_params": {
            "detection": {"iou_threshold": 0.5, "bbox_format": "xywh"}
        },
        "report_settings": {"output_dir": str(test_data_dir / "reports_detection")},
    }
    config_file = test_data_dir / "dummy_detection_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_detection").mkdir(parents=True, exist_ok=True)
    return config_file


@pytest.fixture(scope="session")
def dummy_classification_config_path(test_data_dir: Path) -> Path:
    config_content = {
        "project_info": {
            "project_name": "Pytest Classification Project",
            "model_type": "classification",
            "run_id": "pytest_cls_run_001",
        },
        "data_loader": {
            "field_mapping": {
                "loop": "loop_id",
                "image_id": "img_name",
                "image_path": "path",
                "pre_time_ms": "t_pre",
                "inference_time_ms": "t_inf",
                "post_time_ms": "t_post",
                "total_time_ms": "t_total",
                "classification": {
                    "top_k_id_pattern": "pred_label_top{k}",
                    "top_k_score_pattern": "pred_score_top{k}",
                },
            }
        },
        "evaluation_params": {"classification": {"top_k": [1, 3]}},
        "report_settings": {
            "output_dir": str(test_data_dir / "reports_classification")
        },
    }
    config_file = test_data_dir / "dummy_classification_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)
    (test_data_dir / "reports_classification").mkdir(parents=True, exist_ok=True)
    return config_file


@pytest.fixture
def detection_config(dummy_detection_config_path: Path) -> MainConfig:
    return load_config(dummy_detection_config_path)


@pytest.fixture
def classification_config(dummy_classification_config_path: Path) -> MainConfig:
    return load_config(dummy_classification_config_path)


@pytest.fixture(scope="session")
def dummy_detection_csv_path(test_data_dir: Path) -> Path:
    csv_content = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w,h\n"
        "1,img1.jpg,path1,10,20,5,35,0,0.9,100,100,50,50\n"
        "1,img1.jpg,path1,10,20,5,35,1,0.8,10,10,20,20\n"  # Same image, different object
        "2,img1.jpg,path1,12,22,6,40,0,0.85,102,102,50,50\n"  # Loop 2, obj 1 moved slightly
        "1,img2.jpg,path2,8,18,4,30,0,0.95,200,200,30,30\n"
        "2,img2.jpg,path2,9,19,4,32,0,0.93,201,201,30,30\n"
    )
    csv_file = test_data_dir / "dummy_detection_data.csv"
    with open(csv_file, "w") as f:
        f.write(csv_content)
    return csv_file


@pytest.fixture(scope="session")
def dummy_classification_csv_path(test_data_dir: Path) -> Path:
    csv_content = (
        "loop_id,img_name,path,t_pre,t_inf,t_post,t_total,label_k1,score_k1,label_k3,score_k3\n"
        "1,imgA.jpg,pathA,5,10,2,17,cat,0.9,dog,0.8\n"
        "2,imgA.jpg,pathA,6,11,2,19,cat,0.88,fox,0.75\n"
        "1,imgB.jpg,pathB,7,12,3,22,bird,0.95,fish,0.85\n"
        "2,imgB.jpg,pathB,7,13,3,23,bird,0.93,fish,0.82\n"
    )
    csv_file = test_data_dir / "dummy_classification_data.csv"
    with open(csv_file, "w") as f:
        f.write(csv_content)
    return csv_file


@pytest.fixture
def sample_evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        metrics={"metric1": 1.0, "metric2": "value"},
        extra_data={
            "df1": pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
            "non_df_data": {"x": 100},
        },
    )


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory for caching tests, cleaned up after test."""
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Cleanup reports directories after test session if they were created by fixtures
def pytest_sessionfinish(session, exitstatus):
    # This is a bit broad; ideally, fixtures would handle their own cleanup.
    # For session-scoped tmp_path_factory, pytest handles it.
    # This is for output_dirs specified in dummy configs if they are outside tmp_path_factory.
    # However, the fixtures above place reports inside test_data_dir, which IS from tmp_path_factory.
    # So explicit cleanup here might not be strictly necessary if all outputs go to tmp_path.

    # Example cleanup if output_dir was fixed:
    # report_output_dir = Path("./test_pytest_reports") # Example fixed path
    # if report_output_dir.exists():
    #     print(f"\nCleaning up test report output directory: {report_output_dir}")
    #     shutil.rmtree(report_output_dir)
    pass
