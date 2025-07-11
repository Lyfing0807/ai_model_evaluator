import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.detection import DetectionStabilityEvaluator, calculate_iou
from ai_eval_tool.utils.types import EvaluationResult

@pytest.fixture
def detection_stability_evaluator(detection_config: MainConfig) -> DetectionStabilityEvaluator:
    return DetectionStabilityEvaluator(detection_config)

# Test IoU calculation helper
def test_calculate_iou():
    box1 = [0, 0, 10, 10] # Area 100
    box2 = [5, 5, 15, 15] # Area 100
    # Intersection: [5,5,10,10], Area = 25
    # Union: 100 + 100 - 25 = 175
    # IoU = 25 / 175 = 1/7
    assert calculate_iou(box1, box2) == pytest.approx(1/7)

    box_no_overlap = [0, 0, 1, 1]
    box_far_away = [5, 5, 6, 6]
    assert calculate_iou(box_no_overlap, box_far_away) == 0.0

    box_identical = [0,0,10,10]
    assert calculate_iou(box_identical, box_identical) == 1.0

    box_contained = [0,0,10,10]
    box_inner = [2,2,8,8] # Area 36
    # Intersection Area 36, Union Area 100. IoU = 36/100
    assert calculate_iou(box_contained, box_inner) == pytest.approx(36/100)


def test_track_objects_for_image(detection_stability_evaluator: DetectionStabilityEvaluator):
    # Test data for a single image across multiple loops
    image_data = {
        "image_id": ["imgA"] * 5,
        "loop":     [1, 1, 2, 2, 3],
        "internal_bbox": [
            [10, 10, 20, 20],  # Loop 1, Obj 1 (Track 0)
            [50, 50, 60, 60],  # Loop 1, Obj 2 (Track 1)
            [12, 12, 22, 22],  # Loop 2, Obj 1' (Matches Track 0)
            [80, 80, 90, 90],  # Loop 2, Obj 3 (New Track 2)
            [13, 13, 23, 23],  # Loop 3, Obj 1'' (Matches Track 0)
        ],
        "category_id": [0,1,0,0,0], # Not used in tracking logic directly
        "score": [0.9,0.8,0.85,0.7,0.92] # Not used in tracking logic
    }
    image_df = pl.DataFrame(image_data)

    # Lower IoU threshold for easier matching in test
    detection_stability_evaluator.iou_threshold = 0.5

    tracked_df = detection_stability_evaluator._track_objects_for_image(image_df)

    assert "tracked_object_id" in tracked_df.columns
    assert tracked_df["tracked_object_id"].null_count() == 0

    # Expected track IDs (can vary based on internal tie-breaking if any, but should be consistent)
    # Loop 1, Obj 1 -> Track 0
    # Loop 1, Obj 2 -> Track 1
    # Loop 2, Obj 1' (matches L1O1) -> Track 0
    # Loop 2, Obj 3 (new) -> Track 2
    # Loop 3, Obj 1'' (matches L2O1') -> Track 0

    # Check specific assignments (assuming a stable assignment order from _track_objects_for_image)
    # This depends on the initial assignment of track IDs in the first loop.
    # Let's verify based on the logic: first objects in first loop get IDs 0, 1, ...

    # Get the track ID for the first object in loop 1
    track_id_obj1_l1 = tracked_df.filter((pl.col("loop") == 1) & (pl.col("internal_bbox").list.get(0) == 10.0))["tracked_object_id"][0]
    track_id_obj2_l1 = tracked_df.filter((pl.col("loop") == 1) & (pl.col("internal_bbox").list.get(0) == 50.0))["tracked_object_id"][0]

    # Object 1' in loop 2 should match track_id_obj1_l1
    assert tracked_df.filter((pl.col("loop") == 2) & (pl.col("internal_bbox").list.get(0) == 12.0))["tracked_object_id"][0] == track_id_obj1_l1

    # Object 1'' in loop 3 should also match track_id_obj1_l1
    assert tracked_df.filter((pl.col("loop") == 3) & (pl.col("internal_bbox").list.get(0) == 13.0))["tracked_object_id"][0] == track_id_obj1_l1

    # Object 3 in loop 2 should be a new track ID, different from obj1_l1 and obj2_l1
    track_id_obj3_l2 = tracked_df.filter((pl.col("loop") == 2) & (pl.col("internal_bbox").list.get(0) == 80.0))["tracked_object_id"][0]
    assert track_id_obj3_l2 != track_id_obj1_l1
    assert track_id_obj3_l2 != track_id_obj2_l1

    # Total unique tracks should be 3 (for Obj1-chain, Obj2, Obj3)
    assert tracked_df["tracked_object_id"].n_unique() == 3


def test_calculate_average_box(detection_stability_evaluator: DetectionStabilityEvaluator):
    # Test data for a single tracked object
    object_group_data = {
        "internal_bbox": [
            [10, 10, 20, 20],  # cx=15, cy=15, w=10, h=10
            [12, 12, 22, 22],  # cx=17, cy=17, w=10, h=10
            [14, 14, 24, 24],  # cx=19, cy=19, w=10, h=10
        ]
    }
    object_group_df = pl.DataFrame(object_group_data)
    avg_box = detection_stability_evaluator._calculate_average_box(object_group_df)

    # Expected: avg_cx=17, avg_cy=17, avg_w=10, avg_h=10
    # avg_x_min = 17 - 10/2 = 12
    # avg_y_min = 17 - 10/2 = 12
    # avg_x_max = 17 + 10/2 = 22
    # avg_y_max = 17 + 10/2 = 22
    assert avg_box == pytest.approx([12.0, 12.0, 22.0, 22.0])

def test_detection_stability_full_evaluation(detection_stability_evaluator: DetectionStabilityEvaluator, dummy_detection_csv_path: Path):
    # Use the DataLoader to get the initial DataFrame with 'internal_bbox'
    from ai_eval_tool.data_loader import DataLoader # Local import for test
    data_loader = DataLoader(detection_stability_evaluator.config) # Use config from fixture
    input_df = data_loader.load_data(dummy_detection_csv_path)

    # Lower IoU for test data if needed, default is 0.5 from fixture
    detection_stability_evaluator.iou_threshold = 0.7 # A bit stricter for this test data
                                                    # The dummy_detection_csv data has obj1 moving slightly, IoU should be high.
                                                    # Box1: [75,75,125,125], Box2: [77,77,127,127] (from 102,102,50,50 xywh)
                                                    # IoU([75,75,125,125], [77,77,127,127]) will be high.
                                                    # (50-2)*(50-2) / (50*50 + 50*50 - (50-2)^2) = 48*48 / (2*2500 - 48*48) = 2304 / (5000-2304) = 2304 / 2696 = ~0.85

    result = detection_stability_evaluator.evaluate(input_df)
    metrics = result.metrics
    extra_data = result.extra_data

    assert isinstance(result, EvaluationResult)
    assert "det_stab_mean_appearance_consistency" in metrics
    assert "det_stab_mean_iou_consistency" in metrics
    assert "det_stab_num_total_tracked_instances" in metrics
    assert "object_stability_details_df" in extra_data

    details_df = extra_data["object_stability_details_df"]
    assert isinstance(details_df, pl.DataFrame)

    # Based on dummy_detection_csv_path:
    # img1: loop1 (2 objs), loop2 (1 obj matched, 1 new potentially if not tracked)
    # img2: loop1 (1 obj), loop2 (1 obj matched)
    # Expected tracked instances:
    #   img1, objA (loop1, loop2) -> track0
    #   img1, objB (loop1 only) -> track1
    #   img2, objD (loop1, loop2) -> track2
    # Total 3 unique objects tracked for stability metrics.
    # num_total_tracked_instances is the number of rows in details_df, which is 3.
    assert metrics["det_stab_num_total_tracked_instances"] == 3.0
    # num_unique_objects_tracked_multiple_loops: objA, objD -> 2
    assert metrics["det_stab_num_unique_objects_tracked_multiple_loops"] == 2.0

    # Check appearance consistency for the object that appeared in both loops for img1
    # img1 has 2 loops total. Track 0 (objA) appeared in 2 loops. Consistency = 2/2 = 1.0
    # img1 has 2 loops total. Track 1 (objB) appeared in 1 loop. Consistency = 1/2 = 0.5
    # img2 has 2 loops total. Track 2 (objD) appeared in 2 loops. Consistency = 2/2 = 1.0
    # Mean appearance consistency = (1.0 + 0.5 + 1.0) / 3 = 2.5 / 3 = 0.8333
    assert metrics["det_stab_mean_appearance_consistency"] == pytest.approx(2.5 / 3)

    # Check IoU consistency for one of the stable objects (e.g., track for img2)
    # img2, objD: loop1 [175,175,225,225], loop2 [176,176,226,226]
    # Avg box for img2/objD: cx_avg=200.5, cy_avg=200.5, w_avg=50, h_avg=50
    # avg_box = [175.5, 175.5, 225.5, 225.5]
    # IoU([175,175,225,225], avg_box) and IoU([176,176,226,226], avg_box) should be high.
    # This requires tracing the specific track_id.
    img2_details = details_df.filter(pl.col("image_id") == "img2")
    assert img2_details.height == 1 # Only one tracked object for img2
    assert img2_details[0, "mean_iou_consistency"] > 0.9 # Expect high IoU with its own average

def test_detection_stability_empty_input(detection_stability_evaluator: DetectionStabilityEvaluator):
    empty_df = pl.DataFrame({
        "image_id": [], "loop": [], "internal_bbox": [],
        "category_id": [], "score": []
    }, schema={
        "image_id": pl.Utf8, "loop": pl.Int64, "internal_bbox": pl.List(pl.Float64),
        "category_id": pl.Int64, "score": pl.Float64
    })
    result = detection_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]

def test_detection_stability_missing_tracking_cols(detection_stability_evaluator: DetectionStabilityEvaluator):
    df_missing_bbox = pl.DataFrame({"image_id": ["img1"], "loop": [1], "category_id": [0], "score": [0.9]})
    result = detection_stability_evaluator.evaluate(df_missing_bbox)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]

```
