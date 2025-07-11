import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

from ai_eval_tool.config_manager import MainConfig
from ai_eval_tool.evaluators.stability.classification import ClassificationStabilityEvaluator, jaccard_similarity
from ai_eval_tool.utils.types import EvaluationResult

@pytest.fixture
def classification_stability_evaluator(classification_config: MainConfig) -> ClassificationStabilityEvaluator:
    return ClassificationStabilityEvaluator(classification_config)

# Test Jaccard similarity helper
def test_jaccard_similarity():
    set1 = {1, 2, 3}
    set2 = {2, 3, 4}
    # Intersection = {2,3} (len 2), Union = {1,2,3,4} (len 4) -> J = 2/4 = 0.5
    assert jaccard_similarity(set1, set2) == 0.5

    set_empty1, set_empty2 = set(), set()
    assert jaccard_similarity(set_empty1, set_empty2) == 1.0 # Both empty

    set_disjoint1 = {1,2}
    set_disjoint2 = {3,4}
    assert jaccard_similarity(set_disjoint1, set_disjoint2) == 0.0

    set_identical = {1,2,3}
    assert jaccard_similarity(set_identical, set_identical) == 1.0


def test_classification_stability_metrics(classification_stability_evaluator: ClassificationStabilityEvaluator, dummy_classification_csv_path: Path):
    # classification_config has top_k = [1, 3]
    # DataLoader will produce top_k_labels = [[label_k1, label_k3]], top_k_scores = [[score_k1, score_k3]]
    from ai_eval_tool.data_loader import DataLoader # Local import for test
    data_loader = DataLoader(classification_stability_evaluator.config)
    input_df = data_loader.load_data(dummy_classification_csv_path)

    result = classification_stability_evaluator.evaluate(input_df)
    metrics = result.metrics
    extra_data = result.extra_data

    assert isinstance(result, EvaluationResult)
    assert "cls_stab_mean_top_1_consistency_rate" in metrics
    assert "cls_stab_mean_top_1_confidence_std" in metrics
    assert "cls_stab_mean_jaccard_top_1" in metrics # Based on config top_k=[1,3]
    assert "cls_stab_mean_jaccard_top_3" in metrics
    assert "classification_stability_details_df" in extra_data

    details_df = extra_data["classification_stability_details_df"]
    assert isinstance(details_df, pl.DataFrame)
    assert details_df.shape[0] == 2 # Two unique image_ids (imgA, imgB)

    # --- Manual Calculation for imgA ---
    # Loops: 1, 2
    # L1: labels=["cat", "dog"], scores=[0.9, 0.8] -> Top-1 label="cat", Top-1 score=0.9
    # L2: labels=["cat", "fox"], scores=[0.88, 0.75] -> Top-1 label="cat", Top-1 score=0.88

    # Top-1 Consistency for imgA: "cat" vs "cat" -> Consistent. Rate = 1.0
    imgA_details = details_df.filter(pl.col("image_id") == "imgA")
    assert imgA_details[0, "top_1_consistency_rate"] == 1.0

    # Top-1 Confidence Std for imgA: std of [0.9, 0.88]
    imgA_conf_std = np.std([0.9, 0.88])
    assert imgA_details[0, "top_1_confidence_std"] == pytest.approx(imgA_conf_std)

    # Jaccard for imgA (current simplified impl: set of all labels in the list for that row)
    # L1_set_all = {"cat", "dog"}
    # L2_set_all = {"cat", "fox"}
    # J(L1_set_all, L2_set_all) = 1/3
    # The mean_top_k_jaccard in details_df is a dict: {1: val, 3: val}.
    # Current impl of jaccard in evaluator uses the same set for all K values.
    imgA_jaccard_val = 1/3
    assert imgA_details[0, "mean_top_k_jaccard"][1] == pytest.approx(imgA_jaccard_val) # K=1
    assert imgA_details[0, "mean_top_k_jaccard"][3] == pytest.approx(imgA_jaccard_val) # K=3

    # --- Manual Calculation for imgB ---
    # Loops: 1, 2
    # L1: labels=["bird", "fish"], scores=[0.95, 0.85] -> Top-1 label="bird", Top-1 score=0.95
    # L2: labels=["bird", "fish"], scores=[0.93, 0.82] -> Top-1 label="bird", Top-1 score=0.93

    # Top-1 Consistency for imgB: "bird" vs "bird" -> Consistent. Rate = 1.0
    imgB_details = details_df.filter(pl.col("image_id") == "imgB")
    assert imgB_details[0, "top_1_consistency_rate"] == 1.0

    # Top-1 Confidence Std for imgB: std of [0.95, 0.93]
    imgB_conf_std = np.std([0.95, 0.93])
    assert imgB_details[0, "top_1_confidence_std"] == pytest.approx(imgB_conf_std)

    # Jaccard for imgB
    # L1_set_all = {"bird", "fish"}
    # L2_set_all = {"bird", "fish"}
    # J(L1_set_all, L2_set_all) = 2/2 = 1.0
    imgB_jaccard_val = 1.0
    assert imgB_details[0, "mean_top_k_jaccard"][1] == pytest.approx(imgB_jaccard_val) # K=1
    assert imgB_details[0, "mean_top_k_jaccard"][3] == pytest.approx(imgB_jaccard_val) # K=3

    # --- Aggregated Metrics ---
    assert metrics["cls_stab_mean_top_1_consistency_rate"] == pytest.approx((1.0 + 1.0) / 2)
    assert metrics["cls_stab_mean_top_1_confidence_std"] == pytest.approx((imgA_conf_std + imgB_conf_std) / 2)
    assert metrics["cls_stab_mean_jaccard_top_1"] == pytest.approx((imgA_jaccard_val + imgB_jaccard_val) / 2)
    assert metrics["cls_stab_mean_jaccard_top_3"] == pytest.approx((imgA_jaccard_val + imgB_jaccard_val) / 2)


def test_classification_stability_single_loop(classification_stability_evaluator: ClassificationStabilityEvaluator):
    data = {
        "image_id": ["imgX"], "loop": [1],
        "top_k_labels": [["apple", "banana"]],
        "top_k_scores": [[0.99, 0.88]]
    }
    input_df = pl.DataFrame(data)
    result = classification_stability_evaluator.evaluate(input_df)
    metrics = result.metrics
    details_df = result.extra_data["classification_stability_details_df"]

    assert details_df.shape[0] == 1
    assert details_df[0, "top_1_consistency_rate"] == 1.0
    assert details_df[0, "top_1_confidence_std"] == 0.0
    assert details_df[0, "mean_top_k_jaccard"][1] == 1.0 # For K=1
    assert details_df[0, "mean_top_k_jaccard"][3] == 1.0 # For K=3 (as per config)

    assert metrics["cls_stab_mean_top_1_consistency_rate"] == 1.0
    assert metrics["cls_stab_mean_top_1_confidence_std"] == 0.0

def test_classification_stability_empty_input(classification_stability_evaluator: ClassificationStabilityEvaluator):
    empty_df = pl.DataFrame({
        "image_id": [], "loop": [], "top_k_labels": [], "top_k_scores": []
    }, schema={
        "image_id": pl.Utf8, "loop": pl.Int64,
        "top_k_labels": pl.List(pl.Utf8), "top_k_scores": pl.List(pl.Float64)
    })
    result = classification_stability_evaluator.evaluate(empty_df)
    assert "warning" in result.metrics
    assert "Empty input data" in result.metrics["warning"]

def test_classification_stability_missing_cols(classification_stability_evaluator: ClassificationStabilityEvaluator):
    df_missing_cols = pl.DataFrame({"image_id": ["img1"], "loop": [1]})
    result = classification_stability_evaluator.evaluate(df_missing_cols)
    assert "error" in result.metrics
    assert "Missing columns" in result.metrics["error"]

```
