"""
Data Loader: Handles loading and initial preprocessing of model output data from CSV files.
"""
from typing import List, Tuple, Dict, Any
import polars as pl
from pathlib import Path
from .config_manager import DataLoaderConfig, MainConfig
from .utils.logging_config import get_logger

logger = get_logger(__name__)

class DataLoader:
    """
    Loads data from CSV files using Polars and performs initial transformations
    based on the provided configuration.
    """
    def __init__(self, config: MainConfig): # Takes MainConfig to access model_type for specific parsing
        self.main_config = config
        self.data_loader_config = config.data_loader
        self.model_type = config.project_info.model_type

    def load_data(self, csv_file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Loads the CSV file into a Polars DataFrame and performs initial processing.

        Args:
            csv_file_path: Path to the input CSV file.

        Returns:
            A Polars DataFrame with standardized column names and processed fields.
        """
        logger.info(f"Loading data from CSV: {csv_file_path}")
        try:
            df = pl.read_csv(csv_file_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_file_path}: {e}")
            raise

        df = self._rename_common_columns(df)

        # Model-specific processing
        if self.model_type == "detection":
            df = self._process_detection_fields(df)
        elif self.model_type == "classification":
            df = self._process_classification_fields(df)
        elif self.model_type == "rotated_detection":
            df = self._process_rotated_detection_fields(df)
        elif self.model_type == "pose":
            df = self._process_pose_fields(df)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"DataFrame schema after loading:\n{df.schema}")
        return df

    def _rename_common_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Renames common columns based on field_mapping."""
        common_mapping = self.data_loader_config.field_mapping.model_dump(exclude_none=True)

        rename_dict: Dict[str, str] = {}
        for target_name, source_name in common_mapping.items():
            if isinstance(source_name, str) and source_name in df.columns and target_name != source_name:
                rename_dict[source_name] = target_name

        if rename_dict:
            logger.debug(f"Renaming common columns: {rename_dict}")
            df = df.rename(rename_dict)
        return df

    def _process_detection_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to detection models (e.g., bbox)."""
        if not self.data_loader_config.field_mapping.detection:
            logger.warning("Detection field mapping not found in config, skipping detection field processing.")
            return df

        mapping = self.data_loader_config.field_mapping.detection
        eval_params = self.main_config.evaluation_params.detection
        if not eval_params:
            logger.error("Detection evaluation parameters missing, cannot process bbox format.")
            raise ValueError("Detection evaluation parameters missing for bbox processing.")

        # Rename category_id and score
        rename_specific_dict: Dict[str, str] = {}
        if mapping.category_id in df.columns and mapping.category_id != "category_id":
            rename_specific_dict[mapping.category_id] = "category_id"
        if mapping.score in df.columns and mapping.score != "score":
            rename_specific_dict[mapping.score] = "score"
        if rename_specific_dict:
            df = df.rename(rename_specific_dict)

        # Process bounding boxes
        bbox_cols = mapping.bbox
        if not all(col in df.columns for col in bbox_cols):
            logger.error(f"One or more bbox columns {bbox_cols} not found in DataFrame.")
            raise ValueError(f"Missing bbox columns: {bbox_cols}")

        # Ensure bbox columns are numeric
        for col_name in bbox_cols:
            if not df[col_name].dtype.is_numeric():
                 df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))


        bbox_exprs = [pl.col(c) for c in bbox_cols]

        if eval_params.bbox_format == "xywh":
            # x_center, y_center, width, height -> x_min, y_min, x_max, y_max
            x_c, y_c, w, h = bbox_exprs
            x_min = x_c - w / 2
            y_min = y_c - h / 2
            x_max = x_c + w / 2
            y_max = y_c + h / 2
            df = df.with_columns(internal_bbox=pl.concat_list([x_min, y_min, x_max, y_max]))
            logger.info("Converted 'xywh' bboxes to standardized 'internal_bbox' (xyxy).")

        elif eval_params.bbox_format == "xyxy":
            # x_min, y_min, x_max, y_max -> just concatenate
            df = df.with_columns(internal_bbox=pl.concat_list(bbox_exprs))
            logger.info("Standardized 'xyxy' bboxes to 'internal_bbox'.")
        else:
            logger.error(f"Unsupported bbox_format: {eval_params.bbox_format}")
            raise ValueError(f"Unsupported bbox_format: {eval_params.bbox_format}")

        return df

    def _process_classification_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to classification models (e.g., Top-K)."""
        if not self.data_loader_config.field_mapping.classification:
            logger.warning("Classification field mapping not found, skipping classification field processing.")
            return df
        if not self.main_config.evaluation_params.classification:
            logger.error("Classification evaluation parameters (top_k list) missing.")
            raise ValueError("Classification evaluation parameters missing.")

        mapping = self.data_loader_config.field_mapping.classification
        eval_params = self.main_config.evaluation_params.classification

        top_k_values = eval_params.top_k
        id_pattern = mapping.top_k_id_pattern
        score_pattern = mapping.top_k_score_pattern

        label_cols_to_concat = []
        score_cols_to_concat = []

        for k_val in top_k_values:
            id_col_name = id_pattern.format(k=k_val)
            score_col_name = score_pattern.format(k=k_val)

            if id_col_name not in df.columns:
                logger.error(f"Top-K ID column '{id_col_name}' not found in DataFrame.")
                raise ValueError(f"Missing Top-K ID column: {id_col_name}")
            if score_col_name not in df.columns:
                logger.error(f"Top-K Score column '{score_col_name}' not found in DataFrame.")
                raise ValueError(f"Missing Top-K Score column: {score_col_name}")

            label_cols_to_concat.append(pl.col(id_col_name))
            score_cols_to_concat.append(pl.col(score_col_name))

        if label_cols_to_concat:
            df = df.with_columns(top_k_labels=pl.concat_list(label_cols_to_concat))
            logger.info(f"Aggregated Top-K label columns into 'top_k_labels'. Columns used: {[c.meta.output_name() for c in label_cols_to_concat]}")
        if score_cols_to_concat:
            df = df.with_columns(top_k_scores=pl.concat_list(score_cols_to_concat))
            logger.info(f"Aggregated Top-K score columns into 'top_k_scores'. Columns used: {[c.meta.output_name() for c in score_cols_to_concat]}")

        return df

    def _process_rotated_detection_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to rotated detection models (e.g., rbbox)."""
        if not self.data_loader_config.field_mapping.rotated_detection:
            logger.warning("Rotated detection field mapping not found, skipping processing.")
            return df

        mapping = self.data_loader_config.field_mapping.rotated_detection
        # eval_params = self.main_config.evaluation_params.rotated_detection # May not be needed for loading

        # Rename category_id and score
        rename_specific_dict: Dict[str, str] = {}
        if mapping.category_id in df.columns and mapping.category_id != "category_id":
            rename_specific_dict[mapping.category_id] = "category_id"
        if mapping.score in df.columns and mapping.score != "score":
            rename_specific_dict[mapping.score] = "score"
        if rename_specific_dict:
            df = df.rename(rename_specific_dict)

        # Process rotated bounding boxes (rbbox)
        # Expected format: [center_x, center_y, width, height, angle_degrees]
        rbbox_cols = mapping.rbbox
        if not all(col in df.columns for col in rbbox_cols):
            logger.error(f"One or more rbbox columns {rbbox_cols} not found in DataFrame.")
            raise ValueError(f"Missing rbbox columns: {rbbox_cols}")

        for col_name in rbbox_cols:
            if not df[col_name].dtype.is_numeric():
                 df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))

        rbbox_exprs = [pl.col(c) for c in rbbox_cols]
        df = df.with_columns(internal_rbbox=pl.concat_list(rbbox_exprs))
        logger.info("Standardized rbbox columns to 'internal_rbbox' list column.")
        return df

    def _process_pose_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to pose estimation models (e.g., keypoints string)."""
        if not self.data_loader_config.field_mapping.pose:
            logger.warning("Pose field mapping not found, skipping pose field processing.")
            return df

        mapping = self.data_loader_config.field_mapping.pose
        # eval_params = self.main_config.evaluation_params.pose # For keypoint layout, OKS sigma

        # Rename person_score
        if mapping.person_score in df.columns and mapping.person_score != "person_score":
            df = df.rename({mapping.person_score: "person_score"})

        # Process person_bbox (assuming xywh, convert to xyxy for internal use if needed by matching)
        # For now, just concat them similar to detection bbox if needed for matching.
        # Or, it might be used directly if person matching logic expects this format.
        # Let's assume it's fine as separate columns for now, or it's handled by the evaluator.
        # If standardization is needed like 'internal_bbox':
        person_bbox_cols = mapping.person_bbox
        if not all(col in df.columns for col in person_bbox_cols):
            logger.error(f"One or more person_bbox columns {person_bbox_cols} not found.")
            raise ValueError(f"Missing person_bbox columns: {person_bbox_cols}")

        for col_name in person_bbox_cols:
             if not df[col_name].dtype.is_numeric():
                 df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))

        # For consistency, let's create an internal_person_bbox (xyxy) like detection
        # Assuming person_bbox_cols are [x, y, w, h] from config
        x_c, y_c, w, h = [pl.col(c) for c in person_bbox_cols]
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2
        df = df.with_columns(internal_person_bbox=pl.concat_list([x_min, y_min, x_max, y_max]))
        logger.info("Standardized 'person_bbox' (xywh) to 'internal_person_bbox' (xyxy).")


        # Process keypoints string: "x1,y1,c1;x2,y2,c2;..."
        # This will be parsed into a list of lists/tuples of floats: [[x1,y1,c1], [x2,y2,c2], ...]
        kp_col_name = mapping.keypoints
        if kp_col_name not in df.columns:
            logger.error(f"Keypoints column '{kp_col_name}' not found.")
            raise ValueError(f"Missing keypoints column: {kp_col_name}")

        def parse_keypoints_string(kp_str: str) -> List[List[float]]:
            if not kp_str or not isinstance(kp_str, str):
                return []
            try:
                points = []
                parts = kp_str.split(';')
                for part in parts:
                    coords = [float(c) for c in part.split(',')]
                    if len(coords) == 3: # x, y, confidence/visibility
                        points.append(coords)
                    elif len(coords) == 2: # x,y assuming confidence is 1 or handled elsewhere
                        points.append(coords + [1.0]) # Add default confidence if only 2
                return points
            except ValueError:
                logger.warning(f"Could not parse keypoints string: {kp_str[:50]}...") # Log only a part
                return [] # Return empty list on parsing error for this row

        # The return type for apply is tricky with lists of lists for Polars.
        # It's often better to explode and then group if complex ops are needed,
        # or use a struct if fixed number of keypoints.
        # For now, store as Object type, evaluators will handle it.
        df = df.with_columns(
            internal_keypoints=pl.col(kp_col_name).apply(
                parse_keypoints_string, return_dtype=pl.Object
            )
        )
        logger.info(f"Parsed keypoints string column '{kp_col_name}' into 'internal_keypoints'.")
        return df


if __name__ == "__main__":
    from ai_eval_tool.config_manager import load_config

    # Create dummy CSV and config for testing
    dummy_csv_content = """loop_id,img_name,rel_path,t_pre,t_inf,t_post,t_total,det_cat,det_score,x,y,w,h,class_top_1_id,class_top_1_score,class_top_3_id,class_top_3_score
1,img1.jpg,imgs/img1.jpg,10,20,5,35,0,0.9,100,100,50,50,cat,0.95,dog,0.8
1,img2.jpg,imgs/img2.jpg,12,22,6,40,1,0.8,150,150,60,40,dog,0.88,cat,0.7
2,img1.jpg,imgs/img1.jpg,11,21,5,37,0,0.92,102,102,50,50,cat,0.93,dog,0.82
"""
    dummy_csv_path = Path("dummy_data_loader_test.csv")
    with open(dummy_csv_path, 'w') as f:
        f.write(dummy_csv_content)

    # Test Detection
    dummy_det_config_content = """
project_info:
  project_name: "DataLoader Test Detection"
  model_type: "detection"
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
    detection:
      category_id: "det_cat"
      score: "det_score"
      bbox: ["x", "y", "w", "h"]
evaluation_params:
  detection:
    iou_threshold: 0.5
    bbox_format: "xywh"
report_settings: {}
"""
    dummy_det_config_path = Path("dummy_config_loader_det_test.yaml")
    with open(dummy_det_config_path, 'w') as f:
        f.write(dummy_det_config_content)

    try:
        logger.info("--- Testing Detection DataLoader ---")
        config_det = load_config(dummy_det_config_path)
        loader_det = DataLoader(config_det)
        df_det = loader_det.load_data(dummy_csv_path)
        print("Detection DataFrame head:")
        print(df_det.head(2))
        print("Detection DataFrame schema:")
        print(df_det.schema)
        assert "internal_bbox" in df_det.columns
        assert isinstance(df_det["internal_bbox"][0], list)

    except Exception as e:
        logger.error(f"Error during Detection DataLoader test: {e}", exc_info=True)
    finally:
        if dummy_det_config_path.exists(): dummy_det_config_path.unlink()

    # Test Classification
    dummy_cls_config_content = """
project_info:
  project_name: "DataLoader Test Classification"
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
    top_k: [1, 3]
report_settings: {}
"""
    dummy_cls_config_path = Path("dummy_config_loader_cls_test.yaml")
    with open(dummy_cls_config_path, 'w') as f:
        f.write(dummy_cls_config_content)

    try:
        logger.info("--- Testing Classification DataLoader ---")
        config_cls = load_config(dummy_cls_config_path)
        loader_cls = DataLoader(config_cls)
        df_cls = loader_cls.load_data(dummy_csv_path)
        print("Classification DataFrame head:")
        print(df_cls.head(2))
        print("Classification DataFrame schema:")
        print(df_cls.schema)
        assert "top_k_labels" in df_cls.columns
        assert "top_k_scores" in df_cls.columns
        assert isinstance(df_cls["top_k_labels"][0], list)
        assert isinstance(df_cls["top_k_scores"][0], list)
        print(f"Top K labels example: {df_cls['top_k_labels'][0]}")

    except Exception as e:
        logger.error(f"Error during Classification DataLoader test: {e}", exc_info=True)
    finally:
        if dummy_cls_config_path.exists(): dummy_cls_config_path.unlink()
        if dummy_csv_path.exists(): dummy_csv_path.unlink() # Clean up CSV at the end
```
