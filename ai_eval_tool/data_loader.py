"""
Data Loader: Handles loading and initial preprocessing of model output data from CSV files.
"""

from pathlib import Path
from typing import Dict, List, Union

import polars as pl

from .config_manager import MainConfig
from .utils.logging_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads data from CSV files using Polars and performs initial transformations
    based on the provided configuration.
    """

    def __init__(
        self, config: MainConfig
    ):  # Takes MainConfig to access model_type for specific parsing
        self.main_config = config
        self.data_loader_config = config.data_loader
        self.model_type = config.project_info.model_type

    def load_data(self, csv_file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Loads the CSV file into a Polars DataFrame and performs initial processing.
        Optimized for memory usage with large datasets.

        Args:
            csv_file_path: Path to the input CSV file.

        Returns:
            A Polars DataFrame with standardized column names and processed fields.
        """
        logger.info(f"Loading data from CSV: {csv_file_path}")

        # Validate file existence and format
        self._validate_csv_file(csv_file_path)

        try:
            # Memory optimization: Use lazy loading for large files
            df = self._load_csv_optimized(csv_file_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_file_path}: {e}")
            raise ValueError(f"CSV file reading failed: {e}") from e

        # Validate basic DataFrame structure
        self._validate_dataframe_structure(df)

        df = self._rename_common_columns(df)

        # Model-specific processing
        try:
            if self.model_type == "detection":
                df = self._process_detection_fields(df)
            elif self.model_type == "classification":
                df = self._process_classification_fields(df)
            elif self.model_type == "rotated_detection":
                df = self._process_rotated_detection_fields(df)
            elif self.model_type == "pose":
                df = self._process_pose_fields(df)
            elif self.model_type == "tracking":
                df = self._process_tracking_fields(df)
            elif self.model_type == "ranking":
                df = self._process_ranking_fields(df)
            else:
                logger.warning(
                    f"Unknown model type '{self.model_type}', skipping model-specific processing."
                )
        except Exception as e:
            logger.error(
                f"Failed to process {self.model_type} specific fields: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Model-specific field processing failed for {self.model_type}: {e}"
            ) from e

        # Validate required columns after all processing
        self._validate_required_columns(df)

        # Final validation
        self._validate_final_dataframe(df)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"DataFrame schema after loading: {df.schema}")
        return df

    def _load_csv_optimized(self, csv_file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Optimized CSV loading for large datasets with memory management.

        Args:
            csv_file_path: Path to the input CSV file.

        Returns:
            A Polars DataFrame loaded with memory optimizations.
        """
        file_path = Path(csv_file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_size_mb > 500:  # For files larger than 500MB
            logger.info(
                f"Large file detected ({file_size_mb:.1f}MB). Using optimized loading strategy."
            )
            # Use lazy loading and streaming for very large files
            lazy_df = pl.scan_csv(
                csv_file_path,
                infer_schema_length=10000,  # Limit schema inference for speed
                try_parse_dates=False,
            )  # Skip date parsing for performance
            # Collect with streaming to reduce memory usage
            df = lazy_df.collect(streaming=True)
        elif file_size_mb > 100:  # For files larger than 100MB
            logger.info(
                f"Medium file detected ({file_size_mb:.1f}MB). Using memory-optimized loading."
            )
            # Use chunked reading with optimized dtypes
            df = pl.read_csv(
                csv_file_path,
                low_memory=True,
                infer_schema_length=5000,  # Faster schema inference
                try_parse_dates=False,  # Skip date parsing
                rechunk=True,
            )  # Optimize memory layout
        elif file_size_mb > 10:  # For medium files 10-100MB
            logger.info(
                f"Medium file detected ({file_size_mb:.1f}MB). Using standard optimized loading."
            )
            df = pl.read_csv(
                csv_file_path,
                infer_schema_length=1000,  # Quick schema inference
                try_parse_dates=False,
            )
        else:
            # Standard loading for smaller files
            df = pl.read_csv(csv_file_path, try_parse_dates=False)

        logger.info(
            f"Loaded CSV with shape {df.shape}, memory usage optimized for {file_size_mb:.1f}MB file"
        )
        return self._optimize_memory_usage(df)

    def _optimize_memory_usage(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types where possible.

        Args:
            df: Input DataFrame

        Returns:
            Memory-optimized DataFrame
        """
        logger.debug("Optimizing DataFrame memory usage...")

        if df.height == 0:
            logger.debug("DataFrame is empty, skipping memory optimization.")
            return df

        # Get memory usage before optimization
        initial_memory = self._estimate_memory_usage(df)

        # Batch process columns for better performance
        optimizations = []

        for col_name in df.columns:
            dtype = df[col_name].dtype

            if dtype == pl.Int64:
                # Use lazy evaluation to check min/max only once
                col_stats = df.select(
                    [
                        pl.col(col_name).min().alias("min_val"),
                        pl.col(col_name).max().alias("max_val"),
                    ]
                ).row(0)

                col_min, col_max = col_stats

                if col_min is not None and col_max is not None:
                    new_dtype = self._get_optimal_int_dtype(col_min, col_max)
                    if new_dtype != pl.Int64:
                        optimizations.append((col_name, new_dtype))

            elif dtype == pl.Float64:
                # Check if we can downcast to Float32 without significant precision loss
                if df[col_name].null_count() < df.height:  # Has non-null values
                    # For most ML metrics, Float32 precision is sufficient
                    optimizations.append((col_name, pl.Float32))

            elif dtype == pl.Utf8:
                # Check if string column can be categorical for memory savings
                unique_ratio = df[col_name].n_unique() / df.height
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimizations.append((col_name, pl.Categorical))

        # Apply all optimizations in batch
        if optimizations:
            cast_exprs = [
                pl.col(col_name).cast(new_dtype)
                for col_name, new_dtype in optimizations
            ]
            df = df.with_columns(cast_exprs)

            for col_name, new_dtype in optimizations:
                logger.debug(f"Optimized {col_name} to {new_dtype}")

        # Get memory usage after optimization
        final_memory = self._estimate_memory_usage(df)
        memory_saved = initial_memory - final_memory

        if memory_saved > 0:
            logger.info(
                f"Memory optimization saved {memory_saved:.2f}MB ({memory_saved/initial_memory*100:.1f}%)"
            )

        logger.debug("Memory optimization completed")
        return df

    def _get_optimal_int_dtype(self, min_val: int, max_val: int) -> pl.DataType:
        """Get the optimal integer data type for the given range."""
        if min_val >= 0 and max_val <= 255:
            return pl.UInt8
        elif min_val >= -128 and max_val <= 127:
            return pl.Int8
        elif min_val >= 0 and max_val <= 65535:
            return pl.UInt16
        elif min_val >= -32768 and max_val <= 32767:
            return pl.Int16
        elif min_val >= 0 and max_val <= 4294967295:
            return pl.UInt32
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return pl.Int32
        else:
            return pl.Int64

    def _estimate_memory_usage(self, df: pl.DataFrame) -> float:
        """Estimate DataFrame memory usage in MB."""
        try:
            # Rough estimation based on data types and row count
            memory_bytes = 0
            for col_name in df.columns:
                dtype = df[col_name].dtype
                if dtype in [pl.Int8, pl.UInt8]:
                    memory_bytes += df.height * 1
                elif dtype in [pl.Int16, pl.UInt16]:
                    memory_bytes += df.height * 2
                elif dtype in [pl.Int32, pl.UInt32, pl.Float32]:
                    memory_bytes += df.height * 4
                elif dtype in [pl.Int64, pl.UInt64, pl.Float64]:
                    memory_bytes += df.height * 8
                elif dtype == pl.Utf8:
                    # Rough estimate for strings
                    avg_str_len = df[col_name].str.len_chars().mean() or 10
                    memory_bytes += df.height * avg_str_len
                else:
                    # Default estimate
                    memory_bytes += df.height * 8

            return memory_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0  # Fallback if estimation fails

    def _validate_csv_file(self, csv_file_path: Union[str, Path]) -> None:
        """
        Validates CSV file existence and basic format.

        Args:
            csv_file_path: Path to the CSV file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(csv_file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() not in [".csv", ".tsv"]:
            logger.warning(
                f"File extension '{file_path.suffix}' is not .csv or .tsv, but will attempt to read as CSV"
            )

        # Check if file is empty
        if file_path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {file_path}")

        logger.debug(f"CSV file validation passed: {file_path}")

    def _validate_dataframe_structure(self, df: pl.DataFrame) -> None:
        """
        Validates basic DataFrame structure after loading.

        Args:
            df: Loaded DataFrame

        Raises:
            ValueError: If DataFrame structure is invalid
        """
        if df.height == 0:
            raise ValueError("CSV file contains no data rows")

        if df.width == 0:
            raise ValueError("CSV file contains no columns")

        # Check for completely empty columns
        empty_cols = [col for col in df.columns if df[col].null_count() == df.height]
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicate_cols = [
                col for col in set(df.columns) if df.columns.count(col) > 1
            ]
            raise ValueError(f"Duplicate column names found: {duplicate_cols}")

        logger.debug(f"DataFrame structure validation passed: {df.shape}")

    def _validate_required_columns(self, df: pl.DataFrame) -> None:
        """
        Validates that required columns are present after renaming.

        Args:
            df: DataFrame after column renaming
        Raises:
            ValueError: If required columns are missing
        """
        required_common_cols = []

        # Always require loop column for stability analysis
        if (
            hasattr(self.data_loader_config.field_mapping, "loop")
            and self.data_loader_config.field_mapping.loop
        ):
            required_common_cols.append("loop")

        # Model-specific required columns
        if self.model_type == "tracking":
            required_common_cols.extend(["frame_id", "object_id_pred"])
        elif self.model_type == "ranking":
            required_common_cols.append("query_id")
        elif self.model_type in [
            "detection",
            "classification",
            "pose",
            "rotated_detection",
        ]:
            required_common_cols.append("image_id")

        missing_cols = [col for col in required_common_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing after renaming: {missing_cols}")

        logger.debug(f"Required columns validation passed: {required_common_cols}")

    def _validate_final_dataframe(self, df: pl.DataFrame) -> None:
        """
        Performs final validation on the processed DataFrame.

        Args:
            df: Final processed DataFrame

        Raises:
            ValueError: If final DataFrame is invalid
        """
        # Check for critical model-specific columns
        model_specific_cols = []

        if self.model_type == "detection":
            model_specific_cols = ["internal_bbox", "category_id", "score"]
        elif self.model_type == "classification":
            model_specific_cols = ["top_k_labels", "top_k_scores"]
        elif self.model_type == "rotated_detection":
            model_specific_cols = ["internal_rbbox", "category_id", "score"]
        elif self.model_type == "pose":
            model_specific_cols = [
                "internal_person_bbox",
                "internal_keypoints",
                "person_score",
            ]
        elif self.model_type == "tracking":
            model_specific_cols = ["internal_bbox_pred", "object_id_pred"]
        elif self.model_type == "ranking":
            model_specific_cols = [
                "internal_item_id_pred_list",
                "internal_item_id_gt_list",
            ]

        missing_model_cols = [
            col for col in model_specific_cols if col not in df.columns
        ]
        if missing_model_cols:
            raise ValueError(f"Model-specific columns missing: {missing_model_cols}")

        # Validate data types for critical numeric columns
        numeric_cols = [
            "total_time_ms",
            "pre_time_ms",
            "inference_time_ms",
            "post_time_ms",
        ]
        for col in numeric_cols:
            if col in df.columns and not df[col].dtype.is_numeric():
                logger.warning(f"Column '{col}' is not numeric type: {df[col].dtype}")

        # Check for reasonable data ranges
        if "total_time_ms" in df.columns:
            negative_times = df.filter(pl.col("total_time_ms") < 0).height
            if negative_times > 0:
                logger.warning(
                    f"Found {negative_times} rows with negative total_time_ms"
                )

        logger.debug("Final DataFrame validation passed")

    def _rename_common_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Renames common columns based on field_mapping."""
        common_mapping = self.data_loader_config.field_mapping.model_dump(
            exclude_none=True
        )

        rename_dict: Dict[str, str] = {}
        drop_cols = []
        for target_name, source_name in common_mapping.items():
            # Skip model-specific mappings here, they are handled in their respective functions
            if target_name in ["detection", "classification", "rotated_detection", "pose", "tracking", "ranking"]:
                continue

            if (
                isinstance(source_name, str)
                and source_name in df.columns
                and target_name != source_name
            ):
                rename_dict[source_name] = target_name
                drop_cols.append(source_name) # Add original column to drop list

        if rename_dict:
            logger.debug(f"Renaming common columns: {rename_dict}")
            df = df.rename(rename_dict)
        
        # Drop the original columns that were renamed
        if drop_cols:
            df = df.drop(drop_cols)
            logger.debug(f"Dropped original common columns: {drop_cols}")
        return df

    def _process_detection_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to detection models (e.g., bbox)."""
        logger.debug(f"_process_detection_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_detection_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.detection:
            logger.warning(
                "Detection field mapping not found in config, skipping detection field processing."
            )
            return df

        mapping = self.data_loader_config.field_mapping.detection
        eval_params = self.main_config.evaluation_params.detection
        if not eval_params:
            logger.error(
                "Detection evaluation parameters missing, cannot process bbox format."
            )
            raise ValueError(
                "Detection evaluation parameters missing for bbox processing."
            )

        # Rename category_id and score
        rename_specific_dict: Dict[str, str] = {}
        if mapping.category_id in df.columns and mapping.category_id != "category_id":
            rename_specific_dict[mapping.category_id] = "category_id"
        if mapping.score in df.columns and mapping.score != "score":
            rename_specific_dict[mapping.score] = "score"
        if rename_specific_dict:
            df = df.rename(rename_specific_dict)
        logger.debug(f"_process_detection_fields: After renaming, DataFrame schema:
{df.schema}")

        # Collect columns to drop after processing
        cols_to_drop = []
        if mapping.category_id in df.columns and mapping.category_id != "category_id":
            cols_to_drop.append(mapping.category_id)
        if mapping.score in df.columns and mapping.score != "score":
            cols_to_drop.append(mapping.score)
        cols_to_drop.extend(mapping.bbox) # Use mapping.bbox directly

        # Process bounding boxes
        bbox_cols = mapping.bbox
        if not all(col in df.columns for col in bbox_cols):
            logger.error(
                f"One or more bbox columns {bbox_cols} not found in DataFrame."
            )
            raise ValueError(f"Missing bbox columns: {bbox_cols}")

        # Ensure bbox columns are numeric
        for col_name in bbox_cols:
            logger.debug(f"_process_detection_fields: Checking bbox column '{col_name}' with dtype {df[col_name].dtype}")
            if not df[col_name].dtype.is_numeric():
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )
                    logger.debug(f"Converted bbox column '{col_name}' to Float64")
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert bbox column '{col_name}' to numeric: {e}"
                    ) from e

            # Check for invalid bbox values
            invalid_count = df.filter(
                pl.col(col_name).is_null() | pl.col(col_name).is_infinite()
            ).height
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} invalid values in bbox column '{col_name}'"
                )

        bbox_exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in bbox_cols]
        logger.debug(f"_process_detection_fields: bbox_exprs created.")

        if eval_params.bbox_format == "xywh":
            # x_center, y_center, width, height -> x_min, y_min, x_max, y_max
            x_c, y_c, w, h = bbox_exprs
            x_min = x_c - w / 2
            y_min = y_c - h / 2
            x_max = x_c + w / 2
            y_max = y_c + h / 2
            df = df.with_columns(
                internal_bbox=pl.concat_list([x_min, y_min, x_max, y_max])
            )
            logger.info(
                "Converted 'xywh' bboxes to standardized 'internal_bbox' (xyxy)."
            )

        elif eval_params.bbox_format == "xyxy":
            # x_min, y_min, x_max, y_max -> just concatenate
            df = df.with_columns(internal_bbox=pl.concat_list(bbox_exprs))
            logger.info("Standardized 'xyxy' bboxes to 'internal_bbox'.")
        else:
            logger.error(f"Unsupported bbox_format: {eval_params.bbox_format}")
            raise ValueError(f"Unsupported bbox_format: {eval_params.bbox_format}")

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original detection columns: {cols_to_drop}")

        logger.debug(f"_process_detection_fields: Final DataFrame schema:
{df.schema}")
        logger.debug(f"_process_detection_fields: Final DataFrame head:
{df.head()}")
        return df

    def _process_classification_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to classification models (e.g., Top-K)."""
        logger.debug(f"_process_classification_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_classification_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.classification:
            logger.warning(
                "Classification field mapping not found, skipping classification field processing."
            )
            return df
        if not self.main_config.evaluation_params.classification:
            logger.error("Classification evaluation parameters (top_k list) missing.")
            raise ValueError("Classification evaluation parameters missing.")

        mapping = self.data_loader_config.field_mapping.classification
        eval_params = self.main_config.evaluation_params.classification

        top_k_values = eval_params.top_k
        id_pattern = mapping.top_k_id_pattern
        score_pattern = mapping.top_k_score_pattern
        logger.debug(f"_process_classification_fields: top_k_values={top_k_values}, id_pattern={id_pattern}, score_pattern={score_pattern}")

        label_cols_to_concat = []
        score_cols_to_concat = []
        cols_to_drop = []

        for k_val in top_k_values:
            id_col_name = id_pattern.format(k=k_val)
            score_col_name = score_pattern.format(k=k_val)
            logger.debug(f"_process_classification_fields: Checking for columns: {id_col_name}, {score_col_name}")

            if id_col_name not in df.columns:
                logger.error(f"Top-K ID column '{id_col_name}' not found in DataFrame.")
                raise ValueError(f"Missing Top-K ID column: {id_col_name}")
            if score_col_name not in df.columns:
                logger.error(
                    f"Top-K Score column '{score_col_name}' not found in DataFrame."
                )
                raise ValueError(f"Missing Top-K Score column: {score_col_name}")

            label_cols_to_concat.append(pl.col(id_col_name))
            score_cols_to_concat.append(pl.col(score_col_name).cast(pl.Float64, strict=False)) # Ensure scores are Float64
            cols_to_drop.extend([id_col_name, score_col_name])

        if label_cols_to_concat:
            df = df.with_columns(top_k_labels=pl.concat_list(label_cols_to_concat))
            logger.info(
                f"Aggregated Top-K label columns into 'top_k_labels'. Columns used: {[c.meta.output_name() for c in label_cols_to_concat]}"
            )
        if score_cols_to_concat:
            df = df.with_columns(top_k_scores=pl.concat_list(score_cols_to_concat))
            logger.info(
                f"Aggregated Top-K score columns into 'top_k_scores'. Columns used: {[c.meta.output_name() for c in score_cols_to_concat]}"
            )

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original classification columns: {cols_to_drop}")

        logger.debug(f"_process_classification_fields: Final DataFrame schema:
{df.schema}")
        logger.debug(f"_process_classification_fields: Final DataFrame head:
{df.head()}")
        return df

    def _process_rotated_detection_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to rotated detection models (e.g., rbbox)."""
        logger.debug(f"_process_rotated_detection_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_rotated_detection_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.rotated_detection:
            logger.warning(
                "Rotated detection field mapping not found, skipping processing."
            )
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
        logger.debug(f"_process_rotated_detection_fields: After renaming, DataFrame schema:
{df.schema}")

        # Collect columns to drop after processing
        cols_to_drop = []
        if mapping.category_id in df.columns and mapping.category_id != "category_id":
            cols_to_drop.append(mapping.category_id)
        if mapping.score in df.columns and mapping.score != "score":
            cols_to_drop.append(mapping.score)
        cols_to_drop.extend(mapping.rbbox) # Use mapping.rbbox directly

        # Process rotated bounding boxes (rbbox)
        # Expected format: [center_x, center_y, width, height, angle_degrees]
        rbbox_cols = mapping.rbbox
        if not all(col in df.columns for col in rbbox_cols):
            logger.error(
                f"One or more rbbox columns {rbbox_cols} not found in DataFrame."
            )
            raise ValueError(f"Missing rbbox columns: {rbbox_cols}")

        for col_name in rbbox_cols:
            logger.debug(f"_process_rotated_detection_fields: Checking rbbox column '{col_name}' with dtype {df[col_name].dtype}")
            if not df[col_name].dtype.is_numeric():
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )
                    logger.debug(f"Converted rbbox column '{col_name}' to Float64")
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert rbbox column '{col_name}' to numeric: {e}"
                    ) from e

            # Check for invalid rbbox values
            invalid_count = df.filter(
                pl.col(col_name).is_null() | pl.col(col_name).is_infinite()
            ).height
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} invalid values in rbbox column '{col_name}'"
                )

        rbbox_exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in rbbox_cols]
        logger.debug(f"_process_rotated_detection_fields: rbbox_exprs created.")
        df = df.with_columns(internal_rbbox=pl.concat_list(rbbox_exprs))
        logger.info("Standardized rbbox columns to 'internal_rbbox' list column.")

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original rotated detection columns: {cols_to_drop}")

        logger.debug(f"""_process_rotated_detection_fields: Final DataFrame schema:
{df.schema}""")
        logger.debug(f"""_process_rotated_detection_fields: Final DataFrame head:
{df.head()}""")
        return df

    def _process_pose_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to pose estimation models (e.g., keypoints string)."""
        logger.debug(f"_process_pose_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_pose_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.pose:
            logger.warning(
                "Pose field mapping not found, skipping pose field processing."
            )
            return df

        mapping = self.data_loader_config.field_mapping.pose
        # eval_params = self.main_config.evaluation_params.pose # For keypoint layout, OKS sigma

        # Rename person_score
        cols_to_drop = []
        if (
            mapping.person_score in df.columns
            and mapping.person_score != "person_score"
        ):
            df = df.rename({mapping.person_score: "person_score"})
            cols_to_drop.append(mapping.person_score)
        logger.debug(f"_process_pose_fields: After renaming, DataFrame schema:
{df.schema}")

        # Process person_bbox (assuming xywh, convert to xyxy for internal use if needed by matching)
        # For now, just concat them similar to detection bbox if needed for matching.
        # Or, it might be used directly if person matching logic expects this format.
        # Let's assume it's fine as separate columns for now, or it's handled by the evaluator.
        # If standardization is needed like 'internal_bbox':
        person_bbox_cols = mapping.person_bbox
        if not all(col in df.columns for col in person_bbox_cols):
            logger.error(
                f"One or more person_bbox columns {person_bbox_cols} not found."
            )
            raise ValueError(f"Missing person_bbox columns: {person_bbox_cols}")

        for col_name in person_bbox_cols:
            logger.debug(f"_process_pose_fields: Checking person_bbox column '{col_name}' with dtype {df[col_name].dtype}")
            if not df[col_name].dtype.is_numeric():
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )
                    logger.debug(
                        f"Converted person_bbox column '{col_name}' to Float64"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert person_bbox column '{col_name}' to numeric: {e}"
                    ) from e
        cols_to_drop.extend(person_bbox_cols)

        # For consistency, let's create an internal_person_bbox (xyxy) like detection
        # Assuming person_bbox_cols are [x, y, w, h] from config
        x_c, y_c, w, h = [pl.col(c).cast(pl.Float64, strict=False) for c in person_bbox_cols] # Ensure numeric
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2
        df = df.with_columns(
            internal_person_bbox=pl.concat_list([x_min, y_min, x_max, y_max])
        )
        logger.info(
            "Standardized 'person_bbox' (xywh) to 'internal_person_bbox' (xyxy)."
        )

        # Process keypoints string: "x1,y1,c1;x2,y2,c2;..."
        # This will be parsed into a list of lists/tuples of floats: [[x1,y1,c1], [x2,y2,c2], ...]
        kp_col_name = mapping.keypoints
        if kp_col_name not in df.columns:
            logger.error(f"Keypoints column '{kp_col_name}' not found.")
            raise ValueError(f"Missing keypoints column: {kp_col_name}")
        cols_to_drop.append(kp_col_name)

        def parse_keypoints_string(kp_str: str) -> List[List[float]]:
            if not kp_str or not isinstance(kp_str, str):
                return []
            try:
                points = []
                parts = kp_str.split(";")
                for part in parts:
                    coords = [float(c) for c in part.split(",")]
                    if len(coords) == 3:  # x, y, confidence/visibility
                        points.append(coords)
                    elif (
                        len(coords) == 2
                    ):  # x,y assuming confidence is 1 or handled elsewhere
                        points.append(
                            coords + [1.0]
                        )  # Add default confidence if only 2
                return points
            except ValueError:
                logger.warning(
                    f"Could not parse keypoints string: {kp_str[:50]}..."
                )  # Log only a part
                return []  # Return empty list on parsing error for this row

        # The return type for apply is tricky with lists of lists for Polars.
        # It's often better to explode and then group if complex ops are needed,
        # or use a struct if fixed number of keypoints.
        # For now, store as Object type, evaluators will handle it.
        df = df.with_columns(
            internal_keypoints=pl.col(kp_col_name).map_elements(
                parse_keypoints_string, return_dtype=pl.Object  # type: ignore
            )
        )
        logger.info(
            f"Parsed keypoints string column '{kp_col_name}' into 'internal_keypoints'."
        )

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original pose columns: {cols_to_drop}")

        logger.debug(f"""_process_pose_fields: Final DataFrame schema:
{df.schema}""")
        logger.debug(f"""_process_pose_fields: Final DataFrame head:
{df.head()}""")
        return df

    def _process_tracking_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to tracking models."""
        logger.debug(f"_process_tracking_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_tracking_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.tracking:
            logger.warning(
                "Tracking field mapping not found, skipping tracking field processing."
            )
            return df

        mapping = self.data_loader_config.field_mapping.tracking
        eval_params = self.main_config.evaluation_params.tracking
        if not eval_params:
            logger.error(
                "Tracking evaluation parameters missing, cannot process bbox formats for tracking."
            )
            raise ValueError(
                "Tracking evaluation parameters missing for bbox processing."
            )

        # Rename common tracking fields (object_id_pred, category_id_pred, score_pred)
        rename_dict: Dict[str, str] = {}
        cols_to_drop = []

        if (
            mapping.object_id_pred in df.columns
            and mapping.object_id_pred != "object_id_pred"
        ):
            rename_dict[mapping.object_id_pred] = "object_id_pred"
            cols_to_drop.append(mapping.object_id_pred)
        if (
            mapping.category_id_pred
            and mapping.category_id_pred in df.columns
            and mapping.category_id_pred != "category_id_pred"
        ):
            rename_dict[mapping.category_id_pred] = "category_id_pred"
            cols_to_drop.append(mapping.category_id_pred)
        if (
            mapping.score_pred
            and mapping.score_pred in df.columns
            and mapping.score_pred != "score_pred"
        ):
            rename_dict[mapping.score_pred] = "score_pred"
            cols_to_drop.append(mapping.score_pred)

        # Rename GT fields if present
        if (
            mapping.object_id_gt
            and mapping.object_id_gt in df.columns
            and mapping.object_id_gt != "object_id_gt"
        ):
            rename_dict[mapping.object_id_gt] = "object_id_gt"
            cols_to_drop.append(mapping.object_id_gt)
        if (
            mapping.category_id_gt
            and mapping.category_id_gt in df.columns
            and mapping.category_id_gt != "category_id_gt"
        ):
            rename_dict[mapping.category_id_gt] = "category_id_gt"
            cols_to_drop.append(mapping.category_id_gt)
        if (
            mapping.visibility_gt
            and mapping.visibility_gt in df.columns
            and mapping.visibility_gt != "visibility_gt"
        ):
            rename_dict[mapping.visibility_gt] = "visibility_gt"
            cols_to_drop.append(mapping.visibility_gt)
        if (
            mapping.ignored_gt
            and mapping.ignored_gt in df.columns
            and mapping.ignored_gt != "ignored_gt"
        ):
            rename_dict[mapping.ignored_gt] = "ignored_gt"
            cols_to_drop.append(mapping.ignored_gt)

        if rename_dict:
            df = df.rename(rename_dict)
        logger.debug(f"_process_tracking_fields: After renaming, DataFrame schema:
{df.schema}")

        # Process predicted bboxes
        bbox_pred_cols = mapping.bbox_pred
        if not all(col in df.columns for col in bbox_pred_cols):
            logger.error(f"One or more bbox_pred columns {bbox_pred_cols} not found.")
            raise ValueError(f"Missing bbox_pred columns: {bbox_pred_cols}")
        for col_name in bbox_pred_cols:  # Ensure numeric
            logger.debug(f"_process_tracking_fields: Checking bbox_pred column '{col_name}' with dtype {df[col_name].dtype}")
            if not df[col_name].dtype.is_numeric():
                df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
        cols_to_drop.extend(bbox_pred_cols)

        pred_exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in bbox_pred_cols]
        logger.debug(f"_process_tracking_fields: pred_exprs created.")
        if eval_params.bbox_pred_format == "xywh":
            x_c, y_c, w, h = pred_exprs
            df = df.with_columns(
                internal_bbox_pred=pl.concat_list(
                    [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]
                )
            )
        elif eval_params.bbox_pred_format == "xyxy":
            df = df.with_columns(internal_bbox_pred=pl.concat_list(pred_exprs))
        else:
            raise ValueError(
                f"Unsupported bbox_pred_format: {eval_params.bbox_pred_format}"
            )
        logger.info("Standardized 'bbox_pred' to 'internal_bbox_pred' (xyxy).")

        # Process GT bboxes (if configured and present)
        if mapping.bbox_gt and all(col in df.columns for col in mapping.bbox_gt):
            bbox_gt_cols = mapping.bbox_gt
            for col_name in bbox_gt_cols:  # Ensure numeric
                logger.debug(f"_process_tracking_fields: Checking bbox_gt column '{col_name}' with dtype {df[col_name].dtype}")
                if not df[col_name].dtype.is_numeric():
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )
            cols_to_drop.extend(bbox_gt_cols)

            gt_exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in bbox_gt_cols] # Ensure numeric
            gt_format = (
                eval_params.bbox_gt_format or "xywh"
            )  # Default to xywh if not specified for GT
            if gt_format == "xywh":
                x_c, y_c, w, h = gt_exprs
                df = df.with_columns(
                    internal_bbox_gt=pl.concat_list(
                        [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]
                    )
                )
            elif gt_format == "xyxy":
                df = df.with_columns(internal_bbox_gt=pl.concat_list(gt_exprs))
            else:
                raise ValueError(f"Unsupported bbox_gt_format: {gt_format}")
            logger.info("Standardized 'bbox_gt' to 'internal_bbox_gt' (xyxy).")
        elif mapping.bbox_gt:  # Configured but not all columns present
            logger.warning(
                f"bbox_gt columns {mapping.bbox_gt} configured but not all found in DataFrame. Skipping GT bbox processing."
            )
            df = df.with_columns(
                internal_bbox_gt=pl.lit(None, dtype=pl.List(pl.Float64))
            )

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original tracking columns: {cols_to_drop}")

        # Ensure frame_id is present (renamed from common mapping if needed)
        if "frame_id" not in df.columns:
            logger.error(
                "'frame_id' column (or its mapping) is required for tracking models but not found."
            )
            raise ValueError("'frame_id' column is essential for tracking.")
        # Ensure object_id_pred is present
        if "object_id_pred" not in df.columns:
            logger.error(
                "'object_id_pred' column (or its mapping) is required for tracking models but not found."
            )
            raise ValueError("'object_id_pred' column is essential for tracking.")
        # Ensure query_id is present for ranking models (if common field mapping used)
        if self.model_type == "ranking" and "query_id" not in df.columns:
            logger.error(
                "'query_id' column (or its mapping) is required for ranking models but not found."
            )
            raise ValueError("'query_id' column is essential for ranking.")

        logger.debug(f"""_process_tracking_fields: Final DataFrame schema:
{df.schema}""")
        logger.debug(f"""_process_tracking_fields: Final DataFrame head:
{df.head()}""")
        return df

    def _process_ranking_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Processes fields specific to ranking/recommendation models."""
        logger.debug(f"_process_ranking_fields: Initial DataFrame schema:
{df.schema}")
        logger.debug(f"_process_ranking_fields: Initial DataFrame head:
{df.head()}")

        if not self.data_loader_config.field_mapping.ranking:
            logger.warning(
                "Ranking field mapping not found, skipping ranking field processing."
            )
            return df

        mapping = self.data_loader_config.field_mapping.ranking
        # eval_params = self.main_config.evaluation_params.ranking # For k_values, not needed for loading

        # Common fields like query_id, loop_id should be handled by _rename_common_columns
        # Ensure query_id is present
        if "query_id" not in df.columns:  # Check for the standardized name
            # Check if original name from mapping exists, if so it means renaming failed or wasn't applied yet.
            # This check should ideally be after all renaming.
            # For now, assume common renaming has run.
            logger.error(
                f"'query_id' column (mapped from '{self.data_loader_config.field_mapping.query_id}') not found after common renaming."
            )
            raise ValueError(
                "Standardized 'query_id' column not found for ranking model."
            )

        list_delimiter = mapping.list_delimiter
        logger.debug(f"_process_ranking_fields: list_delimiter={list_delimiter}")

        # Process item_id_pred_list
        pred_list_col = mapping.item_id_pred_list
        cols_to_drop = []
        if pred_list_col in df.columns:
            # Convert comma-separated string to list of strings/ints
            # Assuming item IDs can be strings or ints. If always int, can cast later.
            # Polars str.split returns a list of strings.
            df = df.with_columns(
                internal_item_id_pred_list=pl.col(pred_list_col).str.split(
                    list_delimiter
                )
                # .list.eval(pl.element().cast(pl.Int64, strict=False))  # Optional: if IDs are numeric
            )
            logger.info(
                f"Processed '{pred_list_col}' into 'internal_item_id_pred_list'."
            )
            cols_to_drop.append(pred_list_col)
        else:
            logger.error(f"Predicted item list column '{pred_list_col}' not found.")
            raise ValueError(f"Missing predicted item list column: {pred_list_col}")

        # Process score_pred_list (optional)
        score_list_col = mapping.score_pred_list
        if score_list_col:
            if score_list_col in df.columns:
                df = df.with_columns(
                    internal_score_pred_list=pl.col(score_list_col)
                    .str.split(list_delimiter)
                    .list.eval(
                        pl.element().cast(pl.Float64, strict=False)
                    )  # Scores are usually float
                )
                logger.info(
                    f"Processed '{score_list_col}' into 'internal_score_pred_list'."
                )
                cols_to_drop.append(score_list_col)
            else:
                logger.warning(
                    f"Predicted score list column '{score_list_col}' configured but not found. Skipping."
                )
                df = df.with_columns(
                    internal_score_pred_list=pl.lit(None, dtype=pl.List(pl.Float64))
                )

        # Process item_id_gt_list
        gt_list_col = mapping.item_id_gt_list
        if gt_list_col in df.columns:
            df = df.with_columns(
                internal_item_id_gt_list=pl.col(gt_list_col).str.split(list_delimiter)
                # .list.eval(pl.element().cast(pl.Int64, strict=False))  # Optional: if IDs are numeric
            )
            logger.info(f"Processed '{gt_list_col}' into 'internal_item_id_gt_list'.")
            cols_to_drop.append(gt_list_col)
        else:
            logger.error(f"Ground truth item list column '{gt_list_col}' not found.")
            raise ValueError(f"Missing ground truth item list column: {gt_list_col}")

        # Drop original columns
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug(f"Dropped original ranking columns: {cols_to_drop}")

        logger.debug(f"""_process_ranking_fields: Final DataFrame schema:
{df.schema}""")
        logger.debug(f"""_process_ranking_fields: Final DataFrame head:
{df.head()}""")
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
    with open(dummy_csv_path, "w") as f:
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
    with open(dummy_det_config_path, "w") as f:
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
        if dummy_det_config_path.exists():
            dummy_det_config_path.unlink()

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
      top_k_id_pattern: "pred_label_top{k}"
      top_k_score_pattern: "pred_score_top{k}"
evaluation_params:
  classification:
    top_k: [1, 3]
report_settings: {}
"""
    dummy_cls_config_path = Path("dummy_config_loader_cls_test.yaml")
    with open(dummy_cls_config_path, "w") as f:
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
        if dummy_cls_config_path.exists():
            dummy_cls_config_path.unlink()
        if dummy_csv_path.exists():
            dummy_csv_path.unlink()  # Clean up CSV at the end