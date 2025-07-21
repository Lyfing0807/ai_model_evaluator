"""
Main command-line interface for the AI Model Evaluator.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ai_eval_tool.config_manager import MainConfig, load_config
from ai_eval_tool.data_loader import DataLoader
from ai_eval_tool.engine import EvaluationEngine
from ai_eval_tool.reporting.generator import ReportGenerator
from ai_eval_tool.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def execute_analysis_phase(
    config: MainConfig,
    csv_path: str,
    run_id: str,
    cache_base_path: Path,
    engine: EvaluationEngine,
):
    """Handles data loading and evaluation, then serializes results."""
    logger.info(f"Executing ANALYSIS phase for Run ID: {run_id}")
    data_loader = DataLoader(config)
    data_df = data_loader.load_data(csv_path)
    logger.info(f"Data loaded. Shape: {data_df.shape}")

    evaluation_results = engine.run(data_df)
    logger.info(
        f"Evaluation engine finished. Metrics: {list(evaluation_results.metrics.keys())}"
    )

    engine.serialize_results(evaluation_results, cache_base_path, run_id)
    logger.info(
        f"Analysis results for Run ID {run_id} serialized to {cache_base_path / run_id}"
    )


def execute_report_phase(
    config: MainConfig, run_id: str, cache_base_path: Path, engine: EvaluationEngine
):
    """Handles deserializing results and generating reports."""
    logger.info(f"Executing REPORT phase for Run ID: {run_id}")

    try:
        evaluation_results = engine.deserialize_results(cache_base_path, run_id)
        logger.info(
            f"Successfully deserialized results for Run ID {run_id}. Metrics: {list(evaluation_results.metrics.keys())}"
        )
    except FileNotFoundError:
        logger.error(
            f"Cache for Run ID {run_id} not found at {cache_base_path / run_id}. Cannot generate report."
        )
        logger.error(
            "Please run the analysis phase first (e.g. without --report-only, or with --analyze-only)."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Failed to deserialize results for Run ID {run_id}: {e}", exc_info=True
        )
        sys.exit(1)

    report_generator = ReportGenerator(config)
    report_generator.generate(evaluation_results, run_id)
    logger.info(
        f"Report generation finished for Run ID {run_id}. Reports in: {config.report_settings.output_dir.resolve()}"
    )


def main_flow(args: argparse.Namespace):
    """Orchestrates the main evaluation flow based on CLI arguments."""

    # 1. Load Configuration - needed for all modes to some extent
    logger.info(f"Loading configuration from: {args.config}")
    config: MainConfig = load_config(args.config)
    logger.info(
        f"Project: {config.project_info.project_name}, Model Type: {config.project_info.model_type}"
    )

    # Determine Run ID
    # For analyze_only or full run, generate if not in config.
    # For report_only, run_id MUST come from cache_path or be explicitly given and match a cache.
    run_id: str
    if args.report_only and args.cache_path:
        # Infer run_id from the last component of cache_path if it's a directory
        # Assumes cache_path points to the <run_id> directory within the base cache.
        run_id = Path(args.cache_path).name
        logger.info(f"Using Run ID '{run_id}' from --cache-path for report-only mode.")
    elif config.project_info.run_id:
        run_id = config.project_info.run_id
        logger.info(f"Using Run ID '{run_id}' from configuration file.")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(
            f"Generated Run ID: {run_id} (as it was not in config or inferred for report-only)."
        )

    # Update project_info.run_id in the config object if it was generated, for consistency
    if not config.project_info.run_id:
        config.project_info.run_id = run_id

    # Re-setup logging with final run_id for consistent log file naming
    if hasattr(sys.modules["ai_eval_tool.utils.logging_config"], "_logger_configured"):
        sys.modules["ai_eval_tool.utils.logging_config"]._logger_configured = False
    setup_logging(
        level=args.log_level.upper(),
        log_dir=Path(config.report_settings.output_dir) / run_id / "logs",
        run_id=run_id,
        console=True,
        file=True,
    )
    logger.info(
        f"Logger re-initialized. Run ID: {run_id}. Log files in: {Path(config.report_settings.output_dir) / run_id / 'logs'}"
    )

    engine = EvaluationEngine(config)

    # Determine cache base path. This will be <output_dir_from_config>/.cache/
    # And specific run data will be under <output_dir_from_config>/.cache/<run_id>/
    cache_base_directory = Path(config.report_settings.output_dir) / ".cache"

    if args.analyze_only:
        if not args.csv:
            logger.error("--csv is required for --analyze-only mode.")
            sys.exit(1)
        execute_analysis_phase(config, args.csv, run_id, cache_base_directory, engine)
        logger.info("Analysis phase completed. Results cached.")
    elif args.report_only:
        if (
            not args.cache_path
        ):  # cache_path should point to the specific run_id directory
            logger.error(
                "--cache-path (pointing to the run_id cache directory) is required for --report-only mode."
            )
            sys.exit(1)
        # The run_id was already inferred from cache_path. The cache_base_directory is its parent.
        run_specific_cache_path = Path(args.cache_path)
        if run_specific_cache_path.name != run_id:
            logger.warning(
                f"Run ID inferred from cache path ({run_id}) does not match its directory name ({run_specific_cache_path.name}). Using inferred: {run_id}"
            )

        # For report_only, cache_base_path should be the parent of the run_id directory given in --cache-path
        cache_base_for_report_only = run_specific_cache_path.parent
        execute_report_phase(config, run_id, cache_base_for_report_only, engine)
        logger.info("Report phase completed using cached results.")
    else:  # Full run (analyze and report)
        if not args.csv:
            logger.error("--csv is required for a full run.")
            sys.exit(1)
        execute_analysis_phase(config, args.csv, run_id, cache_base_directory, engine)
        execute_report_phase(
            config, run_id, cache_base_directory, engine
        )  # Report from the analysis just done
        logger.info("Full analysis and report generation completed.")


def main():
    parser = argparse.ArgumentParser(description="AI Model Evaluation Tool")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the input CSV data file. Required unless --report-only.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only the analysis phase and save intermediate results. Output cache will be in <output_dir_from_config>/.cache/<run_id>/.",
    )
    group.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from cached intermediate results. Requires --cache-path.",
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        help="Path to the specific run_id cache directory (e.g., <output_dir>/.cache/<your_run_id>). Required for --report-only. If provided with --analyze-only, this specific path is not directly used, but implies a cache base inside the report's output_dir.",
    )

    args = parser.parse_args()

    if args.report_only and not args.cache_path:
        parser.error("--cache-path is required when using --report-only.")
    if (
        args.analyze_only or not args.report_only
    ) and not args.csv:  # Full run also needs CSV
        parser.error("--csv is required if not using --report-only.")

    # Initial minimal logger setup. Will be reconfigured once run_id is known.
    if hasattr(sys.modules["ai_eval_tool.utils.logging_config"], "_logger_configured"):
        sys.modules["ai_eval_tool.utils.logging_config"]._logger_configured = False
    setup_logging(level=args.log_level.upper(), console=True, file=False)
    logger.info(
        f"Starting AI Model Evaluator. Config: {args.config}, CSV: {args.csv or 'N/A'}"
    )

    try:
        main_flow(args)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration, data, or argument error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
