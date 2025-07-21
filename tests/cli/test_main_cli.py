import subprocess  # For running the CLI as a subprocess
import sys
from pathlib import Path

import yaml

# Define paths relative to the project root (assuming tests are run from root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_CLI_SCRIPT = PROJECT_ROOT / "main_cli.py"


# Helper to run the CLI command
def run_cli_command(args_list):
    command = [sys.executable, str(MAIN_CLI_SCRIPT)] + args_list
    # print(f"Running command: {' '.join(command)}") # For debugging
    return subprocess.run(
        command, capture_output=True, text=True, check=False
    )  # check=False to handle non-zero exits


def test_cli_help_message():
    result = run_cli_command(["--help"])
    assert result.returncode == 0
    assert "AI Model Evaluation Tool" in result.stdout
    assert "--config" in result.stdout
    assert "--csv" in result.stdout


def test_cli_missing_required_args():
    result = run_cli_command([])  # No args
    assert result.returncode != 0  # Should fail
    assert (
        "error: the following arguments are required: --config" in result.stderr.lower()
    )  # argparse error

    result = run_cli_command(["--config", "dummy.yaml"])  # Missing --csv
    assert result.returncode != 0
    assert (
        "error: --csv is required if not using --report-only" in result.stderr.lower()
    )


def test_cli_full_run_detection(
    dummy_detection_config_path: Path,
    dummy_detection_csv_path: Path,
    test_data_dir: Path,
):
    # Ensure the output directory from the config exists and is writable for the test
    config_data = yaml.safe_load(dummy_detection_config_path.read_text())
    report_output_dir_name = Path(
        config_data["report_settings"]["output_dir"]
    ).name  # e.g. "reports_detection"
    report_output_dir = (
        test_data_dir / report_output_dir_name
    )  # This is where reports will go

    # The cache will be under report_output_dir / ".cache" / run_id
    # Logs will be under report_output_dir / run_id / "logs"

    args = [
        "--config",
        str(dummy_detection_config_path),
        "--csv",
        str(dummy_detection_csv_path),
        "--log-level",
        "DEBUG",
    ]
    result = run_cli_command(args)

    # For debugging if it fails:
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert (
        "AI Model Evaluation process completed successfully" in result.stdout
    )  # Check for success message

    # Check for report files (run_id might be generated, so we need to find it or use fixed one)
    run_id = config_data["project_info"]["run_id"]  # "pytest_det_run_001"
    assert (report_output_dir / f"{run_id}_report.md").exists()
    assert (report_output_dir / f"{run_id}_report.html").exists()
    assert (report_output_dir / run_id / "charts" / "latency_distribution.png").exists()
    assert (report_output_dir / run_id / "logs" / f"ai_eval_tool_{run_id}.log").exists()


def test_cli_analyze_only_mode(
    dummy_classification_config_path: Path,
    dummy_classification_csv_path: Path,
    test_data_dir: Path,
):
    config_data = yaml.safe_load(dummy_classification_config_path.read_text())
    report_output_dir_name = Path(config_data["report_settings"]["output_dir"]).name
    report_output_dir = test_data_dir / report_output_dir_name
    run_id = config_data["project_info"]["run_id"]  # "pytest_cls_run_001"

    cache_base_dir = report_output_dir / ".cache"
    expected_run_cache_dir = cache_base_dir / run_id

    args = [
        "--config",
        str(dummy_classification_config_path),
        "--csv",
        str(dummy_classification_csv_path),
        "--analyze-only",
    ]
    result = run_cli_command(args)

    if result.returncode != 0:
        print("STDOUT (analyze-only):", result.stdout)
        print("STDERR (analyze-only):", result.stderr)

    assert result.returncode == 0
    assert "Analysis phase completed. Results cached." in result.stdout
    assert (expected_run_cache_dir / "results.json").exists()
    assert (
        expected_run_cache_dir / "extra_data_deduplicated_perf_df_for_charts.parquet"
    ).exists()
    # Classification stability details df should also be cached
    assert (
        expected_run_cache_dir
        / "extra_data_classification_stability_details_df.parquet"
    ).exists()


def test_cli_report_only_mode(
    dummy_classification_config_path: Path,
    dummy_classification_csv_path: Path,
    test_data_dir: Path,
):
    # First, run analyze-only to create cache
    config_data = yaml.safe_load(dummy_classification_config_path.read_text())
    report_output_dir_name = Path(config_data["report_settings"]["output_dir"]).name
    report_output_dir = test_data_dir / report_output_dir_name
    run_id = config_data["project_info"]["run_id"]

    cache_base_dir = report_output_dir / ".cache"
    run_cache_dir_path_str = str(
        cache_base_dir / run_id
    )  # Path to the specific run's cache

    analyze_args = [
        "--config",
        str(dummy_classification_config_path),
        "--csv",
        str(dummy_classification_csv_path),
        "--analyze-only",
    ]
    analyze_result = run_cli_command(analyze_args)
    assert (
        analyze_result.returncode == 0
    ), f"Analyze-only pre-step failed: {analyze_result.stderr}"

    # Now run report-only using the created cache
    report_args = [
        "--config",
        str(
            dummy_classification_config_path
        ),  # Config still needed for report settings
        "--report-only",
        "--cache-path",
        run_cache_dir_path_str,  # Provide path to the run_id specific cache dir
    ]
    report_result = run_cli_command(report_args)

    if report_result.returncode != 0:
        print("STDOUT (report-only):", report_result.stdout)
        print("STDERR (report-only):", report_result.stderr)

    assert report_result.returncode == 0
    assert "Report phase completed using cached results." in report_result.stdout
    assert (report_output_dir / f"{run_id}_report.html").exists()


def test_cli_report_only_missing_cache(
    dummy_classification_config_path: Path, test_data_dir: Path
):
    # Try to run report-only with a cache path that doesn't exist
    non_existent_cache_path = str(
        test_data_dir / ".cache" / "non_existent_run_id_for_report"
    )

    args = [
        "--config",
        str(dummy_classification_config_path),
        "--report-only",
        "--cache-path",
        non_existent_cache_path,
    ]
    result = run_cli_command(args)
    assert result.returncode != 0
    assert "Cache for Run ID non_existent_run_id_for_report not found" in result.stderr
