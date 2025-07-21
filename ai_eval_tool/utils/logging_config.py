"""
Logging Configuration for the AI Model Evaluator.
Uses Loguru for simple and powerful logging.
"""

import sys
from pathlib import Path

import loguru

# Store a reference to the logger instance to avoid reconfiguring if called multiple times
_logger_configured = False
_logger_instance = None


def setup_logging(
    level: str = "INFO",
    log_dir: Path = Path("log"),  # Default log directory
    run_id: str = None,  # Optional run_id for naming log files
    console: bool = True,
    file: bool = True,
    rotation: str = "10 MB",  # Rotate log files at 10 MB
    retention: str = "5 days",  # Keep logs for 5 days
) -> "loguru.Logger":
    """
    Configures the Loguru logger for the application.

    Args:
        level: Logging level (e.g., "DEBUG", "INFO", "WARNING").
        log_dir: Directory to store log files.
        run_id: Optional unique ID for this run, used in log file naming.
        console: Whether to log to the console.
        file: Whether to log to a file.
        rotation: Log file rotation size or time.
        retention: Log file retention policy.

    Returns:
        The configured Loguru logger instance.
    """
    global _logger_configured, _logger_instance

    # If a logger instance already exists and is configured, return it
    # This might be too simple if dynamic reconfiguration is needed,
    # but for basic setup, it prevents adding multiple handlers.
    if _logger_configured and _logger_instance:
        # To change level of existing logger:
        # _logger_instance.remove() # Remove all handlers
        # _logger_configured = False # Then reconfigure
        # For now, just return if already set up.
        return _logger_instance

    logger = loguru.logger
    logger.remove()  # Remove default handlers, if any

    if console:
        logger.add(
            sys.stderr,
            level=level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            enqueue=True,  # Make it thread-safe
        )

    if file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name_base = "ai_eval_tool"
        if run_id:
            log_file_path = log_dir / f"{log_file_name_base}_{run_id}.log"
        else:
            log_file_path = log_dir / f"{log_file_name_base}.log"

        logger.add(
            log_file_path,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            enqueue=True,  # Make it thread-safe
            encoding="utf-8",
        )

    _logger_instance = logger
    _logger_configured = True
    logger.info(
        f"Logger configured. Level: {level}. Console: {console}. File: {file} (Path: {log_dir if file else 'N/A'})"
    )
    return logger


def get_logger(name: str = None) -> "loguru.Logger":
    """
    Returns a logger instance. If setup_logging hasn't been called,
    it will use Loguru's default configuration or a basic setup.
    It's recommended to call setup_logging once at the application entry point.

    Args:
        name: Optional name for the logger (e.g., __name__ of the calling module).

    Returns:
        A Loguru logger instance.
    """
    global _logger_instance, _logger_configured
    if not _logger_configured or not _logger_instance:
        # Fallback to a default setup if not explicitly configured.
        # This ensures that get_logger() always returns a usable logger.
        # However, it's best to call setup_logging() early in your app.
        # print("Warning: Logger accessed via get_logger() before explicit setup_logging(). Using default/basic config.", file=sys.stderr)
        setup_logging(
            level="DEBUG", console=True, file=False
        )  # Basic console logger if not set up

    if name:
        return _logger_instance.bind(
            name=name
        )  # Not how Loguru typically scopes, but can be used for filtering if needed
    return _logger_instance


# Initialize a default logger when the module is imported,
# but allow setup_logging to reconfigure it.
# This ensures logger is available even if setup_logging is not called first from main.
# However, it's better to call setup_logging explicitly.
if not _logger_configured:
    default_logger = setup_logging(
        level="INFO", console=True, file=False
    )  # Minimal default
    # default_logger.debug("Default logger initialized in logging_config.py. Call setup_logging() for custom config.")


if __name__ == "__main__":
    # Example usage:

    # 1. Basic setup (will use defaults or the one from module import)
    log1 = get_logger(__name__)
    log1.info("This is an info message from log1.")
    log1.debug(
        "This is a debug message from log1 (might not show with default INFO level)."
    )

    # 2. Custom setup
    run_specific_id = "my_test_run_123"
    custom_log_dir = Path("./custom_logs")

    # Reconfigure the logger (or set it up if not done yet)
    # Loguru's logger is global, so this reconfigures the same underlying logger.
    # To avoid issues with multiple handlers, setup_logging now checks _logger_configured.
    # For a clean test, you might want to reset the logger state if possible or test in separate processes.
    # For this example, let's assume it's the first proper setup or we want to override.
    _logger_configured = False  # Force re-configuration for this example
    log2 = setup_logging(
        level="DEBUG",
        log_dir=custom_log_dir,
        run_id=run_specific_id,
        console=True,
        file=True,
    )

    log2.debug(
        f"This is a debug message from log2 (should show now). Log directory: {custom_log_dir.resolve()}"
    )
    log2.info("This is an info message from log2.")
    log2.warning("This is a warning.")
    log2.error("This is an error.")

    # Test another module getting the logger
    another_logger = get_logger("another_module")
    another_logger.info("Message from another_module's logger.")

    print(f"Check for log file in: {custom_log_dir.resolve()}")
    # Clean up dummy log directory if needed for tests
    # import shutil
    # if custom_log_dir.exists():
    #     shutil.rmtree(custom_log_dir)
