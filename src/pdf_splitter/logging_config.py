"""
Centralized logging configuration for PDF Splitter.

Usage:
    from pdf_splitter.logging_config import setup_logging, get_logger

    # In CLI entry point:
    setup_logging(verbose=True)

    # In any module:
    logger = get_logger(__name__)
    logger.info("Processing...")
"""

import logging
import sys

# Package-level logger name
PACKAGE_NAME = "src"

# Default format strings
DEFAULT_FORMAT = "[%(levelname)s] %(message)s"
VERBOSE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
VERBOSE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False, level: int | None = None, stream=None) -> None:
    """
    Configure logging for the PDF Splitter application.

    Should be called once at application startup (e.g., in CLI main()).

    Args:
        verbose: If True, use INFO level with detailed format.
                 If False, use WARNING level with concise format.
        level: Override the logging level (e.g., logging.WARNING).
               If None, determined by verbose flag.
        stream: Output stream (defaults to sys.stdout).
    """
    if level is None:
        level = logging.INFO if verbose else logging.WARNING

    if stream is None:
        stream = sys.stdout

    # Select format based on verbosity
    if verbose:
        fmt = VERBOSE_FORMAT
        datefmt = VERBOSE_DATE_FORMAT
    else:
        fmt = DEFAULT_FORMAT
        datefmt = None

    # Create handler for stdout
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure package logger
    pkg_logger = logging.getLogger(PACKAGE_NAME)
    pkg_logger.setLevel(level)

    # Set all relevant loggers to the same level for consistency
    for logger_name in ["docling", "docling_core", "docling_parse"]:
        logging.getLogger(logger_name).setLevel(level)

    if verbose:
        pkg_logger.debug("Verbose logging enabled")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming within the package.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)
