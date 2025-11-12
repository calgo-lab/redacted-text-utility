from loguru import logger
from pathlib import Path
from typing import Optional

import sys


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

_is_configured = False


def configure_logging(log_level: str = "INFO",
                      json_format: bool = False,
                      log_to_file: bool = False,
                      file_prefix: str = "log") -> None:
    """
    Configure application-wide logging with loguru.
    
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_to_file: Whether to log to files
    :param json_format: Whether to use JSON format (for production)
    :param file_prefix: Prefix for log files
    :return: None
    """
    global _is_configured
    
    if _is_configured:
        logger.info("Logging setup is already configured, skipping ...")
        return
    
    logger.remove()
    
    if json_format:
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,
            backtrace=True,
            diagnose=True
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    if log_to_file:
        if json_format:
            logger.add(LOGS_DIR / f"{file_prefix}_{{time:YYYY-MM-DD}}.json",
                       rotation="00:00",
                       retention="30 days",
                       level=log_level,
                       backtrace=True,
                       diagnose=True,
                       serialize=True,
                       enqueue=True)
        else:
            logger.add(LOGS_DIR / f"{file_prefix}_{{time:YYYY-MM-DD}}.log",
                   rotation="00:00",
                   retention="30 days",
                   level=log_level,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                   backtrace=True,
                   diagnose=True,
                   enqueue=True)

    
    _is_configured = True

    logger.info(
        "Logging setup is completed",
        level=log_level,
        to_file=log_to_file,
        json_format=json_format
    )

def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with optional context binding.
    
    :param name: Module name (usually __name__)
    :return: Logger instance
    """
    if name:
        return logger.bind(module=name)
    return logger


def log_exception(message: str, **kwargs):
    """
    Log an exception with full traceback and context.
    Automatically captures class name, function name, line number, and local variables.
    
    :param message: Custom error message
    :param kwargs: Additional context to log
    :return: None
    """
    logger.opt(exception=True).error(message, **kwargs)