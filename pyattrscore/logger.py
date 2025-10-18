"""
Logger Configuration for PyAttrScore

This module provides centralized logging configuration for the PyAttrScore package.
It supports different logging levels and can be configured for various environments.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class PyAttrScoreLogger:
    """
    Centralized logger for PyAttrScore package.
    
    This class provides a consistent logging interface across all modules
    in the package with configurable log levels and output formats.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str = "pyattrscore") -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Name of the logger (typically module name)
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """
        Create a new logger instance with default configuration.
        
        Args:
            name: Name of the logger
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Avoid adding multiple handlers if logger already exists
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        
        return logger
    
    @classmethod
    def configure_logging(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        include_timestamp: bool = True,
        json_format: bool = False
    ) -> None:
        """
        Configure logging for the entire package.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path to write logs to
            format_string: Custom format string for log messages
            include_timestamp: Whether to include timestamp in logs
            json_format: Whether to use JSON format for structured logging
        """
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Clear existing handlers
        for logger_name in cls._loggers:
            logger = cls._loggers[logger_name]
            logger.handlers.clear()
        
        # Create formatters
        if json_format:
            formatter = JsonFormatter()
        else:
            if format_string is None:
                if include_timestamp:
                    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                else:
                    format_string = '%(name)s - %(levelname)s - %(message)s'
            
            formatter = logging.Formatter(
                format_string,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        
        # Create file handler if specified
        file_handler = None
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
        
        # Configure all existing loggers
        for logger_name in cls._loggers:
            logger = cls._loggers[logger_name]
            logger.setLevel(numeric_level)
            logger.addHandler(console_handler)
            
            if file_handler:
                logger.addHandler(file_handler)
        
        # Set default configuration for new loggers
        cls._default_level = numeric_level
        cls._default_handlers = [console_handler]
        if file_handler:
            cls._default_handlers.append(file_handler)
        
        cls._configured = True
        
        # Log configuration
        logger = cls.get_logger("pyattrscore.logger")
        logger.info(f"Logging configured - Level: {level}, File: {log_file}")
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """
        Set logging level for all loggers.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger_name in cls._loggers:
            logger = cls._loggers[logger_name]
            logger.setLevel(numeric_level)
            
            # Update handler levels
            for handler in logger.handlers:
                handler.setLevel(numeric_level)
    
    @classmethod
    def add_file_handler(cls, log_file: str, level: str = "INFO") -> None:
        """
        Add file handler to all existing loggers.
        
        Args:
            log_file: Path to log file
            level: Logging level for file handler
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        for logger_name in cls._loggers:
            logger = cls._loggers[logger_name]
            logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects, which is useful
    for log aggregation and analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def get_logger(name: str = None) -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Name of the logger (defaults to calling module)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'pyattrscore')
    
    return PyAttrScoreLogger.get_logger(name)


def configure_logging(**kwargs) -> None:
    """
    Convenience function to configure logging.
    
    Args:
        **kwargs: Keyword arguments passed to PyAttrScoreLogger.configure_logging
    """
    PyAttrScoreLogger.configure_logging(**kwargs)


# Initialize default logger
logger = get_logger("pyattrscore")