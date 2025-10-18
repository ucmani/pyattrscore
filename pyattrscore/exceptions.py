"""
Custom Exceptions for PyAttrScore

This module defines custom exception classes for handling various error
scenarios in the attribution modeling package.
"""


class PyAttrScoreError(Exception):
    """Base exception class for all PyAttrScore errors."""
    
    def __init__(self, message: str, error_code: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class InvalidInputError(PyAttrScoreError):
    """
    Raised when input data is invalid or malformed.
    
    This includes missing required columns, invalid data types,
    or data that doesn't meet validation requirements.
    """
    
    def __init__(self, message: str, invalid_fields: list = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            invalid_fields: List of field names that are invalid
        """
        self.invalid_fields = invalid_fields or []
        error_code = "INVALID_INPUT"
        super().__init__(message, error_code)


class AttributionCalculationError(PyAttrScoreError):
    """
    Raised when attribution calculation fails.
    
    This can occur due to mathematical errors, insufficient data,
    or other issues during the attribution calculation process.
    """
    
    def __init__(self, message: str, model_name: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            model_name: Name of the attribution model that failed
        """
        self.model_name = model_name
        error_code = "ATTRIBUTION_CALCULATION_ERROR"
        super().__init__(message, error_code)


class ConfigurationError(PyAttrScoreError):
    """
    Raised when configuration parameters are invalid.
    
    This includes invalid attribution windows, decay rates,
    or other configuration parameters.
    """
    
    def __init__(self, message: str, config_field: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            config_field: Name of the configuration field that is invalid
        """
        self.config_field = config_field
        error_code = "CONFIGURATION_ERROR"
        super().__init__(message, error_code)


class InsufficientDataError(PyAttrScoreError):
    """
    Raised when there is insufficient data for attribution calculation.
    
    This can occur when there are no touchpoints, no conversions,
    or insufficient data points for the chosen attribution model.
    """
    
    def __init__(self, message: str, required_data_points: int = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            required_data_points: Minimum number of data points required
        """
        self.required_data_points = required_data_points
        error_code = "INSUFFICIENT_DATA"
        super().__init__(message, error_code)


class ModelNotImplementedError(PyAttrScoreError):
    """
    Raised when trying to use a model that hasn't been implemented yet.
    
    This is used for placeholder models or features that are planned
    but not yet available.
    """
    
    def __init__(self, message: str, model_name: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            model_name: Name of the model that is not implemented
        """
        self.model_name = model_name
        error_code = "MODEL_NOT_IMPLEMENTED"
        super().__init__(message, error_code)


class DataValidationError(PyAttrScoreError):
    """
    Raised when data validation fails.
    
    This is a more specific version of InvalidInputError for cases
    where the data structure is correct but the values don't pass
    business logic validation.
    """
    
    def __init__(self, message: str, validation_rule: str = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            validation_rule: Name of the validation rule that failed
        """
        self.validation_rule = validation_rule
        error_code = "DATA_VALIDATION_ERROR"
        super().__init__(message, error_code)


class AttributionWindowError(PyAttrScoreError):
    """
    Raised when attribution window configuration is invalid.
    
    This can occur when the attribution window is too large, too small,
    or conflicts with the available data timeframe.
    """
    
    def __init__(self, message: str, window_days: int = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            window_days: The invalid attribution window in days
        """
        self.window_days = window_days
        error_code = "ATTRIBUTION_WINDOW_ERROR"
        super().__init__(message, error_code)


def handle_exception(func):
    """
    Decorator to handle and log exceptions in attribution methods.
    
    This decorator can be used to wrap attribution calculation methods
    to provide consistent error handling and logging.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PyAttrScoreError:
            # Re-raise PyAttrScore exceptions as-is
            raise
        except ValueError as e:
            # Convert ValueError to InvalidInputError
            raise InvalidInputError(f"Invalid input: {str(e)}")
        except Exception as e:
            # Convert other exceptions to generic PyAttrScoreError
            raise PyAttrScoreError(f"Unexpected error in {func.__name__}: {str(e)}")
    
    return wrapper