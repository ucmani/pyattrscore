"""
PyAttrScore - Python Attribution Modeling Package

A production-grade Python package for calculating marketing attribution scores
using multiple models. Includes validation, logging, error handling, and testing.

Author: Mani Gidijala
Version: 1.0.0
License: MIT
"""

__version__ = "0.0.3"
__author__ = "Mani Gidijala"
__email__ = ""
__license__ = "MIT"

# Import core components
from .base import AttributionModel, TouchpointData, AttributionConfig
from .exceptions import (
    PyAttrScoreError,
    InvalidInputError,
    AttributionCalculationError,
    ConfigurationError,
    InsufficientDataError,
    ModelNotImplementedError,
    DataValidationError,
    AttributionWindowError
)
from .logger import get_logger, configure_logging, PyAttrScoreLogger

# Import attribution models
from .first_touch import FirstTouchAttribution
from .last_touch import LastTouchAttribution
from .linear import LinearAttribution
from .exponential_decay import ExponentialDecayAttribution
from .linear_decay import LinearDecayAttribution
from .u_shaped import UShapedAttribution
from .windowed_first_touch import WindowedFirstTouchAttribution
from .football import FootballAttribution, FootballAttributionConfig, FootballMetrics, ChannelArchetype, FootballRole

# Import utilities
from .utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    filter_by_date_range,
    calculate_time_decay_weights,
    normalize_weights,
    group_touchpoints_by_user,
    identify_conversion_touchpoints,
    calculate_attribution_window_bounds,
    validate_attribution_scores,
    create_attribution_result_dataframe,
    safe_divide,
    calculate_summary_statistics
)

# Define what gets imported with "from pyattrscore import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core components
    "AttributionModel",
    "TouchpointData", 
    "AttributionConfig",
    
    # Exceptions
    "PyAttrScoreError",
    "InvalidInputError",
    "AttributionCalculationError",
    "ConfigurationError",
    "InsufficientDataError",
    "ModelNotImplementedError",
    "DataValidationError",
    "AttributionWindowError",
    
    # Logging
    "get_logger",
    "configure_logging",
    "PyAttrScoreLogger",
    
    # Attribution Models
    "FirstTouchAttribution",
    "LastTouchAttribution",
    "LinearAttribution",
    "ExponentialDecayAttribution",
    "LinearDecayAttribution",
    "UShapedAttribution",
    "WindowedFirstTouchAttribution",
    "FootballAttribution",
    "FootballAttributionConfig",
    "FootballMetrics",
    "ChannelArchetype",
    "FootballRole",
    
    # Utilities
    "validate_dataframe_structure",
    "convert_timestamp_column",
    "sort_by_timestamp",
    "filter_by_date_range",
    "calculate_time_decay_weights",
    "normalize_weights",
    "group_touchpoints_by_user",
    "identify_conversion_touchpoints",
    "calculate_attribution_window_bounds",
    "validate_attribution_scores",
    "create_attribution_result_dataframe",
    "safe_divide",
    "calculate_summary_statistics",
]

# Model registry for easy access
ATTRIBUTION_MODELS = {
    'first_touch': FirstTouchAttribution,
    'last_touch': LastTouchAttribution,
    'linear': LinearAttribution,
    'exponential_decay': ExponentialDecayAttribution,
    'linear_decay': LinearDecayAttribution,
    'u_shaped': UShapedAttribution,
    'windowed_first_touch': WindowedFirstTouchAttribution,
    'football': FootballAttribution,
}

def get_model(model_name: str, config: AttributionConfig = None):
    """
    Get an attribution model instance by name.
    
    Args:
        model_name: Name of the attribution model
        config: Configuration object for the model
        
    Returns:
        Attribution model instance
        
    Raises:
        ModelNotImplementedError: If model name is not recognized
        
    Example:
        >>> from pyattrscore import get_model, AttributionConfig
        >>> config = AttributionConfig(attribution_window_days=30)
        >>> model = get_model('linear', config)
        >>> # Use model.calculate_attribution(data)
    """
    if model_name not in ATTRIBUTION_MODELS:
        available_models = list(ATTRIBUTION_MODELS.keys())
        raise ModelNotImplementedError(
            f"Model '{model_name}' not found. Available models: {available_models}",
            model_name=model_name
        )
    
    model_class = ATTRIBUTION_MODELS[model_name]
    return model_class(config)

def list_models():
    """
    List all available attribution models.
    
    Returns:
        List of available model names
        
    Example:
        >>> from pyattrscore import list_models
        >>> models = list_models()
        >>> print(models)
        ['first_touch', 'last_touch', 'linear', ...]
    """
    return list(ATTRIBUTION_MODELS.keys())

def get_model_info(model_name: str = None):
    """
    Get information about attribution models.
    
    Args:
        model_name: Specific model name, or None for all models
        
    Returns:
        Dictionary with model information
        
    Example:
        >>> from pyattrscore import get_model_info
        >>> info = get_model_info('linear')
        >>> print(info['attribution_logic'])
    """
    if model_name is None:
        # Return info for all models
        return {
            name: model_class().get_model_info() 
            for name, model_class in ATTRIBUTION_MODELS.items()
        }
    else:
        if model_name not in ATTRIBUTION_MODELS:
            available_models = list(ATTRIBUTION_MODELS.keys())
            raise ModelNotImplementedError(
                f"Model '{model_name}' not found. Available models: {available_models}",
                model_name=model_name
            )
        
        model_class = ATTRIBUTION_MODELS[model_name]
        return model_class().get_model_info()

# Package metadata
PACKAGE_INFO = {
    'name': 'PyAttrScore',
    'version': __version__,
    'description': 'Python Attribution Modeling Package',
    'long_description': __doc__,
    'author': __author__,
    'email': __email__,
    'license': __license__,
    'keywords': ['attribution', 'marketing', 'analytics', 'conversion', 'touchpoint'],
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    'python_requires': '>=3.8',
    'install_requires': [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'pydantic>=1.8.0',
        'pyyaml>=5.4.0',
    ],
    'extras_require': {
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
    },
}

# Initialize logging with default configuration
try:
    configure_logging(level="INFO")
    logger = get_logger("pyattrscore")
    logger.info(f"PyAttrScore v{__version__} initialized successfully")
except Exception as e:
    # Fallback to basic logging if configuration fails
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyattrscore").warning(f"Failed to configure logging: {e}")