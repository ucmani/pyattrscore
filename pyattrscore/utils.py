"""
Utility Functions for PyAttrScore

This module provides common utility functions used across the attribution
modeling package, including data processing, validation, and mathematical
operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from .exceptions import InvalidInputError, DataValidationError
from .logger import get_logger

logger = get_logger(__name__)


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that a DataFrame has the required columns and structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        InvalidInputError: If DataFrame is invalid or missing required columns
    """
    if df is None:
        raise InvalidInputError("DataFrame cannot be None")
    
    if df.empty:
        from .exceptions import InsufficientDataError
        raise InsufficientDataError("DataFrame cannot be empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidInputError(
            f"Missing required columns: {missing_columns}",
            invalid_fields=missing_columns
        )
    
    logger.debug(f"DataFrame validation passed for columns: {required_columns}")


def convert_timestamp_column(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Convert timestamp column to datetime if it's not already.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of the timestamp column
        
    Returns:
        DataFrame with converted timestamp column
        
    Raises:
        InvalidInputError: If timestamp conversion fails
    """
    if timestamp_col not in df.columns:
        raise InvalidInputError(f"Timestamp column '{timestamp_col}' not found")
    
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            logger.debug(f"Converted {timestamp_col} column to datetime")
        
        return df
    except Exception as e:
        raise InvalidInputError(f"Failed to convert timestamp column: {str(e)}")


def sort_by_timestamp(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Sort DataFrame by timestamp column.
    
    Args:
        df: DataFrame to sort
        timestamp_col: Name of the timestamp column
        
    Returns:
        Sorted DataFrame
    """
    return df.sort_values(by=timestamp_col).reset_index(drop=True)


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame to filter
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)
        timestamp_col: Name of the timestamp column
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if start_date is not None:
        filtered_df = filtered_df[filtered_df[timestamp_col] >= start_date]
    
    if end_date is not None:
        filtered_df = filtered_df[filtered_df[timestamp_col] <= end_date]
    
    logger.debug(f"Filtered DataFrame from {len(df)} to {len(filtered_df)} rows")
    return filtered_df


def calculate_time_decay_weights(
    timestamps: List[datetime],
    reference_time: datetime,
    decay_rate: float = 0.5,
    decay_type: str = 'exponential'
) -> List[float]:
    """
    Calculate time decay weights for touchpoints.
    
    Args:
        timestamps: List of touchpoint timestamps
        reference_time: Reference time (usually conversion time)
        decay_rate: Decay rate (0-1)
        decay_type: Type of decay ('exponential' or 'linear')
        
    Returns:
        List of decay weights
        
    Raises:
        DataValidationError: If parameters are invalid
    """
    if not 0 <= decay_rate <= 1:
        raise DataValidationError("Decay rate must be between 0 and 1")
    
    if decay_type not in ['exponential', 'linear']:
        raise DataValidationError("Decay type must be 'exponential' or 'linear'")
    
    weights = []
    max_time_diff = 0
    
    # Calculate time differences in hours
    time_diffs = []
    for ts in timestamps:
        diff_hours = (reference_time - ts).total_seconds() / 3600
        time_diffs.append(max(0, diff_hours))  # Ensure non-negative
        max_time_diff = max(max_time_diff, diff_hours)
    
    # Calculate weights based on decay type
    for time_diff in time_diffs:
        if decay_type == 'exponential':
            # Exponential decay: weight = decay_rate^(time_diff/max_time_diff)
            if max_time_diff > 0:
                normalized_time = time_diff / max_time_diff
                weight = decay_rate ** normalized_time
            else:
                weight = 1.0
        else:  # linear decay
            # Linear decay: weight = 1 - (1-decay_rate) * (time_diff/max_time_diff)
            if max_time_diff > 0:
                normalized_time = time_diff / max_time_diff
                weight = 1 - (1 - decay_rate) * normalized_time
            else:
                weight = 1.0
        
        weights.append(weight)
    
    logger.debug(f"Calculated {decay_type} decay weights with rate {decay_rate}")
    return weights


def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize weights so they sum to 1.0.
    
    Args:
        weights: List of weights to normalize
        
    Returns:
        List of normalized weights
        
    Raises:
        DataValidationError: If weights are invalid
    """
    if not weights:
        raise DataValidationError("Weights list cannot be empty")
    
    if any(w < 0 for w in weights):
        raise DataValidationError("Weights cannot be negative")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise DataValidationError("Total weight cannot be zero")
    
    normalized = [w / total_weight for w in weights]
    logger.debug(f"Normalized {len(weights)} weights (sum: {sum(normalized):.6f})")
    
    return normalized


def group_touchpoints_by_user(df: pd.DataFrame, user_col: str = 'user_id') -> Dict[str, pd.DataFrame]:
    """
    Group touchpoints by user ID.
    
    Args:
        df: DataFrame with touchpoint data
        user_col: Name of the user ID column
        
    Returns:
        Dictionary mapping user ID to DataFrame of touchpoints
    """
    if user_col not in df.columns:
        raise InvalidInputError(f"User column '{user_col}' not found")
    
    grouped = {}
    for user_id, group_df in df.groupby(user_col):
        grouped[str(user_id)] = group_df.reset_index(drop=True)
    
    logger.debug(f"Grouped touchpoints for {len(grouped)} users")
    return grouped


def identify_conversion_touchpoints(df: pd.DataFrame, conversion_col: str = 'conversion') -> pd.DataFrame:
    """
    Identify touchpoints that led to conversions.
    
    Args:
        df: DataFrame with touchpoint data
        conversion_col: Name of the conversion column
        
    Returns:
        DataFrame containing only conversion touchpoints
    """
    if conversion_col not in df.columns:
        # If no conversion column, assume no conversions
        logger.warning(f"Conversion column '{conversion_col}' not found, assuming no conversions")
        return df.iloc[0:0].copy()  # Return empty DataFrame with same structure
    
    conversions = df[df[conversion_col] == True].copy()
    logger.debug(f"Found {len(conversions)} conversion touchpoints")
    
    return conversions


def calculate_attribution_window_bounds(
    conversion_time: datetime,
    window_days: int
) -> Tuple[datetime, datetime]:
    """
    Calculate the start and end bounds for an attribution window.
    
    Args:
        conversion_time: Time of conversion
        window_days: Attribution window in days
        
    Returns:
        Tuple of (window_start, window_end)
    """
    window_start = conversion_time - timedelta(days=window_days)
    window_end = conversion_time
    
    return window_start, window_end


def validate_attribution_scores(scores: List[float], tolerance: float = 1e-6) -> bool:
    """
    Validate that attribution scores sum to approximately 1.0.
    
    Args:
        scores: List of attribution scores
        tolerance: Tolerance for sum validation
        
    Returns:
        True if scores are valid, False otherwise
    """
    if not scores:
        return False
    
    total_score = sum(scores)
    is_valid = abs(total_score - 1.0) <= tolerance
    
    if not is_valid:
        logger.warning(f"Attribution scores sum to {total_score:.6f}, expected 1.0")
    
    return is_valid


def create_attribution_result_dataframe(
    touchpoints: pd.DataFrame,
    attribution_scores: List[float],
    model_name: str
) -> pd.DataFrame:
    """
    Create a standardized attribution result DataFrame.
    
    Args:
        touchpoints: Original touchpoint data
        attribution_scores: Calculated attribution scores
        model_name: Name of the attribution model used
        
    Returns:
        DataFrame with attribution results
        
    Raises:
        DataValidationError: If inputs don't match
    """
    if len(touchpoints) != len(attribution_scores):
        raise DataValidationError(
            f"Number of touchpoints ({len(touchpoints)}) doesn't match "
            f"number of attribution scores ({len(attribution_scores)})"
        )
    
    result_df = touchpoints.copy()
    result_df['attribution_score'] = attribution_scores
    result_df['model_name'] = model_name
    result_df['attribution_percentage'] = [score * 100 for score in attribution_scores]
    
    # Add attribution value if conversion_value exists
    if 'conversion_value' in result_df.columns:
        # For each user, distribute the conversion value based on attribution scores
        result_df['attribution_value'] = 0.0
        for user_id in result_df['user_id'].unique():
            user_mask = result_df['user_id'] == user_id
            user_data = result_df[user_mask]
            
            # Get the total conversion value for this user
            total_conversion_value = user_data['conversion_value'].fillna(0).sum()
            
            # Distribute the conversion value based on attribution scores
            if total_conversion_value > 0:
                result_df.loc[user_mask, 'attribution_value'] = (
                    result_df.loc[user_mask, 'attribution_score'] * total_conversion_value
                )
    
    logger.debug(f"Created attribution result DataFrame with {len(result_df)} rows")
    return result_df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_summary_statistics(attribution_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for attribution results.
    
    Args:
        attribution_df: DataFrame with attribution results
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_touchpoints': len(attribution_df),
        'unique_users': attribution_df['user_id'].nunique() if 'user_id' in attribution_df.columns else 0,
        'unique_channels': attribution_df['channel'].nunique() if 'channel' in attribution_df.columns else 0,
        'total_conversions': attribution_df['conversion'].sum() if 'conversion' in attribution_df.columns else 0,
        'total_attribution_score': attribution_df['attribution_score'].sum() if 'attribution_score' in attribution_df.columns else 0,
    }
    
    # Add channel-level statistics
    if 'channel' in attribution_df.columns and 'attribution_score' in attribution_df.columns:
        channel_stats = attribution_df.groupby('channel')['attribution_score'].agg(['sum', 'mean', 'count']).to_dict()
        stats['channel_attribution'] = channel_stats
    
    # Add conversion value statistics if available
    if 'attribution_value' in attribution_df.columns:
        stats['total_attribution_value'] = attribution_df['attribution_value'].sum()
        stats['average_attribution_value'] = attribution_df['attribution_value'].mean()
    
    logger.debug("Calculated attribution summary statistics")
    return stats