"""
Exponential Time Decay Attribution Model

This module implements the Exponential Time Decay Attribution model, which
credits interactions based on an exponential decay function where more recent
touchpoints receive higher attribution credit.
"""

import pandas as pd
import math
from typing import List, Dict, Any
from datetime import datetime
from .base import AttributionModel, TouchpointData, AttributionConfig
from .exceptions import InsufficientDataError, AttributionCalculationError, ConfigurationError
from .utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    group_touchpoints_by_user,
    identify_conversion_touchpoints,
    create_attribution_result_dataframe,
    validate_attribution_scores,
    normalize_weights,
    calculate_time_decay_weights
)
from .logger import get_logger

logger = get_logger(__name__)


class ExponentialDecayAttribution(AttributionModel):
    """
    Exponential Time Decay Attribution Model.
    
    This model assigns attribution credit based on an exponential decay function,
    where touchpoints closer to conversion receive exponentially higher credit.
    The decay rate controls how quickly the attribution decreases over time.
    
    Key characteristics:
    - Recent touchpoints receive higher credit
    - Exponential decay function provides smooth weighting
    - Configurable decay rate
    - Respects attribution window settings
    - Good for understanding recency impact
    """
    
    def __init__(self, config: AttributionConfig = None):
        """
        Initialize Exponential Decay Attribution model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(config)
        
        # Validate decay rate
        if not 0 <= self.config.decay_rate <= 1:
            raise ConfigurationError(
                f"Decay rate must be between 0 and 1, got {self.config.decay_rate}",
                config_field="decay_rate"
            )
        
        logger.info(f"Initialized Exponential Decay Attribution model with "
                   f"{self.config.attribution_window_days}-day window and "
                   f"{self.config.decay_rate} decay rate")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exponential decay attribution scores.
        
        This method assigns attribution based on exponential time decay,
        where more recent touchpoints receive higher credit.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting Exponential Decay Attribution calculation")
        
        try:
            # Validate input data structure
            required_columns = ['user_id', 'touchpoint_id', 'channel', 'timestamp']
            validate_dataframe_structure(data, required_columns)
            
            # Convert and sort by timestamp
            data = convert_timestamp_column(data)
            data = sort_by_timestamp(data)
            
            # Validate input data using Pydantic models
            touchpoints = self.validate_input_data(data)
            
            if not touchpoints:
                raise InsufficientDataError("No valid touchpoints found in input data")
            
            # Group touchpoints by user journey
            user_journeys = self.group_by_user_journey(touchpoints)
            
            # Filter to only include users with conversions
            converting_users = self._filter_converting_users(user_journeys)
            
            if not converting_users:
                raise InsufficientDataError("No converting users found in the data")
            
            logger.info(f"Processing {len(converting_users)} converting users")
            
            # Calculate attribution for each user
            all_attributions = []
            
            for user_id, user_touchpoints in converting_users.items():
                user_attribution = self._calculate_user_attribution(user_touchpoints)
                all_attributions.extend(user_attribution)
            
            # Create result DataFrame
            if not all_attributions:
                raise AttributionCalculationError("No attributions calculated")
            
            # Convert back to DataFrame format for result creation
            result_data = []
            attribution_scores = []
            
            for attribution in all_attributions:
                touchpoint_data = attribution['touchpoint'].__dict__
                result_data.append(touchpoint_data)
                attribution_scores.append(attribution['score'])
            
            result_df = pd.DataFrame(result_data)
            
            # Create standardized result
            final_result = create_attribution_result_dataframe(
                result_df, 
                attribution_scores, 
                self.model_name
            )
            
            # Validate results
            self._validate_results(final_result)
            
            logger.info(f"Exponential Decay Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError, ConfigurationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Exponential Decay Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate Exponential Decay Attribution: {str(e)}",
                model_name=self.model_name
            )
    
    def _filter_converting_users(self, user_journeys: Dict[str, List[TouchpointData]]) -> Dict[str, List[TouchpointData]]:
        """
        Filter user journeys to only include those with conversions.
        
        Args:
            user_journeys: Dictionary of user journeys
            
        Returns:
            Dictionary of converting user journeys
        """
        converting_users = {}
        
        for user_id, touchpoints in user_journeys.items():
            # Check if this user has any conversions
            has_conversion = any(tp.conversion for tp in touchpoints)
            
            if has_conversion:
                converting_users[user_id] = touchpoints
        
        logger.debug(f"Filtered to {len(converting_users)} converting users "
                    f"from {len(user_journeys)} total users")
        
        return converting_users
    
    def _calculate_user_attribution(self, touchpoints: List[TouchpointData]) -> List[Dict[str, Any]]:
        """
        Calculate attribution for a single user's journey using exponential decay.
        
        For Exponential Decay Attribution, we:
        1. Filter touchpoints within the attribution window
        2. Calculate exponential decay weights based on time to conversion
        3. Normalize weights to sum to 1.0
        
        Args:
            touchpoints: List of touchpoints for a single user
            
        Returns:
            List of attribution dictionaries
        """
        if not touchpoints:
            return []
        
        # Sort touchpoints by timestamp (should already be sorted, but ensure)
        sorted_touchpoints = sorted(touchpoints, key=lambda x: x.timestamp)
        
        # Find conversion touchpoints to determine attribution window
        conversion_touchpoints = [tp for tp in sorted_touchpoints if tp.conversion]
        
        if not conversion_touchpoints:
            # No conversions found, return all with 0 attribution
            return [{
                'touchpoint': tp,
                'score': 0.0,
                'position': i + 1,
                'in_window': False,
                'decay_weight': 0.0
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Use the latest conversion time as reference
        latest_conversion_time = max(tp.timestamp for tp in conversion_touchpoints)
        
        # Filter touchpoints within attribution window
        windowed_touchpoints = self.filter_by_attribution_window(sorted_touchpoints)
        
        # If no touchpoints in window, return all with 0 attribution
        if not windowed_touchpoints:
            return [{
                'touchpoint': tp,
                'score': 0.0,
                'position': i + 1,
                'in_window': False,
                'decay_weight': 0.0
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Calculate exponential decay weights
        windowed_timestamps = [tp.timestamp for tp in windowed_touchpoints]
        decay_weights = calculate_time_decay_weights(
            windowed_timestamps,
            latest_conversion_time,
            self.config.decay_rate,
            'exponential'
        )
        
        # Normalize weights to sum to 1.0
        normalized_weights = normalize_weights(decay_weights)
        
        # Create attribution results
        attributions = []
        windowed_touchpoint_map = {
            tp.touchpoint_id: (weight, decay_weight) 
            for tp, weight, decay_weight in zip(windowed_touchpoints, normalized_weights, decay_weights)
        }
        
        for i, touchpoint in enumerate(sorted_touchpoints):
            if touchpoint.touchpoint_id in windowed_touchpoint_map:
                score, decay_weight = windowed_touchpoint_map[touchpoint.touchpoint_id]
                in_window = True
            else:
                score = 0.0
                decay_weight = 0.0
                in_window = False
            
            attributions.append({
                'touchpoint': touchpoint,
                'score': score,
                'position': i + 1,
                'in_window': in_window,
                'decay_weight': decay_weight
            })
        
        logger.debug(f"Calculated exponential decay attribution for user with {len(touchpoints)} touchpoints, "
                    f"{len(windowed_touchpoints)} in attribution window")
        
        return attributions
    
    def _validate_results(self, result_df: pd.DataFrame) -> None:
        """
        Validate the attribution results.
        
        Args:
            result_df: DataFrame with attribution results
            
        Raises:
            AttributionCalculationError: If validation fails
        """
        if result_df.empty:
            raise AttributionCalculationError("Attribution results are empty")
        
        # Check that attribution scores are valid
        if 'attribution_score' not in result_df.columns:
            raise AttributionCalculationError("Attribution scores missing from results")
        
        # For each user, check that attribution scores sum to 1.0 (within tolerance)
        tolerance = 1e-6
        
        for user_id in result_df['user_id'].unique():
            user_scores = result_df[result_df['user_id'] == user_id]['attribution_score']
            total_score = user_scores.sum()
            
            # Check if user has any conversions
            user_data = result_df[result_df['user_id'] == user_id]
            has_conversion = user_data['conversion'].any() if 'conversion' in user_data.columns else False
            
            if has_conversion:
                if abs(total_score - 1.0) > tolerance:
                    raise AttributionCalculationError(
                        f"User {user_id} attribution scores sum to {total_score}, expected 1.0"
                    )
            else:
                if total_score > tolerance:
                    raise AttributionCalculationError(
                        f"User {user_id} has no conversions but attribution scores sum to {total_score}"
                    )
        
        logger.debug("Exponential Decay Attribution results validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Exponential Decay Attribution model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Assigns credit based on exponential decay function with recency bias',
            'use_cases': [
                'Understanding recency impact on conversions',
                'Weighting recent touchpoints higher',
                'Time-sensitive attribution analysis'
            ],
            'advantages': [
                'Considers timing of touchpoints',
                'Recent touchpoints get higher credit',
                'Smooth exponential weighting',
                'Configurable decay rate'
            ],
            'limitations': [
                'May undervalue early touchpoints',
                'Requires parameter tuning',
                'Complex to explain to stakeholders'
            ],
            'requires_conversions': True,
            'supports_attribution_window': True,
            'attribution_window_days': self.config.attribution_window_days,
            'decay_rate': self.config.decay_rate,
            'decay_function': 'exponential'
        }
        
        base_info.update(model_specific_info)
        return base_info