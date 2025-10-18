"""
Windowed First Touch Attribution Model

This module implements the Windowed First Touch Attribution model, which assigns
100% of the conversion credit to the first touchpoint that occurred within
a defined attribution window before conversion.
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
from .base import AttributionModel, TouchpointData, AttributionConfig
from .exceptions import InsufficientDataError, AttributionCalculationError
from .utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    group_touchpoints_by_user,
    identify_conversion_touchpoints,
    create_attribution_result_dataframe,
    validate_attribution_scores,
    calculate_attribution_window_bounds
)
from .logger import get_logger

logger = get_logger(__name__)


class WindowedFirstTouchAttribution(AttributionModel):
    """
    Windowed First Touch Attribution Model.
    
    This model assigns 100% of the conversion credit to the first touchpoint
    that occurred within the attribution window before conversion. Unlike
    regular First Touch Attribution, this model only considers touchpoints
    within a specified time window.
    
    Key characteristics:
    - Focuses on first touch within attribution window
    - Ignores touchpoints outside the window
    - Good for understanding recent awareness drivers
    - Respects attribution window settings
    """
    
    def __init__(self, config: AttributionConfig = None):
        """
        Initialize Windowed First Touch Attribution model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(config)
        logger.info(f"Initialized Windowed First Touch Attribution model with "
                   f"{self.config.attribution_window_days}-day window")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate windowed first touch attribution scores.
        
        This method assigns 100% attribution to the first touchpoint
        within the attribution window for each user's conversion journey.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting Windowed First Touch Attribution calculation")
        
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
            
            logger.info(f"Windowed First Touch Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Windowed First Touch Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate Windowed First Touch Attribution: {str(e)}",
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
        Calculate attribution for a single user's journey using windowed first touch.
        
        For Windowed First Touch Attribution, we:
        1. Find the conversion time
        2. Filter touchpoints within the attribution window
        3. Assign 100% credit to the first touchpoint in the window
        
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
                'is_first_in_window': False
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Use the latest conversion time as reference
        latest_conversion_time = max(tp.timestamp for tp in conversion_touchpoints)
        
        # Calculate attribution window bounds
        window_start, window_end = calculate_attribution_window_bounds(
            latest_conversion_time,
            self.config.attribution_window_days
        )
        
        # Filter touchpoints within attribution window
        windowed_touchpoints = [
            tp for tp in sorted_touchpoints
            if window_start <= tp.timestamp <= window_end
        ]
        
        # If no touchpoints in window, return all with 0 attribution
        if not windowed_touchpoints:
            logger.debug(f"No touchpoints found within {self.config.attribution_window_days}-day window")
            return [{
                'touchpoint': tp,
                'score': 0.0,
                'position': i + 1,
                'in_window': False,
                'is_first_in_window': False
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Find the first touchpoint in the window
        first_windowed_touchpoint = min(windowed_touchpoints, key=lambda x: x.timestamp)
        
        # Create attribution results
        attributions = []
        windowed_touchpoint_ids = {tp.touchpoint_id for tp in windowed_touchpoints}
        
        for i, touchpoint in enumerate(sorted_touchpoints):
            in_window = touchpoint.touchpoint_id in windowed_touchpoint_ids
            is_first_in_window = (
                in_window and 
                touchpoint.touchpoint_id == first_windowed_touchpoint.touchpoint_id
            )
            
            # First touchpoint in window gets 100% attribution, others get 0%
            score = 1.0 if is_first_in_window else 0.0
            
            attributions.append({
                'touchpoint': touchpoint,
                'score': score,
                'position': i + 1,
                'in_window': in_window,
                'is_first_in_window': is_first_in_window
            })
        
        logger.debug(f"Calculated windowed first touch attribution for user with {len(touchpoints)} touchpoints, "
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
        
        # For each user, check that at most one touchpoint has score = 1.0
        for user_id in result_df['user_id'].unique():
            user_scores = result_df[result_df['user_id'] == user_id]['attribution_score']
            
            # Count touchpoints with non-zero scores
            non_zero_scores = user_scores[user_scores > 0]
            
            # Check if user has any conversions
            user_data = result_df[result_df['user_id'] == user_id]
            has_conversion = user_data['conversion'].any() if 'conversion' in user_data.columns else False
            
            if has_conversion:
                if len(non_zero_scores) > 1:
                    raise AttributionCalculationError(
                        f"User {user_id} should have at most one touchpoint with attribution, "
                        f"found {len(non_zero_scores)}"
                    )
                
                if len(non_zero_scores) == 1:
                    if not abs(non_zero_scores.iloc[0] - 1.0) < 1e-6:
                        raise AttributionCalculationError(
                            f"User {user_id} first windowed touchpoint should have score 1.0, "
                            f"found {non_zero_scores.iloc[0]}"
                        )
                
                # Total should be 0 or 1
                total_score = user_scores.sum()
                if not (abs(total_score) < 1e-6 or abs(total_score - 1.0) < 1e-6):
                    raise AttributionCalculationError(
                        f"User {user_id} attribution scores sum to {total_score}, expected 0 or 1.0"
                    )
            else:
                # Non-converting users should have 0 attribution
                total_score = user_scores.sum()
                if total_score > 1e-6:
                    raise AttributionCalculationError(
                        f"User {user_id} has no conversions but attribution scores sum to {total_score}"
                    )
        
        logger.debug("Windowed First Touch Attribution results validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Windowed First Touch Attribution model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Assigns 100% credit to the first touchpoint within attribution window',
            'use_cases': [
                'Understanding recent awareness drivers',
                'Focusing on relevant touchpoints',
                'Time-bounded first touch analysis'
            ],
            'advantages': [
                'Focuses on relevant time period',
                'Simple and intuitive',
                'Good for recent awareness campaigns',
                'Respects attribution window'
            ],
            'limitations': [
                'Ignores touchpoints outside window',
                'May miss important early touchpoints',
                'Single touchpoint gets all credit'
            ],
            'requires_conversions': True,
            'supports_attribution_window': True,
            'attribution_window_days': self.config.attribution_window_days
        }
        
        base_info.update(model_specific_info)
        return base_info