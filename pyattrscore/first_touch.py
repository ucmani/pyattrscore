"""
First Touch Attribution Model

This module implements the First Touch Attribution model, which assigns
100% of the conversion credit to the first touchpoint in the customer journey.
"""

import pandas as pd
from typing import List, Dict, Any
from .base import AttributionModel, TouchpointData, AttributionConfig
from .exceptions import InsufficientDataError, AttributionCalculationError
from .utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    group_touchpoints_by_user,
    identify_conversion_touchpoints,
    create_attribution_result_dataframe,
    validate_attribution_scores
)
from .logger import get_logger

logger = get_logger(__name__)


class FirstTouchAttribution(AttributionModel):
    """
    First Touch Attribution Model.
    
    This model assigns 100% of the conversion credit to the first touchpoint
    in each user's conversion journey. It's useful for understanding which
    channels are most effective at initiating customer journeys.
    
    Key characteristics:
    - Simple and easy to understand
    - Gives full credit to awareness-building channels
    - May undervalue nurturing and closing channels
    - Works well for short sales cycles
    """
    
    def __init__(self, config: AttributionConfig = None):
        """
        Initialize First Touch Attribution model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(config)
        logger.info("Initialized First Touch Attribution model")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate first touch attribution scores.
        
        This method assigns 100% attribution to the first touchpoint
        in each user's journey that leads to a conversion.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting First Touch Attribution calculation")
        
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
            
            logger.info(f"First Touch Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in First Touch Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate First Touch Attribution: {str(e)}",
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
        Calculate attribution for a single user's journey.
        
        For First Touch Attribution, we assign 100% credit to the first
        touchpoint in the journey, and 0% to all others.
        
        Args:
            touchpoints: List of touchpoints for a single user
            
        Returns:
            List of attribution dictionaries
        """
        if not touchpoints:
            return []
        
        # Sort touchpoints by timestamp (should already be sorted, but ensure)
        sorted_touchpoints = sorted(touchpoints, key=lambda x: x.timestamp)
        
        attributions = []
        
        for i, touchpoint in enumerate(sorted_touchpoints):
            # First touchpoint gets 100% attribution, others get 0%
            score = 1.0 if i == 0 else 0.0
            
            attributions.append({
                'touchpoint': touchpoint,
                'score': score,
                'position': i + 1,
                'is_first': i == 0
            })
        
        logger.debug(f"Calculated attribution for user with {len(touchpoints)} touchpoints")
        
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
        
        # For each user, check that exactly one touchpoint has score = 1.0
        for user_id in result_df['user_id'].unique():
            user_scores = result_df[result_df['user_id'] == user_id]['attribution_score']
            
            # Count touchpoints with non-zero scores
            non_zero_scores = user_scores[user_scores > 0]
            
            if len(non_zero_scores) != 1:
                raise AttributionCalculationError(
                    f"User {user_id} should have exactly one touchpoint with attribution, "
                    f"found {len(non_zero_scores)}"
                )
            
            if not abs(non_zero_scores.iloc[0] - 1.0) < 1e-6:
                raise AttributionCalculationError(
                    f"User {user_id} first touchpoint should have score 1.0, "
                    f"found {non_zero_scores.iloc[0]}"
                )
        
        logger.debug("First Touch Attribution results validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the First Touch Attribution model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Assigns 100% credit to the first touchpoint',
            'use_cases': [
                'Understanding awareness channel effectiveness',
                'Short sales cycles',
                'Top-of-funnel optimization'
            ],
            'advantages': [
                'Simple and intuitive',
                'Good for awareness campaigns',
                'Easy to explain to stakeholders'
            ],
            'limitations': [
                'Ignores nurturing touchpoints',
                'May undervalue closing channels',
                'Not suitable for long sales cycles'
            ],
            'requires_conversions': True,
            'supports_attribution_window': False
        }
        
        base_info.update(model_specific_info)
        return base_info