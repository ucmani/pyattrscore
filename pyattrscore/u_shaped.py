"""
U-Shaped Attribution Model

This module implements the U-Shaped Attribution model, which assigns
40% credit to the first touchpoint, 40% to the last touchpoint, and
distributes the remaining 20% equally among middle touchpoints.
"""

import pandas as pd
from typing import List, Dict, Any
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
    normalize_weights
)
from .logger import get_logger

logger = get_logger(__name__)


class UShapedAttribution(AttributionModel):
    """
    U-Shaped Attribution Model.
    
    This model assigns attribution credit in a U-shaped pattern:
    - 40% to the first touchpoint (awareness)
    - 40% to the last touchpoint (conversion)
    - 20% distributed equally among middle touchpoints (nurturing)
    
    Key characteristics:
    - Balances first and last touch importance
    - Recognizes middle touchpoints
    - Good for longer customer journeys
    - Configurable first/last touch weights
    """
    
    def __init__(self, config: AttributionConfig = None, first_touch_weight: float = 0.4, last_touch_weight: float = 0.4):
        """
        Initialize U-Shaped Attribution model.
        
        Args:
            config: Configuration object for the model
            first_touch_weight: Weight for first touchpoint (default 0.4)
            last_touch_weight: Weight for last touchpoint (default 0.4)
        """
        super().__init__(config)
        
        self.first_touch_weight = first_touch_weight
        self.last_touch_weight = last_touch_weight
        self.middle_touch_weight = 1.0 - first_touch_weight - last_touch_weight
        
        # Validate weights
        if not 0 <= first_touch_weight <= 1:
            raise ConfigurationError(
                f"First touch weight must be between 0 and 1, got {first_touch_weight}",
                config_field="first_touch_weight"
            )
        
        if not 0 <= last_touch_weight <= 1:
            raise ConfigurationError(
                f"Last touch weight must be between 0 and 1, got {last_touch_weight}",
                config_field="last_touch_weight"
            )
        
        if first_touch_weight + last_touch_weight > 1:
            raise ConfigurationError(
                f"First touch weight ({first_touch_weight}) + Last touch weight ({last_touch_weight}) "
                f"cannot exceed 1.0",
                config_field="u_shaped_weights"
            )
        
        if self.middle_touch_weight < 0:
            raise ConfigurationError(
                f"Middle touch weight cannot be negative. "
                f"First ({first_touch_weight}) + Last ({last_touch_weight}) weights exceed 1.0",
                config_field="u_shaped_weights"
            )
        
        logger.info(f"Initialized U-Shaped Attribution model with weights: "
                   f"First={first_touch_weight}, Last={last_touch_weight}, "
                   f"Middle={self.middle_touch_weight}")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate U-shaped attribution scores.
        
        This method assigns attribution in a U-shaped pattern with
        higher weights for first and last touchpoints.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting U-Shaped Attribution calculation")
        
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
            
            logger.info(f"U-Shaped Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError, ConfigurationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in U-Shaped Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate U-Shaped Attribution: {str(e)}",
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
        Calculate attribution for a single user's journey using U-shaped model.
        
        For U-Shaped Attribution, we:
        1. Assign first_touch_weight to the first touchpoint
        2. Assign last_touch_weight to the last touchpoint before conversion
        3. Distribute middle_touch_weight equally among middle touchpoints
        
        Args:
            touchpoints: List of touchpoints for a single user
            
        Returns:
            List of attribution dictionaries
        """
        if not touchpoints:
            return []
        
        # Sort touchpoints by timestamp (should already be sorted, but ensure)
        sorted_touchpoints = sorted(touchpoints, key=lambda x: x.timestamp)
        
        # Find conversion touchpoints to determine the journey end
        conversion_touchpoints = [tp for tp in sorted_touchpoints if tp.conversion]
        
        if not conversion_touchpoints:
            # No conversions found, return all with 0 attribution
            return [{
                'touchpoint': tp,
                'score': 0.0,
                'position': i + 1,
                'touchpoint_type': 'no_conversion'
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Find the latest conversion time
        latest_conversion_time = max(tp.timestamp for tp in conversion_touchpoints)
        
        # Find all touchpoints up to and including the conversion time
        journey_touchpoints = [
            tp for tp in sorted_touchpoints 
            if tp.timestamp <= latest_conversion_time
        ]
        
        if not journey_touchpoints:
            # This shouldn't happen, but handle gracefully
            return [{
                'touchpoint': tp,
                'score': 0.0,
                'position': i + 1,
                'touchpoint_type': 'no_conversion'
            } for i, tp in enumerate(sorted_touchpoints)]
        
        num_journey_touchpoints = len(journey_touchpoints)
        
        # Calculate attribution based on journey length
        attributions = []
        
        for i, touchpoint in enumerate(sorted_touchpoints):
            # Check if this touchpoint is part of the conversion journey
            if touchpoint.timestamp > latest_conversion_time:
                # Touchpoint after conversion gets 0 attribution
                score = 0.0
                touchpoint_type = 'post_conversion'
            elif num_journey_touchpoints == 1:
                # Single touchpoint gets 100% attribution
                score = 1.0
                touchpoint_type = 'single'
            elif num_journey_touchpoints == 2:
                # Two touchpoints: split between first and last weights
                journey_index = next(
                    j for j, jtp in enumerate(journey_touchpoints) 
                    if jtp.touchpoint_id == touchpoint.touchpoint_id
                )
                if journey_index == 0:
                    score = self.first_touch_weight + self.middle_touch_weight / 2
                    touchpoint_type = 'first_of_two'
                else:
                    score = self.last_touch_weight + self.middle_touch_weight / 2
                    touchpoint_type = 'last_of_two'
            else:
                # Three or more touchpoints: U-shaped distribution
                try:
                    journey_index = next(
                        j for j, jtp in enumerate(journey_touchpoints) 
                        if jtp.touchpoint_id == touchpoint.touchpoint_id
                    )
                    
                    if journey_index == 0:
                        # First touchpoint
                        score = self.first_touch_weight
                        touchpoint_type = 'first'
                    elif journey_index == num_journey_touchpoints - 1:
                        # Last touchpoint
                        score = self.last_touch_weight
                        touchpoint_type = 'last'
                    else:
                        # Middle touchpoint
                        num_middle_touchpoints = num_journey_touchpoints - 2
                        score = self.middle_touch_weight / num_middle_touchpoints if num_middle_touchpoints > 0 else 0.0
                        touchpoint_type = 'middle'
                except StopIteration:
                    # Touchpoint not in journey (shouldn't happen)
                    score = 0.0
                    touchpoint_type = 'not_in_journey'
            
            attributions.append({
                'touchpoint': touchpoint,
                'score': score,
                'position': i + 1,
                'touchpoint_type': touchpoint_type
            })
        
        logger.debug(f"Calculated U-shaped attribution for user with {len(touchpoints)} touchpoints, "
                    f"{num_journey_touchpoints} in conversion journey")
        
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
        
        logger.debug("U-Shaped Attribution results validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the U-Shaped Attribution model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Assigns higher credit to first and last touchpoints, distributes remainder to middle',
            'use_cases': [
                'Balancing awareness and conversion touchpoints',
                'Multi-touch customer journeys',
                'Understanding nurturing touchpoint value'
            ],
            'advantages': [
                'Balances first and last touch importance',
                'Recognizes middle touchpoints',
                'Good for longer customer journeys',
                'Intuitive U-shaped distribution'
            ],
            'limitations': [
                'Fixed weight distribution',
                'May not suit all business models',
                'Arbitrary middle touchpoint weighting'
            ],
            'requires_conversions': True,
            'supports_attribution_window': False,
            'first_touch_weight': self.first_touch_weight,
            'last_touch_weight': self.last_touch_weight,
            'middle_touch_weight': self.middle_touch_weight
        }
        
        base_info.update(model_specific_info)
        return base_info