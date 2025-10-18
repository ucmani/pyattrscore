"""
Linear Attribution Model

This module implements the Linear Attribution model, which distributes
conversion credit equally among all touchpoints within the attribution window.
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
    validate_attribution_scores,
    normalize_weights
)
from .logger import get_logger

logger = get_logger(__name__)


class LinearAttribution(AttributionModel):
    """
    Linear Attribution Model.
    
    This model distributes conversion credit equally among all touchpoints
    within the attribution window. Each touchpoint receives an equal share
    of the attribution credit.
    
    Key characteristics:
    - Fair distribution across all touchpoints
    - Considers the entire customer journey
    - Good for understanding overall channel contribution
    - Respects attribution window settings
    """
    
    def __init__(self, config: AttributionConfig = None):
        """
        Initialize Linear Attribution model.
        
        Args:
            config: Configuration object for the model
        """
        super().__init__(config)
        logger.info(f"Initialized Linear Attribution model with {self.config.attribution_window_days}-day window")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate linear attribution scores.
        
        This method distributes attribution equally among all touchpoints
        within the attribution window for each user's conversion journey.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting Linear Attribution calculation")
        
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
            
            logger.info(f"Linear Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Linear Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate Linear Attribution: {str(e)}",
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
        
        For Linear Attribution, we:
        1. Filter touchpoints within the attribution window
        2. Distribute credit equally among all touchpoints in the window
        
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
                'in_window': False
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
                'in_window': False
            } for i, tp in enumerate(sorted_touchpoints)]
        
        # Calculate equal attribution for touchpoints in window
        num_touchpoints_in_window = len(windowed_touchpoints)
        equal_attribution = 1.0 / num_touchpoints_in_window
        
        # Create attribution results
        attributions = []
        windowed_touchpoint_ids = {tp.touchpoint_id for tp in windowed_touchpoints}
        
        for i, touchpoint in enumerate(sorted_touchpoints):
            in_window = touchpoint.touchpoint_id in windowed_touchpoint_ids
            score = equal_attribution if in_window else 0.0
            
            attributions.append({
                'touchpoint': touchpoint,
                'score': score,
                'position': i + 1,
                'in_window': in_window
            })
        
        logger.debug(f"Calculated linear attribution for user with {len(touchpoints)} touchpoints, "
                    f"{num_touchpoints_in_window} in attribution window")
        
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
        
        logger.debug("Linear Attribution results validation passed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Linear Attribution model.
        
        Returns:
            Dictionary with model information
        """
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Distributes credit equally among all touchpoints within attribution window',
            'use_cases': [
                'Understanding overall channel contribution',
                'Balanced view of customer journey',
                'Multi-touch attribution analysis'
            ],
            'advantages': [
                'Fair distribution across all touchpoints',
                'Considers entire customer journey',
                'Good baseline for comparison',
                'Respects attribution window'
            ],
            'limitations': [
                'May overvalue less important touchpoints',
                'Doesn\'t consider touchpoint timing',
                'All touchpoints treated equally'
            ],
            'requires_conversions': True,
            'supports_attribution_window': True,
            'attribution_window_days': self.config.attribution_window_days
        }
        
        base_info.update(model_specific_info)
        return base_info