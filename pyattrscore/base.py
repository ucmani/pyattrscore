"""
Base Attribution Model Class

This module defines the abstract base class for all attribution models.
All attribution models must inherit from this class and implement the
calculate_attribution method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta


class TouchpointData(BaseModel):
    """
    Pydantic model for validating touchpoint data input.
    
    Attributes:
        user_id: Unique identifier for the user/customer
        touchpoint_id: Unique identifier for the touchpoint
        channel: Marketing channel (e.g., 'email', 'social', 'search')
        timestamp: When the touchpoint occurred
        conversion: Whether this touchpoint led to a conversion (boolean)
        conversion_value: Value of the conversion (optional)
    """
    user_id: str = Field(..., description="Unique user identifier")
    touchpoint_id: str = Field(..., description="Unique touchpoint identifier")
    channel: str = Field(..., description="Marketing channel name")
    timestamp: datetime = Field(..., description="Touchpoint timestamp")
    conversion: bool = Field(default=False, description="Whether this led to conversion")
    conversion_value: Optional[float] = Field(default=None, description="Conversion value")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is not in the future"""
        if v > datetime.now():
            raise ValueError("Timestamp cannot be in the future")
        return v
    
    @validator('conversion_value')
    def validate_conversion_value(cls, v, values):
        """Ensure conversion_value is provided when conversion is True"""
        if values.get('conversion') and v is None:
            raise ValueError("conversion_value must be provided when conversion is True")
        return v


class AttributionConfig(BaseModel):
    """
    Configuration model for attribution parameters.
    
    Attributes:
        attribution_window_days: Number of days to look back for attributions
        decay_rate: Rate of decay for time-based models (0-1)
        include_non_converting_paths: Whether to include paths without conversions
    """
    attribution_window_days: int = Field(default=30, ge=1, le=365)
    decay_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    include_non_converting_paths: bool = Field(default=False)
    
    @validator('attribution_window_days')
    def validate_window(cls, v):
        """Validate attribution window is reasonable"""
        if v < 1 or v > 365:
            raise ValueError("Attribution window must be between 1 and 365 days")
        return v


class AttributionModel(ABC):
    """
    Abstract base class for all attribution models.
    
    This class defines the common interface that all attribution models must implement.
    It provides validation, logging setup, and common utility methods.
    """
    
    def __init__(self, config: Optional[AttributionConfig] = None):
        """
        Initialize the attribution model.
        
        Args:
            config: Configuration object for the model
        """
        self.config = config or AttributionConfig()
        self.model_name = self.__class__.__name__
        
    def validate_input_data(self, data: pd.DataFrame) -> List[TouchpointData]:
        """
        Validate and convert input DataFrame to TouchpointData objects.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            List of validated TouchpointData objects
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        required_columns = ['user_id', 'touchpoint_id', 'channel', 'timestamp']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add default values for optional columns
        if 'conversion' not in data.columns:
            data['conversion'] = False
        if 'conversion_value' not in data.columns:
            data['conversion_value'] = None
            
        # Convert to TouchpointData objects for validation
        touchpoints = []
        for _, row in data.iterrows():
            try:
                touchpoint = TouchpointData(**row.to_dict())
                touchpoints.append(touchpoint)
            except Exception as e:
                raise ValueError(f"Invalid data in row {row.name}: {str(e)}")
                
        return touchpoints
    
    def filter_by_attribution_window(self, touchpoints: List[TouchpointData]) -> List[TouchpointData]:
        """
        Filter touchpoints based on attribution window.
        
        Args:
            touchpoints: List of touchpoint data
            
        Returns:
            Filtered list of touchpoints within attribution window
        """
        if not touchpoints:
            return touchpoints
            
        # Find the latest conversion timestamp
        conversion_touchpoints = [tp for tp in touchpoints if tp.conversion]
        if not conversion_touchpoints:
            return touchpoints
            
        latest_conversion = max(tp.timestamp for tp in conversion_touchpoints)
        window_start = latest_conversion - timedelta(days=self.config.attribution_window_days)
        
        # Filter touchpoints within window
        filtered_touchpoints = [
            tp for tp in touchpoints 
            if tp.timestamp >= window_start and tp.timestamp <= latest_conversion
        ]
        
        return filtered_touchpoints
    
    def group_by_user_journey(self, touchpoints: List[TouchpointData]) -> Dict[str, List[TouchpointData]]:
        """
        Group touchpoints by user journey.
        
        Args:
            touchpoints: List of touchpoint data
            
        Returns:
            Dictionary mapping user_id to list of touchpoints
        """
        user_journeys = {}
        for touchpoint in touchpoints:
            if touchpoint.user_id not in user_journeys:
                user_journeys[touchpoint.user_id] = []
            user_journeys[touchpoint.user_id].append(touchpoint)
        
        # Sort touchpoints by timestamp for each user
        for user_id in user_journeys:
            user_journeys[user_id].sort(key=lambda x: x.timestamp)
            
        return user_journeys
    
    @abstractmethod
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate attribution scores for the given data.
        
        This method must be implemented by all subclasses.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model and its configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'config': self.config.dict(),
            'description': self.__doc__.strip() if self.__doc__ else "No description available"
        }