"""
Unit tests for First Touch Attribution Model
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pyattrscore import FirstTouchAttribution, AttributionConfig, TouchpointData
from pyattrscore.exceptions import InsufficientDataError, AttributionCalculationError


class TestFirstTouchAttribution:
    """Test cases for First Touch Attribution model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AttributionConfig(attribution_window_days=30)
        self.model = FirstTouchAttribution(self.config)
        
        # Sample data for testing
        self.sample_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1, 10, 0),
                'conversion': False,
                'conversion_value': None
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp2',
                'channel': 'social',
                'timestamp': datetime(2023, 1, 2, 10, 0),
                'conversion': False,
                'conversion_value': None
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp3',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 3, 10, 0),
                'conversion': True,
                'conversion_value': 100.0
            }
        ])
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model_name == 'FirstTouchAttribution'
        assert self.model.config.attribution_window_days == 30
    
    def test_single_user_conversion_journey(self):
        """Test attribution for a single user with conversion."""
        result = self.model.calculate_attribution(self.sample_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'attribution_score' in result.columns
        assert 'model_name' in result.columns
        
        # Check attribution scores
        user1_scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        assert user1_scores == [1.0, 0.0, 0.0]  # First touchpoint gets 100%
        
        # Check total attribution sums to 1
        assert abs(sum(user1_scores) - 1.0) < 1e-6
    
    def test_multiple_users(self):
        """Test attribution for multiple users."""
        multi_user_data = pd.DataFrame([
            # User 1 journey
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user1', 'touchpoint_id': 'tp2', 'channel': 'search', 
             'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 100.0},
            
            # User 2 journey
            {'user_id': 'user2', 'touchpoint_id': 'tp3', 'channel': 'social', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user2', 'touchpoint_id': 'tp4', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 2), 'conversion': False},
            {'user_id': 'user2', 'touchpoint_id': 'tp5', 'channel': 'direct', 
             'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 200.0}
        ])
        
        result = self.model.calculate_attribution(multi_user_data)
        
        # Check each user's attribution
        user1_scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        user2_scores = result[result['user_id'] == 'user2']['attribution_score'].tolist()
        
        assert user1_scores == [1.0, 0.0]  # First touchpoint gets 100%
        assert user2_scores == [1.0, 0.0, 0.0]  # First touchpoint gets 100%
    
    def test_single_touchpoint_journey(self):
        """Test attribution for journey with single touchpoint."""
        single_tp_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'direct',
                'timestamp': datetime(2023, 1, 1),
                'conversion': True,
                'conversion_value': 50.0
            }
        ])
        
        result = self.model.calculate_attribution(single_tp_data)
        
        assert len(result) == 1
        assert result.iloc[0]['attribution_score'] == 1.0
    
    def test_no_conversion_data(self):
        """Test handling of data with no conversions."""
        no_conversion_data = pd.DataFrame([
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user1', 'touchpoint_id': 'tp2', 'channel': 'social', 
             'timestamp': datetime(2023, 1, 2), 'conversion': False}
        ])
        
        with pytest.raises(InsufficientDataError):
            self.model.calculate_attribution(no_conversion_data)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['user_id', 'touchpoint_id', 'channel', 'timestamp'])
        
        with pytest.raises(InsufficientDataError):
            self.model.calculate_attribution(empty_data)
    
    def test_missing_required_columns(self):
        """Test handling of data with missing required columns."""
        incomplete_data = pd.DataFrame([
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email'}
            # Missing timestamp column
        ])
        
        with pytest.raises(Exception):  # Should raise validation error
            self.model.calculate_attribution(incomplete_data)
    
    def test_invalid_timestamp_data(self):
        """Test handling of invalid timestamp data."""
        invalid_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': 'invalid_date',
                'conversion': True
            }
        ])
        
        with pytest.raises(Exception):  # Should raise validation error
            self.model.calculate_attribution(invalid_data)
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'attribution_logic' in info
        assert 'requires_conversions' in info
        assert info['model_name'] == 'FirstTouchAttribution'
        assert info['requires_conversions'] is True
    
    def test_attribution_with_conversion_values(self):
        """Test attribution calculation with conversion values."""
        result = self.model.calculate_attribution(self.sample_data)
        
        # Check that attribution values are calculated
        if 'attribution_value' in result.columns:
            user1_values = result[result['user_id'] == 'user1']['attribution_value'].tolist()
            # First touchpoint should get 100% of conversion value
            assert user1_values[0] == 100.0
            assert user1_values[1] == 0.0
            assert user1_values[2] == 0.0
    
    def test_touchpoint_ordering(self):
        """Test that touchpoints are properly ordered by timestamp."""
        # Create data with timestamps out of order
        unordered_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp3',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 3),
                'conversion': True,
                'conversion_value': 100.0
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1),
                'conversion': False
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp2',
                'channel': 'social',
                'timestamp': datetime(2023, 1, 2),
                'conversion': False
            }
        ])
        
        result = self.model.calculate_attribution(unordered_data)
        
        # The first touchpoint chronologically should get 100% attribution
        first_tp_score = result[result['touchpoint_id'] == 'tp1']['attribution_score'].iloc[0]
        assert first_tp_score == 1.0
    
    def test_multiple_conversions_same_user(self):
        """Test handling of multiple conversions for the same user."""
        multi_conversion_data = pd.DataFrame([
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user1', 'touchpoint_id': 'tp2', 'channel': 'social', 
             'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 50.0},
            {'user_id': 'user1', 'touchpoint_id': 'tp3', 'channel': 'search', 
             'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 75.0}
        ])
        
        result = self.model.calculate_attribution(multi_conversion_data)
        
        # First touchpoint should still get 100% attribution
        scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        assert scores == [1.0, 0.0, 0.0]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with valid config
        valid_config = AttributionConfig(attribution_window_days=30)
        model = FirstTouchAttribution(valid_config)
        assert model.config.attribution_window_days == 30
        
        # Test with default config
        model_default = FirstTouchAttribution()
        assert model_default.config is not None