"""
Unit tests for Linear Attribution Model
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pyattrscore import LinearAttribution, AttributionConfig
from pyattrscore.exceptions import InsufficientDataError, AttributionCalculationError


class TestLinearAttribution:
    """Test cases for Linear Attribution model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AttributionConfig(attribution_window_days=30)
        self.model = LinearAttribution(self.config)
        
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
        assert self.model.model_name == 'LinearAttribution'
        assert self.model.config.attribution_window_days == 30
    
    def test_equal_attribution_distribution(self):
        """Test that attribution is distributed equally among all touchpoints."""
        result = self.model.calculate_attribution(self.sample_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'attribution_score' in result.columns
        
        # Check attribution scores - should be equal (1/3 each)
        user1_scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        expected_score = 1.0 / 3.0
        
        for score in user1_scores:
            assert abs(score - expected_score) < 1e-6
        
        # Check total attribution sums to 1
        assert abs(sum(user1_scores) - 1.0) < 1e-6
    
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
    
    def test_two_touchpoint_journey(self):
        """Test attribution for journey with two touchpoints."""
        two_tp_data = pd.DataFrame([
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
                'channel': 'search',
                'timestamp': datetime(2023, 1, 2),
                'conversion': True,
                'conversion_value': 100.0
            }
        ])
        
        result = self.model.calculate_attribution(two_tp_data)
        
        # Each touchpoint should get 50%
        scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        assert len(scores) == 2
        assert abs(scores[0] - 0.5) < 1e-6
        assert abs(scores[1] - 0.5) < 1e-6
    
    def test_attribution_window_filtering(self):
        """Test that attribution window is properly applied."""
        # Create data with touchpoints outside attribution window
        window_config = AttributionConfig(attribution_window_days=5)
        model = LinearAttribution(window_config)
        
        window_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1),  # 10 days before conversion
                'conversion': False
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp2',
                'channel': 'social',
                'timestamp': datetime(2023, 1, 8),  # 3 days before conversion
                'conversion': False
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp3',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 11),  # Conversion day
                'conversion': True,
                'conversion_value': 100.0
            }
        ])
        
        result = model.calculate_attribution(window_data)
        
        # Only touchpoints within 5-day window should get attribution
        scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        
        # tp1 should get 0 (outside window), tp2 and tp3 should get 0.5 each
        assert abs(scores[0] - 0.0) < 1e-6  # tp1 outside window
        assert abs(scores[1] - 0.5) < 1e-6  # tp2 in window
        assert abs(scores[2] - 0.5) < 1e-6  # tp3 in window
    
    def test_multiple_users(self):
        """Test attribution for multiple users."""
        multi_user_data = pd.DataFrame([
            # User 1 journey (3 touchpoints)
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user1', 'touchpoint_id': 'tp2', 'channel': 'social', 
             'timestamp': datetime(2023, 1, 2), 'conversion': False},
            {'user_id': 'user1', 'touchpoint_id': 'tp3', 'channel': 'search', 
             'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 150.0},
            
            # User 2 journey (2 touchpoints)
            {'user_id': 'user2', 'touchpoint_id': 'tp4', 'channel': 'direct', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user2', 'touchpoint_id': 'tp5', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 200.0}
        ])
        
        result = self.model.calculate_attribution(multi_user_data)
        
        # Check each user's attribution
        user1_scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        user2_scores = result[result['user_id'] == 'user2']['attribution_score'].tolist()
        
        # User 1: 3 touchpoints, each gets 1/3
        expected_user1 = 1.0 / 3.0
        for score in user1_scores:
            assert abs(score - expected_user1) < 1e-6
        
        # User 2: 2 touchpoints, each gets 1/2
        expected_user2 = 1.0 / 2.0
        for score in user2_scores:
            assert abs(score - expected_user2) < 1e-6
    
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
    
    def test_attribution_with_conversion_values(self):
        """Test attribution calculation with conversion values."""
        result = self.model.calculate_attribution(self.sample_data)
        
        # Check that attribution values are calculated
        if 'attribution_value' in result.columns:
            user1_values = result[result['user_id'] == 'user1']['attribution_value'].tolist()
            expected_value = 100.0 / 3.0  # Total value divided equally
            
            for value in user1_values:
                assert abs(value - expected_value) < 1e-6
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'attribution_logic' in info
        assert 'supports_attribution_window' in info
        assert info['model_name'] == 'LinearAttribution'
        assert info['supports_attribution_window'] is True
        assert info['attribution_window_days'] == 30
    
    def test_empty_attribution_window(self):
        """Test handling when no touchpoints fall within attribution window."""
        # Set very short attribution window
        short_window_config = AttributionConfig(attribution_window_days=1)
        model = LinearAttribution(short_window_config)
        
        # Create data where touchpoints are outside the window
        outside_window_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1),  # 5 days before conversion
                'conversion': False
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp2',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 6),  # Conversion day
                'conversion': True,
                'conversion_value': 100.0
            }
        ])
        
        result = model.calculate_attribution(outside_window_data)
        
        # Only the conversion touchpoint should be in window and get attribution
        scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        assert abs(scores[0] - 0.0) < 1e-6  # Outside window
        assert abs(scores[1] - 1.0) < 1e-6  # Conversion touchpoint gets all credit
    
    def test_large_number_of_touchpoints(self):
        """Test attribution with many touchpoints."""
        num_touchpoints = 10
        
        # Create data with many touchpoints
        large_data = []
        for i in range(num_touchpoints):
            large_data.append({
                'user_id': 'user1',
                'touchpoint_id': f'tp{i+1}',
                'channel': f'channel{i+1}',
                'timestamp': datetime(2023, 1, 1) + timedelta(days=i),
                'conversion': i == num_touchpoints - 1,  # Last touchpoint is conversion
                'conversion_value': 1000.0 if i == num_touchpoints - 1 else None
            })
        
        large_df = pd.DataFrame(large_data)
        result = self.model.calculate_attribution(large_df)
        
        # Each touchpoint should get equal attribution
        scores = result[result['user_id'] == 'user1']['attribution_score'].tolist()
        expected_score = 1.0 / num_touchpoints
        
        for score in scores:
            assert abs(score - expected_score) < 1e-6
        
        # Total should sum to 1
        assert abs(sum(scores) - 1.0) < 1e-6