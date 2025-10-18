"""
Test cases for Football-Inspired Attribution Model

This module contains comprehensive tests for the FootballAttribution model,
including role assignment, CIS calculation, and football metrics validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from pyattrscore.football import (
    FootballAttribution, 
    FootballAttributionConfig, 
    FootballMetrics,
    FootballRole,
    ChannelArchetype
)
from pyattrscore.exceptions import (
    InsufficientDataError, 
    AttributionCalculationError,
    ConfigurationError
)


class TestFootballAttributionConfig:
    """Test cases for FootballAttributionConfig"""
    
    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = FootballAttributionConfig()
        
        assert config.scorer_weight == 0.25
        assert config.assister_weight == 0.20
        assert config.key_passer_weight == 0.15
        assert config.baseline_weight == 0.1
        assert config.cold_lead_threshold_days == 7
        assert 'organic_search' in config.channel_archetypes
    
    def test_custom_config_creation(self):
        """Test creating config with custom values"""
        config = FootballAttributionConfig(
            scorer_weight=0.3,
            assister_weight=0.25,
            key_passer_weight=0.2,
            most_passes_weight=0.1,
            most_minutes_weight=0.05,
            most_dribbles_weight=0.05,
            participant_weight=0.05,
            baseline_weight=0.15,
            cold_lead_threshold_days=14
        )
        
        assert config.scorer_weight == 0.3
        assert config.baseline_weight == 0.15
        assert config.cold_lead_threshold_days == 14
    
    def test_invalid_weights_sum(self):
        """Test that invalid weight sums raise ConfigurationError"""
        with pytest.raises(ConfigurationError):
            FootballAttributionConfig(
                scorer_weight=0.5,  # This will make total > 1.0
                assister_weight=0.5,
                key_passer_weight=0.2,
                most_passes_weight=0.1,
                most_minutes_weight=0.1,
                most_dribbles_weight=0.1,
                participant_weight=0.1
            )


class TestFootballAttribution:
    """Test cases for FootballAttribution model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample touchpoint data for testing"""
        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1', 'user2', 'user2'],
            'touchpoint_id': ['tp1', 'tp2', 'tp3', 'tp4', 'tp5'],
            'channel': ['organic_search', 'paid_search', 'direct', 'social_media', 'email'],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 11, 0),
                datetime(2024, 1, 3, 12, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 2, 15, 0)
            ],
            'conversion': [False, False, True, False, True],
            'conversion_value': [None, None, 100.0, None, 50.0],
            'engagement_time': [30.0, 45.0, 60.0, 25.0, 35.0]
        })
        return data
    
    @pytest.fixture
    def football_model(self):
        """Create FootballAttribution model instance"""
        config = FootballAttributionConfig()
        return FootballAttribution(config)
    
    def test_model_initialization(self):
        """Test model initialization with default config"""
        model = FootballAttribution()
        assert isinstance(model.football_config, FootballAttributionConfig)
        assert model.model_name == 'FootballAttribution'
    
    def test_model_initialization_with_config(self):
        """Test model initialization with custom config"""
        config = FootballAttributionConfig(scorer_weight=0.3)
        model = FootballAttribution(config)
        assert model.football_config.scorer_weight == 0.3
    
    def test_calculate_attribution_basic(self, football_model, sample_data):
        """Test basic attribution calculation"""
        result = football_model.calculate_attribution(sample_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'attribution_score' in result.columns
        assert 'football_roles' in result.columns
        assert 'channel_archetype' in result.columns
        
        # Check that scores sum to 1.0 for each user
        for user_id in result['user_id'].unique():
            user_scores = result[result['user_id'] == user_id]['attribution_score']
            assert abs(user_scores.sum() - 1.0) < 1e-6
    
    def test_calculate_attribution_no_conversions(self, football_model):
        """Test attribution calculation with no conversions"""
        data = pd.DataFrame({
            'user_id': ['user1', 'user1'],
            'touchpoint_id': ['tp1', 'tp2'],
            'channel': ['organic_search', 'paid_search'],
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'conversion': [False, False],
            'engagement_time': [30.0, 45.0]
        })
        
        with pytest.raises(InsufficientDataError):
            football_model.calculate_attribution(data)
    
    def test_calculate_attribution_empty_data(self, football_model):
        """Test attribution calculation with empty data"""
        data = pd.DataFrame(columns=['user_id', 'touchpoint_id', 'channel', 'timestamp'])
        
        with pytest.raises(InsufficientDataError):
            football_model.calculate_attribution(data)
    
    def test_role_assignment_scorer(self, football_model, sample_data):
        """Test that scorer role is assigned correctly"""
        result = football_model.calculate_attribution(sample_data)
        
        # Check that conversion touchpoints get scorer role
        conversion_rows = result[result['conversion'] == True]
        for _, row in conversion_rows.iterrows():
            roles = row['football_roles']
            assert FootballRole.SCORER.value in roles
    
    def test_role_assignment_assister(self, football_model):
        """Test that assister role is assigned correctly"""
        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'touchpoint_id': ['tp1', 'tp2', 'tp3'],
            'channel': ['organic_search', 'paid_search', 'direct'],
            'timestamp': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3)
            ],
            'conversion': [False, False, True],
            'engagement_time': [30.0, 45.0, 60.0]
        })
        
        result = football_model.calculate_attribution(data)
        
        # The touchpoint before conversion should get assister role
        tp2_row = result[result['touchpoint_id'] == 'tp2']
        assert not tp2_row.empty
        roles = tp2_row.iloc[0]['football_roles']
        assert FootballRole.ASSISTER.value in roles
    
    def test_role_assignment_key_passer(self, football_model, sample_data):
        """Test that key passer role is assigned correctly"""
        result = football_model.calculate_attribution(sample_data)
        
        # First touchpoint should get key passer role (if not scorer/assister)
        for user_id in result['user_id'].unique():
            user_data = result[result['user_id'] == user_id].sort_values('timestamp')
            first_touchpoint = user_data.iloc[0]
            roles = first_touchpoint['football_roles']
            
            # Should have key passer role if not scorer or assister
            if (FootballRole.SCORER.value not in roles and 
                FootballRole.ASSISTER.value not in roles):
                assert FootballRole.KEY_PASSER.value in roles
    
    def test_cis_calculation(self, football_model):
        """Test Channel Impact Score calculation"""
        # Test with known roles
        roles = [FootballRole.SCORER.value]
        cis = football_model._calculate_cis_score(roles)
        
        expected_cis = (football_model.football_config.baseline_weight + 
                       (1 - football_model.football_config.baseline_weight) * 
                       football_model.football_config.scorer_weight)
        
        assert abs(cis - expected_cis) < 1e-6
    
    def test_cis_calculation_multiple_roles(self, football_model):
        """Test CIS calculation with multiple roles"""
        roles = [FootballRole.SCORER.value, FootballRole.MOST_PASSES.value]
        cis = football_model._calculate_cis_score(roles)
        
        expected_role_contribution = (football_model.football_config.scorer_weight + 
                                    football_model.football_config.most_passes_weight)
        expected_cis = (football_model.football_config.baseline_weight + 
                       (1 - football_model.football_config.baseline_weight) * 
                       expected_role_contribution)
        
        assert abs(cis - expected_cis) < 1e-6
    
    def test_channel_performance_summary(self, football_model, sample_data):
        """Test channel performance summary generation"""
        result = football_model.calculate_attribution(sample_data)
        summary = football_model.get_channel_performance_summary(result)
        
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        assert 'channel' in summary.columns
        assert 'channel_goals' in summary.columns
        assert 'channel_assists' in summary.columns
        assert 'attribution_score' in summary.columns
        assert 'channel_archetype' in summary.columns
    
    def test_channel_archetype_assignment(self, football_model, sample_data):
        """Test that channel archetypes are assigned correctly"""
        result = football_model.calculate_attribution(sample_data)
        
        # Check that known channels get correct archetypes
        organic_rows = result[result['channel'] == 'organic_search']
        if not organic_rows.empty:
            assert organic_rows.iloc[0]['channel_archetype'] == 'generator'
        
        direct_rows = result[result['channel'] == 'direct']
        if not direct_rows.empty:
            assert direct_rows.iloc[0]['channel_archetype'] == 'closer'
    
    def test_football_metrics_calculation(self, football_model, sample_data):
        """Test that football metrics are calculated correctly"""
        result = football_model.calculate_attribution(sample_data)
        
        # Check that football metrics columns exist
        expected_columns = [
            'channel_goals', 'channel_assists', 'channel_key_passes',
            'channel_passes', 'channel_minutes', 'channel_dribbles',
            'channel_expected_goals', 'channel_expected_assists'
        ]
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_cold_lead_revival_detection(self, football_model):
        """Test detection of cold lead revivals"""
        # Create data with a gap larger than cold_lead_threshold_days
        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'touchpoint_id': ['tp1', 'tp2', 'tp3'],
            'channel': ['organic_search', 'paid_search', 'direct'],
            'timestamp': [
                datetime(2024, 1, 1),
                datetime(2024, 1, 10),  # 9 days gap > 7 day threshold
                datetime(2024, 1, 11)
            ],
            'conversion': [False, False, True],
            'engagement_time': [30.0, 45.0, 60.0]
        })
        
        result = football_model.calculate_attribution(data)
        
        # tp2 should get most_dribbles role for reviving cold lead
        tp2_row = result[result['touchpoint_id'] == 'tp2']
        if not tp2_row.empty:
            roles = tp2_row.iloc[0]['football_roles']
            # Should have dribbles role or be assister (takes precedence)
            assert (FootballRole.MOST_DRIBBLES.value in roles or 
                   FootballRole.ASSISTER.value in roles)
    
    def test_model_info(self, football_model):
        """Test model information retrieval"""
        info = football_model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'attribution_logic' in info
        assert 'football_roles' in info
        assert 'channel_archetypes' in info
        assert 'role_weights' in info
        assert info['requires_conversions'] is True
        assert info['supports_attribution_window'] is True
    
    def test_attribution_window_filtering(self, football_model):
        """Test that attribution window filtering works correctly"""
        # Create data spanning more than the attribution window
        base_date = datetime(2024, 1, 1)
        data = pd.DataFrame({
            'user_id': ['user1'] * 4,
            'touchpoint_id': ['tp1', 'tp2', 'tp3', 'tp4'],
            'channel': ['organic_search', 'paid_search', 'email', 'direct'],
            'timestamp': [
                base_date,
                base_date + timedelta(days=10),
                base_date + timedelta(days=25),
                base_date + timedelta(days=35)  # Conversion date
            ],
            'conversion': [False, False, False, True],
            'engagement_time': [30.0, 45.0, 35.0, 60.0]
        })
        
        # Set attribution window to 30 days
        config = FootballAttributionConfig(attribution_window_days=30)
        model = FootballAttribution(config)
        
        result = model.calculate_attribution(data)
        
        # Only touchpoints within 30 days of conversion should have attribution > 0
        for _, row in result.iterrows():
            days_before_conversion = (base_date + timedelta(days=35) - row['timestamp']).days
            if days_before_conversion > 30:
                assert row['attribution_score'] == 0.0
            else:
                # Within window touchpoints should have some attribution
                assert row['attribution_score'] >= 0.0


class TestFootballMetrics:
    """Test cases for FootballMetrics dataclass"""
    
    def test_metrics_initialization(self):
        """Test FootballMetrics initialization"""
        metrics = FootballMetrics()
        
        assert metrics.goals == 0
        assert metrics.assists == 0
        assert metrics.key_passes == 0
        assert metrics.passes == 0
        assert metrics.minutes == 0.0
        assert metrics.dribbles == 0
        assert metrics.expected_goals == 0.0
        assert metrics.expected_assists == 0.0
        assert metrics.channel_impact_score == 0.0
    
    def test_metrics_with_values(self):
        """Test FootballMetrics with custom values"""
        metrics = FootballMetrics(
            goals=5,
            assists=3,
            passes=100,
            minutes=1500.0,
            expected_goals=4.2
        )
        
        assert metrics.goals == 5
        assert metrics.assists == 3
        assert metrics.passes == 100
        assert metrics.minutes == 1500.0
        assert metrics.expected_goals == 4.2


class TestIntegration:
    """Integration tests for the complete football attribution workflow"""
    
    def test_end_to_end_attribution(self):
        """Test complete end-to-end attribution workflow"""
        # Create realistic customer journey data
        data = pd.DataFrame({
            'user_id': ['customer1'] * 5 + ['customer2'] * 3,
            'touchpoint_id': ['tp1', 'tp2', 'tp3', 'tp4', 'tp5', 'tp6', 'tp7', 'tp8'],
            'channel': [
                'organic_search', 'paid_search', 'email', 'social_media', 'direct',
                'display', 'referral', 'direct'
            ],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 3, 11, 0),
                datetime(2024, 1, 5, 12, 0),
                datetime(2024, 1, 7, 13, 0),
                datetime(2024, 1, 10, 14, 0),  # Conversion
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 4, 10, 0),
                datetime(2024, 1, 6, 11, 0)   # Conversion
            ],
            'conversion': [False, False, False, False, True, False, False, True],
            'conversion_value': [None, None, None, None, 150.0, None, None, 75.0],
            'engagement_time': [45.0, 60.0, 30.0, 90.0, 120.0, 25.0, 40.0, 80.0]
        })
        
        # Initialize model with custom configuration
        config = FootballAttributionConfig(
            attribution_window_days=30,
            scorer_weight=0.3,
            assister_weight=0.25
        )
        model = FootballAttribution(config)
        
        # Calculate attribution
        result = model.calculate_attribution(data)
        
        # Validate results
        assert not result.empty
        assert len(result) == 8  # All touchpoints should be in result
        
        # Check attribution scores sum to 1.0 for each user
        for user_id in result['user_id'].unique():
            user_scores = result[result['user_id'] == user_id]['attribution_score']
            assert abs(user_scores.sum() - 1.0) < 1e-6
        
        # Generate performance summary
        summary = model.get_channel_performance_summary(result)
        assert not summary.empty
        assert 'channel' in summary.columns
        
        # Validate that all channels are represented
        unique_channels = set(data['channel'].unique())
        summary_channels = set(summary['channel'].unique())
        assert unique_channels == summary_channels
    
    def test_example_from_specification(self):
        """Test the specific example from the specification: Organic → Digital → Referral"""
        data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'touchpoint_id': ['tp1', 'tp2', 'tp3'],
            'channel': ['organic_search', 'paid_search', 'referral'],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 11, 0),
                datetime(2024, 1, 3, 12, 0)
            ],
            'conversion': [False, False, True],
            'conversion_value': [None, None, 100.0],
            'engagement_time': [30.0, 45.0, 60.0]
        })
        
        # Use default configuration
        model = FootballAttribution()
        result = model.calculate_attribution(data)
        
        # Validate the example calculation
        assert len(result) == 3
        
        # Check role assignments
        organic_row = result[result['channel'] == 'organic_search'].iloc[0]
        digital_row = result[result['channel'] == 'paid_search'].iloc[0]
        referral_row = result[result['channel'] == 'referral'].iloc[0]
        
        # Referral should be scorer
        assert FootballRole.SCORER.value in referral_row['football_roles']
        
        # Digital should be assister
        assert FootballRole.ASSISTER.value in digital_row['football_roles']
        
        # Organic should be key passer
        assert FootballRole.KEY_PASSER.value in organic_row['football_roles']
        
        # Check that attribution scores sum to 1.0
        total_attribution = result['attribution_score'].sum()
        assert abs(total_attribution - 1.0) < 1e-6
        
        # Referral (closer) should have highest attribution as scorer
        assert referral_row['attribution_score'] > digital_row['attribution_score']
        assert referral_row['attribution_score'] > organic_row['attribution_score']


if __name__ == '__main__':
    pytest.main([__file__])