"""
Unit tests for Utility Functions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyattrscore.utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    filter_by_date_range,
    calculate_time_decay_weights,
    normalize_weights,
    group_touchpoints_by_user,
    identify_conversion_touchpoints,
    calculate_attribution_window_bounds,
    validate_attribution_scores,
    create_attribution_result_dataframe,
    safe_divide,
    calculate_summary_statistics
)
from pyattrscore.exceptions import InvalidInputError, DataValidationError


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame([
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp1',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1),
                'conversion': False,
                'conversion_value': None
            },
            {
                'user_id': 'user1',
                'touchpoint_id': 'tp2',
                'channel': 'social',
                'timestamp': datetime(2023, 1, 2),
                'conversion': True,
                'conversion_value': 100.0
            },
            {
                'user_id': 'user2',
                'touchpoint_id': 'tp3',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 3),
                'conversion': False,
                'conversion_value': None
            }
        ])
    
    def test_validate_dataframe_structure_valid(self):
        """Test DataFrame structure validation with valid data."""
        required_columns = ['user_id', 'touchpoint_id', 'channel', 'timestamp']
        
        # Should not raise any exception
        validate_dataframe_structure(self.sample_data, required_columns)
    
    def test_validate_dataframe_structure_missing_columns(self):
        """Test DataFrame structure validation with missing columns."""
        required_columns = ['user_id', 'touchpoint_id', 'channel', 'timestamp', 'missing_col']
        
        with pytest.raises(InvalidInputError) as exc_info:
            validate_dataframe_structure(self.sample_data, required_columns)
        
        assert 'missing_col' in str(exc_info.value)
    
    def test_validate_dataframe_structure_empty(self):
        """Test DataFrame structure validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        required_columns = ['user_id']
        
        with pytest.raises(InvalidInputError):
            validate_dataframe_structure(empty_df, required_columns)
    
    def test_convert_timestamp_column_string(self):
        """Test timestamp conversion from string."""
        df_with_string_timestamps = pd.DataFrame([
            {
                'user_id': 'user1',
                'timestamp': '2023-01-01 10:00:00'
            },
            {
                'user_id': 'user2',
                'timestamp': '2023-01-02 11:00:00'
            }
        ])
        
        result = convert_timestamp_column(df_with_string_timestamps)
        
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        assert result['timestamp'].iloc[0] == datetime(2023, 1, 1, 10, 0)
    
    def test_convert_timestamp_column_already_datetime(self):
        """Test timestamp conversion when already datetime."""
        result = convert_timestamp_column(self.sample_data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        # Should return the same data
        pd.testing.assert_frame_equal(result, self.sample_data)
    
    def test_sort_by_timestamp(self):
        """Test sorting by timestamp."""
        # Create unsorted data
        unsorted_data = pd.DataFrame([
            {'user_id': 'user1', 'timestamp': datetime(2023, 1, 3), 'value': 'c'},
            {'user_id': 'user1', 'timestamp': datetime(2023, 1, 1), 'value': 'a'},
            {'user_id': 'user1', 'timestamp': datetime(2023, 1, 2), 'value': 'b'}
        ])
        
        result = sort_by_timestamp(unsorted_data)
        
        expected_order = ['a', 'b', 'c']
        actual_order = result['value'].tolist()
        
        assert actual_order == expected_order
    
    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        result = filter_by_date_range(self.sample_data, start_date, end_date)
        
        assert len(result) == 2  # Should exclude the third row
        assert all(start_date <= ts <= end_date for ts in result['timestamp'])
    
    def test_filter_by_date_range_no_bounds(self):
        """Test filtering with no date bounds."""
        result = filter_by_date_range(self.sample_data)
        
        # Should return all data unchanged
        pd.testing.assert_frame_equal(result, self.sample_data)
    
    def test_calculate_time_decay_weights_exponential(self):
        """Test exponential time decay weight calculation."""
        timestamps = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ]
        reference_time = datetime(2023, 1, 3)
        decay_rate = 0.5
        
        weights = calculate_time_decay_weights(
            timestamps, reference_time, decay_rate, 'exponential'
        )
        
        assert len(weights) == 3
        assert all(0 <= w <= 1 for w in weights)
        # Most recent should have highest weight
        assert weights[2] >= weights[1] >= weights[0]
    
    def test_calculate_time_decay_weights_linear(self):
        """Test linear time decay weight calculation."""
        timestamps = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ]
        reference_time = datetime(2023, 1, 3)
        decay_rate = 0.5
        
        weights = calculate_time_decay_weights(
            timestamps, reference_time, decay_rate, 'linear'
        )
        
        assert len(weights) == 3
        assert all(0 <= w <= 1 for w in weights)
        # Most recent should have highest weight
        assert weights[2] >= weights[1] >= weights[0]
    
    def test_calculate_time_decay_weights_invalid_decay_rate(self):
        """Test time decay weights with invalid decay rate."""
        timestamps = [datetime(2023, 1, 1)]
        reference_time = datetime(2023, 1, 1)
        
        with pytest.raises(DataValidationError):
            calculate_time_decay_weights(timestamps, reference_time, 1.5, 'exponential')
    
    def test_calculate_time_decay_weights_invalid_type(self):
        """Test time decay weights with invalid decay type."""
        timestamps = [datetime(2023, 1, 1)]
        reference_time = datetime(2023, 1, 1)
        
        with pytest.raises(DataValidationError):
            calculate_time_decay_weights(timestamps, reference_time, 0.5, 'invalid')
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = [1.0, 2.0, 3.0]
        normalized = normalize_weights(weights)
        
        assert len(normalized) == 3
        assert abs(sum(normalized) - 1.0) < 1e-6
        assert normalized == [1/6, 2/6, 3/6]
    
    def test_normalize_weights_empty(self):
        """Test weight normalization with empty list."""
        with pytest.raises(DataValidationError):
            normalize_weights([])
    
    def test_normalize_weights_negative(self):
        """Test weight normalization with negative weights."""
        with pytest.raises(DataValidationError):
            normalize_weights([1.0, -1.0, 2.0])
    
    def test_normalize_weights_zero_sum(self):
        """Test weight normalization with zero sum."""
        with pytest.raises(DataValidationError):
            normalize_weights([0.0, 0.0, 0.0])
    
    def test_group_touchpoints_by_user(self):
        """Test grouping touchpoints by user."""
        grouped = group_touchpoints_by_user(self.sample_data)
        
        assert isinstance(grouped, dict)
        assert 'user1' in grouped
        assert 'user2' in grouped
        assert len(grouped['user1']) == 2
        assert len(grouped['user2']) == 1
    
    def test_identify_conversion_touchpoints(self):
        """Test identifying conversion touchpoints."""
        conversions = identify_conversion_touchpoints(self.sample_data)
        
        assert len(conversions) == 1
        assert conversions.iloc[0]['user_id'] == 'user1'
        assert conversions.iloc[0]['touchpoint_id'] == 'tp2'
    
    def test_identify_conversion_touchpoints_no_column(self):
        """Test identifying conversions when column is missing."""
        data_no_conversion = self.sample_data.drop('conversion', axis=1)
        conversions = identify_conversion_touchpoints(data_no_conversion)
        
        assert len(conversions) == 0
    
    def test_calculate_attribution_window_bounds(self):
        """Test attribution window bounds calculation."""
        conversion_time = datetime(2023, 1, 10)
        window_days = 7
        
        start, end = calculate_attribution_window_bounds(conversion_time, window_days)
        
        assert start == datetime(2023, 1, 3)
        assert end == datetime(2023, 1, 10)
    
    def test_validate_attribution_scores_valid(self):
        """Test attribution score validation with valid scores."""
        scores = [0.3, 0.3, 0.4]
        assert validate_attribution_scores(scores) is True
    
    def test_validate_attribution_scores_invalid_sum(self):
        """Test attribution score validation with invalid sum."""
        scores = [0.3, 0.3, 0.3]  # Sum = 0.9
        assert validate_attribution_scores(scores) is False
    
    def test_validate_attribution_scores_empty(self):
        """Test attribution score validation with empty scores."""
        assert validate_attribution_scores([]) is False
    
    def test_create_attribution_result_dataframe(self):
        """Test creating attribution result DataFrame."""
        touchpoints = self.sample_data.iloc[:2]  # First 2 rows
        scores = [0.6, 0.4]
        model_name = 'TestModel'
        
        result = create_attribution_result_dataframe(touchpoints, scores, model_name)
        
        assert 'attribution_score' in result.columns
        assert 'model_name' in result.columns
        assert 'attribution_percentage' in result.columns
        assert result['attribution_score'].tolist() == scores
        assert all(result['model_name'] == model_name)
        assert result['attribution_percentage'].tolist() == [60.0, 40.0]
    
    def test_create_attribution_result_dataframe_mismatched_length(self):
        """Test creating result DataFrame with mismatched lengths."""
        touchpoints = self.sample_data.iloc[:2]
        scores = [0.6, 0.4, 0.0]  # Different length
        model_name = 'TestModel'
        
        with pytest.raises(DataValidationError):
            create_attribution_result_dataframe(touchpoints, scores, model_name)
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=999) == 999
    
    def test_calculate_summary_statistics(self):
        """Test summary statistics calculation."""
        # Create result DataFrame with attribution scores
        result_data = self.sample_data.copy()
        result_data['attribution_score'] = [0.5, 0.5, 0.0]
        result_data['attribution_value'] = [50.0, 50.0, 0.0]
        
        stats = calculate_summary_statistics(result_data)
        
        assert isinstance(stats, dict)
        assert 'total_touchpoints' in stats
        assert 'unique_users' in stats
        assert 'unique_channels' in stats
        assert 'total_conversions' in stats
        assert 'total_attribution_score' in stats
        
        assert stats['total_touchpoints'] == 3
        assert stats['unique_users'] == 2
        assert stats['unique_channels'] == 3
        assert stats['total_conversions'] == 1
        assert stats['total_attribution_score'] == 1.0
    
    def test_calculate_summary_statistics_with_channel_breakdown(self):
        """Test summary statistics with channel-level breakdown."""
        result_data = self.sample_data.copy()
        result_data['attribution_score'] = [0.3, 0.7, 0.0]
        
        stats = calculate_summary_statistics(result_data)
        
        assert 'channel_attribution' in stats
        channel_stats = stats['channel_attribution']
        assert 'sum' in channel_stats
        assert 'mean' in channel_stats
        assert 'count' in channel_stats