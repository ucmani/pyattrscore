import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyattrscore.football import FootballAttribution, FootballAttributionConfig

def test_post_conversion_exclusion():
    """Test that post-conversion touchpoints are excluded when configured."""
    # Create test data with touchpoints before and after conversion
    data = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user1', 'user1', 'user1'],
        'touchpoint_id': ['tp1', 'tp2', 'tp3', 'tp4', 'tp5'],
        'channel': ['email', 'social', 'search', 'direct', 'email'],
        'timestamp': [
            datetime(2023, 1, 1, 10, 0),  # Before conversion
            datetime(2023, 1, 2, 10, 0),  # Before conversion
            datetime(2023, 1, 3, 10, 0),  # Conversion
            datetime(2023, 1, 4, 10, 0),  # After conversion
            datetime(2023, 1, 5, 10, 0),  # After conversion
        ],
        'conversion': [False, False, True, True, True],  # Make post-conversion touchpoints also conversions
        'conversion_value': [0, 0, 100, 50, 25],
        'engagement_time': [30, 45, 60, 20, 15]
    })
    
    # Test with exclude_post_conversion_touchpoints=True
    config_exclude = FootballAttributionConfig(
        attribution_window_days=30,
        exclude_post_conversion_touchpoints=True
    )
    model_exclude = FootballAttribution(config_exclude)
    result_exclude = model_exclude.calculate_attribution(data)
    
    # Test with exclude_post_conversion_touchpoints=False
    config_include = FootballAttributionConfig(
        attribution_window_days=30,
        exclude_post_conversion_touchpoints=False
    )
    model_include = FootballAttribution(config_include)
    result_include = model_include.calculate_attribution(data)
    
    # Check results
    print("\n=== Post-Conversion Exclusion Test ===")
    print(f"With exclusion: {len(result_exclude)} touchpoints")
    print(f"Without exclusion: {len(result_include)} touchpoints")
    print(f"Difference: {len(result_include) - len(result_exclude)} touchpoints")
    
    # The difference should be 2 (tp4 and tp5)
    assert len(result_include) - len(result_exclude) == 2, "Post-conversion exclusion not working correctly"
    
    # Verify the excluded touchpoints are the ones after conversion
    if len(result_exclude) < len(result_include):
        excluded_ids = set(result_include['touchpoint_id']) - set(result_exclude['touchpoint_id'])
        print(f"Excluded touchpoint IDs: {excluded_ids}")
        assert excluded_ids == {'tp4', 'tp5'}, "Wrong touchpoints were excluded"
    
    print("Post-conversion exclusion test passed!")
    return result_exclude, result_include

def test_expected_goals_sum():
    """Test that expected goals don't sum to more than 1.0."""
    # Create test data with multiple touchpoints and conversions
    data = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user1', 'user2', 'user2'],
        'touchpoint_id': ['tp1', 'tp2', 'tp3', 'tp4', 'tp5'],
        'channel': ['email', 'social', 'search', 'direct', 'email'],
        'timestamp': [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 2, 10, 0),
            datetime(2023, 1, 3, 10, 0),
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 2, 10, 0),
        ],
        'conversion': [False, False, True, False, True],
        'conversion_value': [0, 0, 100, 0, 50],
        'engagement_time': [30, 45, 60, 20, 15]
    })
    
    # Run the model
    config = FootballAttributionConfig(
        attribution_window_days=30,
        exclude_post_conversion_touchpoints=True,
        conversion_xg_weight=0.8,
        non_conversion_xg_weight=0.1
    )
    model = FootballAttribution(config)
    result = model.calculate_attribution(data)
    
    # Get channel summary
    channel_summary = model.get_channel_performance_summary(result)
    
    # Check expected goals
    print("\n=== Expected Goals Sum Test ===")
    print("Channel expected goals:")
    for _, row in channel_summary.iterrows():
        print(f"{row['channel']}: {row['channel_expected_goals']}")
    
    # Check that no channel has expected goals > 1.0
    max_xg = channel_summary['channel_expected_goals'].max()
    print(f"Maximum expected goals for any channel: {max_xg}")
    assert max_xg <= 1.0, f"Expected goals should not exceed 1.0, got {max_xg}"
    
    print("Expected goals test passed!")
    return result, channel_summary

if __name__ == "__main__":
    print("Testing PyAttrScore Football Attribution fixes...")
    
    # Test post-conversion exclusion
    result_exclude, result_include = test_post_conversion_exclusion()
    
    # Test expected goals calculation
    result, channel_summary = test_expected_goals_sum()
    
    print("\nAll tests passed! The fixes have been successfully implemented.")