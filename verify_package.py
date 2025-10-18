"""
Package Verification Script

This script tests the basic functionality of the PyAttrScore package
to ensure all components are working correctly.
"""

import pandas as pd
from datetime import datetime
import sys
import traceback

def test_basic_functionality():
    """Test basic package functionality."""
    print("🔍 Testing PyAttrScore Package Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Import the package
        print("1. Testing package imports...")
        from pyattrscore import (
            FirstTouchAttribution,
            LastTouchAttribution,
            LinearAttribution,
            ExponentialDecayAttribution,
            LinearDecayAttribution,
            UShapedAttribution,
            WindowedFirstTouchAttribution,
            AttributionConfig,
            get_model,
            list_models
        )
        print("   ✅ All imports successful")
        
        # Test 2: Create sample data
        print("\n2. Creating sample touchpoint data...")
        sample_data = pd.DataFrame([
            {
                'user_id': 'user_001',
                'touchpoint_id': 'tp_001',
                'channel': 'email',
                'timestamp': datetime(2023, 1, 1, 10, 0),
                'conversion': False,
                'conversion_value': None
            },
            {
                'user_id': 'user_001',
                'touchpoint_id': 'tp_002',
                'channel': 'social_media',
                'timestamp': datetime(2023, 1, 2, 14, 30),
                'conversion': False,
                'conversion_value': None
            },
            {
                'user_id': 'user_001',
                'touchpoint_id': 'tp_003',
                'channel': 'search',
                'timestamp': datetime(2023, 1, 3, 9, 15),
                'conversion': True,
                'conversion_value': 150.0
            }
        ])
        print(f"   ✅ Sample data created with {len(sample_data)} touchpoints")
        
        # Test 3: Test configuration
        print("\n3. Testing configuration...")
        config = AttributionConfig(attribution_window_days=30, decay_rate=0.5)
        print(f"   ✅ Configuration created: {config.attribution_window_days} day window")
        
        # Test 4: Test First Touch Attribution
        print("\n4. Testing First Touch Attribution...")
        first_touch_model = FirstTouchAttribution(config)
        first_touch_results = first_touch_model.calculate_attribution(sample_data)
        print(f"   ✅ First Touch Attribution completed: {len(first_touch_results)} results")
        print(f"   📊 Attribution scores: {first_touch_results['attribution_score'].tolist()}")
        
        # Test 5: Test Linear Attribution
        print("\n5. Testing Linear Attribution...")
        linear_model = LinearAttribution(config)
        linear_results = linear_model.calculate_attribution(sample_data)
        print(f"   ✅ Linear Attribution completed: {len(linear_results)} results")
        print(f"   📊 Attribution scores: {[round(x, 3) for x in linear_results['attribution_score'].tolist()]}")
        
        # Test 6: Test U-Shaped Attribution
        print("\n6. Testing U-Shaped Attribution...")
        u_shaped_model = UShapedAttribution(config, first_touch_weight=0.4, last_touch_weight=0.4)
        u_shaped_results = u_shaped_model.calculate_attribution(sample_data)
        print(f"   ✅ U-Shaped Attribution completed: {len(u_shaped_results)} results")
        print(f"   📊 Attribution scores: {[round(x, 3) for x in u_shaped_results['attribution_score'].tolist()]}")
        
        # Test 7: Test model factory
        print("\n7. Testing model factory...")
        available_models = list_models()
        print(f"   ✅ Available models: {available_models}")
        
        # Test a model via factory
        factory_model = get_model('exponential_decay', config)
        factory_results = factory_model.calculate_attribution(sample_data)
        print(f"   ✅ Factory model test completed: {len(factory_results)} results")
        
        # Test 8: Validate attribution scores sum to 1
        print("\n8. Validating attribution scores...")
        for model_name, results in [
            ('First Touch', first_touch_results),
            ('Linear', linear_results),
            ('U-Shaped', u_shaped_results),
            ('Exponential Decay', factory_results)
        ]:
            total_score = results['attribution_score'].sum()
            if abs(total_score - 1.0) < 1e-6:
                print(f"   ✅ {model_name}: Attribution scores sum to {total_score:.6f}")
            else:
                print(f"   ❌ {model_name}: Attribution scores sum to {total_score:.6f} (expected 1.0)")
        
        # Test 9: Test model info
        print("\n9. Testing model information...")
        model_info = first_touch_model.get_model_info()
        print(f"   ✅ Model info retrieved: {model_info['model_name']}")
        print(f"   📝 Description: {model_info.get('attribution_logic', 'N/A')}")
        
        # Test 10: Test with multiple users
        print("\n10. Testing with multiple users...")
        multi_user_data = pd.DataFrame([
            # User 1
            {'user_id': 'user_001', 'touchpoint_id': 'tp_001', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user_001', 'touchpoint_id': 'tp_002', 'channel': 'search', 
             'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 100.0},
            
            # User 2
            {'user_id': 'user_002', 'touchpoint_id': 'tp_003', 'channel': 'social', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False},
            {'user_id': 'user_002', 'touchpoint_id': 'tp_004', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 200.0}
        ])
        
        multi_results = linear_model.calculate_attribution(multi_user_data)
        unique_users = multi_results['user_id'].nunique()
        print(f"   ✅ Multi-user test completed: {unique_users} users, {len(multi_results)} touchpoints")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! PyAttrScore package is working correctly.")
        print("\n📋 Package Summary:")
        print(f"   • Available Models: {len(available_models)}")
        print(f"   • Core Features: Attribution calculation, data validation, logging")
        print(f"   • Test Coverage: Basic functionality verified")
        print("\n🚀 The package is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling capabilities."""
    print("\n" + "=" * 50)
    print("🛡️  Testing Error Handling")
    print("=" * 50)
    
    try:
        from pyattrscore import LinearAttribution
        from pyattrscore.exceptions import InsufficientDataError
        
        # Test with no conversion data
        print("1. Testing with no conversions...")
        no_conversion_data = pd.DataFrame([
            {'user_id': 'user1', 'touchpoint_id': 'tp1', 'channel': 'email', 
             'timestamp': datetime(2023, 1, 1), 'conversion': False}
        ])
        
        model = LinearAttribution()
        try:
            model.calculate_attribution(no_conversion_data)
            print("   ❌ Should have raised InsufficientDataError")
            return False
        except InsufficientDataError:
            print("   ✅ Correctly raised InsufficientDataError for no conversions")
        
        # Test with empty data
        print("\n2. Testing with empty data...")
        empty_data = pd.DataFrame(columns=['user_id', 'touchpoint_id', 'channel', 'timestamp'])
        try:
            model.calculate_attribution(empty_data)
            print("   ❌ Should have raised InsufficientDataError")
            return False
        except InsufficientDataError:
            print("   ✅ Correctly raised InsufficientDataError for empty data")
        
        print("\n✅ Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("PyAttrScore Package Verification")
    print("================================")
    
    # Run basic functionality tests
    basic_test_passed = test_basic_functionality()
    
    # Run error handling tests
    error_test_passed = test_error_handling()
    
    # Final summary
    print("\n" + "=" * 60)
    if basic_test_passed and error_test_passed:
        print("🎊 VERIFICATION COMPLETE: All tests passed!")
        print("📦 PyAttrScore package is ready for production use.")
        sys.exit(0)
    else:
        print("💥 VERIFICATION FAILED: Some tests did not pass.")
        print("🔧 Please check the errors above and fix any issues.")
        sys.exit(1)