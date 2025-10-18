"""
PyAttrScore - Football-Inspired Attribution Model Demo

This script demonstrates the Football-Inspired Attribution Model,
a unique approach to marketing attribution that treats channels
like football players working together to score conversions.

Run with: python main.py
Add --football flag to run the football model specifically.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys

# Import PyAttrScore components
from pyattrscore import (
    FootballAttribution,
    FootballAttributionConfig,
    get_model,
    list_models,
    get_model_info
)


def create_sample_journey_data():
    """Create sample customer journey data for demonstration."""
    print("🏈 Creating sample customer journey data...")
    
    # Sample data representing the specification example and additional journeys
    data = pd.DataFrame({
        'user_id': [
            # Specification example: Organic → Digital → Referral
            'customer_1', 'customer_1', 'customer_1',
            # Additional realistic journeys
            'customer_2', 'customer_2', 'customer_2', 'customer_2',
            'customer_3', 'customer_3', 'customer_3', 'customer_3', 'customer_3',
            'customer_4', 'customer_4',
            'customer_5', 'customer_5', 'customer_5', 'customer_5', 'customer_5', 'customer_5'
        ],
        'touchpoint_id': [
            'tp_1', 'tp_2', 'tp_3',
            'tp_4', 'tp_5', 'tp_6', 'tp_7',
            'tp_8', 'tp_9', 'tp_10', 'tp_11', 'tp_12',
            'tp_13', 'tp_14',
            'tp_15', 'tp_16', 'tp_17', 'tp_18', 'tp_19', 'tp_20'
        ],
        'channel': [
            # Customer 1: Organic → Paid Search → Referral (from spec)
            'organic_search', 'paid_search', 'referral',
            # Customer 2: Social → Email → Paid Search → Direct
            'social_media', 'email', 'paid_search', 'direct',
            # Customer 3: Display → Organic → Email → Social → Direct
            'display', 'organic_search', 'email', 'social_media', 'direct',
            # Customer 4: Direct → Referral (short journey)
            'direct', 'referral',
            # Customer 5: Organic → Display → Email → Paid → Social → Direct (long journey)
            'organic_search', 'display', 'email', 'paid_search', 'social_media', 'direct'
        ],
        'timestamp': [
            # Customer 1 (3 days)
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 2, 11, 0),
            datetime(2024, 1, 3, 12, 0),
            # Customer 2 (1 week)
            datetime(2024, 1, 5, 9, 0),
            datetime(2024, 1, 7, 14, 0),
            datetime(2024, 1, 10, 16, 0),
            datetime(2024, 1, 12, 11, 0),
            # Customer 3 (2 weeks)
            datetime(2024, 1, 15, 8, 0),
            datetime(2024, 1, 17, 13, 0),
            datetime(2024, 1, 20, 15, 0),
            datetime(2024, 1, 25, 10, 0),
            datetime(2024, 1, 28, 14, 0),
            # Customer 4 (same day)
            datetime(2024, 2, 1, 10, 0),
            datetime(2024, 2, 1, 15, 0),
            # Customer 5 (3 weeks)
            datetime(2024, 2, 5, 9, 0),
            datetime(2024, 2, 8, 11, 0),
            datetime(2024, 2, 12, 14, 0),
            datetime(2024, 2, 18, 16, 0),
            datetime(2024, 2, 22, 13, 0),
            datetime(2024, 2, 25, 12, 0)
        ],
        'conversion': [
            # Customer 1: Converts on referral
            False, False, True,
            # Customer 2: Converts on direct
            False, False, False, True,
            # Customer 3: Converts on direct
            False, False, False, False, True,
            # Customer 4: Converts on referral
            False, True,
            # Customer 5: Converts on direct
            False, False, False, False, False, True
        ],
        'conversion_value': [
            None, None, 100.0,
            None, None, None, 250.0,
            None, None, None, None, 180.0,
            None, 75.0,
            None, None, None, None, None, 320.0
        ],
        'engagement_time': [
            30.0, 45.0, 60.0,
            25.0, 40.0, 35.0, 90.0,
            15.0, 50.0, 30.0, 20.0, 120.0,
            45.0, 80.0,
            35.0, 20.0, 55.0, 40.0, 25.0, 100.0
        ]
    })
    
    print(f"✅ Created {len(data)} touchpoints across {data['user_id'].nunique()} customers")
    return data


def demonstrate_football_model():
    """Demonstrate the Football Attribution Model."""
    print("\n" + "="*70)
    print("🏈 FOOTBALL-INSPIRED ATTRIBUTION MODEL DEMONSTRATION")
    print("="*70)
    print("Transforming marketing attribution into a beautiful team game!")
    
    # Create sample data
    data = create_sample_journey_data()
    
    # Show sample data
    print("\n📋 Sample Customer Journey Data:")
    print(data[['user_id', 'channel', 'timestamp', 'conversion', 'engagement_time']].head(10))
    
    # Configure Football Attribution
    print("\n⚙️ Configuring Football Attribution Model...")
    config = FootballAttributionConfig(
        attribution_window_days=30,
        scorer_weight=0.25,      # Final conversion (like a striker)
        assister_weight=0.20,    # Setup before conversion (like a midfielder)
        key_passer_weight=0.15,  # Journey initiator (like a defender starting play)
        most_passes_weight=0.15, # Most frequent engagement
        most_minutes_weight=0.10, # Longest engagement time
        most_dribbles_weight=0.10, # Cold lead revival
        participant_weight=0.05,  # Supporting role
        baseline_weight=0.1,
        cold_lead_threshold_days=7
    )
    
    print("✅ Football team formation configured!")
    print(f"   🥅 Scorer weight: {config.scorer_weight}")
    print(f"   🎯 Assister weight: {config.assister_weight}")
    print(f"   🚀 Key passer weight: {config.key_passer_weight}")
    
    # Initialize model
    model = FootballAttribution(config)
    
    # Calculate attribution
    print("\n🔄 Calculating Channel Impact Scores (CIS)...")
    try:
        results = model.calculate_attribution(data)
        print(f"✅ Attribution calculated successfully!")
        print(f"📊 Processed {len(results)} touchpoints")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Display results
    print("\n🏆 FOOTBALL ATTRIBUTION RESULTS")
    print("-" * 50)
    
    # Show the specification example first
    print("\n📖 Specification Example (Organic → Digital → Referral):")
    customer_1_results = results[results['user_id'] == 'customer_1'].copy()
    customer_1_results = customer_1_results.sort_values('timestamp')
    
    for _, row in customer_1_results.iterrows():
        roles_str = ', '.join(row['football_roles']) if row['football_roles'] else 'Participant'
        print(f"  {row['channel']:15} | {roles_str:20} | CIS: {row['attribution_score']:.3f} ({row['attribution_score']*100:.1f}%)")
    
    # Verify the calculation
    referral_cis = customer_1_results[customer_1_results['channel'] == 'referral']['attribution_score'].iloc[0]
    digital_cis = customer_1_results[customer_1_results['channel'] == 'paid_search']['attribution_score'].iloc[0]
    organic_cis = customer_1_results[customer_1_results['channel'] == 'organic_search']['attribution_score'].iloc[0]
    
    print(f"\n✅ Specification Example Results:")
    print(f"   Referral (Closer):  {referral_cis:.1%}")
    print(f"   Digital (Assister): {digital_cis:.1%}")
    print(f"   Organic (Generator): {organic_cis:.1%}")
    
    # Generate team performance summary
    print("\n🏟️ TEAM PERFORMANCE SUMMARY")
    print("-" * 50)
    summary = model.get_channel_performance_summary(results)
    
    # Display key metrics
    print(f"{'Channel':<15} {'Archetype':<12} {'Goals':<6} {'Assists':<7} {'Passes':<7} {'Attribution':<12}")
    print("-" * 70)
    
    for _, row in summary.iterrows():
        print(f"{row['channel']:<15} {row['channel_archetype']:<12} "
              f"{row['channel_goals']:<6.0f} {row['channel_assists']:<7.0f} "
              f"{row['channel_passes']:<7.0f} {row['attribution_score']:<12.3f}")
    
    # Show insights
    print("\n⚽ FOOTBALL ANALYTICS INSIGHTS")
    print("-" * 50)
    
    # Top performers
    top_scorer = summary.loc[summary['channel_goals'].idxmax()]
    top_assister = summary.loc[summary['channel_assists'].idxmax()]
    most_active = summary.loc[summary['channel_passes'].idxmax()]
    
    print(f"🥅 Top Scorer (Closer):     {top_scorer['channel']} ({top_scorer['channel_goals']:.0f} goals)")
    print(f"🎯 Top Assister (Setup):    {top_assister['channel']} ({top_assister['channel_assists']:.0f} assists)")
    print(f"🏃 Most Active Player:      {most_active['channel']} ({most_active['channel_passes']:.0f} passes)")
    
    # Team formation analysis
    print(f"\n🏟️ TEAM FORMATION ANALYSIS")
    archetype_performance = summary.groupby('channel_archetype').agg({
        'channel_goals': 'sum',
        'channel_assists': 'sum',
        'attribution_score': 'sum'
    }).round(2)
    
    for archetype, row in archetype_performance.iterrows():
        print(f"   {archetype.upper():<12}: {row['channel_goals']:.0f} goals, "
              f"{row['channel_assists']:.0f} assists, {row['attribution_score']:.2f} total attribution")
    
    return results, summary


def show_model_comparison():
    """Show comparison between different attribution models."""
    print("\n" + "="*70)
    print("📊 ATTRIBUTION MODEL COMPARISON")
    print("="*70)
    
    # Create simple test data
    test_data = pd.DataFrame({
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
    
    models_to_compare = ['first_touch', 'last_touch', 'linear', 'football']
    
    print(f"Journey: {' → '.join(test_data['channel'].tolist())}")
    print(f"{'Model':<15} {'Organic':<12} {'Paid Search':<12} {'Direct':<12}")
    print("-" * 55)
    
    for model_name in models_to_compare:
        try:
            if model_name == 'football':
                model = get_model(model_name, FootballAttributionConfig())
            else:
                model = get_model(model_name)
            
            result = model.calculate_attribution(test_data)
            
            # Extract attribution scores
            organic_score = result[result['channel'] == 'organic_search']['attribution_score'].iloc[0]
            paid_score = result[result['channel'] == 'paid_search']['attribution_score'].iloc[0]
            direct_score = result[result['channel'] == 'direct']['attribution_score'].iloc[0]
            
            print(f"{model_name:<15} {organic_score:<12.3f} {paid_score:<12.3f} {direct_score:<12.3f}")
            
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)[:30]}...")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='PyAttrScore Football Attribution Demo')
    parser.add_argument('--football', action='store_true', 
                       help='Run football attribution model demo')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different attribution models')
    parser.add_argument('--list', action='store_true',
                       help='List all available models')
    
    args = parser.parse_args()
    
    print("🏈 Welcome to PyAttrScore - Football-Inspired Attribution!")
    print("Making marketing attribution as exciting as the beautiful game!")
    
    if args.list:
        print(f"\n📋 Available Attribution Models:")
        models = list_models()
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        return
    
    if args.compare:
        show_model_comparison()
        return
    
    if args.football or len(sys.argv) == 1:
        # Run football demo by default or when --football flag is used
        results, summary = demonstrate_football_model()
        
        if results is not None:
            print("\n" + "="*70)
            print("🎉 DEMO COMPLETE!")
            print("="*70)
            print("🏈 The Football Attribution Model transforms marketing analytics")
            print("   by treating each channel like a player in a football team.")
            print("\n💡 Key Benefits:")
            print("   • Intuitive role-based attribution (Scorer, Assister, Key Passer)")
            print("   • Fair credit distribution using Channel Impact Score (CIS)")
            print("   • Football metrics for easy stakeholder communication")
            print("   • Customizable role weights for different business models")
            print("\n🚀 Next Steps:")
            print("   • Customize channel archetypes for your business")
            print("   • Adjust role weights based on your conversion funnel")
            print("   • Use insights to optimize marketing budget allocation")
            print("   • Run: python football_example.py for detailed analysis")
            
            print(f"\n📊 Quick Stats from this demo:")
            print(f"   • Processed {len(results)} touchpoints")
            print(f"   • Analyzed {results['user_id'].nunique()} customer journeys")
            print(f"   • Evaluated {len(summary)} marketing channels")
            print(f"   • Total conversions: {summary['channel_goals'].sum():.0f}")
    else:
        print("\nUsage:")
        print("  python main.py --football    # Run football attribution demo")
        print("  python main.py --compare     # Compare attribution models")
        print("  python main.py --list        # List available models")
        print("  python football_example.py   # Detailed football analysis")


if __name__ == '__main__':
    main()
