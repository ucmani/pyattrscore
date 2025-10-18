"""
Demo script using sample data with Football Attribution Model

This script demonstrates how to use the sample_data.csv file
with the Football Attribution Model to analyze marketing channel performance.
"""

import pandas as pd
from datetime import datetime
from pyattrscore import FootballAttribution, FootballAttributionConfig, get_model


def load_sample_data():
    """Load the sample data from CSV file."""
    try:
        data = pd.read_csv('sample_data.csv')
        # Convert timestamp column to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        print(f"✅ Loaded {len(data)} touchpoints from sample_data.csv")
        print(f"📊 Data covers {data['user_id'].nunique()} customers")
        print(f"🎯 Total conversions: {data['conversion'].sum()}")
        return data
    except FileNotFoundError:
        print("❌ sample_data.csv not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def demonstrate_football_with_sample_data():
    """Demonstrate Football Attribution with sample data."""
    print("🏈 Football Attribution Demo with Sample Data")
    print("=" * 50)
    
    # Load sample data
    data = load_sample_data()
    if data is None:
        return
    
    # Show data preview
    print("\n📋 Sample Data Preview:")
    print(data.head(10))
    
    # Initialize Football Attribution model
    print("\n⚙️ Initializing Football Attribution Model...")
    model = FootballAttribution()
    
    # Calculate attribution
    print("\n🔄 Calculating Football Attribution...")
    try:
        results = model.calculate_attribution(data)
        print(f"✅ Attribution calculated successfully!")
        print(f"📊 Processed {len(results)} touchpoints")
    except Exception as e:
        print(f"❌ Error calculating attribution: {e}")
        return
    
    # Show results preview
    print("\n🏆 Attribution Results Preview:")
    preview_cols = ['user_id', 'channel', 'attribution_score', 'football_roles', 'channel_archetype']
    print(results[preview_cols].head(10))
    
    # Generate team performance summary
    print("\n🏟️ Team Performance Summary:")
    summary = model.get_channel_performance_summary(results)
    
    display_cols = ['channel', 'channel_archetype', 'channel_goals', 'channel_assists', 
                   'channel_passes', 'attribution_score']
    print(summary[display_cols].round(3))
    
    # Show insights
    print("\n⚽ Football Analytics Insights:")
    print("-" * 40)
    
    # Top performers
    if not summary.empty:
        top_scorer = summary.loc[summary['channel_goals'].idxmax()]
        top_assister = summary.loc[summary['channel_assists'].idxmax()]
        most_active = summary.loc[summary['channel_passes'].idxmax()]
        
        print(f"🥅 Top Scorer (Closer):     {top_scorer['channel']} ({top_scorer['channel_goals']:.0f} goals)")
        print(f"🎯 Top Assister (Setup):    {top_assister['channel']} ({top_assister['channel_assists']:.0f} assists)")
        print(f"🏃 Most Active Player:      {most_active['channel']} ({most_active['channel_passes']:.0f} passes)")
        
        # Team formation analysis
        print(f"\n🏟️ Team Formation Analysis:")
        archetype_performance = summary.groupby('channel_archetype').agg({
            'channel_goals': 'sum',
            'channel_assists': 'sum',
            'attribution_score': 'sum'
        }).round(2)
        
        for archetype, row in archetype_performance.iterrows():
            print(f"   {archetype.upper():<12}: {row['channel_goals']:.0f} goals, "
                  f"{row['channel_assists']:.0f} assists, {row['attribution_score']:.2f} total attribution")
    
    return results, summary


def compare_models_with_sample_data():
    """Compare different attribution models using sample data."""
    print("\n" + "=" * 50)
    print("📊 Model Comparison with Sample Data")
    print("=" * 50)
    
    # Load sample data
    data = load_sample_data()
    if data is None:
        return
    
    # Models to compare
    models_to_compare = ['first_touch', 'last_touch', 'linear', 'football']
    
    print(f"\nComparing models: {', '.join(models_to_compare)}")
    print(f"{'Model':<15} {'Organic':<12} {'Paid Search':<12} {'Direct':<12} {'Referral':<12}")
    print("-" * 65)
    
    for model_name in models_to_compare:
        try:
            model = get_model(model_name)
            result = model.calculate_attribution(data)
            
            # Extract attribution scores for key channels
            channel_scores = result.groupby('channel')['attribution_score'].sum()
            
            organic_score = channel_scores.get('organic_search', 0)
            paid_score = channel_scores.get('paid_search', 0)
            direct_score = channel_scores.get('direct', 0)
            referral_score = channel_scores.get('referral', 0)
            
            print(f"{model_name:<15} {organic_score:<12.3f} {paid_score:<12.3f} {direct_score:<12.3f} {referral_score:<12.3f}")
            
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)[:30]}...")


def main():
    """Main function to run the demo."""
    print("🏈 PyAttrScore Football Attribution Demo")
    print("Using sample_data.csv for realistic customer journey analysis")
    print()
    
    # Run football attribution demo
    results, summary = demonstrate_football_with_sample_data()
    
    if results is not None:
        # Run model comparison
        compare_models_with_sample_data()
        
        print("\n" + "=" * 50)
        print("🎉 Demo Complete!")
        print("=" * 50)
        print("💡 Key Insights:")
        print("• Football Attribution provides intuitive role-based insights")
        print("• Each channel gets credit based on their 'position' in the customer journey")
        print("• Channel Impact Score (CIS) fairly distributes conversion credit")
        print("• Team performance metrics help optimize marketing strategies")
        
        print(f"\n📊 Summary Statistics:")
        if summary is not None and not summary.empty:
            total_goals = summary['channel_goals'].sum()
            total_assists = summary['channel_assists'].sum()
            total_passes = summary['channel_passes'].sum()
            
            print(f"   • Total Goals (Conversions): {total_goals:.0f}")
            print(f"   • Total Assists (Setups): {total_assists:.0f}")
            print(f"   • Total Passes (Touchpoints): {total_passes:.0f}")
            print(f"   • Channels Analyzed: {len(summary)}")
        
        print("\n🚀 Next Steps:")
        print("   • Customize channel archetypes for your business")
        print("   • Adjust role weights based on your conversion funnel")
        print("   • Use insights to optimize marketing budget allocation")
        print("   • Run: python football_example.py for more detailed analysis")


if __name__ == "__main__":
    main()