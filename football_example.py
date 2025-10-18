"""
Football-Inspired Attribution Model Example

This script demonstrates how to use the Football-Inspired Attribution Model
to analyze marketing channel performance using football analytics concepts.

The example shows:
1. Creating sample customer journey data
2. Configuring the football attribution model
3. Calculating Channel Impact Score (CIS)
4. Analyzing football metrics and channel performance
5. Generating insights for marketing optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from pyattrscore import (
    FootballAttribution, 
    FootballAttributionConfig,
    get_model
)


def create_sample_data():
    """
    Create realistic sample data representing customer journeys
    across multiple marketing channels.
    """
    print("🏈 Creating sample customer journey data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define channels and their typical characteristics
    channels = {
        'organic_search': {'freq': 0.25, 'conversion_rate': 0.15, 'engagement': (20, 60)},
        'paid_search': {'freq': 0.20, 'conversion_rate': 0.12, 'engagement': (15, 45)},
        'social_media': {'freq': 0.15, 'conversion_rate': 0.08, 'engagement': (10, 30)},
        'email': {'freq': 0.15, 'conversion_rate': 0.18, 'engagement': (25, 70)},
        'direct': {'freq': 0.10, 'conversion_rate': 0.35, 'engagement': (30, 90)},
        'referral': {'freq': 0.08, 'conversion_rate': 0.25, 'engagement': (20, 80)},
        'display': {'freq': 0.05, 'conversion_rate': 0.05, 'engagement': (5, 20)},
        'video': {'freq': 0.02, 'conversion_rate': 0.10, 'engagement': (40, 120)}
    }
    
    # Generate customer journeys
    data = []
    touchpoint_id = 1
    
    # Create 100 customer journeys
    for customer_id in range(1, 101):
        # Random journey length (1-8 touchpoints)
        journey_length = np.random.randint(1, 9)
        
        # Generate journey timestamps
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 90))
        journey_dates = []
        current_date = start_date
        
        for i in range(journey_length):
            if i == 0:
                journey_dates.append(current_date)
            else:
                # Add 1-7 days between touchpoints
                days_gap = np.random.randint(1, 8)
                current_date += timedelta(days=days_gap)
                journey_dates.append(current_date)
        
        # Select channels for this journey
        channel_names = list(channels.keys())
        channel_weights = [channels[ch]['freq'] for ch in channel_names]
        
        journey_channels = np.random.choice(
            channel_names, 
            size=journey_length, 
            p=np.array(channel_weights) / sum(channel_weights),
            replace=True
        )
        
        # Determine if this journey converts (30% conversion rate)
        converts = np.random.random() < 0.3
        conversion_touchpoint = journey_length - 1 if converts else -1
        
        # Create touchpoints
        for i in range(journey_length):
            channel = journey_channels[i]
            is_conversion = (i == conversion_touchpoint)
            
            # Generate engagement time based on channel characteristics
            min_eng, max_eng = channels[channel]['engagement']
            engagement_time = np.random.uniform(min_eng, max_eng)
            
            # Generate conversion value if this is a conversion
            conversion_value = None
            if is_conversion:
                conversion_value = np.random.uniform(50, 500)
            
            data.append({
                'user_id': f'customer_{customer_id}',
                'touchpoint_id': f'tp_{touchpoint_id}',
                'channel': channel,
                'timestamp': journey_dates[i],
                'conversion': is_conversion,
                'conversion_value': conversion_value,
                'engagement_time': engagement_time
            })
            
            touchpoint_id += 1
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} touchpoints across {df['user_id'].nunique()} customers")
    print(f"📊 Conversion rate: {(df['conversion'].sum() / df['user_id'].nunique()) * 100:.1f}%")
    
    return df


def demonstrate_football_attribution():
    """
    Demonstrate the Football Attribution model with sample data.
    """
    print("\n" + "="*60)
    print("🏈 FOOTBALL-INSPIRED ATTRIBUTION MODEL DEMO")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Display sample of the data
    print("\n📋 Sample Customer Journey Data:")
    print(data.head(10).to_string(index=False))
    
    # Configure the Football Attribution model
    print("\n⚙️ Configuring Football Attribution Model...")
    
    config = FootballAttributionConfig(
        attribution_window_days=30,
        scorer_weight=0.25,      # Final conversion touchpoint
        assister_weight=0.20,    # Setup touchpoint before conversion
        key_passer_weight=0.15,  # Journey initiator
        most_passes_weight=0.15, # Most frequent engagement
        most_minutes_weight=0.10, # Longest engagement time
        most_dribbles_weight=0.10, # Cold lead revival
        participant_weight=0.05,  # Supporting touchpoint
        baseline_weight=0.1,
        cold_lead_threshold_days=7
    )
    
    print(f"✅ Configuration set with {config.attribution_window_days}-day attribution window")
    print(f"🎯 Role weights: Scorer={config.scorer_weight}, Assister={config.assister_weight}")
    
    # Initialize the model
    model = FootballAttribution(config)
    
    # Calculate attribution
    print("\n🔄 Calculating Football Attribution...")
    try:
        results = model.calculate_attribution(data)
        print(f"✅ Attribution calculated for {len(results)} touchpoints")
    except Exception as e:
        print(f"❌ Error calculating attribution: {e}")
        return
    
    # Display attribution results
    print("\n📊 Attribution Results Sample:")
    sample_results = results[['user_id', 'channel', 'attribution_score', 
                             'football_roles', 'channel_archetype']].head(10)
    print(sample_results.to_string(index=False))
    
    # Generate channel performance summary
    print("\n🏆 Channel Performance Summary:")
    summary = model.get_channel_performance_summary(results)
    
    # Display key metrics
    display_columns = [
        'channel', 'channel_archetype', 'channel_goals', 'channel_assists',
        'channel_passes', 'attribution_score', 'goals_per_100_passes'
    ]
    
    summary_display = summary[display_columns].round(3)
    print(summary_display.to_string(index=False))
    
    return results, summary


def analyze_football_insights(results, summary):
    """
    Analyze and display football-inspired insights from the attribution results.
    """
    print("\n" + "="*60)
    print("⚽ FOOTBALL ANALYTICS INSIGHTS")
    print("="*60)
    
    # Top Scorers (Conversion Closers)
    print("\n🥅 TOP SCORERS (Conversion Closers):")
    top_scorers = summary.nlargest(3, 'channel_goals')[['channel', 'channel_goals', 'channel_archetype']]
    for _, row in top_scorers.iterrows():
        print(f"  {row['channel']:15} | {row['channel_goals']:2.0f} goals | {row['channel_archetype']}")
    
    # Top Assisters (Setup Channels)
    print("\n🎯 TOP ASSISTERS (Setup Channels):")
    top_assisters = summary.nlargest(3, 'channel_assists')[['channel', 'channel_assists', 'channel_archetype']]
    for _, row in top_assisters.iterrows():
        print(f"  {row['channel']:15} | {row['channel_assists']:2.0f} assists | {row['channel_archetype']}")
    
    # Most Active Players (Highest Engagement)
    print("\n🏃 MOST ACTIVE PLAYERS (Highest Engagement):")
    most_active = summary.nlargest(3, 'channel_passes')[['channel', 'channel_passes', 'channel_minutes']]
    for _, row in most_active.iterrows():
        print(f"  {row['channel']:15} | {row['channel_passes']:3.0f} passes | {row['channel_minutes']:6.1f} minutes")
    
    # Efficiency Analysis
    print("\n📈 EFFICIENCY ANALYSIS:")
    summary['conversion_rate'] = (summary['channel_goals'] / summary['channel_passes'] * 100).round(1)
    summary['assist_rate'] = (summary['channel_assists'] / summary['channel_passes'] * 100).round(1)
    
    efficiency = summary.nlargest(3, 'conversion_rate')[['channel', 'conversion_rate', 'assist_rate']]
    for _, row in efficiency.iterrows():
        print(f"  {row['channel']:15} | {row['conversion_rate']:4.1f}% conversion | {row['assist_rate']:4.1f}% assist rate")
    
    # Channel Archetype Performance
    print("\n🏟️ TEAM FORMATION ANALYSIS:")
    archetype_performance = summary.groupby('channel_archetype').agg({
        'channel_goals': 'sum',
        'channel_assists': 'sum',
        'channel_passes': 'sum',
        'attribution_score': 'sum'
    }).round(2)
    
    for archetype, row in archetype_performance.iterrows():
        print(f"  {archetype.upper():12} | Goals: {row['channel_goals']:2.0f} | "
              f"Assists: {row['channel_assists']:2.0f} | "
              f"Attribution: {row['attribution_score']:.2f}")


def demonstrate_example_from_spec():
    """
    Demonstrate the specific example from the specification:
    Organic → Digital Marketing → Referral
    """
    print("\n" + "="*60)
    print("📖 SPECIFICATION EXAMPLE DEMONSTRATION")
    print("="*60)
    print("Journey: Organic Search → Digital Marketing → Referral")
    
    # Create the exact example from the specification
    example_data = pd.DataFrame({
        'user_id': ['customer_example', 'customer_example', 'customer_example'],
        'touchpoint_id': ['tp_1', 'tp_2', 'tp_3'],
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
    
    print("\n📋 Example Journey Data:")
    print(example_data.to_string(index=False))
    
    # Use default configuration
    model = FootballAttribution()
    results = model.calculate_attribution(example_data)
    
    print("\n🏈 Football Role Assignments:")
    for _, row in results.iterrows():
        roles_str = ', '.join(row['football_roles'])
        print(f"  {row['channel']:15} | {roles_str:20} | CIS: {row['attribution_score']:.3f}")
    
    print("\n📊 Channel Contribution Analysis:")
    for _, row in results.iterrows():
        percentage = row['attribution_score'] * 100
        print(f"  {row['channel']:15} | {percentage:5.1f}% contribution")
    
    # Verify the calculation matches the specification
    referral_score = results[results['channel'] == 'referral']['attribution_score'].iloc[0]
    digital_score = results[results['channel'] == 'paid_search']['attribution_score'].iloc[0]
    organic_score = results[results['channel'] == 'organic_search']['attribution_score'].iloc[0]
    
    print(f"\n✅ Results Summary:")
    print(f"  Referral (Closer):  {referral_score:.1%}")
    print(f"  Digital (Assister): {digital_score:.1%}")
    print(f"  Organic (Generator): {organic_score:.1%}")
    
    return results


def create_visualization(summary):
    """
    Create visualizations of the football attribution results.
    """
    try:
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Football-Inspired Attribution Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Channel Goals and Assists
        channels = summary['channel']
        goals = summary['channel_goals']
        assists = summary['channel_assists']
        
        x = np.arange(len(channels))
        width = 0.35
        
        ax1.bar(x - width/2, goals, width, label='Goals (Conversions)', color='#2E8B57')
        ax1.bar(x + width/2, assists, width, label='Assists (Setups)', color='#4169E1')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Count')
        ax1.set_title('Goals vs Assists by Channel')
        ax1.set_xticks(x)
        ax1.set_xticklabels(channels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Attribution Score Distribution
        ax2.pie(summary['attribution_score'], labels=summary['channel'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Attribution Score Distribution')
        
        # 3. Channel Efficiency (Goals per 100 Passes)
        efficiency = summary['goals_per_100_passes']
        bars = ax3.bar(channels, efficiency, color='#FF6347')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Goals per 100 Passes')
        ax3.set_title('Channel Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 4. Channel Archetype Performance
        archetype_data = summary.groupby('channel_archetype')['attribution_score'].sum()
        colors = ['#FFD700', '#32CD32', '#FF69B4', '#87CEEB']
        wedges, texts, autotexts = ax4.pie(archetype_data.values, labels=archetype_data.index, 
                                          autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title('Performance by Channel Archetype')
        
        plt.tight_layout()
        plt.savefig('football_attribution_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard saved as 'football_attribution_dashboard.png'")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        print("📊 Matplotlib not available. Skipping visualizations.")
        print("    Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"📊 Error creating visualizations: {e}")


def main():
    """
    Main function to run the complete Football Attribution demonstration.
    """
    print("🏈 Welcome to the Football-Inspired Attribution Model Demo!")
    print("This demo shows how marketing channels work together like a football team.")
    
    try:
        # Run the main demonstration
        results, summary = demonstrate_football_attribution()
        
        # Analyze insights
        analyze_football_insights(results, summary)
        
        # Show the specification example
        demonstrate_example_from_spec()
        
        # Create visualizations
        create_visualization(summary)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("Key Takeaways:")
        print("• Each marketing channel plays a specific role like football positions")
        print("• Channel Impact Score (CIS) fairly distributes conversion credit")
        print("• Football metrics provide intuitive performance insights")
        print("• Role-based attribution helps optimize channel strategies")
        print("\nNext Steps:")
        print("• Customize role weights based on your business model")
        print("• Map your channels to appropriate archetypes")
        print("• Use insights to optimize budget allocation")
        print("• Monitor team performance over time")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your data and configuration.")


if __name__ == "__main__":
    main()