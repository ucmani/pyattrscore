# PyAttrScore - Python Attribution Modeling Package

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pyattrscore.svg)](https://badge.fury.io/py/pyattrscore)
[![Build Status](https://github.com/pyattrscore/pyattrscore/workflows/CI/badge.svg)](https://github.com/pyattrscore/pyattrscore/actions)
[![Coverage Status](https://codecov.io/gh/pyattrscore/pyattrscore/branch/main/graph/badge.svg)](https://codecov.io/gh/pyattrscore/pyattrscore)

PyAttrScore is a Python package designed to calculate marketing attribution scores using multiple models. It includes validation, logging, error handling, and comprehensive testing modules, making it ready for integration into analytics pipelines to measure channel effectiveness.

## 🚀 Features

- **Multiple Attribution Models**: First Touch, Last Touch, Linear, Time Decay (Exponential & Linear), U-Shaped, Windowed First Touch, and **Football-Inspired Attribution**
- **🏈 Football Attribution Model**: Treats marketing channels as football players with distinct roles (Scorer, Assister, Key Passer, Most Passes, Most Minutes, Most Dribbles, Participant) and calculates a Channel Impact Score (CIS) based on role weights
- **Role-Based Attribution**: Assigns credit based on channel roles in the customer journey, providing intuitive team-based insights
- **Channel Archetypes**: Classifies channels into Generator, Assister, Closer, and Participant archetypes for strategic analysis
- **Configurable Role Weights**: Customize the impact of each football role on the CIS calculation
- **Comprehensive Channel Metrics**: Includes goals, assists, key passes, engagement time, expected goals, and more
- **Production Ready**: Robust error handling, logging, and validation for reliable use in analytics pipelines
- **Flexible Configuration**: YAML-based and programmatic configuration options for all models
- **Data Validation**: Built-in Pydantic models ensure input data integrity
- **Comprehensive Testing**: Over 90% test coverage with pytest for confidence in results
- **Easy Integration**: Simple API design for seamless integration into existing workflows
- **Performance Optimized**: Efficient algorithms designed for large-scale data processing
- **Advanced Analytics**: Team performance summaries, role-based channel analysis, and batch processing support

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install pyattrscore
```

### From Source

```bash
git clone https://github.com/pyattrscore/pyattrscore.git
cd pyattrscore
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/pyattrscore/pyattrscore.git
cd pyattrscore
pip install -e ".[dev]"
```

## 🏃‍♂️ Quick Start

### 🏈 Football Attribution Demo

Experience the revolutionary Football-Inspired Attribution model:

```bash
# Run the football attribution demo
python main.py --football

# Compare all attribution models
python main.py --compare

# Run detailed football analysis
python football_example.py

# Use sample data
python main.py --football --data sample_data.csv
```

### Basic Usage

```python
import pandas as pd
from datetime import datetime
from pyattrscore import FirstTouchAttribution, AttributionConfig

# Sample touchpoint data
data = pd.DataFrame([
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

# Initialize attribution model
config = AttributionConfig(attribution_window_days=30)
model = FirstTouchAttribution(config)

# Calculate attribution
results = model.calculate_attribution(data)
print(results)
```

### Using Different Models

```python
from pyattrscore import (
    LinearAttribution,
    ExponentialDecayAttribution,
    UShapedAttribution,
    FootballAttribution,
    get_model
)

# Method 1: Direct instantiation
linear_model = LinearAttribution(config)
results_linear = linear_model.calculate_attribution(data)

# Method 2: Using model factory
decay_model = get_model('exponential_decay', config)
results_decay = decay_model.calculate_attribution(data)

# Method 3: Football Attribution
football_model = get_model('football')
results_football = football_model.calculate_attribution(data)

# Method 4: U-Shaped with custom weights
u_shaped_model = UShapedAttribution(
    config, 
    first_touch_weight=0.3, 
    last_touch_weight=0.5
)
results_u_shaped = u_shaped_model.calculate_attribution(data)
```

## 📊 Attribution Models

### 1. First Touch Attribution
Assigns 100% credit to the first touchpoint in the customer journey.

```python
from pyattrscore import FirstTouchAttribution

model = FirstTouchAttribution()
results = model.calculate_attribution(data)
```

**Use Cases:**
- Understanding awareness channel effectiveness
- Short sales cycles
- Top-of-funnel optimization

### 2. Last Touch Attribution
Assigns 100% credit to the last touchpoint before conversion.

```python
from pyattrscore import LastTouchAttribution

model = LastTouchAttribution()
results = model.calculate_attribution(data)
```

**Use Cases:**
- Understanding closing channel effectiveness
- Bottom-of-funnel optimization
- Direct response campaigns

### 3. Linear Attribution
Distributes credit equally among all touchpoints within the attribution window.

```python
from pyattrscore import LinearAttribution, AttributionConfig

config = AttributionConfig(attribution_window_days=30)
model = LinearAttribution(config)
results = model.calculate_attribution(data)
```

**Use Cases:**
- Balanced view of customer journey
- Multi-touch attribution analysis
- Understanding overall channel contribution

### 4. Time Decay Attribution
Credits touchpoints based on their proximity to conversion.

```python
from pyattrscore import ExponentialDecayAttribution, LinearDecayAttribution

# Exponential decay
config = AttributionConfig(attribution_window_days=30, decay_rate=0.5)
exp_model = ExponentialDecayAttribution(config)
results_exp = exp_model.calculate_attribution(data)

# Linear decay
linear_decay_model = LinearDecayAttribution(config)
results_linear_decay = linear_decay_model.calculate_attribution(data)
```

**Use Cases:**
- Understanding recency impact
- Time-sensitive attribution analysis
- Weighting recent touchpoints higher

### 5. U-Shaped Attribution
Assigns higher credit to first and last touchpoints, distributing remainder to middle touchpoints.

```python
from pyattrscore import UShapedAttribution

model = UShapedAttribution(
    first_touch_weight=0.4,
    last_touch_weight=0.4
    # Remaining 20% distributed to middle touchpoints
)
results = model.calculate_attribution(data)
```

**Use Cases:**
- Balancing awareness and conversion touchpoints
- Multi-touch customer journeys
- Understanding nurturing touchpoint value

### 6. Windowed First Touch Attribution
Assigns 100% credit to the first touchpoint within the attribution window.

```python
from pyattrscore import WindowedFirstTouchAttribution, AttributionConfig

config = AttributionConfig(attribution_window_days=14)
model = WindowedFirstTouchAttribution(config)
results = model.calculate_attribution(data)
```

**Use Cases:**
- Understanding recent awareness drivers
- Time-bounded first touch analysis
- Focusing on relevant touchpoints

### 7. 🏈 Football-Based Attribution Model (Improved Definition)

The Football-Based Attribution Model applies a football (soccer) metaphor to marketing attribution, treating marketing channels as players on a football team. Each channel is assigned a role based on its contribution to the customer journey, and a Channel Impact Score (CIS) is calculated to quantify its overall impact.

```python
from pyattrscore import FootballAttribution, FootballAttributionConfig

# Configure the football model
config = FootballAttributionConfig(
    attribution_window_days=30,
    scorer_weight=0.25,      # Final conversion touchpoint
    assister_weight=0.20,    # Setup touchpoint before conversion
    key_passer_weight=0.15,  # Journey initiator
    most_passes_weight=0.14, # Most frequent engagement
    most_minutes_weight=0.10, # Longest engagement time
    most_dribbles_weight=0.10, # Cold lead revival
    participant_weight=0.06,  # Supporting touchpoint
    baseline_weight=0.1,
    cold_lead_threshold_days=7
)

model = FootballAttribution(config)
results = model.calculate_attribution(data)

# Get team performance summary
summary = model.get_channel_performance_summary(results)
print(summary)
```

### Football Roles and Their Marketing Analogies

- **Scorer**: The final touchpoint that directly leads to conversion, analogous to the striker who scores the goal.
- **Assister**: The touchpoint immediately preceding the conversion, setting up the "goal," similar to a midfielder providing an assist.
- **Key Passer**: The journey initiator, the first touchpoint that starts the conversion build-up, like a defender or playmaker starting the play.
- **Most Passes**: The channel with the highest frequency of engagement, representing consistent involvement.
- **Most Minutes**: The channel with the longest engagement time, indicating sustained interaction.
- **Most Dribbles**: The channel that revives cold leads, re-engaging users after inactivity.
- **Participant**: Supporting touchpoints that contribute but do not fit the above roles.

### Channel Archetypes

Channels are classified into archetypes based on their typical marketing role:

- **Generator**: Creates awareness and initiates plays (e.g., Organic Search, Social Media).
- **Assister**: Nurtures and sets up conversions (e.g., Email, Paid Search).
- **Closer**: Finishes conversions (e.g., Direct, Referral).
- **Participant**: Supporting roles that assist the team.

### Channel Impact Score (CIS) Formula

The CIS quantifies the contribution of each channel by combining a baseline weight with weighted role contributions:

```
CIS = baseline_weight + (1 - baseline_weight) × Σ(role_weight × role_indicator)
```

Where:

- `baseline_weight` is a minimum credit assigned to all touchpoints.
- `role_weight` is the predefined weight for each football role.
- `role_indicator` is 1 if the channel has the role, 0 otherwise.

This formula ensures that channels with key roles receive higher attribution while all channels receive some baseline credit.


## ⚙️ Configuration

### Using Configuration Objects

```python
from pyattrscore import AttributionConfig, FootballAttributionConfig

# Standard configuration
config = AttributionConfig(
    attribution_window_days=30,
    decay_rate=0.6,
    include_non_converting_paths=False
)

# Football-specific configuration
football_config = FootballAttributionConfig(
    attribution_window_days=30,
    scorer_weight=0.25,
    assister_weight=0.20,
    baseline_weight=0.1,
    channel_archetypes={
        'organic_search': 'generator',
        'paid_search': 'assister',
        'direct': 'closer',
        'referral': 'closer'
    }
)
```

### Using YAML Configuration

```yaml
# config.yaml
global:
  attribution_window_days: 30
  log_level: "INFO"

models:
  linear:
    use_attribution_window: true
  
  exponential_decay:
    decay_rate: 0.5
    use_attribution_window: true
    
  football:
    role_weights:
      scorer_weight: 0.25
      assister_weight: 0.20
      key_passer_weight: 0.15
    baseline_weight: 0.1
    cold_lead_threshold_days: 7
    channel_archetypes:
      organic_search: "generator"
      paid_search: "assister"
      direct: "closer"
```

```python
import yaml
from pyattrscore import AttributionConfig, FootballAttributionConfig

with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = AttributionConfig(**config_dict['global'])
football_config = FootballAttributionConfig(**config_dict['models']['football'])
```

## 📈 Advanced Usage

### Football Attribution Analysis

```python
from pyattrscore import FootballAttribution
import pandas as pd

# Load your data
data = pd.read_csv('sample_data.csv')

# Initialize football model
model = FootballAttribution()
results = model.calculate_attribution(data)

# Analyze team performance
summary = model.get_channel_performance_summary(results)

# Top performers
print("🥅 Top Scorers (Closers):")
top_scorers = summary.nlargest(3, 'channel_goals')
print(top_scorers[['channel', 'channel_goals', 'channel_archetype']])

print("\n🎯 Top Assisters (Setup Channels):")
top_assisters = summary.nlargest(3, 'channel_assists')
print(top_assisters[['channel', 'channel_assists', 'channel_archetype']])

# Team formation analysis
print("\n🏟️ Team Formation Performance:")
archetype_performance = summary.groupby('channel_archetype').agg({
    'channel_goals': 'sum',
    'channel_assists': 'sum',
    'attribution_score': 'sum'
}).round(2)
print(archetype_performance)
```

### Model Comparison

```python
from pyattrscore import get_model, list_models

# Compare multiple models including football
models_to_compare = ['first_touch', 'last_touch', 'linear', 'u_shaped', 'football']
results_comparison = {}

for model_name in models_to_compare:
    model = get_model(model_name, config)
    results = model.calculate_attribution(data)
    
    # Aggregate by channel
    channel_attribution = results.groupby('channel')['attribution_score'].sum()
    results_comparison[model_name] = channel_attribution

comparison_df = pd.DataFrame(results_comparison).fillna(0)
print(comparison_df)

# Football-specific analysis
if 'football' in models_to_compare:
    football_model = get_model('football')
    football_results = football_model.calculate_attribution(data)
    team_summary = football_model.get_channel_performance_summary(football_results)
    print("\n🏈 Team Performance Summary:")
    print(team_summary[['channel', 'channel_archetype', 'channel_goals', 'channel_assists']])
```

### Batch Processing Multiple Users

```python
import pandas as pd
from pyattrscore import FootballAttribution

# Large dataset with multiple users
large_data = pd.DataFrame([
    # User 1 journey
    {'user_id': 'user_001', 'touchpoint_id': 'tp_001', 'channel': 'email', 
     'timestamp': datetime(2023, 1, 1), 'conversion': False, 'engagement_time': 30.0},
    {'user_id': 'user_001', 'touchpoint_id': 'tp_002', 'channel': 'search', 
     'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 100.0, 'engagement_time': 60.0},
    
    # User 2 journey
    {'user_id': 'user_002', 'touchpoint_id': 'tp_003', 'channel': 'social', 
     'timestamp': datetime(2023, 1, 1), 'conversion': False, 'engagement_time': 25.0},
    {'user_id': 'user_002', 'touchpoint_id': 'tp_004', 'channel': 'email', 
     'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 200.0, 'engagement_time': 45.0},
])

model = FootballAttribution()
results = model.calculate_attribution(large_data)

# Analyze results by channel
channel_performance = results.groupby('channel').agg({
    'attribution_score': 'sum',
    'attribution_value': 'sum',
    'user_id': 'nunique',
    'channel_goals': 'first',
    'channel_assists': 'first'
}).round(4)

print(channel_performance)
```

## 🔧 Data Requirements

### Required Columns

Your input DataFrame must contain these columns:

- `user_id` (str): Unique identifier for each user/customer
- `touchpoint_id` (str): Unique identifier for each touchpoint
- `channel` (str): Marketing channel name (e.g., 'email', 'search', 'social')
- `timestamp` (datetime): When the touchpoint occurred

### Optional Columns

- `conversion` (bool): Whether this touchpoint led to a conversion
- `conversion_value` (float): Monetary value of the conversion
- `engagement_time` (float): Time spent on the touchpoint (recommended for Football Attribution)

### Sample Data File

Use the provided `sample_data.csv` for testing:

```python
import pandas as pd
from pyattrscore import FootballAttribution

# Load sample data
data = pd.read_csv('sample_data.csv')
print(data.head())

# Run football attribution
model = FootballAttribution()
results = model.calculate_attribution(data)
```

### Data Validation

PyAttrScore automatically validates your data:

```python
from pyattrscore.exceptions import InvalidInputError

try:
    results = model.calculate_attribution(invalid_data)
except InvalidInputError as e:
    print(f"Data validation failed: {e}")
    print(f"Invalid fields: {e.invalid_fields}")
```

## 📊 Output Format

Attribution results include:

```python
# Standard columns
results.columns
# ['user_id', 'touchpoint_id', 'channel', 'timestamp', 'conversion',
#  'attribution_score', 'attribution_percentage', 'model_name', 'attribution_value']

# Football-specific columns (when using FootballAttribution)
# ['football_roles', 'channel_archetype', 'channel_goals', 'channel_assists', 
#  'channel_passes', 'channel_minutes', 'channel_expected_goals']

# Example output
print(results.head())
#    user_id touchpoint_id      channel  ... football_roles channel_archetype
# 0  user_001       tp_001        email  ...   [assister]        assister
# 1  user_001       tp_002  social_media ...   [key_passer]      generator
# 2  user_001       tp_003       search  ...   [scorer]          closer
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyattrscore --cov-report=html

# Run football attribution tests
pytest tests/test_football.py

# Run with verbose output
pytest -v
```

## 🏈 Football Attribution Examples

### Example 1: Specification Example
```python
# The classic example from the specification
data = pd.DataFrame({
    'user_id': ['customer_1', 'customer_1', 'customer_1'],
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

model = FootballAttribution()
results = model.calculate_attribution(data)

# Expected results:
# Referral (Closer): ~39%
# Paid Search (Assister): ~26%  
# Organic Search (Generator): ~35%
```

### Example 2: Multi-Customer Analysis
```python
# Run the comprehensive example
python football_example.py

# This will show:
# - Role assignments for each touchpoint
# - Channel performance metrics
# - Team formation analysis
# - Football analytics insights
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), and [pydantic](https://pydantic-docs.helpmanual.io/)
- Inspired by marketing attribution research and football analytics
