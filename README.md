# PyAttrScore - Python Attribution Modeling Package

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pyattrscore.svg)](https://badge.fury.io/py/pyattrscore)
[![Build Status](https://github.com/pyattrscore/pyattrscore/workflows/CI/badge.svg)](https://github.com/pyattrscore/pyattrscore/actions)
[![Coverage Status](https://codecov.io/gh/pyattrscore/pyattrscore/branch/main/graph/badge.svg)](https://codecov.io/gh/pyattrscore/pyattrscore)

PyAttrScore is a production-grade Python package designed to calculate marketing attribution scores using multiple models. It includes validation, logging, error handling, and comprehensive testing modules, making it ready for integration into analytics pipelines to measure channel effectiveness.

## 🚀 Features

- **Multiple Attribution Models**: First Touch, Last Touch, Linear, Time Decay (Exponential & Linear), U-Shaped, and Windowed First Touch
- **Production Ready**: Comprehensive error handling, logging, and validation
- **Flexible Configuration**: YAML-based configuration with customizable parameters
- **Data Validation**: Built-in Pydantic models for robust data validation
- **Comprehensive Testing**: 90%+ test coverage with pytest
- **Easy Integration**: Simple API for integration into existing analytics pipelines
- **Performance Optimized**: Efficient algorithms for large-scale data processing

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
    get_model
)

# Method 1: Direct instantiation
linear_model = LinearAttribution(config)
results_linear = linear_model.calculate_attribution(data)

# Method 2: Using model factory
decay_model = get_model('exponential_decay', config)
results_decay = decay_model.calculate_attribution(data)

# Method 3: U-Shaped with custom weights
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

## ⚙️ Configuration

### Using Configuration Objects

```python
from pyattrscore import AttributionConfig

config = AttributionConfig(
    attribution_window_days=30,
    decay_rate=0.6,
    include_non_converting_paths=False
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
```

```python
import yaml
from pyattrscore import AttributionConfig

with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = AttributionConfig(**config_dict['global'])
```

## 📈 Advanced Usage

### Batch Processing Multiple Users

```python
import pandas as pd
from pyattrscore import LinearAttribution

# Large dataset with multiple users
large_data = pd.DataFrame([
    # User 1 journey
    {'user_id': 'user_001', 'touchpoint_id': 'tp_001', 'channel': 'email', 
     'timestamp': datetime(2023, 1, 1), 'conversion': False},
    {'user_id': 'user_001', 'touchpoint_id': 'tp_002', 'channel': 'search', 
     'timestamp': datetime(2023, 1, 2), 'conversion': True, 'conversion_value': 100.0},
    
    # User 2 journey
    {'user_id': 'user_002', 'touchpoint_id': 'tp_003', 'channel': 'social', 
     'timestamp': datetime(2023, 1, 1), 'conversion': False},
    {'user_id': 'user_002', 'touchpoint_id': 'tp_004', 'channel': 'email', 
     'timestamp': datetime(2023, 1, 3), 'conversion': True, 'conversion_value': 200.0},
])

model = LinearAttribution()
results = model.calculate_attribution(large_data)

# Analyze results by channel
channel_performance = results.groupby('channel').agg({
    'attribution_score': 'sum',
    'attribution_value': 'sum',
    'user_id': 'nunique'
}).round(4)

print(channel_performance)
```

### Model Comparison

```python
from pyattrscore import get_model, list_models

# Compare multiple models
models_to_compare = ['first_touch', 'last_touch', 'linear', 'u_shaped']
results_comparison = {}

for model_name in models_to_compare:
    model = get_model(model_name, config)
    results = model.calculate_attribution(data)
    
    # Aggregate by channel
    channel_attribution = results.groupby('channel')['attribution_score'].sum()
    results_comparison[model_name] = channel_attribution

comparison_df = pd.DataFrame(results_comparison).fillna(0)
print(comparison_df)
```

### Custom Logging Configuration

```python
from pyattrscore import configure_logging

# Configure logging
configure_logging(
    level="DEBUG",
    log_file="attribution_analysis.log",
    json_format=True
)

# Now all PyAttrScore operations will be logged
model = LinearAttribution()
results = model.calculate_attribution(data)
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

# Example output
print(results.head())
#    user_id touchpoint_id      channel  ... attribution_percentage model_name  attribution_value
# 0  user_001       tp_001        email  ...                  33.33     Linear              33.33
# 1  user_001       tp_002  social_media ...                  33.33     Linear              33.33
# 2  user_001       tp_003       search  ...                  33.34     Linear              33.34
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyattrscore --cov-report=html

# Run specific test file
pytest tests/test_linear.py

# Run with verbose output
pytest -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/pyattrscore/pyattrscore.git
cd pyattrscore
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest
black .
flake8
mypy pyattrscore
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://pyattrscore.readthedocs.io/](https://pyattrscore.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/pyattrscore/pyattrscore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pyattrscore/pyattrscore/discussions)

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), and [pydantic](https://pydantic-docs.helpmanual.io/)
- Inspired by marketing attribution research and industry best practices
- Thanks to all contributors and the open-source community

## 📚 Citation

If you use PyAttrScore in your research, please cite:

```bibtex
@software{pyattrscore,
  title={PyAttrScore: Python Attribution Modeling Package},
  author={PyAttrScore Team},
  year={2023},
  url={https://github.com/pyattrscore/pyattrscore}
}
```

---

**Made with ❤️ for the marketing analytics community**