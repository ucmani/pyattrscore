"""
Setup script for PyAttrScore package
"""

from setuptools import setup, find_packages
import os
import sys

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    """Get version from package __init__.py"""
    version_file = os.path.join(this_directory, 'pyattrscore', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError('Unable to find version string.')

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError('PyAttrScore requires Python 3.8 or later.')

setup(
    name='pyattrscore',
    version=get_version(),
    author='Mani Gidijala',
    description='Python Attribution Modeling Package for Marketing Analytics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_data={
        'pyattrscore': ['config.yaml'],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Typing :: Typed',
    ],
    keywords=[
        'attribution',
        'marketing',
        'analytics',
        'conversion',
        'touchpoint',
        'customer-journey',
        'marketing-mix-modeling',
        'data-science',
        'machine-learning'
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'pydantic>=1.8.0,<3.0.0',
        'pyyaml>=5.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'pytest-xdist>=2.0.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'isort>=5.0.0',
            'pre-commit>=2.0.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinx-autodoc-typehints>=1.12.0',
            'myst-parser>=0.15.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
        ],
        'performance': [
            'numba>=0.50.0',
            'dask[dataframe]>=2021.0.0',
        ],
        'all': [
            # Include all optional dependencies
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'pytest-xdist>=2.0.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'isort>=5.0.0',
            'pre-commit>=2.0.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinx-autodoc-typehints>=1.12.0',
            'myst-parser>=0.15.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0',
            'numba>=0.50.0',
            'dask[dataframe]>=2021.0.0',
        ],
    },
    zip_safe=False,
)