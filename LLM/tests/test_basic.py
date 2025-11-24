"""Simple test to verify pytest setup."""
import sys
import pytest

def test_python_version():
    """Test Python version is 3.8 or higher."""
    assert sys.version_info >= (3, 8)

def test_imports():
    """Test basic imports work."""
    try:
        import numpy
        import pandas
        import torch
        assert True
    except ImportError as e:
        pytest.skip(f"Optional dependency not installed: {e}")

def test_basic_math():
    """Test basic functionality."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
