"""Pytest configuration to add src to path."""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))