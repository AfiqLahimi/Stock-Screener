"""Tests run offline on the built-in synthetic data (DEMO_DATA=1); no network needed."""
import os
import sys

os.environ["DEMO_DATA"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
