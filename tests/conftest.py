# tests/conftest.py
import sys, os

# Compute project root as one level up from tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Prepend it so `import src.*` works in all tests
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
