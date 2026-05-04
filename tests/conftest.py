"""Shared pytest fixtures for the pyreite test suite."""
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed numpy and stdlib RNGs before each test for reproducibility."""
    np.random.seed(42)
    random.seed(42)
