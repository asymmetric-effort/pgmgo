"""
Fixture generator for src/ci_tests package.

Generates test cases by exercising pgmpy's ChiSquare CI test on synthetic data
where X-Y are dependent and X-Z are independent, capturing test statistics,
p-values, and independence results.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pgmpy.ci_tests import ChiSquare


def generate() -> list[dict]:
    """Generate test cases for CI tests."""
    test_cases = []

    data = _build_data()

    test_cases.append(_test_chi_square_dependent(data))
    test_cases.append(_test_chi_square_independent(data))
    test_cases.append(_test_chi_square_conditional(data))

    return test_cases


def _build_data():
    """Build synthetic data: X-Y dependent, X-Z independent."""
    np.random.seed(42)
    n = 2000

    # X and Y are dependent (correlated).
    X = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    Y = (X + np.random.choice([0, 1], size=n, p=[0.7, 0.3])) % 2
    # Z is independent of X and Y.
    Z = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def _test_chi_square_dependent(data):
    """Test chi_square on X and Y (dependent pair)."""
    test = ChiSquare(data=data)
    result = test(X="X", Y="Y", Z=[], significance_level=0.05)

    return {
        "name": "chi_square_dependent",
        "description": "Chi-square test on dependent pair X-Y (no conditioning)",
        "input": {
            "x": "X",
            "y": "Y",
            "z": [],
            "significance_level": 0.05,
            "data_columns": ["X", "Y", "Z"],
            "data_rows": data[["X", "Y", "Z"]].values.tolist(),
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_chi_square_independent(data):
    """Test chi_square on X and Z (independent pair)."""
    test = ChiSquare(data=data)
    result = test(X="X", Y="Z", Z=[], significance_level=0.05)

    return {
        "name": "chi_square_independent",
        "description": "Chi-square test on independent pair X-Z (no conditioning)",
        "input": {
            "x": "X",
            "y": "Z",
            "z": [],
            "significance_level": 0.05,
            "data_columns": ["X", "Y", "Z"],
            "data_rows": data[["X", "Y", "Z"]].values.tolist(),
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_chi_square_conditional(data):
    """Test chi_square on X and Y conditioned on Z."""
    test = ChiSquare(data=data)
    result = test(X="X", Y="Y", Z=["Z"], significance_level=0.05)

    return {
        "name": "chi_square_conditional",
        "description": "Chi-square test on X-Y conditioned on Z",
        "input": {
            "x": "X",
            "y": "Y",
            "z": ["Z"],
            "significance_level": 0.05,
            "data_columns": ["X", "Y", "Z"],
            "data_rows": data[["X", "Y", "Z"]].values.tolist(),
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }
