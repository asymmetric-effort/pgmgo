"""
Extended fixture generator for src/ci_tests package.

Covers ChiSquare, GSq, FisherZ on same discrete data,
Pearsonr on continuous data, and multiple conditioning sets.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pgmpy.ci_tests import ChiSquare, GSq, FisherZ, Pearsonr


def generate() -> list[dict]:
    """Generate extended CI test cases."""
    test_cases = []

    data = _build_discrete_data()
    cdata = _build_continuous_data()

    # ChiSquare, GSq, FisherZ on same discrete data pair (X, Y dependent)
    test_cases.append(_test_chi_square_dep(data))
    test_cases.append(_test_gsq_dep(data))
    test_cases.append(_test_fisher_z_dep(data))

    # Same tests on independent pair (X, Z)
    test_cases.append(_test_chi_square_indep(data))
    test_cases.append(_test_gsq_indep(data))
    test_cases.append(_test_fisher_z_indep(data))

    # Pearsonr on continuous data
    test_cases.append(_test_pearsonr_dependent(cdata))
    test_cases.append(_test_pearsonr_independent(cdata))

    # Multiple conditioning sets
    test_cases.append(_test_chi_square_cond_z(data))
    test_cases.append(_test_gsq_cond_z(data))

    return test_cases


def _build_discrete_data():
    """Build synthetic discrete data: X-Y dependent, X-Z independent."""
    np.random.seed(42)
    n = 2000
    X = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    Y = (X + np.random.choice([0, 1], size=n, p=[0.7, 0.3])) % 2
    Z = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def _build_continuous_data():
    """Build synthetic continuous data: X-Y correlated, X-Z independent."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n)
    Y = 0.7 * X + 0.3 * np.random.randn(n)
    Z = np.random.randn(n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def _serialize_data(data, columns):
    return {
        "data_columns": columns,
        "data_rows": data[columns].values.tolist(),
    }


def _test_chi_square_dep(data):
    test = ChiSquare(data=data)
    result = test(X="X", Y="Y", Z=[], significance_level=0.05)
    return {
        "name": "chi_square_extended_dep",
        "description": "ChiSquare on dependent X-Y",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_gsq_dep(data):
    test = GSq(data=data)
    result = test(X="X", Y="Y", Z=[], significance_level=0.05)
    return {
        "name": "gsq_dependent",
        "description": "G-squared on dependent X-Y",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_fisher_z_dep(data):
    test = FisherZ(data=data)
    result = test(X="X", Y="Y", Z=[], significance_level=0.05)
    return {
        "name": "fisher_z_dependent",
        "description": "Fisher-Z on dependent X-Y (discrete data)",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "independent": bool(result),
        },
    }


def _test_chi_square_indep(data):
    test = ChiSquare(data=data)
    result = test(X="X", Y="Z", Z=[], significance_level=0.05)
    return {
        "name": "chi_square_extended_indep",
        "description": "ChiSquare on independent X-Z",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Z", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_gsq_indep(data):
    test = GSq(data=data)
    result = test(X="X", Y="Z", Z=[], significance_level=0.05)
    return {
        "name": "gsq_independent",
        "description": "G-squared on independent X-Z",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Z", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_fisher_z_indep(data):
    test = FisherZ(data=data)
    result = test(X="X", Y="Z", Z=[], significance_level=0.05)
    return {
        "name": "fisher_z_independent",
        "description": "Fisher-Z on independent X-Z",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Z", "z": [],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "independent": bool(result),
        },
    }


def _test_pearsonr_dependent(cdata):
    test = Pearsonr(data=cdata)
    result = test(X="X", Y="Y", Z=[], significance_level=0.05)
    return {
        "name": "pearsonr_dependent",
        "description": "Pearsonr on correlated continuous X-Y",
        "input": {
            **_serialize_data(cdata, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": [],
            "significance_level": 0.05,
            "continuous": True,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "independent": bool(result),
        },
    }


def _test_pearsonr_independent(cdata):
    test = Pearsonr(data=cdata)
    result = test(X="X", Y="Z", Z=[], significance_level=0.05)
    return {
        "name": "pearsonr_independent",
        "description": "Pearsonr on independent continuous X-Z",
        "input": {
            **_serialize_data(cdata, ["X", "Y", "Z"]),
            "x": "X", "y": "Z", "z": [],
            "significance_level": 0.05,
            "continuous": True,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "independent": bool(result),
        },
    }


def _test_chi_square_cond_z(data):
    """ChiSquare: X-Y conditioned on Z."""
    test = ChiSquare(data=data)
    result = test(X="X", Y="Y", Z=["Z"], significance_level=0.05)
    return {
        "name": "chi_square_cond_z",
        "description": "ChiSquare X-Y | Z",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": ["Z"],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }


def _test_gsq_cond_z(data):
    """GSq: X-Y conditioned on Z."""
    test = GSq(data=data)
    result = test(X="X", Y="Y", Z=["Z"], significance_level=0.05)
    return {
        "name": "gsq_cond_z",
        "description": "G-squared X-Y | Z",
        "input": {
            **_serialize_data(data, ["X", "Y", "Z"]),
            "x": "X", "y": "Y", "z": ["Z"],
            "significance_level": 0.05,
        },
        "expected": {
            "statistic": float(test.statistic_),
            "p_value": float(test.p_value_),
            "dof": int(test.dof_),
            "independent": bool(result),
        },
    }
