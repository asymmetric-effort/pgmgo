"""
Fixture generator for src/factors package.

Generates test cases by exercising pgmpy's factor classes,
capturing inputs and expected outputs as fixture data.
"""

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
import numpy as np


def generate() -> list[dict]:
    """Generate test cases for the factors package."""
    test_cases = []

    test_cases.append(_test_discrete_factor_creation())
    test_cases.append(_test_discrete_factor_marginalize())
    test_cases.append(_test_discrete_factor_reduce())
    test_cases.append(_test_discrete_factor_product())
    test_cases.append(_test_tabular_cpd_creation())

    return test_cases


def _test_discrete_factor_creation():
    """Test basic DiscreteFactor creation."""
    phi = DiscreteFactor(["x1", "x2"], [2, 3], [0.5, 0.8, 0.1, 0.0, 0.3, 0.9])

    return {
        "name": "discrete_factor_creation",
        "description": "Create a DiscreteFactor and verify properties",
        "input": {
            "variables": ["x1", "x2"],
            "cardinality": [2, 3],
            "values": [0.5, 0.8, 0.1, 0.0, 0.3, 0.9],
        },
        "expected": {
            "variables": list(phi.variables),
            "cardinality": list(phi.cardinality),
            "values": phi.values.flatten().tolist(),
        },
    }


def _test_discrete_factor_marginalize():
    """Test factor marginalization."""
    phi = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2],
                         list(range(12)))
    phi_marg = phi.marginalize(["x3"], inplace=False)

    return {
        "name": "discrete_factor_marginalize",
        "description": "Marginalize x3 out of a 3-variable factor",
        "input": {
            "variables": ["x1", "x2", "x3"],
            "cardinality": [2, 3, 2],
            "values": list(range(12)),
            "marginalize": ["x3"],
        },
        "expected": {
            "variables": list(phi_marg.variables),
            "cardinality": list(phi_marg.cardinality),
            "values": phi_marg.values.flatten().tolist(),
        },
    }


def _test_discrete_factor_reduce():
    """Test factor reduction (fixing variable values)."""
    phi = DiscreteFactor(["x1", "x2", "x3"], [2, 3, 2],
                         list(range(12)))
    phi_red = phi.reduce([("x1", 0)], inplace=False)

    return {
        "name": "discrete_factor_reduce",
        "description": "Reduce factor by fixing x1=0",
        "input": {
            "variables": ["x1", "x2", "x3"],
            "cardinality": [2, 3, 2],
            "values": list(range(12)),
            "reduce": [["x1", 0]],
        },
        "expected": {
            "variables": list(phi_red.variables),
            "cardinality": list(phi_red.cardinality),
            "values": phi_red.values.flatten().tolist(),
        },
    }


def _test_discrete_factor_product():
    """Test factor product."""
    phi1 = DiscreteFactor(["x1", "x2"], [2, 2], [0.5, 0.8, 0.1, 0.0])
    phi2 = DiscreteFactor(["x2", "x3"], [2, 2], [0.5, 0.8, 0.1, 0.0])
    phi_prod = phi1 * phi2

    return {
        "name": "discrete_factor_product",
        "description": "Multiply two factors with shared variable x2",
        "input": {
            "factor1": {
                "variables": ["x1", "x2"],
                "cardinality": [2, 2],
                "values": [0.5, 0.8, 0.1, 0.0],
            },
            "factor2": {
                "variables": ["x2", "x3"],
                "cardinality": [2, 2],
                "values": [0.5, 0.8, 0.1, 0.0],
            },
        },
        "expected": {
            "variables": list(phi_prod.variables),
            "cardinality": list(phi_prod.cardinality),
            "values": phi_prod.values.flatten().tolist(),
        },
    }


def _test_tabular_cpd_creation():
    """Test TabularCPD creation and validation."""
    cpd = TabularCPD(
        variable="G",
        variable_card=3,
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7, 0.02, 0.2]],
        evidence=["D", "I"],
        evidence_card=[2, 2],
    )

    return {
        "name": "tabular_cpd_creation",
        "description": "Create TabularCPD with evidence and verify",
        "input": {
            "variable": "G",
            "variable_card": 3,
            "values": [[0.3, 0.05, 0.9, 0.5],
                       [0.4, 0.25, 0.08, 0.3],
                       [0.3, 0.7, 0.02, 0.2]],
            "evidence": ["D", "I"],
            "evidence_card": [2, 2],
        },
        "expected": {
            "variable": cpd.variable,
            "variables": list(cpd.variables),
            "cardinality": list(cpd.cardinality),
            "values": cpd.get_values().tolist(),
        },
    }
