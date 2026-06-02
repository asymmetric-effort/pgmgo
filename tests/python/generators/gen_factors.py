"""
Fixture generator for src/factors package.

Generates test cases by exercising pgmpy's factor classes,
capturing inputs and expected outputs as fixture data.
"""

from pgmpy.factors.discrete import DiscreteFactor, JointProbabilityDistribution, TabularCPD
import numpy as np


def _factor_to_dict(phi):
    """Convert a factor to a deterministic dict representation.

    We record variable order as-is but sort the variable list separately
    for consistent comparisons. The values array follows the factor's
    internal ordering which is consistent within a single pgmpy version.
    """
    variables = list(phi.variables)
    cardinality = [int(c) for c in phi.cardinality]
    values = phi.values.flatten().tolist()
    # Sort variables and their cardinalities together for canonical output
    paired = sorted(zip(variables, cardinality))
    return {
        "variables": [v for v, _ in paired],
        "cardinality": [c for _, c in paired],
        "values": sorted(values),  # sort values for order-independent comparison
    }


def generate() -> list[dict]:
    """Generate test cases for the factors package."""
    test_cases = []

    test_cases.append(_test_discrete_factor_creation())
    test_cases.append(_test_discrete_factor_marginalize())
    test_cases.append(_test_discrete_factor_reduce())
    test_cases.append(_test_discrete_factor_product())
    test_cases.append(_test_tabular_cpd_creation())
    test_cases.append(_test_joint_probability_distribution())

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
        "expected": _factor_to_dict(phi),
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
        "expected": _factor_to_dict(phi_marg),
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
        "expected": _factor_to_dict(phi_red),
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
        "expected": _factor_to_dict(phi_prod),
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
            "variables": sorted(list(cpd.variables)),
            "cardinality": [int(c) for c in cpd.cardinality],
            "values": cpd.get_values().tolist(),
        },
    }


def _test_joint_probability_distribution():
    """Test JointProbabilityDistribution creation and marginal computation."""
    jpd = JointProbabilityDistribution(
        ["X", "Y"], [2, 2], [0.3, 0.1, 0.2, 0.4],
    )

    marginal_x = jpd.marginal_distribution(["X"], inplace=False)
    marginal_y = jpd.marginal_distribution(["Y"], inplace=False)

    return {
        "name": "joint_probability_distribution",
        "description": "Create 2-variable JPD and compute marginals",
        "input": {
            "variables": ["X", "Y"],
            "cardinality": [2, 2],
            "values": [0.3, 0.1, 0.2, 0.4],
        },
        "expected": {
            "variables": sorted(list(jpd.variables)),
            "cardinality": [int(c) for c in jpd.cardinality],
            "values": jpd.values.flatten().tolist(),
            "marginal_x": {
                "variables": sorted(list(marginal_x.variables)),
                "values": marginal_x.values.flatten().tolist(),
            },
            "marginal_y": {
                "variables": sorted(list(marginal_y.variables)),
                "values": marginal_y.values.flatten().tolist(),
            },
        },
    }
