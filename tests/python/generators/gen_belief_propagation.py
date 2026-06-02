"""
Fixture generator for belief propagation inference.

Generates test cases by exercising pgmpy's BeliefPropagation algorithm,
capturing inputs and expected outputs as fixture data.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation


def generate() -> list[dict]:
    """Generate test cases for belief propagation."""
    test_cases = []

    test_cases.append(_test_bp_query_g_given_d0_i1())
    test_cases.append(_test_bp_query_l_given_i0())

    return test_cases


def _build_student_bn():
    """Build the classic student Bayesian network."""
    edges = [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")]
    bn = BayesianNetwork(edges)

    cpd_d = TabularCPD("D", 2, [[0.6], [0.4]])
    cpd_i = TabularCPD("I", 2, [[0.7], [0.3]])
    cpd_g = TabularCPD(
        "G", 3,
        [[0.3, 0.05, 0.9, 0.5],
         [0.4, 0.25, 0.08, 0.3],
         [0.3, 0.7, 0.02, 0.2]],
        evidence=["D", "I"],
        evidence_card=[2, 2],
    )
    cpd_l = TabularCPD(
        "L", 2,
        [[0.1, 0.4, 0.99],
         [0.9, 0.6, 0.01]],
        evidence=["G"],
        evidence_card=[3],
    )
    cpd_s = TabularCPD(
        "S", 2,
        [[0.95, 0.2],
         [0.05, 0.8]],
        evidence=["I"],
        evidence_card=[2],
    )

    bn.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
    return bn


def _student_bn_input():
    """Return the standard student BN input dict."""
    return {
        "edges": [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")],
        "cpds": {
            "D": {"variable_card": 2, "values": [[0.6], [0.4]]},
            "I": {"variable_card": 2, "values": [[0.7], [0.3]]},
            "G": {
                "variable_card": 3,
                "values": [[0.3, 0.05, 0.9, 0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7, 0.02, 0.2]],
                "evidence": ["D", "I"],
                "evidence_card": [2, 2],
            },
            "L": {
                "variable_card": 2,
                "values": [[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                "evidence": ["G"],
                "evidence_card": [3],
            },
            "S": {
                "variable_card": 2,
                "values": [[0.95, 0.2],
                           [0.05, 0.8]],
                "evidence": ["I"],
                "evidence_card": [2],
            },
        },
    }


def _test_bp_query_g_given_d0_i1():
    """Test BP query P(G | D=0, I=1) -- should match VE result."""
    bn = _build_student_bn()
    bp = BeliefPropagation(bn)

    result = bp.query(variables=["G"], evidence={"D": 0, "I": 1})

    return {
        "name": "belief_propagation_query_g",
        "description": "BP query P(G | D=0, I=1) on student network",
        "input": {
            **_student_bn_input(),
            "query_variables": ["G"],
            "evidence": {"D": 0, "I": 1},
        },
        "expected": {
            "variables": sorted(list(result.variables)),
            "values": [round(v, 10) for v in result.values.flatten().tolist()],
        },
    }


def _test_bp_query_l_given_i0():
    """Test BP query P(L | I=0)."""
    bn = _build_student_bn()
    bp = BeliefPropagation(bn)

    result = bp.query(variables=["L"], evidence={"I": 0})

    return {
        "name": "belief_propagation_query_l",
        "description": "BP query P(L | I=0) on student network",
        "input": {
            **_student_bn_input(),
            "query_variables": ["L"],
            "evidence": {"I": 0},
        },
        "expected": {
            "variables": sorted(list(result.variables)),
            "values": [round(v, 10) for v in result.values.flatten().tolist()],
        },
    }
