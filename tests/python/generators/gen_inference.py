"""
Fixture generator for src/inference package.

Generates test cases by exercising pgmpy's inference algorithms,
capturing inputs and expected outputs as fixture data.
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def generate() -> list[dict]:
    """Generate test cases for the inference package."""
    test_cases = []

    test_cases.append(_test_variable_elimination_query())
    test_cases.append(_test_variable_elimination_map())

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


def _test_variable_elimination_query():
    """Test VE marginal query with evidence."""
    bn = _build_student_bn()
    ve = VariableElimination(bn)

    result = ve.query(variables=["G"], evidence={"D": 0, "I": 1})

    return {
        "name": "variable_elimination_query",
        "description": "Query P(G | D=0, I=1) on student network",
        "input": {
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
            "query_variables": ["G"],
            "evidence": {"D": 0, "I": 1},
        },
        "expected": {
            "variables": list(result.variables),
            "values": result.values.flatten().tolist(),
        },
    }


def _test_variable_elimination_map():
    """Test VE MAP query."""
    bn = _build_student_bn()
    ve = VariableElimination(bn)

    result = ve.map_query(variables=["G", "L"], evidence={"D": 0, "I": 0})

    return {
        "name": "variable_elimination_map",
        "description": "MAP query for G, L given D=0, I=0 on student network",
        "input": {
            "edges": [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")],
            "query_variables": ["G", "L"],
            "evidence": {"D": 0, "I": 0},
        },
        "expected": {
            "map_assignment": dict(result),
        },
    }
