"""
Fixture generator for causal inference cross-validation.

Generates test cases by exercising pgmpy's CausalInference (do-calculus)
and VariableElimination (observational) on the student BN, capturing
inputs and expected outputs as fixture data.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference, VariableElimination


def generate() -> list[dict]:
    """Generate test cases for causal inference."""
    test_cases = []

    test_cases.append(_test_causal_do_query())
    test_cases.append(_test_observational_query())
    test_cases.append(_test_causal_do_query_l())

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


def _cpds_dict():
    """Return CPD definitions as a serialisable dict."""
    return {
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
    }


def _test_causal_do_query():
    """Test CausalInference do-calculus query: P(G | do(D=0))."""
    bn = _build_student_bn()
    ci = CausalInference(bn)

    result = ci.query(["G"], do={"D": 0})

    return {
        "name": "causal_do_query_g",
        "description": "Causal query P(G | do(D=0)) on student network",
        "input": {
            "edges": [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")],
            "cpds": _cpds_dict(),
            "query_variables": ["G"],
            "do_variables": {"D": 0},
            "evidence": {},
        },
        "expected": {
            "variables": sorted(list(result.variables)),
            "values": [round(v, 10) for v in result.values.flatten().tolist()],
        },
    }


def _test_observational_query():
    """Test observational query P(G | D=0) for comparison with do-query."""
    bn = _build_student_bn()
    ve = VariableElimination(bn)

    result = ve.query(variables=["G"], evidence={"D": 0})

    return {
        "name": "observational_query_g",
        "description": "Observational query P(G | D=0) on student network for comparison",
        "input": {
            "edges": [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")],
            "cpds": _cpds_dict(),
            "query_variables": ["G"],
            "evidence": {"D": 0},
        },
        "expected": {
            "variables": sorted(list(result.variables)),
            "values": [round(v, 10) for v in result.values.flatten().tolist()],
        },
    }


def _test_causal_do_query_l():
    """Test CausalInference do-calculus query: P(L | do(D=0))."""
    bn = _build_student_bn()
    ci = CausalInference(bn)

    result = ci.query(["L"], do={"D": 0})

    return {
        "name": "causal_do_query_l",
        "description": "Causal query P(L | do(D=0)) on student network",
        "input": {
            "edges": [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")],
            "cpds": _cpds_dict(),
            "query_variables": ["L"],
            "do_variables": {"D": 0},
            "evidence": {},
        },
        "expected": {
            "variables": sorted(list(result.variables)),
            "values": [round(v, 10) for v in result.values.flatten().tolist()],
        },
    }
