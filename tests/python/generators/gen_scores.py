"""
Fixture generator for structure scoring functions.

Computes BIC, AIC, BDeu, BDs, K2 local scores and full scores on the
student BN using pgmpy, for cross-validation with the Go implementation.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BIC, AIC, BDeu, BDs, K2


def generate() -> list[dict]:
    """Generate scoring test cases."""
    test_cases = []

    bn, data = _build_student_data()

    test_cases.append(_test_local_scores(bn, data))
    test_cases.append(_test_full_scores(bn, data))
    test_cases.append(_test_local_scores_all_nodes(bn, data))

    return test_cases


def _build_student_data():
    """Build student BN and simulate data."""
    edges = [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")]
    bn = BayesianNetwork(edges)
    cpd_d = TabularCPD("D", 2, [[0.6], [0.4]])
    cpd_i = TabularCPD("I", 2, [[0.7], [0.3]])
    cpd_g = TabularCPD(
        "G", 3,
        [[0.3, 0.05, 0.9, 0.5],
         [0.4, 0.25, 0.08, 0.3],
         [0.3, 0.7, 0.02, 0.2]],
        evidence=["D", "I"], evidence_card=[2, 2],
    )
    cpd_l = TabularCPD(
        "L", 2,
        [[0.1, 0.4, 0.99],
         [0.9, 0.6, 0.01]],
        evidence=["G"], evidence_card=[3],
    )
    cpd_s = TabularCPD(
        "S", 2,
        [[0.95, 0.2],
         [0.05, 0.8]],
        evidence=["I"], evidence_card=[2],
    )
    bn.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
    data = bn.simulate(5000, seed=42)
    return bn, data


def _test_local_scores(bn, data):
    """Compute local scores for G with parents [D, I]."""
    scorers = {
        "BIC": BIC(data),
        "AIC": AIC(data),
        "BDeu": BDeu(data),
        "BDs": BDs(data),
        "K2": K2(data),
    }

    scores = {}
    for name, scorer in scorers.items():
        scores[name] = float(scorer.local_score("G", ["D", "I"]))

    return {
        "name": "local_scores_g_di",
        "description": "Local scores for G with parents [D,I] on student data",
        "input": {
            "variable": "G",
            "parents": ["D", "I"],
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "scores": scores,
        },
    }


def _test_full_scores(bn, data):
    """Compute full structure scores (sum of local scores) for entire BN."""
    scorers = {
        "BIC": BIC(data),
        "AIC": AIC(data),
        "BDeu": BDeu(data),
        "BDs": BDs(data),
        "K2": K2(data),
    }

    scores = {}
    for name, scorer in scorers.items():
        scores[name] = float(scorer.score(bn))

    return {
        "name": "full_scores_student_bn",
        "description": "Full structure scores for entire student BN",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "parent_map": {
                "D": [],
                "I": [],
                "G": ["D", "I"],
                "L": ["G"],
                "S": ["I"],
            },
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "scores": scores,
        },
    }


def _test_local_scores_all_nodes(bn, data):
    """Compute local scores for each node with its actual parents."""
    parent_map = {
        "D": [],
        "I": [],
        "G": ["D", "I"],
        "L": ["G"],
        "S": ["I"],
    }

    node_scores = {}
    for node in sorted(parent_map.keys()):
        parents = parent_map[node]
        node_scores[node] = {}
        for name, ScorerClass in [("BIC", BIC), ("AIC", AIC), ("BDeu", BDeu),
                                   ("BDs", BDs), ("K2", K2)]:
            scorer = ScorerClass(data)
            node_scores[node][name] = float(scorer.local_score(node, parents))

    return {
        "name": "local_scores_all_nodes",
        "description": "Local scores for each node with its actual parents",
        "input": {
            "parent_map": parent_map,
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "node_scores": node_scores,
        },
    }
