"""
Fixture generator for structure learning (HillClimb, PC).

Generates test cases by exercising pgmpy's HillClimbSearch (with BIC scoring)
and PC algorithm on a simple A->B->C network, capturing learned edges as
expected outputs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch, PC, BIC


def generate() -> list[dict]:
    """Generate test cases for structure learning."""
    test_cases = []

    bn, data = _build_network_and_data()

    test_cases.append(_test_hill_climb_bic(data))
    test_cases.append(_test_pc_chi_square(data))

    return test_cases


def _build_network_and_data():
    """Build a simple A->B->C network with known CPDs and sample 5000 rows."""
    bn = BayesianNetwork([("A", "B"), ("B", "C")])

    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD(
        "B", 2,
        [[0.8, 0.2],
         [0.2, 0.8]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        "C", 2,
        [[0.9, 0.3],
         [0.1, 0.7]],
        evidence=["B"],
        evidence_card=[2],
    )
    bn.add_cpds(cpd_a, cpd_b, cpd_c)
    assert bn.check_model()

    data = bn.simulate(5000, seed=42)
    # Reorder columns for determinism.
    data = data[["A", "B", "C"]]
    return bn, data


def _test_hill_climb_bic(data):
    """Run HillClimbSearch with BIC scoring on the data."""
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BIC(data), show_progress=False)
    learned_edges = sorted([list(e) for e in model.edges()])

    return {
        "name": "hill_climb_bic",
        "description": "HillClimb with BIC on A->B->C network (5000 samples)",
        "input": {
            "variables": ["A", "B", "C"],
            "node_cards": {"A": 2, "B": 2, "C": 2},
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
        },
        "expected": {
            "learned_edges": learned_edges,
            "skeleton_edges": sorted([sorted(e) for e in learned_edges]),
        },
    }


def _test_pc_chi_square(data):
    """Run PC algorithm with chi_square CI test on the data."""
    pc = PC(data)
    pdag = pc.estimate(
        ci_test="chi_square",
        significance_level=0.05,
        return_type="pdag",
        show_progress=False,
    )

    directed = sorted([list(e) for e in pdag.directed_edges])
    undirected = sorted([sorted(list(e)) for e in pdag.undirected_edges])

    # Skeleton = union of all edges (undirected representation).
    skeleton = set()
    for e in pdag.directed_edges:
        skeleton.add(tuple(sorted(e)))
    for e in pdag.undirected_edges:
        skeleton.add(tuple(sorted(e)))
    skeleton_edges = sorted([list(e) for e in skeleton])

    return {
        "name": "pc_chi_square",
        "description": "PC with chi_square on A->B->C network (5000 samples)",
        "input": {
            "variables": ["A", "B", "C"],
            "node_cards": {"A": 2, "B": 2, "C": 2},
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
            "significance_level": 0.05,
        },
        "expected": {
            "directed_edges": directed,
            "undirected_edges": undirected,
            "skeleton_edges": skeleton_edges,
        },
    }
