"""
Fixture generator for src/models package.

Generates test cases by exercising pgmpy's BayesianNetwork and related
model classes, capturing inputs and expected outputs as fixture data.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np


def generate() -> list[dict]:
    """Generate test cases for the models package."""
    test_cases = []

    # Test: create a simple Bayesian network and verify structure
    test_cases.append(_test_bayesian_network_structure())

    # Test: add CPDs and check model validity
    test_cases.append(_test_bayesian_network_cpds())

    # Test: get independencies
    test_cases.append(_test_bayesian_network_independencies())

    return test_cases


def _test_bayesian_network_structure():
    """Test basic BN creation with nodes and edges."""
    edges = [("Rain", "WetGrass"), ("Sprinkler", "WetGrass")]
    bn = BayesianNetwork(edges)

    return {
        "name": "bayesian_network_structure",
        "description": "Create BN with edges and verify structure",
        "input": {
            "edges": edges,
        },
        "expected": {
            "nodes": sorted(bn.nodes()),
            "edges": sorted([list(e) for e in bn.edges()]),
            "num_nodes": len(bn.nodes()),
            "num_edges": len(bn.edges()),
            "parents": {
                node: sorted(list(bn.get_parents(node)))
                for node in sorted(bn.nodes())
            },
            "children": {
                node: sorted(list(bn.get_children(node)))
                for node in sorted(bn.nodes())
            },
        },
    }


def _test_bayesian_network_cpds():
    """Test BN with CPDs and model validation."""
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
    is_valid = bn.check_model()

    return {
        "name": "bayesian_network_cpds",
        "description": "Create student BN with CPDs and validate",
        "input": {
            "edges": edges,
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
        },
        "expected": {
            "is_valid": is_valid,
            "nodes": sorted(bn.nodes()),
            "num_cpds": len(bn.cpds),
        },
    }


def _test_bayesian_network_independencies():
    """Test independence computation."""
    edges = [("X", "Y"), ("Y", "Z")]
    bn = BayesianNetwork(edges)

    independencies = bn.get_independencies()
    assertions = []
    for assertion in independencies.get_assertions():
        assertions.append({
            "event1": sorted(list(assertion.event1)),
            "event2": sorted(list(assertion.event2)),
            "event3": sorted(list(assertion.event3)),
        })

    return {
        "name": "bayesian_network_independencies",
        "description": "Get independencies from a chain X->Y->Z",
        "input": {
            "edges": edges,
        },
        "expected": {
            "independencies": sorted(assertions, key=lambda x: str(x)),
        },
    }
