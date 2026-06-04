"""
Extended fixture generator for src/models package.

Covers NaiveBayes (fit + CPDs), MarkovChain, BayesianNetwork.Predict,
GetMarkovBlanket, GetIndependencies, Do, MarkovNetwork partition function,
and DynamicBN unrolling.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pgmpy.models import (
    DiscreteBayesianNetwork as BayesianNetwork,
    NaiveBayes,
    DynamicBayesianNetwork as DBN,
    DiscreteMarkovNetwork,
)
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination


def generate() -> list[dict]:
    """Generate extended model test cases."""
    test_cases = []

    test_cases.append(_test_naive_bayes_fit())
    test_cases.append(_test_bn_get_markov_blanket())
    test_cases.append(_test_bn_get_independencies())
    test_cases.append(_test_bn_do_g())
    test_cases.append(_test_markov_network_partition())
    test_cases.append(_test_dbn_structure())
    test_cases.append(_test_bn_predict())

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
    return bn


def _student_cpds_dict():
    return {
        "D": {"variable_card": 2, "values": [[0.6], [0.4]]},
        "I": {"variable_card": 2, "values": [[0.7], [0.3]]},
        "G": {
            "variable_card": 3,
            "values": [[0.3, 0.05, 0.9, 0.5],
                       [0.4, 0.25, 0.08, 0.3],
                       [0.3, 0.7, 0.02, 0.2]],
            "evidence": ["D", "I"], "evidence_card": [2, 2],
        },
        "L": {
            "variable_card": 2,
            "values": [[0.1, 0.4, 0.99],
                       [0.9, 0.6, 0.01]],
            "evidence": ["G"], "evidence_card": [3],
        },
        "S": {
            "variable_card": 2,
            "values": [[0.95, 0.2],
                       [0.05, 0.8]],
            "evidence": ["I"], "evidence_card": [2],
        },
    }


def _test_naive_bayes_fit():
    """NaiveBayes: Fit on data, store CPDs (predict broken in this pgmpy)."""
    np.random.seed(42)
    n = 600
    Y = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    X1 = np.where(Y == 0,
                  np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
                  np.random.choice([0, 1], size=n, p=[0.3, 0.7]))
    X2 = np.where(Y == 0,
                  np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
                  np.random.choice([0, 1], size=n, p=[0.2, 0.8]))
    data = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})

    nb = NaiveBayes()
    nb.fit(data, "Y")

    cpds = {}
    for cpd in sorted(nb.cpds, key=lambda c: c.variable):
        cpds[cpd.variable] = {
            "variable_card": int(cpd.variable_card),
            "values": cpd.get_values().tolist(),
        }

    return {
        "name": "naive_bayes_fit",
        "description": "NaiveBayes fit on X1,X2->Y data, store CPDs",
        "input": {
            "class_variable": "Y",
            "features": ["X1", "X2"],
            "data_columns": ["X1", "X2", "Y"],
            "data_rows": data[["X1", "X2", "Y"]].values.tolist(),
        },
        "expected": {
            "nodes": sorted(nb.nodes()),
            "edges": sorted([list(e) for e in nb.edges()]),
            "cpds": cpds,
        },
    }


def _test_bn_get_markov_blanket():
    """GetMarkovBlanket for each node in student BN."""
    bn = _build_student_bn()
    blankets = {}
    for node in sorted(bn.nodes()):
        mb = bn.get_markov_blanket(node)
        blankets[node] = sorted(mb)

    return {
        "name": "bn_get_markov_blanket",
        "description": "GetMarkovBlanket for each node in student BN",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _student_cpds_dict(),
        },
        "expected": {
            "blankets": blankets,
        },
    }


def _test_bn_get_independencies():
    """GetIndependencies: store all independence assertions for student BN."""
    bn = _build_student_bn()
    independencies = bn.get_independencies()

    assertions = []
    for assertion in independencies.get_assertions():
        assertions.append({
            "event1": sorted(list(assertion.event1)),
            "event2": sorted(list(assertion.event2)),
            "event3": sorted(list(assertion.event3)),
        })

    return {
        "name": "bn_get_independencies",
        "description": "All independence assertions for student BN",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
        },
        "expected": {
            "independencies": sorted(assertions, key=lambda x: str(x)),
            "num_assertions": len(assertions),
        },
    }


def _test_bn_do_g():
    """BN.Do: do(G) on student BN -- removes incoming edges to G."""
    bn = _build_student_bn()
    mutilated = bn.do(["G"])

    return {
        "name": "bn_do_g",
        "description": "do(G) on student BN: removes D->G and I->G edges",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _student_cpds_dict(),
            "do_nodes": ["G"],
        },
        "expected": {
            "mutilated_edges": sorted([list(e) for e in mutilated.edges()]),
            "mutilated_nodes": sorted(mutilated.nodes()),
            "num_edges": len(mutilated.edges()),
        },
    }


def _test_markov_network_partition():
    """MarkovNetwork: GetPartitionFunction on triangle MRF."""
    mn = DiscreteMarkovNetwork([("A", "B"), ("B", "C"), ("A", "C")])
    f_ab = DiscreteFactor(["A", "B"], [2, 2], [10, 1, 1, 10])
    f_bc = DiscreteFactor(["B", "C"], [2, 2], [5, 1, 1, 5])
    f_ac = DiscreteFactor(["A", "C"], [2, 2], [3, 2, 2, 3])
    mn.add_factors(f_ab, f_bc, f_ac)
    Z = mn.get_partition_function()

    return {
        "name": "markov_network_partition_extended",
        "description": "GetPartitionFunction on triangle MRF (A-B-C)",
        "input": {
            "edges": [["A", "B"], ["B", "C"], ["A", "C"]],
            "factors": [
                {"variables": ["A", "B"], "cardinality": [2, 2],
                 "values": [10.0, 1.0, 1.0, 10.0]},
                {"variables": ["B", "C"], "cardinality": [2, 2],
                 "values": [5.0, 1.0, 1.0, 5.0]},
                {"variables": ["A", "C"], "cardinality": [2, 2],
                 "values": [3.0, 2.0, 2.0, 3.0]},
            ],
        },
        "expected": {
            "partition_function": float(Z),
        },
    }


def _test_dbn_structure():
    """DynamicBN: build 2-slice DBN, store intra/inter edges."""
    dbn = DBN()
    dbn.add_edges_from([
        (("D", 0), ("G", 0)),
        (("I", 0), ("G", 0)),
        (("D", 0), ("D", 1)),
        (("G", 0), ("G", 1)),
    ])

    intra_0 = [(str(e[0]), str(e[1])) for e in dbn.get_intra_edges(0)]
    inter = [(str(e[0]), str(e[1])) for e in dbn.get_inter_edges()]

    # Serialize nodes as (name, timeslice) tuples
    all_nodes = []
    for n in sorted(dbn.nodes(), key=str):
        all_nodes.append([str(n.name) if hasattr(n, 'name') else str(n), int(n.time_slice) if hasattr(n, 'time_slice') else 0])

    return {
        "name": "dbn_structure",
        "description": "DynamicBN with 2 slices: D->G intra, D->D and G->G inter",
        "input": {
            "edges": [
                [["D", 0], ["G", 0]],
                [["I", 0], ["G", 0]],
                [["D", 0], ["D", 1]],
                [["G", 0], ["G", 1]],
            ],
        },
        "expected": {
            "num_nodes": len(dbn.nodes()),
            "intra_edges_0": sorted([sorted(e) for e in intra_0]),
            "inter_edges": sorted([list(e) for e in inter]),
        },
    }


def _test_bn_predict():
    """BN.Predict: predict missing values on student BN using VE."""
    bn = _build_student_bn()
    ve = VariableElimination(bn)

    # Predict G given D=0, I=1
    result = ve.map_query(variables=["G"], evidence={"D": 0, "I": 1})
    pred_g_d0_i1 = int(result["G"])

    # Predict L given G=0
    result2 = ve.map_query(variables=["L"], evidence={"G": 0})
    pred_l_g0 = int(result2["L"])

    # Predict S given I=0
    result3 = ve.map_query(variables=["S"], evidence={"I": 0})
    pred_s_i0 = int(result3["S"])

    return {
        "name": "bn_predict",
        "description": "Predict missing values on student BN via MAP",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _student_cpds_dict(),
            "predictions": [
                {"query": ["G"], "evidence": {"D": 0, "I": 1}},
                {"query": ["L"], "evidence": {"G": 0}},
                {"query": ["S"], "evidence": {"I": 0}},
            ],
        },
        "expected": {
            "predictions": [
                {"G": pred_g_d0_i1},
                {"L": pred_l_g0},
                {"S": pred_s_i0},
            ],
        },
    }
