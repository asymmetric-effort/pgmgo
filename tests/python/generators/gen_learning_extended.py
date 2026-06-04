"""
Extended fixture generator for src/learning package.

Covers MLE for all student BN nodes, BayesianEstimator with different ESS,
HillClimbSearch with BIC/AIC/BDeu, PC with different significance levels,
and TreeSearch (Chow-Liu).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import (
    MaximumLikelihoodEstimator,
    BayesianEstimator,
    HillClimbSearch,
    PC,
    TreeSearch,
    BIC,
    AIC,
    BDeu,
)


def generate() -> list[dict]:
    """Generate extended learning test cases."""
    test_cases = []

    bn, data = _build_student_data()

    test_cases.append(_test_mle_all_nodes(bn, data))
    test_cases.append(_test_bayesian_ess5(bn, data))
    test_cases.append(_test_bayesian_ess50(bn, data))

    # Structure learning on A->B->C data
    sl_bn, sl_data = _build_chain_data()
    test_cases.append(_test_hc_bic(sl_data))
    test_cases.append(_test_hc_aic(sl_data))
    test_cases.append(_test_hc_bdeu(sl_data))
    test_cases.append(_test_pc_001(sl_data))
    test_cases.append(_test_pc_005(sl_data))
    test_cases.append(_test_pc_010(sl_data))
    test_cases.append(_test_tree_search(sl_data))

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


def _build_chain_data():
    """Build A->B->C chain and simulate data."""
    bn = BayesianNetwork([("A", "B"), ("B", "C")])
    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD("B", 2, [[0.8, 0.2], [0.2, 0.8]],
                        evidence=["A"], evidence_card=[2])
    cpd_c = TabularCPD("C", 2, [[0.9, 0.3], [0.1, 0.7]],
                        evidence=["B"], evidence_card=[2])
    bn.add_cpds(cpd_a, cpd_b, cpd_c)
    data = bn.simulate(5000, seed=42)
    data = data[["A", "B", "C"]]
    return bn, data


def _cpd_values_as_list(cpd):
    return cpd.get_values().tolist()


def _test_mle_all_nodes(bn, data):
    """MLE: estimate ALL node CPDs on student data."""
    mle = MaximumLikelihoodEstimator(bn, data)

    cpds = {}
    for node in sorted(bn.nodes()):
        cpd = mle.estimate_cpd(node)
        evidence = list(cpd.get_evidence()) if hasattr(cpd, 'get_evidence') else []
        evidence_card = [int(c) for c in cpd.cardinality[1:]] if len(cpd.cardinality) > 1 else []
        cpds[node] = {
            "variable": node,
            "variable_card": int(cpd.variable_card),
            "values": _cpd_values_as_list(cpd),
            "evidence": sorted(bn.get_parents(node)),
            "evidence_card": [int(bn.get_cpds(p).variable_card) for p in sorted(bn.get_parents(node))],
        }

    return {
        "name": "mle_all_nodes",
        "description": "MLE estimation on all student BN nodes (5000 samples)",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "node_cards": {"D": 2, "G": 3, "I": 2, "L": 2, "S": 2},
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "cpds": cpds,
        },
    }


def _test_bayesian_ess5(bn, data):
    """BayesianEstimator with BDeu ESS=5."""
    be = BayesianEstimator(bn, data)
    cpds = {}
    for node in sorted(bn.nodes()):
        cpd = be.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=5)
        cpds[node] = {
            "variable": node,
            "variable_card": int(cpd.variable_card),
            "values": _cpd_values_as_list(cpd),
            "evidence": sorted(bn.get_parents(node)),
            "evidence_card": [int(bn.get_cpds(p).variable_card) for p in sorted(bn.get_parents(node))],
        }

    return {
        "name": "bayesian_bdeu_ess5",
        "description": "BayesianEstimator BDeu ESS=5 on all student BN nodes",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "node_cards": {"D": 2, "G": 3, "I": 2, "L": 2, "S": 2},
            "equivalent_sample_size": 5,
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "cpds": cpds,
        },
    }


def _test_bayesian_ess50(bn, data):
    """BayesianEstimator with BDeu ESS=50."""
    be = BayesianEstimator(bn, data)
    cpds = {}
    for node in sorted(bn.nodes()):
        cpd = be.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=50)
        cpds[node] = {
            "variable": node,
            "variable_card": int(cpd.variable_card),
            "values": _cpd_values_as_list(cpd),
            "evidence": sorted(bn.get_parents(node)),
            "evidence_card": [int(bn.get_cpds(p).variable_card) for p in sorted(bn.get_parents(node))],
        }

    return {
        "name": "bayesian_bdeu_ess50",
        "description": "BayesianEstimator BDeu ESS=50 on all student BN nodes",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "node_cards": {"D": 2, "G": 3, "I": 2, "L": 2, "S": 2},
            "equivalent_sample_size": 50,
            "data_columns": sorted(bn.nodes()),
            "data_rows": data[sorted(bn.nodes())].values.tolist(),
        },
        "expected": {
            "cpds": cpds,
        },
    }


def _test_hc_bic(data):
    """HillClimbSearch with BIC scoring."""
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BIC(data), show_progress=False)
    learned_edges = sorted([list(e) for e in model.edges()])
    return {
        "name": "hc_bic",
        "description": "HillClimb with BIC on A->B->C data",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
        },
        "expected": {
            "learned_edges": learned_edges,
            "skeleton_edges": sorted([sorted(e) for e in learned_edges]),
        },
    }


def _test_hc_aic(data):
    """HillClimbSearch with AIC scoring."""
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=AIC(data), show_progress=False)
    learned_edges = sorted([list(e) for e in model.edges()])
    return {
        "name": "hc_aic",
        "description": "HillClimb with AIC on A->B->C data",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
        },
        "expected": {
            "learned_edges": learned_edges,
            "skeleton_edges": sorted([sorted(e) for e in learned_edges]),
        },
    }


def _test_hc_bdeu(data):
    """HillClimbSearch with BDeu scoring."""
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BDeu(data), show_progress=False)
    learned_edges = sorted([list(e) for e in model.edges()])
    return {
        "name": "hc_bdeu",
        "description": "HillClimb with BDeu on A->B->C data",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
        },
        "expected": {
            "learned_edges": learned_edges,
            "skeleton_edges": sorted([sorted(e) for e in learned_edges]),
        },
    }


def _test_pc_001(data):
    """PC with significance=0.01."""
    pc = PC(data)
    pdag = pc.estimate(ci_test="chi_square", significance_level=0.01,
                       return_type="pdag", show_progress=False)
    skeleton = set()
    for e in pdag.directed_edges:
        skeleton.add(tuple(sorted(e)))
    for e in pdag.undirected_edges:
        skeleton.add(tuple(sorted(e)))
    return {
        "name": "pc_sig001",
        "description": "PC with chi_square significance=0.01",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
            "significance_level": 0.01,
        },
        "expected": {
            "skeleton_edges": sorted([list(e) for e in skeleton]),
        },
    }


def _test_pc_005(data):
    """PC with significance=0.05."""
    pc = PC(data)
    pdag = pc.estimate(ci_test="chi_square", significance_level=0.05,
                       return_type="pdag", show_progress=False)
    skeleton = set()
    for e in pdag.directed_edges:
        skeleton.add(tuple(sorted(e)))
    for e in pdag.undirected_edges:
        skeleton.add(tuple(sorted(e)))
    return {
        "name": "pc_sig005",
        "description": "PC with chi_square significance=0.05",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
            "significance_level": 0.05,
        },
        "expected": {
            "skeleton_edges": sorted([list(e) for e in skeleton]),
        },
    }


def _test_pc_010(data):
    """PC with significance=0.10."""
    pc = PC(data)
    pdag = pc.estimate(ci_test="chi_square", significance_level=0.10,
                       return_type="pdag", show_progress=False)
    skeleton = set()
    for e in pdag.directed_edges:
        skeleton.add(tuple(sorted(e)))
    for e in pdag.undirected_edges:
        skeleton.add(tuple(sorted(e)))
    return {
        "name": "pc_sig010",
        "description": "PC with chi_square significance=0.10",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
            "significance_level": 0.10,
        },
        "expected": {
            "skeleton_edges": sorted([list(e) for e in skeleton]),
        },
    }


def _test_tree_search(data):
    """TreeSearch (Chow-Liu) on A->B->C data."""
    ts = TreeSearch(data[["A", "B", "C"]])
    model = ts.estimate(estimator_type="chow-liu", show_progress=False)
    learned_edges = sorted([sorted(list(e)) for e in model.edges()])
    return {
        "name": "tree_search_chow_liu",
        "description": "Chow-Liu tree on A->B->C data",
        "input": {
            "variables": ["A", "B", "C"],
            "data_columns": ["A", "B", "C"],
            "data_rows": data[["A", "B", "C"]].values.tolist(),
        },
        "expected": {
            "tree_edges": learned_edges,
        },
    }
