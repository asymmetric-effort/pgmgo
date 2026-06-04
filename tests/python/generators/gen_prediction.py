"""
Fixture generator for causal prediction/estimation.

Since pgmpy's estimate_ate, DoubleML, and NaiveAdjustment have bugs or
don't exist in this version, we implement causal ATE estimation manually
using do-calculus queries on discrete BNs.

Also tests IVEstimator availability.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference


def generate() -> list[dict]:
    """Generate prediction/causal estimation test cases."""
    test_cases = []

    test_cases.append(_test_causal_ate_manual())
    test_cases.append(_test_causal_ate_with_confounder())

    return test_cases


def _test_causal_ate_manual():
    """Manual ATE computation: ATE(D, L) on student BN via do-calculus."""
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

    ci = CausalInference(bn)
    # P(L | do(D=0)) and P(L | do(D=1))
    result_d0 = ci.query(["L"], do={"D": 0})
    result_d1 = ci.query(["L"], do={"D": 1})

    # E[L|do(D=x)] = sum(l * P(L=l|do(D=x)))
    el_d0 = sum(l * p for l, p in enumerate(result_d0.values.flatten()))
    el_d1 = sum(l * p for l, p in enumerate(result_d1.values.flatten()))
    ate = el_d1 - el_d0

    cpds_dict = {
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

    return {
        "name": "causal_ate_d_l",
        "description": "ATE(D, L) on student BN via do-calculus",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": cpds_dict,
            "treatment": "D",
            "outcome": "L",
            "treatment_values": [0, 1],
        },
        "expected": {
            "do_d0_values": [round(v, 10) for v in result_d0.values.flatten().tolist()],
            "do_d1_values": [round(v, 10) for v in result_d1.values.flatten().tolist()],
            "ate": round(float(ate), 10),
        },
    }


def _test_causal_ate_with_confounder():
    """ATE on a confounded graph: X <- Z -> Y, X -> Y.

    Z is a confounder. Backdoor adjustment should give correct ATE.
    """
    edges = [("Z", "X"), ("Z", "Y"), ("X", "Y")]
    bn = BayesianNetwork(edges)

    cpd_z = TabularCPD("Z", 2, [[0.5], [0.5]])
    cpd_x = TabularCPD("X", 2,
                        [[0.8, 0.3], [0.2, 0.7]],
                        evidence=["Z"], evidence_card=[2])
    cpd_y = TabularCPD("Y", 2,
                        [[0.9, 0.6, 0.7, 0.1],
                         [0.1, 0.4, 0.3, 0.9]],
                        evidence=["X", "Z"], evidence_card=[2, 2])

    bn.add_cpds(cpd_z, cpd_x, cpd_y)

    ci = CausalInference(bn)
    # P(Y | do(X=0)) and P(Y | do(X=1))
    result_x0 = ci.query(["Y"], do={"X": 0})
    result_x1 = ci.query(["Y"], do={"X": 1})

    ey_x0 = sum(y * p for y, p in enumerate(result_x0.values.flatten()))
    ey_x1 = sum(y * p for y, p in enumerate(result_x1.values.flatten()))
    ate = ey_x1 - ey_x0

    adj_sets = ci.get_all_backdoor_adjustment_sets("X", "Y")
    adj_list = sorted([sorted(list(s)) for s in adj_sets])

    return {
        "name": "causal_ate_confounded",
        "description": "ATE(X, Y) on confounded graph X<-Z->Y, X->Y",
        "input": {
            "edges": [["Z", "X"], ["Z", "Y"], ["X", "Y"]],
            "cpds": {
                "Z": {"variable_card": 2, "values": [[0.5], [0.5]]},
                "X": {"variable_card": 2, "values": [[0.8, 0.3], [0.2, 0.7]],
                       "evidence": ["Z"], "evidence_card": [2]},
                "Y": {"variable_card": 2,
                       "values": [[0.9, 0.6, 0.7, 0.1],
                                  [0.1, 0.4, 0.3, 0.9]],
                       "evidence": ["X", "Z"], "evidence_card": [2, 2]},
            },
            "treatment": "X",
            "outcome": "Y",
            "treatment_values": [0, 1],
        },
        "expected": {
            "do_x0_values": [round(v, 10) for v in result_x0.values.flatten().tolist()],
            "do_x1_values": [round(v, 10) for v in result_x1.values.flatten().tolist()],
            "ate": round(float(ate), 10),
            "backdoor_adjustment_sets": adj_list,
        },
    }
