"""
Extended fixture generator for src/inference package.

Covers VE queries for every variable, VE MAP for multiple combos,
BP matching VE, ApproxInference marginals, CausalInference do-queries,
and DBNInference forward queries.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import (
    VariableElimination,
    BeliefPropagation,
    CausalInference,
    ApproxInference,
)


def generate() -> list[dict]:
    """Generate extended inference test cases."""
    test_cases = []

    bn = _build_student_bn()

    # VE: query every variable with no evidence
    test_cases.extend(_test_ve_all_variables_no_evidence(bn))

    # VE: query every variable with evidence D=0
    test_cases.extend(_test_ve_all_variables_with_evidence(bn))

    # VE: MAP for variable combinations
    test_cases.extend(_test_ve_map_combos(bn))

    # BP: same queries as VE (should match)
    test_cases.extend(_test_bp_matches_ve(bn))

    # ApproxInference: marginal with large n_samples
    test_cases.append(_test_approx_inference(bn))

    # CausalInference: do-calculus queries
    test_cases.append(_test_causal_ate(bn))
    test_cases.append(_test_causal_backdoor(bn))

    return test_cases


def _build_student_bn():
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


def _cpds_dict():
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


def _test_ve_all_variables_no_evidence(bn):
    """VE query for every variable with no evidence."""
    ve = VariableElimination(bn)
    cases = []
    for var in sorted(bn.nodes()):
        result = ve.query(variables=[var], evidence={})
        cases.append({
            "name": f"ve_query_{var}_no_evidence",
            "description": f"VE query P({var}) with no evidence",
            "input": {
                "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
                "cpds": _cpds_dict(),
                "query_variables": [var],
                "evidence": {},
            },
            "expected": {
                "variables": sorted(list(result.variables)),
                "values": [round(v, 10) for v in result.values.flatten().tolist()],
            },
        })
    return cases


def _test_ve_all_variables_with_evidence(bn):
    """VE query for every non-evidence variable with D=0."""
    ve = VariableElimination(bn)
    evidence = {"D": 0}
    cases = []
    for var in sorted(bn.nodes()):
        if var in evidence:
            continue
        result = ve.query(variables=[var], evidence=evidence)
        cases.append({
            "name": f"ve_query_{var}_given_D0",
            "description": f"VE query P({var} | D=0)",
            "input": {
                "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
                "cpds": _cpds_dict(),
                "query_variables": [var],
                "evidence": {"D": 0},
            },
            "expected": {
                "variables": sorted(list(result.variables)),
                "values": [round(v, 10) for v in result.values.flatten().tolist()],
            },
        })
    return cases


def _test_ve_map_combos(bn):
    """VE MAP queries for multiple variable combinations."""
    ve = VariableElimination(bn)
    combos = [
        (["G"], {"D": 0, "I": 0}),
        (["G"], {"D": 1, "I": 0}),
        (["G"], {"D": 0, "I": 1}),
        (["G"], {"D": 1, "I": 1}),
        (["L"], {"G": 0}),
        (["L"], {"G": 1}),
        (["L"], {"G": 2}),
        (["G", "L"], {"D": 0, "I": 0}),
        (["S"], {"I": 0}),
        (["S"], {"I": 1}),
    ]
    cases = []
    for i, (qvars, ev) in enumerate(combos):
        result = ve.map_query(variables=qvars, evidence=ev)
        ev_str = "_".join(f"{k}{v}" for k, v in sorted(ev.items()))
        qv_str = "_".join(qvars)
        cases.append({
            "name": f"ve_map_{qv_str}_given_{ev_str}",
            "description": f"VE MAP({','.join(qvars)} | {ev})",
            "input": {
                "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
                "cpds": _cpds_dict(),
                "query_variables": qvars,
                "evidence": ev,
            },
            "expected": {
                "map_assignment": {k: int(v) for k, v in result.items()},
            },
        })
    return cases


def _test_bp_matches_ve(bn):
    """BP queries that should match VE exactly."""
    ve = VariableElimination(bn)
    bp = BeliefPropagation(bn)

    queries = [
        (["G"], {"D": 0, "I": 1}),
        (["L"], {"I": 0}),
        (["S"], {}),
        (["D"], {"G": 1}),
    ]
    cases = []
    for qvars, ev in queries:
        ve_result = ve.query(variables=qvars, evidence=ev)
        bp_result = bp.query(variables=qvars, evidence=ev)
        ev_str = "_".join(f"{k}{v}" for k, v in sorted(ev.items())) if ev else "none"
        cases.append({
            "name": f"bp_vs_ve_{'_'.join(qvars)}_given_{ev_str}",
            "description": f"BP query P({','.join(qvars)} | {ev}) matches VE",
            "input": {
                "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
                "cpds": _cpds_dict(),
                "query_variables": qvars,
                "evidence": ev,
            },
            "expected": {
                "variables": sorted(list(ve_result.variables)),
                "ve_values": [round(v, 10) for v in ve_result.values.flatten().tolist()],
                "bp_values": [round(v, 10) for v in bp_result.values.flatten().tolist()],
            },
        })
    return cases


def _test_approx_inference(bn):
    """ApproxInference: marginal query with large n_samples."""
    ve = VariableElimination(bn)
    exact = ve.query(variables=["G"], evidence={"D": 0, "I": 1})

    ai = ApproxInference(bn)
    approx = ai.query(["G"], evidence={"D": 0, "I": 1}, n_samples=100000, seed=42)

    return {
        "name": "approx_inference_marginal",
        "description": "ApproxInference P(G | D=0, I=1) with 100k samples",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _cpds_dict(),
            "query_variables": ["G"],
            "evidence": {"D": 0, "I": 1},
            "n_samples": 100000,
        },
        "expected": {
            "exact_values": [round(v, 10) for v in exact.values.flatten().tolist()],
            "approx_values": [round(v, 6) for v in approx.values.flatten().tolist()],
            "tolerance": 0.05,
        },
    }


def _test_causal_ate(bn):
    """CausalInference: do-calculus for ATE(D, G, [0,1])."""
    ci = CausalInference(bn)

    # P(G | do(D=0))
    result_d0 = ci.query(["G"], do={"D": 0})
    # P(G | do(D=1))
    result_d1 = ci.query(["G"], do={"D": 1})

    # ATE = E[G|do(D=1)] - E[G|do(D=0)]
    # For discrete: E[G|do(D=x)] = sum(g * P(G=g|do(D=x)))
    eg_d0 = sum(g * p for g, p in enumerate(result_d0.values.flatten()))
    eg_d1 = sum(g * p for g, p in enumerate(result_d1.values.flatten()))
    ate = eg_d1 - eg_d0

    return {
        "name": "causal_ate_d_g",
        "description": "CausalInference ATE(D, G, [0,1]) on student BN",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _cpds_dict(),
            "treatment": "D",
            "outcome": "G",
            "treatment_values": [0, 1],
        },
        "expected": {
            "do_d0_values": [round(v, 10) for v in result_d0.values.flatten().tolist()],
            "do_d1_values": [round(v, 10) for v in result_d1.values.flatten().tolist()],
            "ate": round(float(ate), 10),
        },
    }


def _test_causal_backdoor(bn):
    """CausalInference: backdoor adjustment sets."""
    ci = CausalInference(bn)

    # Get all backdoor adjustment sets for D->G
    adj_sets = ci.get_all_backdoor_adjustment_sets("D", "G")
    # Convert frozensets to sorted lists
    adj_list = sorted([sorted(list(s)) for s in adj_sets])

    return {
        "name": "causal_backdoor_d_g",
        "description": "CausalInference backdoor adjustment sets for D->G",
        "input": {
            "edges": [["D", "G"], ["I", "G"], ["G", "L"], ["I", "S"]],
            "cpds": _cpds_dict(),
            "treatment": "D",
            "outcome": "G",
        },
        "expected": {
            "adjustment_sets": adj_list,
        },
    }
