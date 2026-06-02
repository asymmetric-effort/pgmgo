"""
Fixture generator for src/sampling package.

Generates test cases by exercising pgmpy's sampling algorithms,
capturing inputs and expected outputs as fixture data.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import numpy as np


def generate() -> list[dict]:
    """Generate test cases for the sampling package."""
    test_cases = []

    test_cases.append(_test_forward_sampling())
    test_cases.append(_test_likelihood_weighted_sampling())

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


def _compute_empirical_marginals(bn, samples, weight_col=None):
    """Compute empirical marginal distributions from samples."""
    marginals = {}
    for var in sorted(bn.nodes()):
        card = bn.get_cpds(var).variable_card
        dist = []
        for val in range(card):
            mask = samples[var] == val
            if weight_col is not None:
                weights = samples[weight_col]
                prob = float(weights[mask].sum() / weights.sum())
            else:
                prob = float(mask.sum() / len(samples))
            dist.append(prob)
        marginals[var] = dist
    return marginals


def _test_forward_sampling():
    """Test forward sampling with seed for reproducibility."""
    bn = _build_student_bn()
    sampler = BayesianModelSampling(bn)

    seed = 42
    n_samples = 10000
    samples = sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)

    marginals = _compute_empirical_marginals(bn, samples)

    return {
        "name": "forward_sampling",
        "description": "Forward sample 10000 from student BN with seed=42, check empirical marginals",
        "input": {
            **_student_bn_input(),
            "seed": seed,
            "n_samples": n_samples,
        },
        "expected": {
            "marginals": marginals,
            "tolerance": 0.05,
        },
    }


def _test_likelihood_weighted_sampling():
    """Test likelihood-weighted sampling with evidence."""
    bn = _build_student_bn()
    sampler = BayesianModelSampling(bn)

    seed = 42
    n_samples = 10000
    evidence = [("D", 0), ("I", 1)]
    samples = sampler.likelihood_weighted_sample(
        evidence=evidence, size=n_samples, seed=seed, show_progress=False,
    )

    marginals = _compute_empirical_marginals(bn, samples, weight_col="_weight")

    return {
        "name": "likelihood_weighted_sampling",
        "description": "Likelihood-weighted sample with evidence D=0, I=1",
        "input": {
            **_student_bn_input(),
            "seed": seed,
            "n_samples": n_samples,
            "evidence": {"D": 0, "I": 1},
        },
        "expected": {
            "marginals": marginals,
            "tolerance": 0.05,
        },
    }
