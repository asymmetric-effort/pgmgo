#!/usr/bin/env python3
"""Generate cross-validation fixtures for scigo vs scipy.

Produces a JSON file with deterministic outputs from scipy functions.
The Go test suite loads this file and compares scigo against each value.
"""

import json
import math
import numpy as np
from scipy import stats, special, optimize, linalg, signal, interpolate, integrate, spatial

fixtures = {}

# ---------------------------------------------------------------------------
# Distributions: PDF, CDF, PPF, Mean, Var at specific known inputs
# ---------------------------------------------------------------------------

def dist_fixture(name, rv, x_vals, p_vals, is_discrete=False):
    """Build a fixture dict for a distribution."""
    entry = {
        "mean": float(rv.mean()),
        "var": float(rv.var()),
    }
    if is_discrete:
        entry["pmf"] = {str(int(x)): float(rv.pmf(x)) for x in x_vals}
        entry["cdf"] = {str(int(x)): float(rv.cdf(x)) for x in x_vals}
    else:
        entry["pdf"] = {str(x): float(rv.pdf(x)) for x in x_vals}
        entry["cdf"] = {str(x): float(rv.cdf(x)) for x in x_vals}
        entry["ppf"] = {str(p): float(rv.ppf(p)) for p in p_vals}
    return entry


# Normal(0, 1)
fixtures["normal_0_1"] = dist_fixture(
    "normal_0_1", stats.norm(0, 1),
    x_vals=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
    p_vals=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
)

# Normal(5, 2)
fixtures["normal_5_2"] = dist_fixture(
    "normal_5_2", stats.norm(5, 2),
    x_vals=[1.0, 3.0, 5.0, 7.0, 9.0],
    p_vals=[0.1, 0.5, 0.9],
)

# ChiSquared(5)
fixtures["chi2_5"] = dist_fixture(
    "chi2_5", stats.chi2(5),
    x_vals=[1.0, 3.0, 5.0, 7.0, 10.0],
    p_vals=[0.1, 0.5, 0.9],
)

# Beta(2, 5)
fixtures["beta_2_5"] = dist_fixture(
    "beta_2_5", stats.beta(2, 5),
    x_vals=[0.1, 0.2, 0.3, 0.5, 0.8],
    p_vals=[0.1, 0.5, 0.9],
)

# Gamma(3, scale=2) -- scipy uses shape a, scale
fixtures["gamma_3_2"] = dist_fixture(
    "gamma_3_2", stats.gamma(3, scale=2),
    x_vals=[1.0, 3.0, 5.0, 7.0, 10.0],
    p_vals=[0.1, 0.5, 0.9],
)

# Exponential(rate=1.5) -- scipy parameterizes as scale=1/rate
fixtures["exponential_1_5"] = dist_fixture(
    "exponential_1_5", stats.expon(scale=1.0/1.5),
    x_vals=[0.0, 0.5, 1.0, 2.0, 3.0],
    p_vals=[0.1, 0.5, 0.9],
)

# Student-t(10)
fixtures["t_10"] = dist_fixture(
    "t_10", stats.t(10),
    x_vals=[-2.0, -1.0, 0.0, 1.0, 2.0],
    p_vals=[0.05, 0.1, 0.5, 0.9, 0.95],
)

# F(5, 10)
fixtures["f_5_10"] = dist_fixture(
    "f_5_10", stats.f(5, 10),
    x_vals=[0.5, 1.0, 2.0, 3.0, 5.0],
    p_vals=[0.1, 0.5, 0.9],
)

# Poisson(4.0)
rv_poisson = stats.poisson(4.0)
fixtures["poisson_4"] = dist_fixture(
    "poisson_4", rv_poisson,
    x_vals=[0, 1, 2, 3, 4, 5, 8, 10],
    p_vals=[],
    is_discrete=True,
)

# Binomial(20, 0.3)
rv_binom = stats.binom(20, 0.3)
fixtures["binomial_20_0_3"] = dist_fixture(
    "binomial_20_0_3", rv_binom,
    x_vals=[0, 2, 5, 6, 8, 10, 15, 20],
    p_vals=[],
    is_discrete=True,
)

# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------

# ttest_ind
x_ttest = [2.1, 2.5, 2.7, 3.0, 3.2, 3.5, 3.8]
y_ttest = [3.0, 3.2, 3.5, 3.7, 4.0, 4.2, 4.5]
t_stat, t_pval = stats.ttest_ind(x_ttest, y_ttest)
fixtures["ttest_ind"] = {
    "x": x_ttest,
    "y": y_ttest,
    "statistic": float(t_stat),
    "pvalue": float(t_pval),
}

# chi2_contingency
observed_table = np.array([[10, 20, 30], [6, 9, 17]])
chi2_stat, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(observed_table)
fixtures["chi2_contingency"] = {
    "observed": observed_table.tolist(),
    "statistic": float(chi2_stat),
    "pvalue": float(chi2_p),
    "dof": int(chi2_dof),
    "expected": chi2_expected.tolist(),
}

# ks_2samp
x_ks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
y_ks = [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.5]
ks_stat, ks_pval = stats.ks_2samp(x_ks, y_ks)
fixtures["ks_2samp"] = {
    "x": x_ks,
    "y": y_ks,
    "statistic": float(ks_stat),
    "pvalue": float(ks_pval),
}

# pearsonr
x_pear = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
y_pear = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]
pr_r, pr_p = stats.pearsonr(x_pear, y_pear)
fixtures["pearsonr"] = {
    "x": x_pear,
    "y": y_pear,
    "statistic": float(pr_r),
    "pvalue": float(pr_p),
}

# spearmanr
sr_r, sr_p = stats.spearmanr(x_pear, y_pear)
fixtures["spearmanr"] = {
    "x": x_pear,
    "y": y_pear,
    "statistic": float(sr_r),
    "pvalue": float(sr_p),
}

# ---------------------------------------------------------------------------
# Special Functions
# ---------------------------------------------------------------------------
fixtures["special"] = {
    "gammaln_5": float(special.gammaln(5)),
    "digamma_3": float(special.digamma(3)),
    "erf_1": float(special.erf(1)),
    "erfinv_0_5": float(special.erfinv(0.5)),
    "betaln_2_3": float(special.betaln(2, 3)),
    "comb_10_3": float(special.comb(10, 3, exact=True)),
    "factorial_10": float(special.factorial(10, exact=True)),
}

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

# minimize x^2, starting at x0=5
res_min = optimize.minimize(lambda x: x[0]**2, [5.0], method="Nelder-Mead")
fixtures["minimize_x2"] = {
    "x0": [5.0],
    "result_x": float(res_min.x[0]),
    "result_fun": float(res_min.fun),
}

# root_scalar: x^2 - 4 = 0 in [0, 3]
res_root = optimize.root_scalar(lambda x: x**2 - 4, bracket=[0, 3], method="brentq")
fixtures["root_scalar_x2_minus_4"] = {
    "bracket": [0.0, 3.0],
    "result": float(res_root.root),
}

# ---------------------------------------------------------------------------
# Linear Algebra
# ---------------------------------------------------------------------------

mat_a = [[2.0, 1.0, 1.0],
         [4.0, 3.0, 3.0],
         [8.0, 7.0, 9.0]]

# LU decomposition
p_lu, l_lu, u_lu = linalg.lu(mat_a)
fixtures["lu"] = {
    "matrix": mat_a,
    "p": p_lu.tolist(),
    "l": l_lu.tolist(),
    "u": u_lu.tolist(),
}

# Cholesky (need a positive-definite matrix)
mat_pd = [[4.0, 2.0, 1.0],
          [2.0, 5.0, 3.0],
          [1.0, 3.0, 6.0]]
cho_l = linalg.cholesky(mat_pd, lower=True)
fixtures["cholesky"] = {
    "matrix": mat_pd,
    "l": cho_l.tolist(),
}

# Determinant
det_val = linalg.det(mat_a)
fixtures["det"] = {
    "matrix": mat_a,
    "result": float(det_val),
}

# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

conv_a = [1.0, 2.0, 3.0]
conv_b = [0.0, 1.0, 0.5]
conv_result = signal.convolve(conv_a, conv_b).tolist()
fixtures["convolve"] = {
    "a": conv_a,
    "b": conv_b,
    "result": [float(v) for v in conv_result],
}

# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

interp_x = [0.0, 1.0, 2.0]
interp_y = [0.0, 1.0, 4.0]
f_interp = interpolate.interp1d(interp_x, interp_y, kind="linear")
fixtures["interp1d_linear"] = {
    "x": interp_x,
    "y": interp_y,
    "queries": {
        "0.5": float(f_interp(0.5)),
        "1.5": float(f_interp(1.5)),
    },
}

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

quad_result, quad_err = integrate.quad(math.sin, 0, math.pi)
fixtures["quad_sin_0_pi"] = {
    "result": float(quad_result),
    "error": float(quad_err),
}

# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

cd_xa = [[0.0, 0.0], [1.0, 1.0]]
cd_xb = [[2.0, 2.0]]
cd_result = spatial.distance.cdist(cd_xa, cd_xb, "euclidean").tolist()
fixtures["cdist_euclidean"] = {
    "xa": cd_xa,
    "xb": cd_xb,
    "result": [[float(v) for v in row] for row in cd_result],
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

output_path = "tests/python/generators/scigo_fixtures.json"
with open(output_path, "w") as f:
    json.dump(fixtures, f, indent=2)

print(f"Wrote {len(fixtures)} fixture groups to {output_path}")
