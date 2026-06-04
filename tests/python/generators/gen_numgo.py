"""
NumPy cross-validation fixture generator for numgo.

Generates deterministic test cases exercising numpy functions and stores
inputs + expected outputs as JSON fixtures for Go cross-validation tests.
"""

import numpy as np
import json


def _a(x):
    """Convert ndarray to nested list for JSON serialization."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def generate():
    test_cases = []

    # =========================================================================
    # CREATION
    # =========================================================================

    test_cases.append({
        "name": "zeros_2x3",
        "category": "creation",
        "input": {"shape": [2, 3]},
        "expected": {"result": np.zeros((2, 3)).tolist()}
    })

    test_cases.append({
        "name": "ones_3x2",
        "category": "creation",
        "input": {"shape": [3, 2]},
        "expected": {"result": np.ones((3, 2)).tolist()}
    })

    test_cases.append({
        "name": "eye_3",
        "category": "creation",
        "input": {"n": 3},
        "expected": {"result": np.eye(3).tolist()}
    })

    test_cases.append({
        "name": "arange_0_10_2",
        "category": "creation",
        "input": {"start": 0.0, "stop": 10.0, "step": 2.0},
        "expected": {"result": np.arange(0.0, 10.0, 2.0).tolist()}
    })

    test_cases.append({
        "name": "linspace_0_1_5",
        "category": "creation",
        "input": {"start": 0.0, "stop": 1.0, "num": 5},
        "expected": {"result": np.linspace(0.0, 1.0, 5).tolist()}
    })

    test_cases.append({
        "name": "logspace_0_2_5",
        "category": "creation",
        "input": {"start": 0.0, "stop": 2.0, "num": 5},
        "expected": {"result": np.logspace(0.0, 2.0, 5).tolist()}
    })

    # diag from 1D
    v = np.array([1.0, 2.0, 3.0])
    test_cases.append({
        "name": "diag_1d",
        "category": "creation",
        "input": {"a": v.tolist(), "k": 0},
        "expected": {"result": np.diag(v).tolist()}
    })

    # diag extract from 2D
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_cases.append({
        "name": "diag_2d_extract",
        "category": "creation",
        "input": {"a": M.tolist(), "k": 0},
        "expected": {"result": np.diag(M).tolist()}
    })

    # tri
    test_cases.append({
        "name": "tri_3x3_k0",
        "category": "creation",
        "input": {"n": 3, "m": 3, "k": 0},
        "expected": {"result": np.tri(3, 3, 0).tolist()}
    })

    # tril
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_cases.append({
        "name": "tril_3x3",
        "category": "creation",
        "input": {"a": A.tolist(), "k": 0},
        "expected": {"result": np.tril(A, 0).tolist()}
    })

    # triu
    test_cases.append({
        "name": "triu_3x3",
        "category": "creation",
        "input": {"a": A.tolist(), "k": 0},
        "expected": {"result": np.triu(A, 0).tolist()}
    })

    # =========================================================================
    # ARITHMETIC (with broadcasting)
    # =========================================================================

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    test_cases.append({
        "name": "add_2x2",
        "category": "arithmetic",
        "input": {"a": A.tolist(), "b": B.tolist()},
        "expected": {"result": (A + B).tolist()}
    })

    test_cases.append({
        "name": "sub_2x2",
        "category": "arithmetic",
        "input": {"a": A.tolist(), "b": B.tolist()},
        "expected": {"result": (A - B).tolist()}
    })

    test_cases.append({
        "name": "mul_2x2",
        "category": "arithmetic",
        "input": {"a": A.tolist(), "b": B.tolist()},
        "expected": {"result": (A * B).tolist()}
    })

    test_cases.append({
        "name": "div_2x2",
        "category": "arithmetic",
        "input": {"a": A.tolist(), "b": B.tolist()},
        "expected": {"result": (A / B).tolist()}
    })

    # Broadcasting: 2x3 + 1x3
    C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    D = np.array([[10.0, 20.0, 30.0]])
    test_cases.append({
        "name": "add_broadcast_2x3_1x3",
        "category": "arithmetic",
        "input": {
            "a": C.tolist(), "a_shape": [2, 3],
            "b": D.tolist(), "b_shape": [1, 3],
        },
        "expected": {"result": (C + D).tolist()}
    })

    # Broadcasting: 3x1 + 1x3
    E = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[10.0, 20.0, 30.0]])
    test_cases.append({
        "name": "add_broadcast_3x1_1x3",
        "category": "arithmetic",
        "input": {
            "a": E.tolist(), "a_shape": [3, 1],
            "b": F.tolist(), "b_shape": [1, 3],
        },
        "expected": {"result": (E + F).tolist()}
    })

    # Sum, Prod
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    test_cases.append({
        "name": "sum_all",
        "category": "arithmetic",
        "input": {"a": X.tolist()},
        "expected": {"result": float(np.sum(X))}
    })

    test_cases.append({
        "name": "prod_all",
        "category": "arithmetic",
        "input": {"a": X.tolist()},
        "expected": {"result": float(np.prod(X))}
    })

    test_cases.append({
        "name": "sum_axis0",
        "category": "arithmetic",
        "input": {"a": X.tolist(), "axis": 0},
        "expected": {"result": np.sum(X, axis=0).tolist()}
    })

    test_cases.append({
        "name": "sum_axis1",
        "category": "arithmetic",
        "input": {"a": X.tolist(), "axis": 1},
        "expected": {"result": np.sum(X, axis=1).tolist()}
    })

    # =========================================================================
    # MATH
    # =========================================================================

    vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    test_cases.append({
        "name": "sin",
        "category": "math",
        "input": {"a": vals.tolist()},
        "expected": {"result": np.sin(vals).tolist()}
    })

    test_cases.append({
        "name": "cos",
        "category": "math",
        "input": {"a": vals.tolist()},
        "expected": {"result": np.cos(vals).tolist()}
    })

    test_cases.append({
        "name": "exp",
        "category": "math",
        "input": {"a": vals.tolist()},
        "expected": {"result": np.exp(vals).tolist()}
    })

    pos_vals = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    test_cases.append({
        "name": "log",
        "category": "math",
        "input": {"a": pos_vals.tolist()},
        "expected": {"result": np.log(pos_vals).tolist()}
    })

    test_cases.append({
        "name": "sqrt",
        "category": "math",
        "input": {"a": pos_vals.tolist()},
        "expected": {"result": np.sqrt(pos_vals).tolist()}
    })

    signed = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    test_cases.append({
        "name": "abs",
        "category": "math",
        "input": {"a": signed.tolist()},
        "expected": {"result": np.abs(signed).tolist()}
    })

    test_cases.append({
        "name": "sign",
        "category": "math",
        "input": {"a": signed.tolist()},
        "expected": {"result": np.sign(signed).tolist()}
    })

    test_cases.append({
        "name": "clip",
        "category": "math",
        "input": {"a": signed.tolist(), "min": -1.0, "max": 3.0},
        "expected": {"result": np.clip(signed, -1.0, 3.0).tolist()}
    })

    frac_vals = np.array([-1.7, -0.3, 0.0, 0.7, 2.5])
    test_cases.append({
        "name": "floor",
        "category": "math",
        "input": {"a": frac_vals.tolist()},
        "expected": {"result": np.floor(frac_vals).tolist()}
    })

    test_cases.append({
        "name": "ceil",
        "category": "math",
        "input": {"a": frac_vals.tolist()},
        "expected": {"result": np.ceil(frac_vals).tolist()}
    })

    test_cases.append({
        "name": "round",
        "category": "math",
        "input": {"a": np.array([1.235, 2.675, 3.145]).tolist(), "decimals": 2},
        "expected": {"result": np.around(np.array([1.235, 2.675, 3.145]), 2).tolist()}
    })

    # =========================================================================
    # LINEAR ALGEBRA
    # =========================================================================

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    test_cases.append({
        "name": "matmul_2x2",
        "category": "linalg",
        "input": {"a": A.tolist(), "b": B.tolist()},
        "expected": {"result": (A @ B).tolist()}
    })

    # Matmul 3x2 @ 2x4
    A32 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    B24 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    test_cases.append({
        "name": "matmul_3x2_2x4",
        "category": "linalg",
        "input": {"a": A32.tolist(), "b": B24.tolist()},
        "expected": {"result": (A32 @ B24).tolist()}
    })

    # Dot product 1D
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    test_cases.append({
        "name": "dot_1d",
        "category": "linalg",
        "input": {"a": u.tolist(), "b": v.tolist()},
        "expected": {"result": float(np.dot(u, v))}
    })

    # Det
    A_det = np.array([[1.0, 2.0], [3.0, 4.0]])
    test_cases.append({
        "name": "det_2x2",
        "category": "linalg",
        "input": {"a": A_det.tolist()},
        "expected": {"result": float(np.linalg.det(A_det))}
    })

    A_det3 = np.array([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]])
    test_cases.append({
        "name": "det_3x3",
        "category": "linalg",
        "input": {"a": A_det3.tolist()},
        "expected": {"result": float(np.linalg.det(A_det3))}
    })

    # Inv
    A_inv = np.array([[4.0, 7.0], [2.0, 6.0]])
    test_cases.append({
        "name": "inv_2x2",
        "category": "linalg",
        "input": {"a": A_inv.tolist()},
        "expected": {"result": np.linalg.inv(A_inv).tolist()}
    })

    # Solve
    A_solve = np.array([[3.0, 1.0], [1.0, 2.0]])
    b_solve = np.array([9.0, 8.0])
    test_cases.append({
        "name": "solve_2x2",
        "category": "linalg",
        "input": {"a": A_solve.tolist(), "b": b_solve.tolist()},
        "expected": {"result": np.linalg.solve(A_solve, b_solve).tolist()}
    })

    # SVD
    A_svd = np.array([[1.0, 0.0, 0.0, 0.0, 2.0],
                       [0.0, 0.0, 3.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0, 0.0, 0.0]])
    U, S, Vt = np.linalg.svd(A_svd, full_matrices=True)
    test_cases.append({
        "name": "svd_4x5",
        "category": "linalg",
        "input": {"a": A_svd.tolist()},
        "expected": {
            "s": S.tolist(),  # just check singular values
        }
    })

    # Simpler SVD for full reconstruction check
    A_svd2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _, S2, _ = np.linalg.svd(A_svd2, full_matrices=True)
    test_cases.append({
        "name": "svd_3x2",
        "category": "linalg",
        "input": {"a": A_svd2.tolist()},
        "expected": {"s": S2.tolist()}
    })

    # Eig (symmetric for reliable real eigenvalues)
    A_eig = np.array([[2.0, 1.0], [1.0, 3.0]])
    eigvals, _ = np.linalg.eig(A_eig)
    eigvals_sorted = np.sort(eigvals).tolist()
    test_cases.append({
        "name": "eig_symmetric_2x2",
        "category": "linalg",
        "input": {"a": A_eig.tolist()},
        "expected": {"eigenvalues_sorted": eigvals_sorted}
    })

    A_eig3 = np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]])
    eigvals3, _ = np.linalg.eig(A_eig3)
    test_cases.append({
        "name": "eig_symmetric_3x3",
        "category": "linalg",
        "input": {"a": A_eig3.tolist()},
        "expected": {"eigenvalues_sorted": np.sort(eigvals3).tolist()}
    })

    # Cholesky
    A_chol = np.array([[4.0, 2.0], [2.0, 3.0]])
    L = np.linalg.cholesky(A_chol)
    test_cases.append({
        "name": "cholesky_2x2",
        "category": "linalg",
        "input": {"a": A_chol.tolist()},
        "expected": {"result": L.tolist()}
    })

    A_chol3 = np.array([[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]])
    L3 = np.linalg.cholesky(A_chol3)
    test_cases.append({
        "name": "cholesky_3x3",
        "category": "linalg",
        "input": {"a": A_chol3.tolist()},
        "expected": {"result": L3.tolist()}
    })

    # QR
    A_qr = np.array([[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0], [1.0, -1.0, 0.0]])
    Q, R = np.linalg.qr(A_qr)
    test_cases.append({
        "name": "qr_4x3",
        "category": "linalg",
        "input": {"a": A_qr.tolist()},
        "expected": {
            "q": Q.tolist(),
            "r": R.tolist(),
        }
    })

    # Lstsq (overdetermined system)
    A_lstsq = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b_lstsq = np.array([1.0, 2.0, 2.0])
    x_lstsq, _, _, _ = np.linalg.lstsq(A_lstsq, b_lstsq, rcond=None)
    test_cases.append({
        "name": "lstsq_3x2",
        "category": "linalg",
        "input": {"a": A_lstsq.tolist(), "b": b_lstsq.tolist()},
        "expected": {"result": x_lstsq.tolist()}
    })

    # Norm (vector, ord=2)
    v_norm = np.array([3.0, 4.0])
    test_cases.append({
        "name": "norm_vec2",
        "category": "linalg",
        "input": {"a": v_norm.tolist(), "ord": 2},
        "expected": {"result": float(np.linalg.norm(v_norm, ord=2))}
    })

    # Norm (vector, ord=1)
    test_cases.append({
        "name": "norm_vec1",
        "category": "linalg",
        "input": {"a": v_norm.tolist(), "ord": 1},
        "expected": {"result": float(np.linalg.norm(v_norm, ord=1))}
    })

    # Trace
    T = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_cases.append({
        "name": "trace_3x3",
        "category": "linalg",
        "input": {"a": T.tolist()},
        "expected": {"result": float(np.trace(T))}
    })

    # =========================================================================
    # STATISTICS
    # =========================================================================

    S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    test_cases.append({
        "name": "mean_all",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.mean(S))}
    })

    test_cases.append({
        "name": "mean_axis0",
        "category": "stats",
        "input": {"a": S.tolist(), "axis": 0},
        "expected": {"result": np.mean(S, axis=0).tolist()}
    })

    test_cases.append({
        "name": "mean_axis1",
        "category": "stats",
        "input": {"a": S.tolist(), "axis": 1},
        "expected": {"result": np.mean(S, axis=1).tolist()}
    })

    # Std (population, ddof=0)
    test_cases.append({
        "name": "std_all",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.std(S))}
    })

    # Var (population, ddof=0)
    test_cases.append({
        "name": "var_all",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.var(S))}
    })

    test_cases.append({
        "name": "min_all",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.min(S))}
    })

    test_cases.append({
        "name": "max_all",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.max(S))}
    })

    test_cases.append({
        "name": "sum_all_stats",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.sum(S))}
    })

    test_cases.append({
        "name": "prod_all_stats",
        "category": "stats",
        "input": {"a": S.tolist()},
        "expected": {"result": float(np.prod(S))}
    })

    # Cumsum
    C_cs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_cases.append({
        "name": "cumsum_1d",
        "category": "stats",
        "input": {"a": C_cs.tolist()},
        "expected": {"result": np.cumsum(C_cs).tolist()}
    })

    # Cumprod
    test_cases.append({
        "name": "cumprod_1d",
        "category": "stats",
        "input": {"a": C_cs.tolist()},
        "expected": {"result": np.cumprod(C_cs).tolist()}
    })

    # Percentile
    P = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    test_cases.append({
        "name": "percentile_50",
        "category": "stats",
        "input": {"a": P.tolist(), "q": 50.0},
        "expected": {"result": float(np.percentile(P, 50.0))}
    })

    test_cases.append({
        "name": "percentile_25",
        "category": "stats",
        "input": {"a": P.tolist(), "q": 25.0},
        "expected": {"result": float(np.percentile(P, 25.0))}
    })

    # Median
    test_cases.append({
        "name": "median",
        "category": "stats",
        "input": {"a": P.tolist()},
        "expected": {"result": float(np.median(P))}
    })

    # Corrcoef
    x_corr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_corr = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
    test_cases.append({
        "name": "corrcoef",
        "category": "stats",
        "input": {"x": x_corr.tolist(), "y": y_corr.tolist()},
        "expected": {"result": np.corrcoef(x_corr, y_corr).tolist()}
    })

    # Cov (2D: each row is a variable)
    cov_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    test_cases.append({
        "name": "cov_2d",
        "category": "stats",
        "input": {"a": cov_data.tolist()},
        "expected": {"result": np.cov(cov_data).tolist()}
    })

    # =========================================================================
    # SORTING
    # =========================================================================

    unsorted = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    test_cases.append({
        "name": "sort_1d",
        "category": "sorting",
        "input": {"a": unsorted.tolist()},
        "expected": {"result": np.sort(unsorted).tolist()}
    })

    test_cases.append({
        "name": "argsort_1d",
        "category": "sorting",
        "input": {"a": unsorted.tolist()},
        "expected": {"result": np.argsort(unsorted).tolist()}
    })

    dup = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
    test_cases.append({
        "name": "unique",
        "category": "sorting",
        "input": {"a": dup.tolist()},
        "expected": {"result": np.unique(dup).tolist()}
    })

    sorted_arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    search_vals = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    test_cases.append({
        "name": "searchsorted",
        "category": "sorting",
        "input": {"sorted": sorted_arr.tolist(), "values": search_vals.tolist()},
        "expected": {"result": np.searchsorted(sorted_arr, search_vals).tolist()}
    })

    # Sort 2D along axis 0
    M2d = np.array([[3.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
    test_cases.append({
        "name": "sort_2d_axis0",
        "category": "sorting",
        "input": {"a": M2d.tolist(), "axis": 0},
        "expected": {"result": np.sort(M2d, axis=0).tolist()}
    })

    test_cases.append({
        "name": "sort_2d_axis1",
        "category": "sorting",
        "input": {"a": M2d.tolist(), "axis": 1},
        "expected": {"result": np.sort(M2d, axis=1).tolist()}
    })

    # =========================================================================
    # LOGIC
    # =========================================================================

    all_true = np.array([1.0, 2.0, 3.0])
    has_zero = np.array([1.0, 0.0, 3.0])
    test_cases.append({
        "name": "all_true",
        "category": "logic",
        "input": {"a": all_true.tolist()},
        "expected": {"result": bool(np.all(all_true))}
    })

    test_cases.append({
        "name": "all_false",
        "category": "logic",
        "input": {"a": has_zero.tolist()},
        "expected": {"result": bool(np.all(has_zero))}
    })

    test_cases.append({
        "name": "any_true",
        "category": "logic",
        "input": {"a": has_zero.tolist()},
        "expected": {"result": bool(np.any(has_zero))}
    })

    all_zeros = np.array([0.0, 0.0, 0.0])
    test_cases.append({
        "name": "any_false",
        "category": "logic",
        "input": {"a": all_zeros.tolist()},
        "expected": {"result": bool(np.any(all_zeros))}
    })

    # allclose
    a_close = np.array([1.0, 2.0, 3.0])
    b_close = np.array([1.0, 2.0000001, 3.0])
    test_cases.append({
        "name": "allclose_true",
        "category": "logic",
        "input": {"a": a_close.tolist(), "b": b_close.tolist(), "atol": 1e-6, "rtol": 1e-5},
        "expected": {"result": bool(np.allclose(a_close, b_close, atol=1e-6, rtol=1e-5))}
    })

    b_far = np.array([1.0, 2.1, 3.0])
    test_cases.append({
        "name": "allclose_false",
        "category": "logic",
        "input": {"a": a_close.tolist(), "b": b_far.tolist(), "atol": 1e-6, "rtol": 1e-5},
        "expected": {"result": bool(np.allclose(a_close, b_far, atol=1e-6, rtol=1e-5))}
    })

    # isnan, isinf
    # Use string markers for special floats since JSON cannot represent NaN/Inf.
    nan_arr = np.array([1.0, float('nan'), 3.0, float('inf'), float('-inf')])
    test_cases.append({
        "name": "isnan",
        "category": "logic",
        "input": {"a_special": [1.0, "NaN", 3.0, "Inf", "-Inf"]},
        "expected": {"result": np.isnan(nan_arr).astype(float).tolist()}
    })

    test_cases.append({
        "name": "isinf",
        "category": "logic",
        "input": {"a_special": [1.0, "NaN", 3.0, "Inf", "-Inf"]},
        "expected": {"result": np.isinf(nan_arr).astype(float).tolist()}
    })

    # greater, less, equal
    ga = np.array([1.0, 5.0, 3.0])
    gb = np.array([2.0, 3.0, 3.0])
    test_cases.append({
        "name": "greater",
        "category": "logic",
        "input": {"a": ga.tolist(), "b": gb.tolist()},
        "expected": {"result": np.greater(ga, gb).astype(float).tolist()}
    })

    test_cases.append({
        "name": "less",
        "category": "logic",
        "input": {"a": ga.tolist(), "b": gb.tolist()},
        "expected": {"result": np.less(ga, gb).astype(float).tolist()}
    })

    test_cases.append({
        "name": "equal",
        "category": "logic",
        "input": {"a": ga.tolist(), "b": gb.tolist()},
        "expected": {"result": np.equal(ga, gb).astype(float).tolist()}
    })

    # =========================================================================
    # SET OPERATIONS
    # =========================================================================

    sa = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sb = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    test_cases.append({
        "name": "intersect1d",
        "category": "setops",
        "input": {"a": sa.tolist(), "b": sb.tolist()},
        "expected": {"result": np.intersect1d(sa, sb).tolist()}
    })

    test_cases.append({
        "name": "union1d",
        "category": "setops",
        "input": {"a": sa.tolist(), "b": sb.tolist()},
        "expected": {"result": np.union1d(sa, sb).tolist()}
    })

    test_cases.append({
        "name": "setdiff1d",
        "category": "setops",
        "input": {"a": sa.tolist(), "b": sb.tolist()},
        "expected": {"result": np.setdiff1d(sa, sb).tolist()}
    })

    test_cases.append({
        "name": "in1d",
        "category": "setops",
        "input": {"a": sa.tolist(), "b": sb.tolist()},
        "expected": {"result": np.isin(sa, sb).astype(float).tolist()}
    })

    # =========================================================================
    # EINSUM
    # =========================================================================

    # matmul via einsum
    A_ein = np.array([[1.0, 2.0], [3.0, 4.0]])
    B_ein = np.array([[5.0, 6.0], [7.0, 8.0]])
    test_cases.append({
        "name": "einsum_matmul",
        "category": "einsum",
        "input": {
            "notation": "ij,jk->ik",
            "operands": [A_ein.tolist(), B_ein.tolist()]
        },
        "expected": {"result": np.einsum("ij,jk->ik", A_ein, B_ein).tolist()}
    })

    # trace via einsum
    T_ein = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    test_cases.append({
        "name": "einsum_trace",
        "category": "einsum",
        "input": {
            "notation": "ii->",
            "operands": [T_ein.tolist()]
        },
        "expected": {"result": float(np.einsum("ii->", T_ein))}
    })

    # outer product via einsum
    u_ein = np.array([1.0, 2.0, 3.0])
    v_ein = np.array([4.0, 5.0])
    test_cases.append({
        "name": "einsum_outer",
        "category": "einsum",
        "input": {
            "notation": "i,j->ij",
            "operands": [u_ein.tolist(), v_ein.tolist()]
        },
        "expected": {"result": np.einsum("i,j->ij", u_ein, v_ein).tolist()}
    })

    # batch matmul via einsum
    batch_a = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    batch_b = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 1.0], [1.0, 2.0]],
    ])
    test_cases.append({
        "name": "einsum_batch_matmul",
        "category": "einsum",
        "input": {
            "notation": "bij,bjk->bik",
            "operands": [batch_a.tolist(), batch_b.tolist()]
        },
        "expected": {"result": np.einsum("bij,bjk->bik", batch_a, batch_b).tolist()}
    })

    # row sums via einsum
    test_cases.append({
        "name": "einsum_row_sums",
        "category": "einsum",
        "input": {
            "notation": "ij->i",
            "operands": [A_ein.tolist()]
        },
        "expected": {"result": np.einsum("ij->i", A_ein).tolist()}
    })

    # column sums via einsum
    test_cases.append({
        "name": "einsum_col_sums",
        "category": "einsum",
        "input": {
            "notation": "ij->j",
            "operands": [A_ein.tolist()]
        },
        "expected": {"result": np.einsum("ij->j", A_ein).tolist()}
    })

    # dot product via einsum
    test_cases.append({
        "name": "einsum_dot",
        "category": "einsum",
        "input": {
            "notation": "i,i->",
            "operands": [u_ein.tolist(), np.array([4.0, 5.0, 6.0]).tolist()]
        },
        "expected": {"result": float(np.einsum("i,i->", u_ein, np.array([4.0, 5.0, 6.0])))}
    })

    return test_cases


if __name__ == "__main__":
    cases = generate()
    print(json.dumps({"test_cases": cases}, indent=2))
