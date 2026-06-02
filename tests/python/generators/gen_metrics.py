"""
Fixture generator for src/metrics package.

Generates test cases for structural Hamming distance (SHD) and confusion
matrices by comparing a true graph against an estimated graph with known
differences.
"""

import networkx as nx


def generate() -> list[dict]:
    """Generate test cases for the metrics package."""
    test_cases = []

    test_cases.append(_test_shd_one_extra_edge())
    test_cases.append(_test_shd_reversal())
    test_cases.append(_test_adjacency_confusion())
    test_cases.append(_test_orientation_confusion())

    return test_cases


def _test_shd_one_extra_edge():
    """SHD between true graph and estimated graph with one extra edge."""
    true_edges = [("A", "B"), ("B", "C"), ("A", "C")]
    est_edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")]

    true_g = nx.DiGraph(true_edges)
    est_g = nx.DiGraph(est_edges)

    # Compute SHD manually: one extra edge C->D in estimated.
    shd = _compute_shd(true_g, est_g)

    return {
        "name": "shd_one_extra_edge",
        "description": "SHD between true graph (3 edges) and estimated with one extra edge",
        "input": {
            "true_edges": true_edges,
            "estimated_edges": est_edges,
            "true_nodes": sorted(true_g.nodes()),
            "estimated_nodes": sorted(est_g.nodes()),
        },
        "expected": {
            "shd": shd,
        },
    }


def _test_shd_reversal():
    """SHD between true graph and estimated graph with one reversed edge."""
    true_edges = [("A", "B"), ("B", "C")]
    est_edges = [("B", "A"), ("B", "C")]

    true_g = nx.DiGraph(true_edges)
    est_g = nx.DiGraph(est_edges)

    shd = _compute_shd(true_g, est_g)

    return {
        "name": "shd_reversal",
        "description": "SHD between true graph and estimated with one reversed edge (A->B vs B->A)",
        "input": {
            "true_edges": true_edges,
            "estimated_edges": est_edges,
            "true_nodes": sorted(true_g.nodes()),
            "estimated_nodes": sorted(est_g.nodes()),
        },
        "expected": {
            "shd": shd,
        },
    }


def _test_adjacency_confusion():
    """Adjacency confusion matrix test."""
    true_edges = [("A", "B"), ("B", "C"), ("A", "C")]
    est_edges = [("A", "B"), ("B", "C"), ("C", "D")]

    true_g = nx.DiGraph(true_edges)
    est_g = nx.DiGraph(est_edges)
    all_nodes = sorted(set(true_g.nodes()) | set(est_g.nodes()))

    # Build undirected adjacency sets.
    true_adj = set()
    for u, v in true_g.edges():
        true_adj.add((min(u, v), max(u, v)))
    est_adj = set()
    for u, v in est_g.edges():
        est_adj.add((min(u, v), max(u, v)))

    tp = fp = tn = fn = 0
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            pair = (all_nodes[i], all_nodes[j])
            in_true = pair in true_adj
            in_est = pair in est_adj
            if in_true and in_est:
                tp += 1
            elif not in_true and in_est:
                fp += 1
            elif in_true and not in_est:
                fn += 1
            else:
                tn += 1

    return {
        "name": "adjacency_confusion",
        "description": "Adjacency confusion matrix: true has A->C, est has C->D instead",
        "input": {
            "true_edges": true_edges,
            "estimated_edges": est_edges,
            "true_nodes": sorted(true_g.nodes()),
            "estimated_nodes": sorted(est_g.nodes()),
        },
        "expected": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def _test_orientation_confusion():
    """Orientation confusion matrix test among adjacency TPs."""
    true_edges = [("A", "B"), ("B", "C")]
    est_edges = [("A", "B"), ("C", "B")]

    true_g = nx.DiGraph(true_edges)
    est_g = nx.DiGraph(est_edges)

    # Adjacency TPs: both have (A,B) and (B,C) as undirected edges.
    # For (A,B): true has A->B, est has A->B => TP for A->B, TN for B->A.
    # For (B,C): true has B->C, est has C->B => FN for B->C, FP for C->B.
    tp = fp = tn = fn = 0

    true_adj = set()
    for u, v in true_g.edges():
        true_adj.add((min(u, v), max(u, v)))
    est_adj = set()
    for u, v in est_g.edges():
        est_adj.add((min(u, v), max(u, v)))

    adj_tp_pairs = []
    for pair in true_adj:
        if pair in est_adj:
            adj_tp_pairs.append(pair)

    for a, b in adj_tp_pairs:
        for u, v in [(a, b), (b, a)]:
            in_true = true_g.has_edge(u, v)
            in_est = est_g.has_edge(u, v)
            if in_true and in_est:
                tp += 1
            elif not in_true and in_est:
                fp += 1
            elif in_true and not in_est:
                fn += 1
            else:
                tn += 1

    return {
        "name": "orientation_confusion",
        "description": "Orientation confusion matrix: B->C reversed to C->B",
        "input": {
            "true_edges": true_edges,
            "estimated_edges": est_edges,
            "true_nodes": sorted(true_g.nodes()),
            "estimated_nodes": sorted(est_g.nodes()),
        },
        "expected": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def _compute_shd(true_g: nx.DiGraph, est_g: nx.DiGraph) -> int:
    """
    Compute Structural Hamming Distance manually.
    Counts edge additions, deletions, and reversals (each reversal = 1).
    """
    true_edges = set(true_g.edges())
    est_edges = set(est_g.edges())

    # Collect all undirected pairs.
    all_pairs = set()
    for u, v in true_edges | est_edges:
        all_pairs.add((min(u, v), max(u, v)))

    dist = 0
    for a, b in all_pairs:
        t_ab = (a, b) in true_edges
        t_ba = (b, a) in true_edges
        e_ab = (a, b) in est_edges
        e_ba = (b, a) in est_edges

        if t_ab == e_ab and t_ba == e_ba:
            continue

        diffs = 0
        if t_ab != e_ab:
            diffs += 1
        if t_ba != e_ba:
            diffs += 1

        if diffs == 2:
            if (t_ab and e_ba and not t_ba and not e_ab) or \
               (t_ba and e_ab and not t_ab and not e_ba):
                dist += 1  # reversal
            else:
                dist += 2
        else:
            dist += 1

    return dist
