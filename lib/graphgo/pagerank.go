package graphgo

// PageRank computes the PageRank of each node in a directed graph using the
// power iteration method. The damping factor is 0.85 and the algorithm runs
// for a maximum of 100 iterations or until convergence (epsilon = 1e-6).
func PageRank(g *DiGraph) map[string]float64 {
	return PageRankWithParams(g, 0.85, 100, 1e-6)
}

// PageRankWithParams computes PageRank with configurable parameters.
func PageRankWithParams(g *DiGraph, damping float64, maxIter int, epsilon float64) map[string]float64 {
	n := len(g.succ)
	if n == 0 {
		return make(map[string]float64)
	}

	nodes := g.Nodes()
	rank := make(map[string]float64, n)
	initial := 1.0 / float64(n)
	for _, node := range nodes {
		rank[node] = initial
	}

	for iter := 0; iter < maxIter; iter++ {
		newRank := make(map[string]float64, n)
		// Sum of ranks of dangling nodes (no outgoing edges).
		danglingSum := 0.0
		for _, node := range nodes {
			if len(g.succ[node]) == 0 {
				danglingSum += rank[node]
			}
		}

		for _, node := range nodes {
			s := 0.0
			for p := range g.pred[node] {
				outDeg := len(g.succ[p])
				if outDeg > 0 {
					s += rank[p] / float64(outDeg)
				}
			}
			newRank[node] = (1-damping)/float64(n) + damping*(s+danglingSum/float64(n))
		}

		// Check convergence.
		maxDiff := 0.0
		for _, node := range nodes {
			diff := newRank[node] - rank[node]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		rank = newRank
		if maxDiff < epsilon {
			break
		}
	}

	return rank
}
