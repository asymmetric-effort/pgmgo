package graphgo

import "sort"

// Moralize converts a directed graph into an undirected moral graph.
// For each node, all pairs of parents are connected (married), then
// all edge directions are dropped.
func Moralize(g *DiGraph) *Graph {
	moral := NewGraph()

	// Add all nodes.
	for _, n := range g.Nodes() {
		moral.AddNode(n)
	}

	// Add undirected version of every directed edge.
	for _, e := range g.Edges() {
		moral.AddEdge(e.Src, e.Dst)
	}

	// Marry co-parents: for each node, connect all pairs of its parents.
	for _, n := range g.Nodes() {
		parents := g.Parents(n)
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				moral.AddEdge(parents[i], parents[j])
			}
		}
	}

	return moral
}

// Triangulate makes an undirected graph chordal using the given elimination
// order. For each node in order, all its current neighbors are connected to
// each other (fill edges), then the node is eliminated. The returned graph
// is a copy of the original with all fill edges added.
func Triangulate(g *Graph, order []string) *Graph {
	// Build the result as a copy of the input.
	result := g.Copy()

	// Build a working adjacency structure for the elimination process.
	adj := make(map[string]map[string]bool, len(order))
	for _, n := range g.Nodes() {
		adj[n] = make(map[string]bool)
	}
	for _, e := range g.Edges() {
		adj[e.A][e.B] = true
		adj[e.B][e.A] = true
	}

	for _, node := range order {
		neighbors := make([]string, 0, len(adj[node]))
		for nb := range adj[node] {
			neighbors = append(neighbors, nb)
		}

		// Connect all pairs of neighbors (fill edges).
		for i := 0; i < len(neighbors); i++ {
			for j := i + 1; j < len(neighbors); j++ {
				a, b := neighbors[i], neighbors[j]
				if !adj[a][b] {
					adj[a][b] = true
					adj[b][a] = true
					result.AddEdge(a, b)
				}
			}
		}

		// Eliminate node from working graph.
		for nb := range adj[node] {
			delete(adj[nb], node)
		}
		delete(adj, node)
	}

	return result
}

// IsChordal checks whether an undirected graph is chordal (triangulated).
// A graph is chordal if every cycle of length >= 4 has a chord.
// This uses the maximum cardinality search (MCS) algorithm: produce a
// perfect elimination ordering via MCS, then verify it.
func IsChordal(g *Graph) bool {
	nodes := g.Nodes()
	n := len(nodes)
	if n <= 3 {
		return true
	}

	// Maximum cardinality search to find a candidate perfect elimination ordering.
	order := make([]string, 0, n)
	weight := make(map[string]int, n)
	numbered := make(map[string]bool, n)

	for _, nd := range nodes {
		weight[nd] = 0
	}

	for i := 0; i < n; i++ {
		// Pick unnumbered node with max weight (break ties by lexicographic order for determinism).
		best := ""
		bestW := -1
		candidates := make([]string, 0)
		for _, nd := range nodes {
			if numbered[nd] {
				continue
			}
			if weight[nd] > bestW {
				bestW = weight[nd]
				candidates = candidates[:0]
				candidates = append(candidates, nd)
			} else if weight[nd] == bestW {
				candidates = append(candidates, nd)
			}
		}
		sort.Strings(candidates)
		best = candidates[0]

		order = append(order, best)
		numbered[best] = true
		for _, nb := range g.Neighbors(best) {
			if !numbered[nb] {
				weight[nb]++
			}
		}
	}

	// MCS selects nodes assigning labels n, n-1, ..., 1 in selection order.
	// The perfect elimination ordering is the reverse of the selection order
	// (node labeled 1 first, node labeled n last).
	// Reverse the order to get the PEO.
	peo := make([]string, n)
	for i, nd := range order {
		peo[n-1-i] = nd
	}

	// Verify PEO: for each node v in the PEO, the neighbors of v that
	// come later in the PEO must form a clique.
	pos := make(map[string]int, n)
	for i, nd := range peo {
		pos[nd] = i
	}

	for idx, v := range peo {
		// Collect neighbors that appear later in the PEO.
		var later []string
		for _, nb := range g.Neighbors(v) {
			if pos[nb] > idx {
				later = append(later, nb)
			}
		}
		// Check that later neighbors form a clique.
		for i := 0; i < len(later); i++ {
			for j := i + 1; j < len(later); j++ {
				if !g.HasEdge(later[i], later[j]) {
					return false
				}
			}
		}
	}

	return true
}
