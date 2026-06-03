package graphgo

// NumberOfSelfLoops returns the number of self-loops in a directed graph.
func NumberOfSelfLoops(g *DiGraph) int {
	count := 0
	for node := range g.succ {
		if g.succ[node][node] {
			count++
		}
	}
	return count
}

// SelfLoops returns the nodes that have self-loops in a directed graph.
func SelfLoops(g *DiGraph) []string {
	var nodes []string
	for node := range g.succ {
		if g.succ[node][node] {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// NumberOfSelfLoopsUndirected returns the number of self-loops in an undirected graph.
func NumberOfSelfLoopsUndirected(g *Graph) int {
	count := 0
	for node := range g.adj {
		if g.adj[node][node] {
			count++
		}
	}
	return count
}

// Density returns the edge density of a directed graph.
// density = |E| / (|V| * (|V| - 1)) for directed graphs.
// Returns 0 for graphs with fewer than 2 nodes.
func Density(g *DiGraph) float64 {
	n := len(g.succ)
	if n < 2 {
		return 0
	}
	e := g.NumberOfEdges()
	return float64(e) / float64(n*(n-1))
}

// DensityUndirected returns the edge density of an undirected graph.
// density = 2 * |E| / (|V| * (|V| - 1)) for undirected graphs.
// Returns 0 for graphs with fewer than 2 nodes.
func DensityUndirected(g *Graph) float64 {
	n := len(g.adj)
	if n < 2 {
		return 0
	}
	e := len(g.Edges())
	return 2.0 * float64(e) / float64(n*(n-1))
}

// IsTree returns true if the directed graph is a tree (a connected DAG where
// each node except the root has exactly one parent).
func IsTree(g *DiGraph) bool {
	n := len(g.succ)
	if n == 0 {
		return false
	}
	// Must be a DAG.
	if !IsDAG(g) {
		return false
	}
	// Must be weakly connected.
	if !IsWeaklyConnected(g) {
		return false
	}
	// Number of edges must be n-1.
	if g.NumberOfEdges() != n-1 {
		return false
	}
	return true
}

// IsForest returns true if the directed graph is a forest (a DAG where
// each weakly connected component is a tree).
func IsForest(g *DiGraph) bool {
	if len(g.succ) == 0 {
		return true
	}
	if !IsDAG(g) {
		return false
	}
	// Each weakly connected component must have exactly n-1 edges.
	components := WeaklyConnectedComponents(g)
	for _, comp := range components {
		nodeSet := make(map[string]bool, len(comp))
		for _, n := range comp {
			nodeSet[n] = true
		}
		edgeCount := 0
		for _, n := range comp {
			for s := range g.succ[n] {
				if nodeSet[s] {
					edgeCount++
				}
			}
		}
		if edgeCount != len(comp)-1 {
			return false
		}
	}
	return true
}

// IsTreeUndirected returns true if the undirected graph is a tree
// (connected and has exactly n-1 edges).
func IsTreeUndirected(g *Graph) bool {
	n := len(g.adj)
	if n == 0 {
		return false
	}
	if !g.IsConnected() {
		return false
	}
	return len(g.Edges()) == n-1
}

// IsForestUndirected returns true if the undirected graph is a forest
// (each connected component is a tree).
func IsForestUndirected(g *Graph) bool {
	if len(g.adj) == 0 {
		return true
	}
	components := g.ConnectedComponents()
	edgeSet := make(map[string]bool)
	for _, e := range g.Edges() {
		edgeSet[undirectedEdgeKey(e.A, e.B)] = true
	}

	for _, comp := range components {
		nodeSet := make(map[string]bool, len(comp))
		for _, n := range comp {
			nodeSet[n] = true
		}
		edgeCount := 0
		for _, n := range comp {
			for nb := range g.adj[n] {
				if nodeSet[nb] && n < nb {
					edgeCount++
				}
			}
		}
		if edgeCount != len(comp)-1 {
			return false
		}
	}
	return true
}
