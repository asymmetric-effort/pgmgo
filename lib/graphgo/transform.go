package graphgo

import "sort"

// Reverse returns a new DiGraph with all edges reversed.
func (g *DiGraph) Reverse() *DiGraph {
	r := NewDiGraph()
	for n := range g.succ {
		r.AddNode(n)
		for k, v := range g.nodeAttrs[n] {
			r.nodeAttrs[n][k] = v
		}
	}
	for src, dsts := range g.succ {
		for dst := range dsts {
			r.AddEdge(dst, src)
			for k, v := range g.edgeAttrs[edgeKey(src, dst)] {
				r.edgeAttrs[edgeKey(dst, src)][k] = v
			}
		}
	}
	return r
}

// ContractNodes merges the given nodes into a single node (the first node in the list).
// All edges to/from the merged nodes are redirected to the surviving node.
// Self-loops created by contraction are not added.
// Returns a new DiGraph.
func ContractNodes(g *DiGraph, nodes []string) *DiGraph {
	if len(nodes) == 0 {
		return g.Copy()
	}

	surviving := nodes[0]
	mergeSet := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		mergeSet[n] = true
	}

	result := NewDiGraph()

	// Add all nodes, mapping merged nodes to surviving.
	for n := range g.succ {
		if mergeSet[n] {
			result.AddNode(surviving)
		} else {
			result.AddNode(n)
		}
	}

	// Add edges with remapping.
	for src, dsts := range g.succ {
		from := src
		if mergeSet[from] {
			from = surviving
		}
		for dst := range dsts {
			to := dst
			if mergeSet[to] {
				to = surviving
			}
			if from != to { // skip self-loops from contraction
				result.AddEdge(from, to)
			}
		}
	}

	return result
}

// ContractNodesUndirected merges the given nodes in an undirected graph.
func ContractNodesUndirected(g *Graph, nodes []string) *Graph {
	if len(nodes) == 0 {
		return g.Copy()
	}

	surviving := nodes[0]
	mergeSet := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		mergeSet[n] = true
	}

	result := NewGraph()
	for n := range g.adj {
		if mergeSet[n] {
			result.AddNode(surviving)
		} else {
			result.AddNode(n)
		}
	}

	for _, e := range g.Edges() {
		a, b := e.A, e.B
		if mergeSet[a] {
			a = surviving
		}
		if mergeSet[b] {
			b = surviving
		}
		if a != b {
			result.AddEdge(a, b)
		}
	}

	return result
}

// LineGraph returns the line graph of an undirected graph.
// In the line graph, each edge of the original graph becomes a node,
// and two nodes in the line graph are connected if the corresponding
// edges in the original graph share an endpoint.
// Nodes in the line graph are named "A-B" (with A < B lexicographically).
func LineGraph(g *Graph) *Graph {
	edges := g.Edges()
	lg := NewGraph()

	// Create a node for each edge.
	type edgeInfo struct {
		a, b string
		name string
	}
	edgeNodes := make([]edgeInfo, len(edges))
	for i, e := range edges {
		a, b := e.A, e.B
		if a > b {
			a, b = b, a
		}
		name := a + "-" + b
		edgeNodes[i] = edgeInfo{a: a, b: b, name: name}
		lg.AddNode(name)
	}

	// Connect edge-nodes that share an endpoint.
	for i := 0; i < len(edgeNodes); i++ {
		for j := i + 1; j < len(edgeNodes); j++ {
			ei, ej := edgeNodes[i], edgeNodes[j]
			if ei.a == ej.a || ei.a == ej.b || ei.b == ej.a || ei.b == ej.b {
				lg.AddEdge(ei.name, ej.name)
			}
		}
	}

	return lg
}

// LineGraphDirected returns the line graph of a directed graph.
// Each edge (u,v) becomes a node "u->v". Two nodes in the line graph are
// connected by an edge if the destination of the first equals the source
// of the second.
func LineGraphDirected(g *DiGraph) *DiGraph {
	edges := g.Edges()
	lg := NewDiGraph()

	type edgeInfo struct {
		src, dst string
		name     string
	}
	edgeNodes := make([]edgeInfo, len(edges))
	for i, e := range edges {
		name := e.Src + "->" + e.Dst
		edgeNodes[i] = edgeInfo{src: e.Src, dst: e.Dst, name: name}
		lg.AddNode(name)
	}

	// Sort for deterministic output.
	sort.Slice(edgeNodes, func(i, j int) bool {
		return edgeNodes[i].name < edgeNodes[j].name
	})

	for i := 0; i < len(edgeNodes); i++ {
		for j := 0; j < len(edgeNodes); j++ {
			if i == j {
				continue
			}
			if edgeNodes[i].dst == edgeNodes[j].src {
				lg.AddEdge(edgeNodes[i].name, edgeNodes[j].name)
			}
		}
	}

	return lg
}
