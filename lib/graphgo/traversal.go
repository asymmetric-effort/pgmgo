package graphgo

import "sort"

// BFS returns nodes reachable from source in breadth-first order in a directed graph.
// The source node is included as the first element.
func BFS(g *DiGraph, source string) []string {
	if !g.HasNode(source) {
		return nil
	}
	visited := map[string]bool{source: true}
	queue := []string{source}
	var order []string
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		order = append(order, curr)
		// Sort successors for deterministic output.
		succs := make([]string, 0, len(g.succ[curr]))
		for s := range g.succ[curr] {
			succs = append(succs, s)
		}
		sort.Strings(succs)
		for _, s := range succs {
			if !visited[s] {
				visited[s] = true
				queue = append(queue, s)
			}
		}
	}
	return order
}

// DFS returns nodes reachable from source in depth-first order in a directed graph.
// The source node is included as the first element.
func DFS(g *DiGraph, source string) []string {
	if !g.HasNode(source) {
		return nil
	}
	visited := make(map[string]bool)
	var order []string
	var visit func(string)
	visit = func(node string) {
		visited[node] = true
		order = append(order, node)
		// Sort successors for deterministic output.
		succs := make([]string, 0, len(g.succ[node]))
		for s := range g.succ[node] {
			succs = append(succs, s)
		}
		sort.Strings(succs)
		for _, s := range succs {
			if !visited[s] {
				visit(s)
			}
		}
	}
	visit(source)
	return order
}

// BFSUndirected returns nodes reachable from source in breadth-first order
// in an undirected graph.
func BFSUndirected(g *Graph, source string) []string {
	if !g.HasNode(source) {
		return nil
	}
	visited := map[string]bool{source: true}
	queue := []string{source}
	var order []string
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		order = append(order, curr)
		nbs := make([]string, 0, len(g.adj[curr]))
		for nb := range g.adj[curr] {
			nbs = append(nbs, nb)
		}
		sort.Strings(nbs)
		for _, nb := range nbs {
			if !visited[nb] {
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	return order
}

// DFSUndirected returns nodes reachable from source in depth-first order
// in an undirected graph.
func DFSUndirected(g *Graph, source string) []string {
	if !g.HasNode(source) {
		return nil
	}
	visited := make(map[string]bool)
	var order []string
	var visit func(string)
	visit = func(node string) {
		visited[node] = true
		order = append(order, node)
		nbs := make([]string, 0, len(g.adj[node]))
		for nb := range g.adj[node] {
			nbs = append(nbs, nb)
		}
		sort.Strings(nbs)
		for _, nb := range nbs {
			if !visited[nb] {
				visit(nb)
			}
		}
	}
	visit(source)
	return order
}
