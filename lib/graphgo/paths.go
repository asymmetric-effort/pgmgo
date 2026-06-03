package graphgo

import "fmt"

// ShortestPath returns the shortest path (by number of edges) from source to
// target in a directed graph using BFS. Returns an error if no path exists.
func ShortestPath(g *DiGraph, source, target string) ([]string, error) {
	if !g.HasNode(source) {
		return nil, fmt.Errorf("graphgo: source node %q not found", source)
	}
	if !g.HasNode(target) {
		return nil, fmt.Errorf("graphgo: target node %q not found", target)
	}
	if source == target {
		return []string{source}, nil
	}

	prev := make(map[string]string)
	visited := map[string]bool{source: true}
	queue := []string{source}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for s := range g.succ[curr] {
			if !visited[s] {
				visited[s] = true
				prev[s] = curr
				if s == target {
					return reconstructPath(prev, source, target), nil
				}
				queue = append(queue, s)
			}
		}
	}
	return nil, fmt.Errorf("graphgo: no path from %q to %q", source, target)
}

// reconstructPath builds a path from the predecessor map.
func reconstructPath(prev map[string]string, source, target string) []string {
	var path []string
	curr := target
	for curr != source {
		path = append(path, curr)
		curr = prev[curr]
	}
	path = append(path, source)
	// reverse
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

// AllShortestPaths returns all shortest paths from source to target in a directed
// graph using BFS. Returns nil if no path exists.
func AllShortestPaths(g *DiGraph, source, target string) [][]string {
	if !g.HasNode(source) || !g.HasNode(target) {
		return nil
	}
	if source == target {
		return [][]string{{source}}
	}

	// BFS to find distances and all predecessors at shortest distance.
	dist := map[string]int{source: 0}
	preds := make(map[string][]string) // node -> list of predecessors on shortest paths
	queue := []string{source}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		d := dist[curr]
		for s := range g.succ[curr] {
			if _, ok := dist[s]; !ok {
				dist[s] = d + 1
				preds[s] = []string{curr}
				queue = append(queue, s)
			} else if dist[s] == d+1 {
				preds[s] = append(preds[s], curr)
			}
		}
	}

	if _, ok := dist[target]; !ok {
		return nil
	}

	// Reconstruct all paths by backtracking from target.
	var result [][]string
	var backtrack func(node string, path []string)
	backtrack = func(node string, path []string) {
		newPath := make([]string, len(path)+1)
		newPath[0] = node
		copy(newPath[1:], path)
		if node == source {
			result = append(result, newPath)
			return
		}
		for _, p := range preds[node] {
			backtrack(p, newPath)
		}
	}
	backtrack(target, nil)
	return result
}

// HasPath returns true if there is a directed path from source to target.
func HasPath(g *DiGraph, source, target string) bool {
	if !g.HasNode(source) || !g.HasNode(target) {
		return false
	}
	if source == target {
		return true
	}
	visited := map[string]bool{source: true}
	queue := []string{source}
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for s := range g.succ[curr] {
			if s == target {
				return true
			}
			if !visited[s] {
				visited[s] = true
				queue = append(queue, s)
			}
		}
	}
	return false
}

// ShortestPathUndirected returns the shortest path in an undirected graph using BFS.
func ShortestPathUndirected(g *Graph, source, target string) ([]string, error) {
	if !g.HasNode(source) {
		return nil, fmt.Errorf("graphgo: source node %q not found", source)
	}
	if !g.HasNode(target) {
		return nil, fmt.Errorf("graphgo: target node %q not found", target)
	}
	if source == target {
		return []string{source}, nil
	}

	prev := make(map[string]string)
	visited := map[string]bool{source: true}
	queue := []string{source}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for nb := range g.adj[curr] {
			if !visited[nb] {
				visited[nb] = true
				prev[nb] = curr
				if nb == target {
					return reconstructPath(prev, source, target), nil
				}
				queue = append(queue, nb)
			}
		}
	}
	return nil, fmt.Errorf("graphgo: no path from %q to %q", source, target)
}
