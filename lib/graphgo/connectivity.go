package graphgo

// WeaklyConnectedComponents returns the weakly connected components of a directed graph.
// Each component is a list of node names. A weakly connected component is a maximal
// set of nodes such that there is an undirected path between every pair.
func WeaklyConnectedComponents(g *DiGraph) [][]string {
	visited := make(map[string]bool, len(g.succ))
	var components [][]string

	for node := range g.succ {
		if visited[node] {
			continue
		}
		// BFS treating edges as undirected.
		var component []string
		queue := []string{node}
		visited[node] = true
		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			component = append(component, curr)
			for s := range g.succ[curr] {
				if !visited[s] {
					visited[s] = true
					queue = append(queue, s)
				}
			}
			for p := range g.pred[curr] {
				if !visited[p] {
					visited[p] = true
					queue = append(queue, p)
				}
			}
		}
		components = append(components, component)
	}
	return components
}

// StronglyConnectedComponents returns the strongly connected components of a directed
// graph using Tarjan's algorithm. Each component is a list of node names.
func StronglyConnectedComponents(g *DiGraph) [][]string {
	index := 0
	nodeIndex := make(map[string]int)
	lowlink := make(map[string]int)
	onStack := make(map[string]bool)
	var stack []string
	var components [][]string

	var strongconnect func(v string)
	strongconnect = func(v string) {
		nodeIndex[v] = index
		lowlink[v] = index
		index++
		stack = append(stack, v)
		onStack[v] = true

		for w := range g.succ[v] {
			if _, ok := nodeIndex[w]; !ok {
				strongconnect(w)
				if lowlink[w] < lowlink[v] {
					lowlink[v] = lowlink[w]
				}
			} else if onStack[w] {
				if nodeIndex[w] < lowlink[v] {
					lowlink[v] = nodeIndex[w]
				}
			}
		}

		if lowlink[v] == nodeIndex[v] {
			var component []string
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				component = append(component, w)
				if w == v {
					break
				}
			}
			components = append(components, component)
		}
	}

	for node := range g.succ {
		if _, ok := nodeIndex[node]; !ok {
			strongconnect(node)
		}
	}
	return components
}

// IsWeaklyConnected returns true if the directed graph is weakly connected,
// i.e. the underlying undirected graph is connected.
func IsWeaklyConnected(g *DiGraph) bool {
	if len(g.succ) == 0 {
		return false
	}
	components := WeaklyConnectedComponents(g)
	return len(components) == 1
}

// IsStronglyConnected returns true if the directed graph is strongly connected,
// i.e. there is a directed path between every pair of nodes.
func IsStronglyConnected(g *DiGraph) bool {
	if len(g.succ) == 0 {
		return false
	}
	components := StronglyConnectedComponents(g)
	return len(components) == 1
}

// IsConnected returns true if the undirected graph is connected.
func (g *Graph) IsConnected() bool {
	if len(g.adj) == 0 {
		return false
	}
	// BFS from any node.
	var start string
	for n := range g.adj {
		start = n
		break
	}
	visited := make(map[string]bool, len(g.adj))
	queue := []string{start}
	visited[start] = true
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for nb := range g.adj[curr] {
			if !visited[nb] {
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	return len(visited) == len(g.adj)
}

// ConnectedComponents returns the connected components of an undirected graph.
// Each component is a list of node names.
func (g *Graph) ConnectedComponents() [][]string {
	visited := make(map[string]bool, len(g.adj))
	var components [][]string

	for node := range g.adj {
		if visited[node] {
			continue
		}
		var component []string
		queue := []string{node}
		visited[node] = true
		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			component = append(component, curr)
			for nb := range g.adj[curr] {
				if !visited[nb] {
					visited[nb] = true
					queue = append(queue, nb)
				}
			}
		}
		components = append(components, component)
	}
	return components
}
