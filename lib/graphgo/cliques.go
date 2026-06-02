package graphgo

import "sort"

// MaxCliques finds all maximal cliques in an undirected graph using an
// iterative version of the Bron-Kerbosch algorithm with pivoting.
func MaxCliques(g *Graph) [][]string {
	// Build sorted node list and adjacency as sets for fast lookup.
	nodes := g.Nodes()
	if len(nodes) == 0 {
		return nil
	}
	sort.Strings(nodes)

	adj := make(map[string]map[string]bool, len(nodes))
	for _, n := range nodes {
		adj[n] = make(map[string]bool)
	}
	for _, e := range g.Edges() {
		adj[e.A][e.B] = true
		adj[e.B][e.A] = true
	}

	type frame struct {
		R map[string]bool
		P map[string]bool
		X map[string]bool
	}

	var result [][]string

	// Initial frame.
	pInit := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		pInit[n] = true
	}
	stack := []frame{{
		R: make(map[string]bool),
		P: pInit,
		X: make(map[string]bool),
	}}

	for len(stack) > 0 {
		// Pop frame.
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if len(f.P) == 0 && len(f.X) == 0 {
			// R is a maximal clique.
			clique := make([]string, 0, len(f.R))
			for n := range f.R {
				clique = append(clique, n)
			}
			sort.Strings(clique)
			result = append(result, clique)
			continue
		}

		if len(f.P) == 0 {
			continue
		}

		// Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|.
		pivot := ""
		pivotCount := -1
		union := make(map[string]bool, len(f.P)+len(f.X))
		for n := range f.P {
			union[n] = true
		}
		for n := range f.X {
			union[n] = true
		}
		// Sort candidates for determinism.
		candidates := make([]string, 0, len(union))
		for n := range union {
			candidates = append(candidates, n)
		}
		sort.Strings(candidates)
		for _, u := range candidates {
			count := 0
			for nb := range adj[u] {
				if f.P[nb] {
					count++
				}
			}
			if count > pivotCount {
				pivotCount = count
				pivot = u
			}
		}

		// Collect vertices in P that are NOT neighbors of pivot.
		// Process in sorted order for determinism.
		var vertices []string
		for n := range f.P {
			if !adj[pivot][n] {
				vertices = append(vertices, n)
			}
		}
		sort.Strings(vertices)

		// Process vertices in reverse order so the stack processes them
		// in the original sorted order (LIFO).
		for i := len(vertices) - 1; i >= 0; i-- {
			v := vertices[i]

			newR := make(map[string]bool, len(f.R)+1)
			for n := range f.R {
				newR[n] = true
			}
			newR[v] = true

			newP := make(map[string]bool)
			for n := range f.P {
				if adj[v][n] {
					newP[n] = true
				}
			}

			newX := make(map[string]bool)
			for n := range f.X {
				if adj[v][n] {
					newX[n] = true
				}
			}

			stack = append(stack, frame{R: newR, P: newP, X: newX})

			// Move v from P to X for subsequent iterations.
			delete(f.P, v)
			f.X[v] = true
		}
	}

	// Sort result for deterministic output.
	sort.Slice(result, func(i, j int) bool {
		a, b := result[i], result[j]
		minLen := len(a)
		if len(b) < minLen {
			minLen = len(b)
		}
		for k := 0; k < minLen; k++ {
			if a[k] < b[k] {
				return true
			}
			if a[k] > b[k] {
				return false
			}
		}
		return len(a) < len(b)
	})

	return result
}

// BuildJunctionTree constructs a junction tree (clique tree) from a set of
// cliques. It builds a weighted complete graph on the cliques (weights are
// the intersection sizes of each pair), then finds a maximum spanning tree
// using Kruskal's algorithm. It returns the tree as an undirected Graph
// (nodes are clique labels like "0", "1", ...) and a map from edge keys
// to separator sets.
func BuildJunctionTree(cliques [][]string) (*Graph, map[string][]string) {
	n := len(cliques)
	if n == 0 {
		return NewGraph(), nil
	}

	// Create clique label for each index.
	labels := make([]string, n)
	for i := 0; i < n; i++ {
		labels[i] = cliqueLabel(i)
	}

	// Build sets for each clique.
	sets := make([]map[string]bool, n)
	for i, c := range cliques {
		sets[i] = make(map[string]bool, len(c))
		for _, v := range c {
			sets[i][v] = true
		}
	}

	// Build weighted edges between all pairs of cliques.
	type wedge struct {
		i, j   int
		weight int
		sep    []string
	}
	var edges []wedge
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sep := intersection(sets[i], sets[j])
			if len(sep) > 0 {
				edges = append(edges, wedge{i: i, j: j, weight: len(sep), sep: sep})
			}
		}
	}

	// Sort edges by weight descending (for maximum spanning tree).
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].weight != edges[b].weight {
			return edges[a].weight > edges[b].weight
		}
		if edges[a].i != edges[b].i {
			return edges[a].i < edges[b].i
		}
		return edges[a].j < edges[b].j
	})

	// Kruskal's with union-find for maximum spanning tree.
	parent := make([]int, n)
	rank := make([]int, n)
	for i := range parent {
		parent[i] = i
	}

	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}

	unionSets := func(a, b int) bool {
		ra, rb := find(a), find(b)
		if ra == rb {
			return false
		}
		if rank[ra] < rank[rb] {
			ra, rb = rb, ra
		}
		parent[rb] = ra
		if rank[ra] == rank[rb] {
			rank[ra]++
		}
		return true
	}

	tree := NewGraph()
	for _, l := range labels {
		tree.AddNode(l)
	}

	separators := make(map[string][]string)

	for _, e := range edges {
		if unionSets(e.i, e.j) {
			a, b := labels[e.i], labels[e.j]
			tree.AddEdge(a, b)
			k := undirectedEdgeKey(a, b)
			separators[k] = e.sep
		}
	}

	return tree, separators
}

// cliqueLabel returns a string label for a clique index.
func cliqueLabel(i int) string {
	// Use simple numeric string labels.
	buf := make([]byte, 0, 4)
	if i == 0 {
		return "0"
	}
	for i > 0 {
		buf = append(buf, byte('0'+i%10))
		i /= 10
	}
	// Reverse.
	for l, r := 0, len(buf)-1; l < r; l, r = l+1, r-1 {
		buf[l], buf[r] = buf[r], buf[l]
	}
	return string(buf)
}

// intersection returns the sorted intersection of two sets.
func intersection(a, b map[string]bool) []string {
	var result []string
	for k := range a {
		if b[k] {
			result = append(result, k)
		}
	}
	sort.Strings(result)
	return result
}
