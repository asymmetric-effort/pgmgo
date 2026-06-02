package learning

import (
	"fmt"
	"math"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// TreeSearchOption is a functional option for configuring TreeSearch.
type TreeSearchOption func(*TreeSearch)

// TreeSearch implements the Chow-Liu tree algorithm and its TAN (Tree-Augmented
// Naive Bayes) extension. The Chow-Liu algorithm finds the maximum weight
// spanning tree based on pairwise mutual information, then orients edges from
// a chosen root. TAN adds a class variable as a parent of all feature nodes.
type TreeSearch struct {
	data     *tabgo.DataFrame
	root     string // root variable for edge orientation (optional)
	classVar string // class variable for TAN (optional)
}

// NewTreeSearch creates a new TreeSearch instance.
func NewTreeSearch(data *tabgo.DataFrame, opts ...TreeSearchOption) *TreeSearch {
	ts := &TreeSearch{
		data: data,
	}
	for _, opt := range opts {
		opt(ts)
	}
	return ts
}

// WithRoot sets the root variable for edge orientation in the Chow-Liu tree.
func WithRoot(root string) TreeSearchOption {
	return func(ts *TreeSearch) {
		ts.root = root
	}
}

// WithClassVariable sets the class variable for TAN (Tree-Augmented Naive Bayes).
// When set, the class variable becomes a parent of all feature nodes.
func WithClassVariable(classVar string) TreeSearchOption {
	return func(ts *TreeSearch) {
		ts.classVar = classVar
	}
}

// weightedEdge represents an edge with a weight for the spanning tree algorithm.
type weightedEdge struct {
	u, v   string
	weight float64
}

// Estimate runs the Chow-Liu (or TAN) algorithm and returns the learned
// BayesianNetwork structure.
//
// Chow-Liu:
//  1. Compute mutual information for all variable pairs.
//  2. Find the maximum weight spanning tree using Kruskal's algorithm.
//  3. Orient edges from root (default: variable with most total MI).
//
// TAN extension: after building the tree over feature variables, add the class
// variable as a parent of all feature nodes.
func (ts *TreeSearch) Estimate() (*models.BayesianNetwork, error) {
	columns := ts.data.Columns()
	if len(columns) < 2 {
		return nil, fmt.Errorf("learning: tree search requires at least 2 variables, got %d", len(columns))
	}

	// Determine feature variables and the tree variables.
	var treeVars []string
	if ts.classVar != "" {
		found := false
		for _, col := range columns {
			if col == ts.classVar {
				found = true
			} else {
				treeVars = append(treeVars, col)
			}
		}
		if !found {
			return nil, fmt.Errorf("learning: class variable %q not found in data", ts.classVar)
		}
		if len(treeVars) < 2 {
			return nil, fmt.Errorf("learning: TAN requires at least 2 feature variables")
		}
	} else {
		treeVars = make([]string, len(columns))
		copy(treeVars, columns)
	}

	// Validate root if specified.
	if ts.root != "" {
		found := false
		for _, v := range treeVars {
			if v == ts.root {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("learning: root variable %q not found among tree variables", ts.root)
		}
	}

	// Compute pairwise mutual information.
	miEdges := ts.computeAllMI(treeVars)

	// Find maximum weight spanning tree (Kruskal's algorithm).
	treeEdges := ts.kruskalMaxSpanningTree(treeVars, miEdges)

	// Choose root: use specified root or pick the variable with highest total MI.
	root := ts.root
	if root == "" {
		root = ts.pickRoot(treeVars, miEdges)
	}

	// Orient edges from root using BFS.
	directedEdges := ts.orientFromRoot(treeVars, treeEdges, root)

	// Build BayesianNetwork.
	bn := models.NewBayesianNetwork()
	for _, col := range columns {
		if err := bn.AddNode(col); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}

	for _, e := range directedEdges {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}

	// TAN: add class variable as parent of all feature nodes.
	if ts.classVar != "" {
		for _, feat := range treeVars {
			if err := bn.AddEdge(ts.classVar, feat); err != nil {
				return nil, fmt.Errorf("learning: %w", err)
			}
		}
	}

	return bn, nil
}

// computeAllMI computes mutual information for all pairs of variables.
func (ts *TreeSearch) computeAllMI(vars []string) []weightedEdge {
	n := ts.data.Len()
	if n == 0 {
		return nil
	}

	var edges []weightedEdge
	for i := 0; i < len(vars); i++ {
		for j := i + 1; j < len(vars); j++ {
			mi := ts.mutualInformation(vars[i], vars[j], n)
			edges = append(edges, weightedEdge{u: vars[i], v: vars[j], weight: mi})
		}
	}
	return edges
}

// mutualInformation computes the MI between two variables from the data.
// MI(X,Y) = sum_{x,y} P(x,y) * log(P(x,y) / (P(x)*P(y)))
func (ts *TreeSearch) mutualInformation(x, y string, n int) float64 {
	xVals := ts.data.Column(x).Values()
	yVals := ts.data.Column(y).Values()

	// Count joint and marginal frequencies.
	type pair struct {
		a, b interface{}
	}
	jointCount := make(map[pair]int)
	xCount := make(map[interface{}]int)
	yCount := make(map[interface{}]int)

	for i := 0; i < n; i++ {
		xv, yv := xVals[i], yVals[i]
		jointCount[pair{xv, yv}]++
		xCount[xv]++
		yCount[yv]++
	}

	nf := float64(n)
	mi := 0.0
	for p, jc := range jointCount {
		pxy := float64(jc) / nf
		px := float64(xCount[p.a]) / nf
		py := float64(yCount[p.b]) / nf
		if pxy > 0 && px > 0 && py > 0 {
			mi += pxy * math.Log(pxy/(px*py))
		}
	}
	return mi
}

// kruskalMaxSpanningTree finds the maximum weight spanning tree using Kruskal's
// algorithm (sort edges by weight descending, add if no cycle).
func (ts *TreeSearch) kruskalMaxSpanningTree(vars []string, edges []weightedEdge) []weightedEdge {
	// Sort edges by weight descending.
	sorted := make([]weightedEdge, len(edges))
	copy(sorted, edges)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].weight > sorted[j].weight
	})

	// Union-Find.
	parent := make(map[string]string)
	rank := make(map[string]int)
	for _, v := range vars {
		parent[v] = v
		rank[v] = 0
	}

	var find func(string) string
	find = func(x string) string {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}

	union := func(x, y string) bool {
		rx, ry := find(x), find(y)
		if rx == ry {
			return false
		}
		if rank[rx] < rank[ry] {
			rx, ry = ry, rx
		}
		parent[ry] = rx
		if rank[rx] == rank[ry] {
			rank[rx]++
		}
		return true
	}

	var tree []weightedEdge
	for _, e := range sorted {
		if union(e.u, e.v) {
			tree = append(tree, e)
			if len(tree) == len(vars)-1 {
				break
			}
		}
	}
	return tree
}

// pickRoot picks the variable with the highest total mutual information.
func (ts *TreeSearch) pickRoot(vars []string, edges []weightedEdge) string {
	totalMI := make(map[string]float64)
	for _, e := range edges {
		totalMI[e.u] += e.weight
		totalMI[e.v] += e.weight
	}

	best := vars[0]
	bestMI := totalMI[best]
	for _, v := range vars[1:] {
		if totalMI[v] > bestMI {
			best = v
			bestMI = totalMI[v]
		}
	}
	return best
}

// orientFromRoot orients tree edges from root via BFS, returning directed edges.
func (ts *TreeSearch) orientFromRoot(vars []string, treeEdges []weightedEdge, root string) [][2]string {
	// Build adjacency list.
	adj := make(map[string][]string)
	for _, v := range vars {
		adj[v] = nil
	}
	for _, e := range treeEdges {
		adj[e.u] = append(adj[e.u], e.v)
		adj[e.v] = append(adj[e.v], e.u)
	}

	visited := make(map[string]bool)
	var directed [][2]string

	// BFS from root.
	queue := []string{root}
	visited[root] = true

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]

		neighbors := adj[node]
		sort.Strings(neighbors)
		for _, nb := range neighbors {
			if !visited[nb] {
				visited[nb] = true
				directed = append(directed, [2]string{node, nb})
				queue = append(queue, nb)
			}
		}
	}

	return directed
}
