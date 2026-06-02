package learning

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// CITestFunc tests conditional independence of x and y given z in data.
// Returns the test statistic, p-value, and whether x and y are independent
// given z at the specified significance level.
type CITestFunc func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (statistic float64, pvalue float64, independent bool)

// PCOption configures optional parameters for the PC algorithm.
type PCOption func(*PCAlgorithm)

// WithMaxCondSetSize sets the maximum conditioning set size. If not set or set
// to a negative value, the algorithm uses the graph's maximum degree as the
// upper bound.
func WithMaxCondSetSize(n int) PCOption {
	return func(pc *PCAlgorithm) {
		pc.maxCondSetSize = n
	}
}

// PCAlgorithm implements the PC (Peter-Clark) constraint-based structure
// learning algorithm. It discovers causal structure from observational data
// by performing conditional independence tests.
type PCAlgorithm struct {
	data           *tabgo.DataFrame
	ciTest         CITestFunc
	significance   float64
	maxCondSetSize int // -1 means no limit (use graph degree)
}

// NewPC creates a new PCAlgorithm instance.
//
// Parameters:
//   - data: the observational dataset with columns as variables
//   - ciTest: a conditional independence test function
//   - significance: the significance level (alpha) for independence tests
//   - opts: optional configuration (e.g., WithMaxCondSetSize)
func NewPC(data *tabgo.DataFrame, ciTest CITestFunc, significance float64, opts ...PCOption) *PCAlgorithm {
	pc := &PCAlgorithm{
		data:           data,
		ciTest:         ciTest,
		significance:   significance,
		maxCondSetSize: -1,
	}
	for _, opt := range opts {
		opt(pc)
	}
	return pc
}

// sepSetKey creates a canonical key for the separating set of a pair (u, v)
// with u < v lexicographically.
func sepSetKey(u, v string) [2]string {
	if u > v {
		u, v = v, u
	}
	return [2]string{u, v}
}

// Estimate runs the PC algorithm and returns a PDAG (CPDAG) representing the
// learned Markov equivalence class.
//
// The algorithm proceeds in three phases:
//  1. Skeleton discovery via conditional independence tests
//  2. V-structure orientation
//  3. Meek rule application
func (pc *PCAlgorithm) Estimate() (*graphgo.PDAG, error) {
	variables := pc.data.Columns()
	if len(variables) < 2 {
		return nil, fmt.Errorf("learning: PC algorithm requires at least 2 variables, got %d", len(variables))
	}

	// Phase 1: Skeleton discovery.
	// Start with a complete undirected graph.
	pdag := graphgo.NewPDAG()
	for _, v := range variables {
		pdag.AddNode(v)
	}
	for i := 0; i < len(variables); i++ {
		for j := i + 1; j < len(variables); j++ {
			pdag.AddUndirectedEdge(variables[i], variables[j])
		}
	}

	// sepSets stores the separating set for each removed edge.
	sepSets := make(map[[2]string][]string)

	// Iterate over conditioning set sizes d = 0, 1, 2, ...
	for d := 0; ; d++ {
		// Check termination: if d exceeds max conditioning set size option.
		if pc.maxCondSetSize >= 0 && d > pc.maxCondSetSize {
			break
		}

		// Compute the maximum adjacency degree in the current skeleton.
		maxDegree := 0
		for _, node := range variables {
			deg := pc.adjacencyCount(pdag, node)
			if deg > maxDegree {
				maxDegree = deg
			}
		}
		if d > maxDegree {
			break
		}

		// Collect current undirected edges (snapshot to avoid mutation during iteration).
		edges := pdag.UndirectedEdges()

		for _, edge := range edges {
			x, y := edge[0], edge[1]

			// Skip if edge was already removed during this iteration.
			if !pdag.HasUndirectedEdge(x, y) {
				continue
			}

			// Try to find a separating set of size d in adj(x)\{y}.
			adjX := pc.undirectedNeighbors(pdag, x)
			adjXMinusY := removeFromSlice(adjX, y)

			if found, subset := pc.findSepSet(x, y, adjXMinusY, d); found {
				pdag.RemoveUndirectedEdge(x, y)
				key := sepSetKey(x, y)
				sepSets[key] = subset
				continue
			}

			// Also try adj(y)\{x}.
			adjY := pc.undirectedNeighbors(pdag, y)
			adjYMinusX := removeFromSlice(adjY, x)

			if found, subset := pc.findSepSet(x, y, adjYMinusX, d); found {
				pdag.RemoveUndirectedEdge(x, y)
				key := sepSetKey(x, y)
				sepSets[key] = subset
				continue
			}
		}
	}

	// Phase 2: V-structure orientation.
	// For each unshielded triple x - z - y (where x and y are not adjacent),
	// if z is NOT in sepSet(x,y), orient as x -> z <- y.
	for _, z := range variables {
		// Collect all undirected neighbors of z.
		neighborsZ := pc.undirectedNeighbors(pdag, z)
		if len(neighborsZ) < 2 {
			continue
		}
		for i := 0; i < len(neighborsZ); i++ {
			for j := i + 1; j < len(neighborsZ); j++ {
				x, y := neighborsZ[i], neighborsZ[j]
				// Check that x and y are NOT adjacent (unshielded triple).
				if pdag.Adjacent(x, y) {
					continue
				}
				// Check if z is NOT in the separating set of (x, y).
				key := sepSetKey(x, y)
				ss, exists := sepSets[key]
				if !exists {
					// No separating set means they were never separated;
					// they should still be adjacent. Skip.
					continue
				}
				if !containsString(ss, z) {
					// Orient x -> z <- y.
					if pdag.HasUndirectedEdge(x, z) {
						pdag.RemoveUndirectedEdge(x, z)
						pdag.AddDirectedEdge(x, z)
					}
					if pdag.HasUndirectedEdge(y, z) {
						pdag.RemoveUndirectedEdge(y, z)
						pdag.AddDirectedEdge(y, z)
					}
				}
			}
		}
	}

	// Phase 3: Meek rules.
	graphgo.ApplyMeekRules(pdag)

	return pdag, nil
}

// EstimateBN runs the PC algorithm via Estimate(), converts the resulting
// PDAG to a DAG by orienting remaining undirected edges consistently, and
// returns a BayesianNetwork (without CPDs — only structure).
func (pc *PCAlgorithm) EstimateBN() (*models.BayesianNetwork, error) {
	pdag, err := pc.Estimate()
	if err != nil {
		return nil, err
	}

	// Convert PDAG to DAG: orient remaining undirected edges.
	// We orient them following a topological heuristic: pick a consistent
	// ordering of nodes and orient undirected edges from earlier to later.
	dag, err := pdagToDAG(pdag)
	if err != nil {
		return nil, fmt.Errorf("learning: failed to convert PDAG to DAG: %w", err)
	}

	return dag, nil
}

// pdagToDAG converts a PDAG to a BayesianNetwork (DAG) by orienting any
// remaining undirected edges in a consistent acyclic manner.
//
// Strategy: use a greedy approach. Process undirected edges sorted
// lexicographically. For each edge (u, v) with u < v, try orienting u->v
// first; if that would create a cycle, orient v->u.
func pdagToDAG(pdag *graphgo.PDAG) (*models.BayesianNetwork, error) {
	bn := models.NewBayesianNetwork()
	nodes := pdag.Nodes()
	for _, n := range nodes {
		_ = bn.AddNode(n)
	}

	// First add all already-directed edges.
	for _, e := range pdag.DirectedEdges() {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			return nil, fmt.Errorf("directed edge (%s, %s): %w", e[0], e[1], err)
		}
	}

	// Orient remaining undirected edges.
	undirected := pdag.UndirectedEdges() // sorted, u < v
	for _, e := range undirected {
		u, v := e[0], e[1]
		// Try u -> v first.
		err := bn.AddEdge(u, v)
		if err == nil {
			continue
		}
		// Try v -> u.
		err = bn.AddEdge(v, u)
		if err != nil {
			return nil, fmt.Errorf("cannot orient undirected edge (%s, %s) in either direction: %w", u, v, err)
		}
	}

	return bn, nil
}

// adjacencyCount returns the number of undirected neighbors of node in the PDAG.
func (pc *PCAlgorithm) adjacencyCount(pdag *graphgo.PDAG, node string) int {
	count := 0
	for _, n := range pdag.Nodes() {
		if n != node && pdag.HasUndirectedEdge(node, n) {
			count++
		}
	}
	return count
}

// undirectedNeighbors returns sorted undirected neighbors of node in the PDAG.
func (pc *PCAlgorithm) undirectedNeighbors(pdag *graphgo.PDAG, node string) []string {
	var neighbors []string
	for _, n := range pdag.Nodes() {
		if n != node && pdag.HasUndirectedEdge(node, n) {
			neighbors = append(neighbors, n)
		}
	}
	sort.Strings(neighbors)
	return neighbors
}

// findSepSet searches for a subset of candidates of size d that makes x and y
// conditionally independent. Returns (true, subset) if found, (false, nil) otherwise.
func (pc *PCAlgorithm) findSepSet(x, y string, candidates []string, d int) (bool, []string) {
	if d > len(candidates) {
		return false, nil
	}
	if d == 0 {
		stat, pval, indep := pc.ciTest(x, y, nil, pc.data, pc.significance)
		_ = stat
		_ = pval
		if indep {
			return true, nil
		}
		return false, nil
	}

	// Enumerate all subsets of candidates of size d.
	subsets := combinations(candidates, d)
	for _, subset := range subsets {
		stat, pval, indep := pc.ciTest(x, y, subset, pc.data, pc.significance)
		_ = stat
		_ = pval
		if indep {
			return true, subset
		}
	}
	return false, nil
}

// combinations returns all C(n,k) subsets of the given slice.
func combinations(items []string, k int) [][]string {
	if k == 0 {
		return [][]string{{}}
	}
	if k > len(items) {
		return nil
	}
	var result [][]string
	combinationsHelper(items, k, 0, nil, &result)
	return result
}

func combinationsHelper(items []string, k, start int, current []string, result *[][]string) {
	if len(current) == k {
		combo := make([]string, k)
		copy(combo, current)
		*result = append(*result, combo)
		return
	}
	remaining := k - len(current)
	for i := start; i <= len(items)-remaining; i++ {
		combinationsHelper(items, k, i+1, append(current, items[i]), result)
	}
}

// removeFromSlice returns a new slice with the target element removed.
func removeFromSlice(items []string, target string) []string {
	result := make([]string, 0, len(items))
	for _, item := range items {
		if item != target {
			result = append(result, item)
		}
	}
	return result
}

// containsString returns true if the slice contains the target string.
func containsString(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}
