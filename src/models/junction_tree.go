package models

import (
	"fmt"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// JunctionTree represents a junction tree (clique tree) — a tree of cliques
// with separator sets, used for exact inference in graphical models.
type JunctionTree struct {
	cliques       [][]string                        // each clique is a sorted list of variable names
	tree          *graphgo.Graph                    // tree structure over clique labels ("0", "1", ...)
	separators    map[string][]string               // edge key -> separator set
	cliqueFactors map[int][]*factors.DiscreteFactor // clique index -> assigned factors
}

// NewJunctionTreeFromBN builds a junction tree from a BayesianNetwork.
//
// Steps:
//  1. Get factors via bn.ToMarkovFactors()
//  2. Build moral graph via graphgo.Moralize()
//  3. Find elimination order (min-degree heuristic)
//  4. Triangulate moral graph
//  5. Find max cliques via graphgo.MaxCliques()
//  6. Build junction tree via graphgo.BuildJunctionTree()
//  7. Assign factors to cliques (each factor goes to the smallest clique containing all its variables)
func NewJunctionTreeFromBN(bn *BayesianNetwork) (*JunctionTree, error) {
	// Step 1: Get factors.
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, fmt.Errorf("models: junction tree: %w", err)
	}

	// Step 2: Build moral graph. We need a graphgo.DiGraph from the BN.
	dg := graphgo.NewDiGraph()
	for _, node := range bn.Nodes() {
		dg.AddNode(node)
	}
	for _, e := range bn.Edges() {
		dg.AddEdge(e[0], e[1])
	}
	moral := graphgo.Moralize(dg)

	// Step 3: Find elimination order using min-degree heuristic.
	order := minDegreeOrder(moral)

	// Step 4: Triangulate the moral graph.
	triangulated := graphgo.Triangulate(moral, order)

	// Step 5: Find maximal cliques.
	cliques := graphgo.MaxCliques(triangulated)
	if len(cliques) == 0 {
		// Degenerate case: no cliques (empty network).
		return &JunctionTree{
			cliques:       nil,
			tree:          graphgo.NewGraph(),
			separators:    make(map[string][]string),
			cliqueFactors: make(map[int][]*factors.DiscreteFactor),
		}, nil
	}

	// Step 6: Build junction tree.
	tree, separators := graphgo.BuildJunctionTree(cliques)

	// Step 7: Assign factors to cliques.
	cliqueFactors := assignFactorsToCliques(markovFactors, cliques)

	return &JunctionTree{
		cliques:       cliques,
		tree:          tree,
		separators:    separators,
		cliqueFactors: cliqueFactors,
	}, nil
}

// Cliques returns a copy of all cliques in the junction tree.
func (jt *JunctionTree) Cliques() [][]string {
	result := make([][]string, len(jt.cliques))
	for i, c := range jt.cliques {
		cp := make([]string, len(c))
		copy(cp, c)
		result[i] = cp
	}
	return result
}

// SeparatorSets returns a copy of the separator sets, keyed by edge key.
func (jt *JunctionTree) SeparatorSets() map[string][]string {
	result := make(map[string][]string, len(jt.separators))
	for k, v := range jt.separators {
		cp := make([]string, len(v))
		copy(cp, v)
		result[k] = cp
	}
	return result
}

// GetCliqueFactors returns the factors assigned to a clique identified by its
// variable set. The clique must match exactly (order-independent).
func (jt *JunctionTree) GetCliqueFactors(clique []string) []*factors.DiscreteFactor {
	sorted := make([]string, len(clique))
	copy(sorted, clique)
	sort.Strings(sorted)
	key := strings.Join(sorted, "\x00")

	for i, c := range jt.cliques {
		ck := strings.Join(c, "\x00") // cliques are already sorted
		if ck == key {
			fs := jt.cliqueFactors[i]
			result := make([]*factors.DiscreteFactor, len(fs))
			copy(result, fs)
			return result
		}
	}
	return nil
}

// CheckModel verifies the running intersection property: for every variable,
// the set of cliques containing that variable forms a connected subtree.
func (jt *JunctionTree) CheckModel() error {
	if len(jt.cliques) <= 1 {
		return nil
	}

	// Build variable -> set of clique indices.
	varCliques := make(map[string][]int)
	for i, c := range jt.cliques {
		for _, v := range c {
			varCliques[v] = append(varCliques[v], i)
		}
	}

	// Build adjacency for the tree using clique indices.
	treeAdj := make(map[int][]int)
	for _, e := range jt.tree.Edges() {
		a := labelToIndex(e.A)
		b := labelToIndex(e.B)
		treeAdj[a] = append(treeAdj[a], b)
		treeAdj[b] = append(treeAdj[b], a)
	}

	// For each variable, check that the subgraph induced by its cliques is connected.
	for v, indices := range varCliques {
		if len(indices) <= 1 {
			continue
		}
		idxSet := make(map[int]bool, len(indices))
		for _, idx := range indices {
			idxSet[idx] = true
		}

		// BFS from the first clique containing v, only traversing cliques that also contain v.
		visited := make(map[int]bool)
		queue := []int{indices[0]}
		visited[indices[0]] = true

		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for _, nb := range treeAdj[cur] {
				if idxSet[nb] && !visited[nb] {
					visited[nb] = true
					queue = append(queue, nb)
				}
			}
		}

		if len(visited) != len(idxSet) {
			return fmt.Errorf("models: running intersection property violated for variable %q", v)
		}
	}

	return nil
}

// minDegreeOrder returns an elimination order using the min-degree heuristic.
// At each step, the node with the fewest neighbors in the remaining graph is
// eliminated.
func minDegreeOrder(g *graphgo.Graph) []string {
	nodes := g.Nodes()
	sort.Strings(nodes)

	// Build working adjacency.
	adj := make(map[string]map[string]bool, len(nodes))
	for _, n := range nodes {
		adj[n] = make(map[string]bool)
	}
	for _, e := range g.Edges() {
		adj[e.A][e.B] = true
		adj[e.B][e.A] = true
	}

	remaining := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		remaining[n] = true
	}

	order := make([]string, 0, len(nodes))
	for len(remaining) > 0 {
		// Find node with minimum degree (break ties lexicographically).
		best := ""
		bestDeg := -1
		for _, n := range nodes {
			if !remaining[n] {
				continue
			}
			deg := len(adj[n])
			if best == "" || deg < bestDeg || (deg == bestDeg && n < best) {
				best = n
				bestDeg = deg
			}
		}

		order = append(order, best)
		delete(remaining, best)

		// Remove node from adjacency.
		for nb := range adj[best] {
			delete(adj[nb], best)
		}
		delete(adj, best)
	}

	return order
}

// assignFactorsToCliques assigns each factor to the smallest clique that
// contains all of the factor's variables.
func assignFactorsToCliques(facs []*factors.DiscreteFactor, cliques [][]string) map[int][]*factors.DiscreteFactor {
	result := make(map[int][]*factors.DiscreteFactor)

	// Pre-build clique sets.
	cliqueSets := make([]map[string]bool, len(cliques))
	for i, c := range cliques {
		s := make(map[string]bool, len(c))
		for _, v := range c {
			s[v] = true
		}
		cliqueSets[i] = s
	}

	for _, f := range facs {
		vars := f.Variables()
		bestIdx := -1
		bestSize := -1

		for i, cs := range cliqueSets {
			// Check if clique contains all factor variables.
			containsAll := true
			for _, v := range vars {
				if !cs[v] {
					containsAll = false
					break
				}
			}
			if !containsAll {
				continue
			}
			// Pick the smallest clique.
			if bestIdx == -1 || len(cliques[i]) < bestSize {
				bestIdx = i
				bestSize = len(cliques[i])
			}
		}

		if bestIdx >= 0 {
			result[bestIdx] = append(result[bestIdx], f)
		}
	}

	return result
}

// labelToIndex converts a clique label string back to an integer index.
func labelToIndex(label string) int {
	n := 0
	for _, ch := range label {
		n = n*10 + int(ch-'0')
	}
	return n
}
