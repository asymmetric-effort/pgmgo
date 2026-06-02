package base

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// MAG is a Maximal Ancestral Graph. It extends ADMG with the constraint that
// every missing edge corresponds to a valid m-separation statement, and every
// non-ancestor in the directed part has no directed edges into it from other
// non-ancestors (the "ancestral" property). A MAG has no directed cycles and
// satisfies maximality: if two nodes are not adjacent, they can be
// m-separated by some subset of the remaining nodes.
type MAG struct {
	*ADMG
}

// NewMAG creates a new empty MAG.
func NewMAG() *MAG {
	return &MAG{ADMG: NewADMG()}
}

// MSeparation tests whether node sets x and y are m-separated given set z in
// this MAG. M-separation generalises d-separation to mixed graphs.
//
// The algorithm works by converting the relevant sub-MAG to a moral graph
// (augmented graph) and then checking path connectivity, following the
// m-separation criterion for ancestral graphs.
//
// The criterion: X and Y are m-separated by Z if and only if they are
// separated by Z in the augmented graph over An(X ∪ Y ∪ Z).
func (m *MAG) MSeparation(x, y, z map[string]bool) bool {
	// Step 1: Compute the ancestral set of X ∪ Y ∪ Z in the directed subgraph.
	target := make(map[string]bool)
	for n := range x {
		target[n] = true
	}
	for n := range y {
		target[n] = true
	}
	for n := range z {
		target[n] = true
	}

	ancestral := make(map[string]bool)
	for n := range target {
		ancestral[n] = true
		for a := range graphgo.Ancestors(m.directed, n) {
			ancestral[a] = true
		}
	}

	// Step 2: Build the augmented (moral) graph over the ancestral set.
	// The augmented graph is an undirected graph where:
	//   (a) For every directed edge u→v in the ancestral subgraph, add u—v.
	//   (b) For every bidirected edge u↔v in the ancestral subgraph, add u—v.
	//   (c) For every node v in the ancestral set, if v has multiple "into"
	//       edges (parents via → and siblings via ↔), marry them (add edges
	//       between all pairs).
	adj := make(map[string]map[string]bool)
	for n := range ancestral {
		adj[n] = make(map[string]bool)
	}

	addUndirected := func(u, v string) {
		if u != v {
			adj[u][v] = true
			adj[v][u] = true
		}
	}

	// (a) Directed edges within the ancestral set.
	for _, e := range m.directed.Edges() {
		if ancestral[e.Src] && ancestral[e.Dst] {
			addUndirected(e.Src, e.Dst)
		}
	}

	// (b) Bidirected edges within the ancestral set.
	for _, e := range m.BidirectedEdges() {
		if ancestral[e.Src] && ancestral[e.Dst] {
			addUndirected(e.Src, e.Dst)
		}
	}

	// (c) Moralisation: for each node, collect all "into" endpoints (parents
	// via directed edges and siblings via bidirected edges) and connect them.
	for n := range ancestral {
		var into []string
		// Parents in directed subgraph.
		for _, p := range m.directed.Parents(n) {
			if ancestral[p] {
				into = append(into, p)
			}
		}
		// Siblings in bidirected subgraph.
		for _, s := range m.bidirected.Successors(n) {
			if ancestral[s] {
				into = append(into, s)
			}
		}
		// Marry all "into" nodes pairwise.
		for i := 0; i < len(into); i++ {
			for j := i + 1; j < len(into); j++ {
				addUndirected(into[i], into[j])
			}
		}
	}

	// Step 3: Check if any x node can reach any y node in the augmented graph
	// without passing through z.
	visited := make(map[string]bool)
	queue := make([]string, 0, len(x))
	for n := range x {
		if !z[n] {
			visited[n] = true
			queue = append(queue, n)
		}
	}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if y[curr] {
			return false // x and y are connected => not m-separated
		}

		for nb := range adj[curr] {
			if !visited[nb] && !z[nb] {
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}

	return true
}

// FromADMG converts an ADMG to a MAG by computing the maximal ancestral graph.
// The conversion latent-projects out any nodes that violate the ancestral
// property, producing a MAG over the same node set.
//
// For a valid ADMG that is already ancestral and maximal, this simply wraps it.
// The conversion adds bidirected edges where inducing paths exist between
// non-adjacent nodes.
func FromADMG(admg *ADMG) (*MAG, error) {
	if admg == nil {
		return nil, fmt.Errorf("base: nil ADMG")
	}

	mag := &MAG{ADMG: admg.Copy()}

	// Verify acyclicity of the directed part.
	if !graphgo.IsDAG(mag.directed) {
		return nil, fmt.Errorf("base: ADMG directed edges contain a cycle")
	}

	// For each pair of non-adjacent nodes, check if an inducing path exists.
	// An inducing path between u and v is a path where every non-endpoint node
	// is an ancestor of u or v and is connected to its neighbors on the path
	// via appropriate edge types. If such a path exists, add a bidirected edge.
	nodes := mag.Nodes()
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			u, v := nodes[i], nodes[j]
			// Skip if already adjacent (directed or bidirected).
			if mag.directed.HasEdge(u, v) || mag.directed.HasEdge(v, u) ||
				mag.bidirected.HasEdge(u, v) {
				continue
			}
			if hasInducingPath(mag.ADMG, u, v) {
				_ = mag.AddBidirectedEdge(u, v)
			}
		}
	}

	return mag, nil
}

// hasInducingPath checks whether there is an inducing path between u and v in
// the ADMG. An inducing path is a path where every non-endpoint node is both
// an ancestor of u or v and a collider on the path. For simplicity, this
// implements a BFS-based check.
func hasInducingPath(a *ADMG, u, v string) bool {
	// Compute ancestors of u and ancestors of v.
	ancU := graphgo.Ancestors(a.directed, u)
	ancU[u] = true
	ancV := graphgo.Ancestors(a.directed, v)
	ancV[v] = true

	// We look for paths from u to v in the combined (directed + bidirected)
	// graph where every intermediate node is an ancestor of {u, v}.
	ancestorOfEndpoints := make(map[string]bool)
	for n := range ancU {
		ancestorOfEndpoints[n] = true
	}
	for n := range ancV {
		ancestorOfEndpoints[n] = true
	}

	// BFS: track (current_node, arrived_via) to handle collider logic.
	type state struct {
		node string
		via  string // "directed_child", "directed_parent", "bidirected"
	}

	visited := make(map[state]bool)
	queue := []state{}

	// From u, we can leave via:
	// - directed edges (as parent, going to children)
	// - directed edges (as child, going to parents) -- but only at endpoint
	// - bidirected edges
	for _, c := range a.directed.Children(u) {
		s := state{node: c, via: "directed_child"}
		if !visited[s] {
			visited[s] = true
			queue = append(queue, s)
		}
	}
	for _, p := range a.directed.Parents(u) {
		s := state{node: p, via: "directed_parent"}
		if !visited[s] {
			visited[s] = true
			queue = append(queue, s)
		}
	}
	for _, sib := range a.bidirected.Successors(u) {
		s := state{node: sib, via: "bidirected"}
		if !visited[s] {
			visited[s] = true
			queue = append(queue, s)
		}
	}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr.node == v {
			return true
		}

		// Intermediate node must be an ancestor of u or v.
		if !ancestorOfEndpoints[curr.node] {
			continue
		}

		// On an inducing path, every non-endpoint must be a collider.
		// A collider has arrowheads on both sides. Arrowheads point into a node
		// via: directed edge (as child) or bidirected edge.
		// So the "arrival" must be via directed_child or bidirected.
		if curr.via == "directed_parent" {
			// Arrived as a parent (tail at this node) => non-collider => blocked.
			continue
		}

		// This is a collider: can leave via directed_child or bidirected (arrowhead out).
		for _, c := range a.directed.Children(curr.node) {
			s := state{node: c, via: "directed_child"}
			if !visited[s] {
				visited[s] = true
				queue = append(queue, s)
			}
		}
		for _, p := range a.directed.Parents(curr.node) {
			s := state{node: p, via: "directed_parent"}
			if !visited[s] {
				visited[s] = true
				queue = append(queue, s)
			}
		}
		for _, sib := range a.bidirected.Successors(curr.node) {
			s := state{node: sib, via: "bidirected"}
			if !visited[s] {
				visited[s] = true
				queue = append(queue, s)
			}
		}
	}

	return false
}

// magNodes is a helper that returns sorted nodes (for deterministic output).
func magNodes(nodes []string) []string {
	sort.Strings(nodes)
	return nodes
}
