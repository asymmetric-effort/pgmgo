package base

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// ADMG is an Acyclic Directed Mixed Graph. It supports directed edges (→)
// representing direct causal effects and bidirected edges (↔) representing
// latent common causes. Acyclicity is enforced on the directed edges only.
type ADMG struct {
	directed   *graphgo.DiGraph // directed edges (→)
	bidirected *graphgo.DiGraph // bidirected edges (↔), stored symmetrically
}

// NewADMG creates a new empty ADMG.
func NewADMG() *ADMG {
	return &ADMG{
		directed:   graphgo.NewDiGraph(),
		bidirected: graphgo.NewDiGraph(),
	}
}

// AddNode adds a node to the ADMG. Returns an error if the node already exists.
func (a *ADMG) AddNode(node string) error {
	if a.directed.HasNode(node) {
		return fmt.Errorf("base: node %q already exists", node)
	}
	a.directed.AddNode(node)
	a.bidirected.AddNode(node)
	return nil
}

// AddDirectedEdge adds a directed edge from → to. Both nodes must exist.
// Returns an error if the edge would create a cycle in the directed subgraph,
// if either node does not exist, or if the edge already exists.
func (a *ADMG) AddDirectedEdge(from, to string) error {
	if !a.directed.HasNode(from) {
		return fmt.Errorf("base: node %q not found", from)
	}
	if !a.directed.HasNode(to) {
		return fmt.Errorf("base: node %q not found", to)
	}
	if a.directed.HasEdge(from, to) {
		return fmt.Errorf("base: directed edge (%q, %q) already exists", from, to)
	}

	// Add and check acyclicity.
	a.directed.AddEdge(from, to)
	if !graphgo.IsDAG(a.directed) {
		_ = a.directed.RemoveEdge(from, to)
		return fmt.Errorf("base: directed edge (%q, %q) would create a cycle", from, to)
	}
	return nil
}

// AddBidirectedEdge adds a bidirected edge u ↔ v representing a latent common
// cause. Both nodes must exist. The edge is stored symmetrically. Returns an
// error if either node does not exist or the edge already exists.
func (a *ADMG) AddBidirectedEdge(u, v string) error {
	if !a.directed.HasNode(u) {
		return fmt.Errorf("base: node %q not found", u)
	}
	if !a.directed.HasNode(v) {
		return fmt.Errorf("base: node %q not found", v)
	}
	if a.bidirected.HasEdge(u, v) {
		return fmt.Errorf("base: bidirected edge (%q, %q) already exists", u, v)
	}
	a.bidirected.AddEdge(u, v)
	a.bidirected.AddEdge(v, u)
	return nil
}

// HasNode returns true if the node exists.
func (a *ADMG) HasNode(node string) bool {
	return a.directed.HasNode(node)
}

// Nodes returns a sorted list of all nodes.
func (a *ADMG) Nodes() []string {
	nodes := a.directed.Nodes()
	sort.Strings(nodes)
	return nodes
}

// DirectedEdges returns all directed edges, sorted lexicographically.
func (a *ADMG) DirectedEdges() []graphgo.Edge {
	edges := a.directed.Edges()
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Src != edges[j].Src {
			return edges[i].Src < edges[j].Src
		}
		return edges[i].Dst < edges[j].Dst
	})
	return edges
}

// BidirectedEdges returns all bidirected edges as unordered pairs (u, v) with
// u < v, sorted lexicographically. Each edge appears exactly once.
func (a *ADMG) BidirectedEdges() []graphgo.Edge {
	seen := make(map[string]bool)
	var edges []graphgo.Edge
	for _, e := range a.bidirected.Edges() {
		// Canonical key: smaller node first.
		u, v := e.Src, e.Dst
		if u > v {
			u, v = v, u
		}
		key := u + "\x00" + v
		if !seen[key] {
			seen[key] = true
			edges = append(edges, graphgo.Edge{Src: u, Dst: v})
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Src != edges[j].Src {
			return edges[i].Src < edges[j].Src
		}
		return edges[i].Dst < edges[j].Dst
	})
	return edges
}

// Parents returns the sorted parents of a node via directed edges.
func (a *ADMG) Parents(node string) []string {
	p := a.directed.Parents(node)
	sort.Strings(p)
	return p
}

// Children returns the sorted children of a node via directed edges.
func (a *ADMG) Children(node string) []string {
	c := a.directed.Children(node)
	sort.Strings(c)
	return c
}

// Siblings returns the sorted set of nodes connected to node via bidirected
// edges (i.e., nodes sharing a latent common cause).
func (a *ADMG) Siblings(node string) []string {
	// In the bidirected graph, successors of node are siblings (symmetric).
	sibs := a.bidirected.Successors(node)
	sort.Strings(sibs)
	return sibs
}

// Districts returns the connected components of the bidirected subgraph.
// Each district is a sorted slice of node names; the list of districts is
// sorted by the first element.
func (a *ADMG) Districts() [][]string {
	nodes := a.Nodes()
	visited := make(map[string]bool)
	var districts [][]string

	for _, n := range nodes {
		if visited[n] {
			continue
		}
		// BFS in the bidirected graph.
		var component []string
		queue := []string{n}
		visited[n] = true
		for len(queue) > 0 {
			curr := queue[0]
			queue = queue[1:]
			component = append(component, curr)
			for _, sib := range a.bidirected.Successors(curr) {
				if !visited[sib] {
					visited[sib] = true
					queue = append(queue, sib)
				}
			}
		}
		sort.Strings(component)
		districts = append(districts, component)
	}

	sort.Slice(districts, func(i, j int) bool {
		return districts[i][0] < districts[j][0]
	})
	return districts
}

// Copy returns a deep copy of the ADMG.
func (a *ADMG) Copy() *ADMG {
	return &ADMG{
		directed:   a.directed.Copy(),
		bidirected: a.bidirected.Copy(),
	}
}
