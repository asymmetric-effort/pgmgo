package base

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// PDAG is a Partially Directed Acyclic Graph that supports both directed and
// undirected edges. It wraps a graphgo.PDAG and is used to represent Markov
// equivalence classes (CPDAGs) in causal inference.
type PDAG struct {
	p *graphgo.PDAG
}

// NewPDAG creates a new empty PDAG.
func NewPDAG() *PDAG {
	return &PDAG{p: graphgo.NewPDAG()}
}

// FromDAG converts a DAG to its CPDAG (Completed Partially Directed Acyclic
// Graph) representation using graphgo.DAGToPDAG. The resulting PDAG represents
// the Markov equivalence class of the given DAG.
func FromDAG(dag *DAG) *PDAG {
	return &PDAG{p: graphgo.DAGToPDAG(dag.g)}
}

// AddNode adds a node to the PDAG. Returns an error if the node already exists.
func (pd *PDAG) AddNode(node string) error {
	if pd.p.HasNode(node) {
		return fmt.Errorf("base: node %q already exists", node)
	}
	pd.p.AddNode(node)
	return nil
}

// AddDirectedEdge adds a directed edge from -> to. Both nodes must already
// exist. Returns an error if either node does not exist or the edge already exists.
func (pd *PDAG) AddDirectedEdge(from, to string) error {
	if !pd.p.HasNode(from) {
		return fmt.Errorf("base: node %q not found", from)
	}
	if !pd.p.HasNode(to) {
		return fmt.Errorf("base: node %q not found", to)
	}
	if pd.p.HasDirectedEdge(from, to) {
		return fmt.Errorf("base: directed edge (%q, %q) already exists", from, to)
	}
	pd.p.AddDirectedEdge(from, to)
	return nil
}

// AddUndirectedEdge adds an undirected edge between u and v. Both nodes must
// already exist. Returns an error if either node does not exist or the edge
// already exists.
func (pd *PDAG) AddUndirectedEdge(u, v string) error {
	if !pd.p.HasNode(u) {
		return fmt.Errorf("base: node %q not found", u)
	}
	if !pd.p.HasNode(v) {
		return fmt.Errorf("base: node %q not found", v)
	}
	if pd.p.HasUndirectedEdge(u, v) {
		return fmt.Errorf("base: undirected edge (%q, %q) already exists", u, v)
	}
	pd.p.AddUndirectedEdge(u, v)
	return nil
}

// RemoveDirectedEdge removes a directed edge from -> to. Returns an error if
// the edge does not exist.
func (pd *PDAG) RemoveDirectedEdge(from, to string) error {
	if !pd.p.HasDirectedEdge(from, to) {
		return fmt.Errorf("base: directed edge (%q, %q) not found", from, to)
	}
	pd.p.RemoveDirectedEdge(from, to)
	return nil
}

// RemoveUndirectedEdge removes an undirected edge between u and v. Returns an
// error if the edge does not exist.
func (pd *PDAG) RemoveUndirectedEdge(u, v string) error {
	if !pd.p.HasUndirectedEdge(u, v) {
		return fmt.Errorf("base: undirected edge (%q, %q) not found", u, v)
	}
	pd.p.RemoveUndirectedEdge(u, v)
	return nil
}

// HasNode returns true if the node exists in the PDAG.
func (pd *PDAG) HasNode(node string) bool {
	return pd.p.HasNode(node)
}

// HasDirectedEdge returns true if the directed edge from -> to exists.
func (pd *PDAG) HasDirectedEdge(from, to string) bool {
	return pd.p.HasDirectedEdge(from, to)
}

// HasUndirectedEdge returns true if the undirected edge between u and v exists.
func (pd *PDAG) HasUndirectedEdge(u, v string) bool {
	return pd.p.HasUndirectedEdge(u, v)
}

// HasEdge returns true if any edge (directed or undirected) exists between u
// and v in either direction.
func (pd *PDAG) HasEdge(u, v string) bool {
	return pd.p.HasEdge(u, v)
}

// Nodes returns a sorted list of all nodes.
func (pd *PDAG) Nodes() []string {
	return pd.p.Nodes()
}

// DirectedEdges returns all directed edges as [from, to] pairs, sorted
// lexicographically.
func (pd *PDAG) DirectedEdges() [][2]string {
	return pd.p.DirectedEdges()
}

// UndirectedEdges returns all undirected edges as [u, v] pairs (each edge
// once, u < v), sorted lexicographically.
func (pd *PDAG) UndirectedEdges() [][2]string {
	return pd.p.UndirectedEdges()
}

// Parents returns the sorted parents (predecessors via directed edges) of a node.
func (pd *PDAG) Parents(node string) []string {
	return pd.p.Parents(node)
}

// Children returns the sorted children (successors via directed edges) of a node.
func (pd *PDAG) Children(node string) []string {
	return pd.p.Children(node)
}

// Skeleton returns an undirected Graph that ignores edge directions.
func (pd *PDAG) Skeleton() *graphgo.Graph {
	return pd.p.Skeleton()
}

// ToDAG orients all undirected edges in the PDAG to produce a consistent DAG.
// It works on a copy: for each undirected edge, it tries both orientations,
// picks one that preserves acyclicity, and applies Meek rules. Returns an error
// if no valid orientation can be found.
func (pd *PDAG) ToDAG() (*DAG, error) {
	work := pd.p.Copy()

	for {
		undirected := work.UndirectedEdges()
		if len(undirected) == 0 {
			break
		}

		edge := undirected[0]
		u, v := edge[0], edge[1]

		oriented := false
		// Try u -> v first.
		if tryOrient(work, u, v) {
			oriented = true
		} else if tryOrient(work, v, u) {
			oriented = true
		}

		if !oriented {
			return nil, fmt.Errorf("base: cannot orient edge (%q, %q) without creating a cycle", u, v)
		}
	}

	// Build a DAG from the fully directed PDAG.
	dag := NewDAG()
	nodes := work.Nodes()
	sort.Strings(nodes)
	for _, n := range nodes {
		dag.g.AddNode(n)
	}
	for _, e := range work.DirectedEdges() {
		dag.g.AddEdge(e[0], e[1])
	}

	// Validate acyclicity.
	if !graphgo.IsDAG(dag.g) {
		return nil, fmt.Errorf("base: resulting graph is not a DAG")
	}

	return dag, nil
}

// tryOrient attempts to orient the undirected edge u—v as u→v on the given
// PDAG. If the resulting directed graph (after Meek rules) is acyclic, it
// returns true. Otherwise it reverts the change and returns false.
func tryOrient(p *graphgo.PDAG, u, v string) bool {
	backup := p.Copy()

	p.RemoveUndirectedEdge(u, v)
	p.AddDirectedEdge(u, v)
	graphgo.ApplyMeekRules(p)

	// Check acyclicity: build a DiGraph from directed edges and verify.
	dg := graphgo.NewDiGraph()
	for _, n := range p.Nodes() {
		dg.AddNode(n)
	}
	for _, e := range p.DirectedEdges() {
		dg.AddEdge(e[0], e[1])
	}
	if graphgo.IsDAG(dg) {
		return true
	}

	// Revert: restore from backup.
	revertPDAG(p, backup)
	return false
}

// revertPDAG restores p to the state captured in backup. This is done by
// clearing and rebuilding since graphgo.PDAG fields are unexported.
func revertPDAG(p, backup *graphgo.PDAG) {
	// Remove all current nodes (which removes all edges too).
	for _, n := range p.Nodes() {
		p.RemoveNode(n)
	}
	// Rebuild from backup.
	for _, n := range backup.Nodes() {
		p.AddNode(n)
	}
	for _, e := range backup.DirectedEdges() {
		p.AddDirectedEdge(e[0], e[1])
	}
	for _, e := range backup.UndirectedEdges() {
		p.AddUndirectedEdge(e[0], e[1])
	}
}

// ApplyMeekRules applies the four Meek orientation rules iteratively until no
// more undirected edges can be oriented. Returns true if any changes were made.
func (pd *PDAG) ApplyMeekRules() bool {
	return graphgo.ApplyMeekRules(pd.p)
}

// Copy returns a deep copy of the PDAG.
func (pd *PDAG) Copy() *PDAG {
	return &PDAG{p: pd.p.Copy()}
}
