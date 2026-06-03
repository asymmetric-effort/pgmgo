package base

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// InDegree returns the in-degree (number of parents) of a single node.
// Returns -1 if the node does not exist.
func (d *DAG) InDegree(node string) int {
	if !d.g.HasNode(node) {
		return -1
	}
	return len(d.Parents(node))
}

// OutDegree returns the out-degree (number of children) of a single node.
// Returns -1 if the node does not exist.
func (d *DAG) OutDegree(node string) int {
	if !d.g.HasNode(node) {
		return -1
	}
	return len(d.Children(node))
}

// NumberOfNodes returns the number of nodes in the DAG.
func (d *DAG) NumberOfNodes() int {
	return len(d.Nodes())
}

// NumberOfEdges returns the number of directed edges in the DAG.
func (d *DAG) NumberOfEdges() int {
	return len(d.Edges())
}

// HasPath returns true if there is a directed path from source to target.
func (d *DAG) HasPath(source, target string) bool {
	if !d.g.HasNode(source) || !d.g.HasNode(target) {
		return false
	}
	if source == target {
		return true
	}
	desc := graphgo.Descendants(d.g, source)
	return desc[target]
}

// NumberOfNodes returns the number of nodes in the undirected graph.
func (u *UndirectedGraph) NumberOfNodes() int {
	return len(u.Nodes())
}

// NumberOfEdges returns the number of undirected edges in the graph.
func (u *UndirectedGraph) NumberOfEdges() int {
	return len(u.Edges())
}

// HasPath returns true if there is a path between source and target in the
// undirected graph.
func (u *UndirectedGraph) HasPath(source, target string) bool {
	if !u.g.HasNode(source) || !u.g.HasNode(target) {
		return false
	}
	if source == target {
		return true
	}
	// BFS
	visited := map[string]bool{source: true}
	queue := []string{source}
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if curr == target {
			return true
		}
		for _, nb := range u.g.Neighbors(curr) {
			if !visited[nb] {
				visited[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	return false
}

// DegreeIter returns a map from each node to its degree.
func (u *UndirectedGraph) DegreeIter() map[string]int {
	result := make(map[string]int)
	for _, n := range u.Nodes() {
		result[n] = u.g.Degree(n)
	}
	return result
}

// IsConnected returns true if the undirected graph is connected.
func (u *UndirectedGraph) IsConnected() bool {
	return u.g.IsConnected()
}

// AddNodes adds multiple nodes. Returns an error if any node already exists.
func (u *UndirectedGraph) AddNodes(nodes ...string) error {
	for _, n := range nodes {
		if err := u.AddNode(n); err != nil {
			return err
		}
	}
	return nil
}

// Graph returns the underlying graphgo.Graph.
func (u *UndirectedGraph) Graph() *graphgo.Graph {
	return u.g
}

// ADMG additional parity methods.

// NumberOfNodes returns the number of nodes in the ADMG.
func (a *ADMG) NumberOfNodes() int {
	return len(a.Nodes())
}

// NumberOfDirectedEdges returns the number of directed edges in the ADMG.
func (a *ADMG) NumberOfDirectedEdges() int {
	return len(a.DirectedEdges())
}

// NumberOfBidirectedEdges returns the number of bidirected edges in the ADMG.
func (a *ADMG) NumberOfBidirectedEdges() int {
	return len(a.BidirectedEdges())
}

// HasDirectedEdge returns true if the directed edge from -> to exists.
func (a *ADMG) HasDirectedEdge(from, to string) bool {
	return a.directed.HasEdge(from, to)
}

// HasBidirectedEdge returns true if the bidirected edge u <-> v exists.
func (a *ADMG) HasBidirectedEdge(u, v string) bool {
	return a.bidirected.HasEdge(u, v)
}

// RemoveNode removes a node and all its incident edges (both directed and
// bidirected) from the ADMG.
func (a *ADMG) RemoveNode(node string) error {
	if !a.directed.HasNode(node) {
		return fmt.Errorf("base: node %q not found", node)
	}
	a.directed.RemoveNode(node)
	a.bidirected.RemoveNode(node)
	return nil
}

// MAG additional parity methods.

// NumberOfNodes returns the number of nodes in the MAG.
func (m *MAG) NumberOfNodes() int {
	return m.ADMG.NumberOfNodes()
}

// PDAG additional parity methods.

// NumberOfNodes returns the number of nodes in the PDAG.
func (pd *PDAG) NumberOfNodes() int {
	return len(pd.Nodes())
}

// NumberOfDirectedEdges returns the number of directed edges in the PDAG.
func (pd *PDAG) NumberOfDirectedEdges() int {
	return len(pd.DirectedEdges())
}

// NumberOfUndirectedEdges returns the number of undirected edges in the PDAG.
func (pd *PDAG) NumberOfUndirectedEdges() int {
	return len(pd.UndirectedEdges())
}
