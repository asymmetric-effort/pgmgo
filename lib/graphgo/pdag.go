package graphgo

import "sort"

// PDAG is a Partially Directed Acyclic Graph that supports both directed (→)
// and undirected (—) edges. It is used to represent Markov equivalence classes
// (CPDAGs) in causal inference.
type PDAG struct {
	nodes      map[string]bool
	directed   map[string]map[string]bool // from -> set of to (directed edges)
	undirected map[string]map[string]bool // node -> set of neighbors (undirected edges, stored symmetrically)
}

// NewPDAG creates an empty PDAG.
func NewPDAG() *PDAG {
	return &PDAG{
		nodes:      make(map[string]bool),
		directed:   make(map[string]map[string]bool),
		undirected: make(map[string]map[string]bool),
	}
}

// AddNode adds a node to the PDAG. If the node already exists, this is a no-op.
func (p *PDAG) AddNode(node string) {
	if !p.nodes[node] {
		p.nodes[node] = true
		p.directed[node] = make(map[string]bool)
		p.undirected[node] = make(map[string]bool)
	}
}

// AddNodes adds multiple nodes.
func (p *PDAG) AddNodes(nodes ...string) {
	for _, n := range nodes {
		p.AddNode(n)
	}
}

// RemoveNode removes a node and all its incident edges (directed and undirected).
func (p *PDAG) RemoveNode(node string) {
	if !p.nodes[node] {
		return
	}
	// Remove directed edges from this node.
	for dst := range p.directed[node] {
		// No need to remove reverse; directed edges are one-way in the map.
		_ = dst
	}
	// Remove directed edges to this node from other nodes.
	for other := range p.nodes {
		delete(p.directed[other], node)
	}
	// Remove undirected edges.
	for neighbor := range p.undirected[node] {
		delete(p.undirected[neighbor], node)
	}
	delete(p.directed, node)
	delete(p.undirected, node)
	delete(p.nodes, node)
}

// AddDirectedEdge adds a directed edge from -> to. Nodes are created if they don't exist.
func (p *PDAG) AddDirectedEdge(from, to string) {
	p.AddNode(from)
	p.AddNode(to)
	p.directed[from][to] = true
}

// AddUndirectedEdge adds an undirected edge between u and v. Nodes are created if they don't exist.
func (p *PDAG) AddUndirectedEdge(u, v string) {
	p.AddNode(u)
	p.AddNode(v)
	p.undirected[u][v] = true
	p.undirected[v][u] = true
}

// RemoveDirectedEdge removes a directed edge from -> to.
func (p *PDAG) RemoveDirectedEdge(from, to string) {
	if s, ok := p.directed[from]; ok {
		delete(s, to)
	}
}

// RemoveUndirectedEdge removes an undirected edge between u and v.
func (p *PDAG) RemoveUndirectedEdge(u, v string) {
	if s, ok := p.undirected[u]; ok {
		delete(s, v)
	}
	if s, ok := p.undirected[v]; ok {
		delete(s, u)
	}
}

// HasNode returns true if the node exists.
func (p *PDAG) HasNode(node string) bool {
	return p.nodes[node]
}

// HasDirectedEdge returns true if the directed edge from -> to exists.
func (p *PDAG) HasDirectedEdge(from, to string) bool {
	if s, ok := p.directed[from]; ok {
		return s[to]
	}
	return false
}

// HasUndirectedEdge returns true if the undirected edge between u and v exists.
func (p *PDAG) HasUndirectedEdge(u, v string) bool {
	if s, ok := p.undirected[u]; ok {
		return s[v]
	}
	return false
}

// HasEdge returns true if any edge (directed or undirected) exists between u and v.
func (p *PDAG) HasEdge(u, v string) bool {
	return p.HasDirectedEdge(u, v) || p.HasDirectedEdge(v, u) || p.HasUndirectedEdge(u, v)
}

// Adjacent returns true if u and v are adjacent (connected by any edge in either direction).
func (p *PDAG) Adjacent(u, v string) bool {
	return p.HasEdge(u, v)
}

// Nodes returns all nodes sorted lexicographically.
func (p *PDAG) Nodes() []string {
	nodes := make([]string, 0, len(p.nodes))
	for n := range p.nodes {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)
	return nodes
}

// DirectedEdges returns all directed edges as [from, to] pairs.
func (p *PDAG) DirectedEdges() [][2]string {
	var edges [][2]string
	for from, dsts := range p.directed {
		for to := range dsts {
			edges = append(edges, [2]string{from, to})
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i][0] != edges[j][0] {
			return edges[i][0] < edges[j][0]
		}
		return edges[i][1] < edges[j][1]
	})
	return edges
}

// UndirectedEdges returns all undirected edges as [u, v] pairs (each edge once, u < v).
func (p *PDAG) UndirectedEdges() [][2]string {
	seen := make(map[[2]string]bool)
	var edges [][2]string
	for u, neighbors := range p.undirected {
		for v := range neighbors {
			var key [2]string
			if u < v {
				key = [2]string{u, v}
			} else {
				key = [2]string{v, u}
			}
			if !seen[key] {
				seen[key] = true
				edges = append(edges, key)
			}
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i][0] != edges[j][0] {
			return edges[i][0] < edges[j][0]
		}
		return edges[i][1] < edges[j][1]
	})
	return edges
}

// Neighbors returns all nodes adjacent to node via any edge (directed or undirected), sorted.
func (p *PDAG) Neighbors(node string) []string {
	set := make(map[string]bool)
	// Directed successors.
	for dst := range p.directed[node] {
		set[dst] = true
	}
	// Directed predecessors.
	for other := range p.nodes {
		if p.directed[other][node] {
			set[other] = true
		}
	}
	// Undirected neighbors.
	for n := range p.undirected[node] {
		set[n] = true
	}
	out := make([]string, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Parents returns nodes with a directed edge pointing to node (predecessors), sorted.
func (p *PDAG) Parents(node string) []string {
	var parents []string
	for other := range p.nodes {
		if p.directed[other][node] {
			parents = append(parents, other)
		}
	}
	sort.Strings(parents)
	return parents
}

// Children returns nodes that node has a directed edge to (successors), sorted.
func (p *PDAG) Children(node string) []string {
	children := make([]string, 0, len(p.directed[node]))
	for dst := range p.directed[node] {
		children = append(children, dst)
	}
	sort.Strings(children)
	return children
}

// Skeleton returns an undirected Graph that ignores edge directions.
func (p *PDAG) Skeleton() *Graph {
	g := NewGraph()
	for n := range p.nodes {
		g.AddNode(n)
	}
	// Add directed edges as undirected.
	for from, dsts := range p.directed {
		for to := range dsts {
			g.AddEdge(from, to)
		}
	}
	// Add undirected edges.
	for u, neighbors := range p.undirected {
		for v := range neighbors {
			if u < v {
				g.AddEdge(u, v)
			}
		}
	}
	return g
}

// Copy returns a deep copy of the PDAG.
func (p *PDAG) Copy() *PDAG {
	c := NewPDAG()
	for n := range p.nodes {
		c.AddNode(n)
	}
	for from, dsts := range p.directed {
		for to := range dsts {
			c.directed[from][to] = true
		}
	}
	for u, neighbors := range p.undirected {
		for v := range neighbors {
			c.undirected[u][v] = true
		}
	}
	return c
}
