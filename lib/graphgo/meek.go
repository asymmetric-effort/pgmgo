package graphgo

// DAGToPDAG converts a DAG (DiGraph) to its CPDAG (Completed Partially Directed
// Acyclic Graph) representation. The CPDAG represents the Markov equivalence class
// of the given DAG. Edges that are compelled (part of a v-structure or forced by
// Meek rules) remain directed; all other edges become undirected.
func DAGToPDAG(g *DiGraph) *PDAG {
	p := NewPDAG()

	// Add all nodes.
	for _, n := range g.Nodes() {
		p.AddNode(n)
	}

	// Start with all edges undirected.
	for _, e := range g.Edges() {
		p.AddUndirectedEdge(e.Src, e.Dst)
	}

	// Orient v-structures: for each node c, if parents a and b are not adjacent
	// in the original DAG, then a→c←b is a v-structure.
	for _, c := range g.Nodes() {
		parents := g.Parents(c)
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				a, b := parents[i], parents[j]
				if !g.HasEdge(a, b) && !g.HasEdge(b, a) {
					// Orient a→c and b→c.
					p.RemoveUndirectedEdge(a, c)
					p.AddDirectedEdge(a, c)
					p.RemoveUndirectedEdge(b, c)
					p.AddDirectedEdge(b, c)
				}
			}
		}
	}

	// Apply Meek rules until convergence.
	ApplyMeekRules(p)

	return p
}

// ApplyMeekRules applies all four Meek orientation rules iteratively until no
// more edges can be oriented. Returns true if any changes were made.
//
// The rules orient undirected edges while preserving the acyclicity and the
// Markov equivalence class:
//
//	R1: Orient u—v into u→v if ∃ w→u and w is not adjacent to v.
//	R2: Orient u—v into u→v if ∃ chain u→w→v.
//	R3: Orient u—v into u→v if ∃ w1—u, w2—u, w1→v, w2→v, and w1 not adj w2.
//	R4: Orient u—v into u→v if ∃ w—u, w→x→v.
func ApplyMeekRules(p *PDAG) bool {
	anyChanged := false
	for {
		changed := false
		changed = applyR1(p) || changed
		changed = applyR2(p) || changed
		changed = applyR3(p) || changed
		changed = applyR4(p) || changed
		if changed {
			anyChanged = true
		} else {
			break
		}
	}
	return anyChanged
}

// orient converts an undirected edge u—v into a directed edge u→v.
// Returns true if the orientation was performed.
func orient(p *PDAG, u, v string) bool {
	if !p.HasUndirectedEdge(u, v) {
		return false
	}
	p.RemoveUndirectedEdge(u, v)
	p.AddDirectedEdge(u, v)
	return true
}

// applyR1: Orient u—v into u→v if ∃ w→u and w not adjacent to v.
func applyR1(p *PDAG) bool {
	changed := false
	for _, edge := range p.UndirectedEdges() {
		u, v := edge[0], edge[1]
		// Try orienting u→v: need w→u where w not adj v.
		if r1Check(p, u, v) {
			orient(p, u, v)
			changed = true
			continue
		}
		// Try orienting v→u: need w→v where w not adj u.
		if r1Check(p, v, u) {
			orient(p, v, u)
			changed = true
		}
	}
	return changed
}

// r1Check returns true if there exists w→u where w is not adjacent to v.
func r1Check(p *PDAG, u, v string) bool {
	for _, w := range p.Parents(u) {
		if w != v && !p.Adjacent(w, v) {
			return true
		}
	}
	return false
}

// applyR2: Orient u—v into u→v if ∃ chain u→w→v.
func applyR2(p *PDAG) bool {
	changed := false
	for _, edge := range p.UndirectedEdges() {
		u, v := edge[0], edge[1]
		if r2Check(p, u, v) {
			orient(p, u, v)
			changed = true
			continue
		}
		if r2Check(p, v, u) {
			orient(p, v, u)
			changed = true
		}
	}
	return changed
}

// r2Check returns true if there exists w such that u→w→v (both directed).
func r2Check(p *PDAG, u, v string) bool {
	for _, w := range p.Children(u) {
		if p.HasDirectedEdge(w, v) {
			return true
		}
	}
	return false
}

// applyR3: Orient u—v into u→v if ∃ w1—u, w2—u, w1→v, w2→v, w1 not adj w2.
func applyR3(p *PDAG) bool {
	changed := false
	for _, edge := range p.UndirectedEdges() {
		u, v := edge[0], edge[1]
		if r3Check(p, u, v) {
			orient(p, u, v)
			changed = true
			continue
		}
		if r3Check(p, v, u) {
			orient(p, v, u)
			changed = true
		}
	}
	return changed
}

// r3Check returns true if ∃ w1, w2 such that w1—u, w2—u, w1→v, w2→v, w1 not adj w2.
func r3Check(p *PDAG, u, v string) bool {
	// Collect nodes w that satisfy: w—u (undirected) and w→v (directed).
	var candidates []string
	for _, n := range p.Nodes() {
		if n == u || n == v {
			continue
		}
		if p.HasUndirectedEdge(n, u) && p.HasDirectedEdge(n, v) {
			candidates = append(candidates, n)
		}
	}
	// Check if any pair of candidates is non-adjacent.
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if !p.Adjacent(candidates[i], candidates[j]) {
				return true
			}
		}
	}
	return false
}

// applyR4: Orient u—v into u→v if ∃ w—u, w→x→v.
func applyR4(p *PDAG) bool {
	changed := false
	for _, edge := range p.UndirectedEdges() {
		u, v := edge[0], edge[1]
		if r4Check(p, u, v) {
			orient(p, u, v)
			changed = true
			continue
		}
		if r4Check(p, v, u) {
			orient(p, v, u)
			changed = true
		}
	}
	return changed
}

// r4Check returns true if ∃ w such that w—u (undirected), and ∃ x such that w→x→v.
func r4Check(p *PDAG, u, v string) bool {
	// For each w with w—u (undirected):
	for w := range p.undirected[u] {
		if w == v {
			continue
		}
		// For each x that w→x (directed):
		for x := range p.directed[w] {
			if x == u || x == v {
				continue
			}
			// Check x→v (directed).
			if p.HasDirectedEdge(x, v) {
				return true
			}
		}
	}
	return false
}
