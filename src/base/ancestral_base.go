package base

// AncestralBase provides common ancestral-graph operations for graph types
// that expose directed parent/child relationships and a node list (e.g. ADMG,
// MAG). It is designed to be embedded or composed into concrete graph types.
type AncestralBase struct {
	// NodesFn returns all nodes in the graph.
	NodesFn func() []string
	// ParentsFn returns the directed parents of a node.
	ParentsFn func(string) []string
	// ChildrenFn returns the directed children of a node.
	ChildrenFn func(string) []string
}

// Ancestors returns the set of all ancestors of node reachable by following
// directed edges backwards. The node itself is NOT included.
func (ab *AncestralBase) Ancestors(node string) map[string]bool {
	result := make(map[string]bool)
	ab.ancestorsDFS(node, result)
	return result
}

func (ab *AncestralBase) ancestorsDFS(node string, visited map[string]bool) {
	for _, p := range ab.ParentsFn(node) {
		if !visited[p] {
			visited[p] = true
			ab.ancestorsDFS(p, visited)
		}
	}
}

// Descendants returns the set of all descendants of node reachable by
// following directed edges forwards. The node itself is NOT included.
func (ab *AncestralBase) Descendants(node string) map[string]bool {
	result := make(map[string]bool)
	ab.descendantsDFS(node, result)
	return result
}

func (ab *AncestralBase) descendantsDFS(node string, visited map[string]bool) {
	for _, c := range ab.ChildrenFn(node) {
		if !visited[c] {
			visited[c] = true
			ab.descendantsDFS(c, visited)
		}
	}
}

// IsAncestor returns true if ancestor is an ancestor of descendant (i.e.
// there is a directed path from ancestor to descendant).
func (ab *AncestralBase) IsAncestor(ancestor, descendant string) bool {
	anc := ab.Ancestors(descendant)
	return anc[ancestor]
}

// AnteriorNodes returns the given nodes together with all their ancestors.
func (ab *AncestralBase) AnteriorNodes(nodes []string) map[string]bool {
	result := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		result[n] = true
		for a := range ab.Ancestors(n) {
			result[a] = true
		}
	}
	return result
}
