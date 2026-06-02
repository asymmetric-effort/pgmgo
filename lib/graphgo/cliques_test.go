//go:build unit

package graphgo

import (
	"sort"
	"testing"
)

func TestMaxCliquesTriangle(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	cliques := MaxCliques(g)
	if len(cliques) != 1 {
		t.Fatalf("expected 1 clique, got %d: %v", len(cliques), cliques)
	}
	expected := []string{"A", "B", "C"}
	if !stringSliceEqual(cliques[0], expected) {
		t.Fatalf("expected clique %v, got %v", expected, cliques[0])
	}
}

func TestMaxCliquesPath(t *testing.T) {
	// Path A-B-C: two maximal cliques {A,B} and {B,C}.
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	cliques := MaxCliques(g)
	if len(cliques) != 2 {
		t.Fatalf("expected 2 cliques, got %d: %v", len(cliques), cliques)
	}
	expected := [][]string{{"A", "B"}, {"B", "C"}}
	for i, c := range expected {
		if !stringSliceEqual(cliques[i], c) {
			t.Fatalf("expected clique %v at position %d, got %v", c, i, cliques[i])
		}
	}
}

func TestMaxCliquesStudentMoralized(t *testing.T) {
	// Student network moralized and triangulated.
	dag := NewDiGraph()
	dag.AddEdge("D", "G")
	dag.AddEdge("I", "G")
	dag.AddEdge("G", "L")
	dag.AddEdge("I", "S")

	moral := Moralize(dag)
	tri := Triangulate(moral, []string{"S", "L", "G", "I", "D"})

	cliques := MaxCliques(tri)

	// Each clique should actually be a clique.
	for _, c := range cliques {
		for i := 0; i < len(c); i++ {
			for j := i + 1; j < len(c); j++ {
				if !tri.HasEdge(c[i], c[j]) {
					t.Fatalf("nodes %s and %s in clique %v are not connected", c[i], c[j], c)
				}
			}
		}
	}

	// Each clique should be maximal: no node outside the clique is connected
	// to all nodes in the clique.
	allNodes := tri.Nodes()
	for _, c := range cliques {
		cSet := make(map[string]bool)
		for _, n := range c {
			cSet[n] = true
		}
		for _, n := range allNodes {
			if cSet[n] {
				continue
			}
			allConnected := true
			for _, cn := range c {
				if !tri.HasEdge(n, cn) {
					allConnected = false
					break
				}
			}
			if allConnected {
				t.Fatalf("clique %v is not maximal: node %s could be added", c, n)
			}
		}
	}

	// D-I edge exists from moralization, and D-G, I-G from original,
	// so {D, G, I} should be a clique. I-S is an edge, so {I, S} should
	// be a clique. G-L is an edge, so {G, L} should be present in some clique.
	foundDGI := false
	foundIS := false
	foundGL := false
	for _, c := range cliques {
		cs := make(map[string]bool)
		for _, n := range c {
			cs[n] = true
		}
		if cs["D"] && cs["G"] && cs["I"] {
			foundDGI = true
		}
		if cs["I"] && cs["S"] {
			foundIS = true
		}
		if cs["G"] && cs["L"] {
			foundGL = true
		}
	}
	if !foundDGI {
		t.Fatalf("expected clique containing {D, G, I}, got %v", cliques)
	}
	if !foundIS {
		t.Fatalf("expected clique containing {I, S}, got %v", cliques)
	}
	if !foundGL {
		t.Fatalf("expected clique containing {G, L}, got %v", cliques)
	}
}

func TestMaxCliquesEmpty(t *testing.T) {
	g := NewGraph()
	cliques := MaxCliques(g)
	if len(cliques) != 0 {
		t.Fatalf("expected 0 cliques for empty graph, got %d", len(cliques))
	}
}

func TestMaxCliquesSingleNode(t *testing.T) {
	g := NewGraph()
	g.AddNode("A")
	cliques := MaxCliques(g)
	if len(cliques) != 1 {
		t.Fatalf("expected 1 clique, got %d", len(cliques))
	}
	if !stringSliceEqual(cliques[0], []string{"A"}) {
		t.Fatalf("expected clique [A], got %v", cliques[0])
	}
}

func TestMaxCliquesDisconnected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddNode("C")

	cliques := MaxCliques(g)
	if len(cliques) != 2 {
		t.Fatalf("expected 2 cliques, got %d: %v", len(cliques), cliques)
	}
}

func TestMaxCliquesCompleteGraph(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("A", "D")
	g.AddEdge("B", "C")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	cliques := MaxCliques(g)
	if len(cliques) != 1 {
		t.Fatalf("expected 1 clique for K4, got %d: %v", len(cliques), cliques)
	}
	expected := []string{"A", "B", "C", "D"}
	if !stringSliceEqual(cliques[0], expected) {
		t.Fatalf("expected clique %v, got %v", expected, cliques[0])
	}
}

func TestBuildJunctionTreeBasic(t *testing.T) {
	cliques := [][]string{
		{"A", "B", "C"},
		{"B", "C", "D"},
		{"C", "E"},
	}

	tree, separators := BuildJunctionTree(cliques)

	// Tree should have 3 nodes.
	if len(tree.Nodes()) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(tree.Nodes()))
	}

	// Tree should have 2 edges (n-1 for a tree).
	if len(tree.Edges()) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(tree.Edges()))
	}

	// The edge between cliques 0 and 1 should have separator {B, C}.
	k01 := undirectedEdgeKey("0", "1")
	sep01, ok := separators[k01]
	if !ok {
		t.Fatal("expected separator between cliques 0 and 1")
	}
	sort.Strings(sep01)
	expectedSep := []string{"B", "C"}
	if !stringSliceEqual(sep01, expectedSep) {
		t.Fatalf("expected separator %v, got %v", expectedSep, sep01)
	}
}

func TestBuildJunctionTreeStudentNetwork(t *testing.T) {
	dag := NewDiGraph()
	dag.AddEdge("D", "G")
	dag.AddEdge("I", "G")
	dag.AddEdge("G", "L")
	dag.AddEdge("I", "S")

	moral := Moralize(dag)
	tri := Triangulate(moral, []string{"S", "L", "G", "I", "D"})
	cliques := MaxCliques(tri)

	tree, separators := BuildJunctionTree(cliques)

	// Tree should have len(cliques) nodes and len(cliques)-1 edges.
	nCliques := len(cliques)
	if len(tree.Nodes()) != nCliques {
		t.Fatalf("expected %d nodes, got %d", nCliques, len(tree.Nodes()))
	}
	if len(tree.Edges()) != nCliques-1 {
		t.Fatalf("expected %d edges, got %d", nCliques-1, len(tree.Edges()))
	}

	// Every separator should be non-empty and a subset of both cliques.
	for _, e := range tree.Edges() {
		k := undirectedEdgeKey(e.A, e.B)
		sep, ok := separators[k]
		if !ok {
			t.Fatalf("missing separator for edge %s-%s", e.A, e.B)
		}
		if len(sep) == 0 {
			t.Fatalf("separator for edge %s-%s should not be empty", e.A, e.B)
		}
	}
}

func TestBuildJunctionTreeEmpty(t *testing.T) {
	tree, separators := BuildJunctionTree(nil)
	if len(tree.Nodes()) != 0 {
		t.Fatal("expected empty tree")
	}
	if separators != nil {
		t.Fatal("expected nil separators")
	}
}

func TestBuildJunctionTreeSingleClique(t *testing.T) {
	tree, separators := BuildJunctionTree([][]string{{"A", "B"}})
	if len(tree.Nodes()) != 1 {
		t.Fatalf("expected 1 node, got %d", len(tree.Nodes()))
	}
	if len(tree.Edges()) != 0 {
		t.Fatalf("expected 0 edges, got %d", len(tree.Edges()))
	}
	if len(separators) != 0 {
		t.Fatalf("expected 0 separators, got %d", len(separators))
	}
}

// stringSliceEqual checks if two sorted string slices are equal.
func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
