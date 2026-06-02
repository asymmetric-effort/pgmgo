//go:build unit

package graphgo

import (
	"sort"
	"testing"
)

// studentNetwork builds the classic student Bayesian network:
// D→G, I→G, G→L, I→S
func studentNetwork() *DiGraph {
	g := NewDiGraph()
	g.AddEdge("D", "G")
	g.AddEdge("I", "G")
	g.AddEdge("G", "L")
	g.AddEdge("I", "S")
	return g
}

func TestMoralizeStudentNetwork(t *testing.T) {
	dag := studentNetwork()
	moral := Moralize(dag)

	// All original nodes should be present.
	nodes := moral.Nodes()
	sort.Strings(nodes)
	expected := []string{"D", "G", "I", "L", "S"}
	if len(nodes) != len(expected) {
		t.Fatalf("expected %d nodes, got %d", len(expected), len(nodes))
	}
	for i, n := range expected {
		if nodes[i] != n {
			t.Fatalf("expected node %s at position %d, got %s", n, i, nodes[i])
		}
	}

	// All original edges should be present (undirected).
	originalEdges := [][2]string{{"D", "G"}, {"I", "G"}, {"G", "L"}, {"I", "S"}}
	for _, e := range originalEdges {
		if !moral.HasEdge(e[0], e[1]) {
			t.Fatalf("expected edge %s-%s from original graph", e[0], e[1])
		}
	}

	// Moral edge: D and I should be married (co-parents of G).
	if !moral.HasEdge("D", "I") {
		t.Fatal("expected moral edge D-I (co-parents of G)")
	}

	// Count edges: 4 original + 1 moral = 5
	edges := moral.Edges()
	if len(edges) != 5 {
		t.Fatalf("expected 5 edges, got %d", len(edges))
	}
}

func TestMoralizeNoCoParents(t *testing.T) {
	// Chain: A→B→C. No co-parents, so no moral edges added.
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	moral := Moralize(g)
	if len(moral.Edges()) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(moral.Edges()))
	}
}

func TestMoralizeMultipleCoParents(t *testing.T) {
	// Three parents: A→D, B→D, C→D. Should marry A-B, A-C, B-C.
	g := NewDiGraph()
	g.AddEdge("A", "D")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	moral := Moralize(g)
	if !moral.HasEdge("A", "B") {
		t.Fatal("expected moral edge A-B")
	}
	if !moral.HasEdge("A", "C") {
		t.Fatal("expected moral edge A-C")
	}
	if !moral.HasEdge("B", "C") {
		t.Fatal("expected moral edge B-C")
	}
	// 3 original + 3 moral = 6
	if len(moral.Edges()) != 6 {
		t.Fatalf("expected 6 edges, got %d", len(moral.Edges()))
	}
}

func TestMoralizeEmpty(t *testing.T) {
	g := NewDiGraph()
	moral := Moralize(g)
	if len(moral.Nodes()) != 0 {
		t.Fatal("expected empty moral graph")
	}
}

func TestTriangulateStudentNetwork(t *testing.T) {
	dag := studentNetwork()
	moral := Moralize(dag)

	// Use elimination order: S, L, G, I, D
	order := []string{"S", "L", "G", "I", "D"}
	tri := Triangulate(moral, order)

	// The triangulated graph should be chordal.
	if !IsChordal(tri) {
		t.Fatal("triangulated graph should be chordal")
	}

	// All original moral edges should still be present.
	for _, e := range moral.Edges() {
		if !tri.HasEdge(e.A, e.B) {
			t.Fatalf("expected edge %s-%s from moral graph", e.A, e.B)
		}
	}

	// All original nodes should still be present.
	for _, n := range moral.Nodes() {
		if !tri.HasNode(n) {
			t.Fatalf("expected node %s in triangulated graph", n)
		}
	}
}

func TestTriangulateAlreadyChordal(t *testing.T) {
	// A complete graph on 3 nodes is already chordal.
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	tri := Triangulate(g, []string{"A", "B", "C"})
	if !IsChordal(tri) {
		t.Fatal("complete graph should remain chordal")
	}
	if len(tri.Edges()) != 3 {
		t.Fatalf("expected 3 edges, got %d", len(tri.Edges()))
	}
}

func TestTriangulateFourCycle(t *testing.T) {
	// A 4-cycle A-B-C-D-A is not chordal.
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "D")
	g.AddEdge("D", "A")

	if IsChordal(g) {
		t.Fatal("4-cycle should not be chordal")
	}

	tri := Triangulate(g, []string{"A", "B", "C", "D"})
	if !IsChordal(tri) {
		t.Fatal("triangulated 4-cycle should be chordal")
	}
}

func TestIsChordalCompleteGraph(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("A", "D")
	g.AddEdge("B", "C")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	if !IsChordal(g) {
		t.Fatal("complete graph should be chordal")
	}
}

func TestIsChordalEmpty(t *testing.T) {
	g := NewGraph()
	if !IsChordal(g) {
		t.Fatal("empty graph should be chordal")
	}
}

func TestIsChordalSingleNode(t *testing.T) {
	g := NewGraph()
	g.AddNode("A")
	if !IsChordal(g) {
		t.Fatal("single node should be chordal")
	}
}

func TestIsChordalTree(t *testing.T) {
	// Trees are always chordal.
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "D")
	g.AddEdge("B", "E")

	if !IsChordal(g) {
		t.Fatal("tree should be chordal")
	}
}

func TestIsChordalFiveCycle(t *testing.T) {
	// 5-cycle without chords is not chordal.
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "D")
	g.AddEdge("D", "E")
	g.AddEdge("E", "A")

	if IsChordal(g) {
		t.Fatal("5-cycle should not be chordal")
	}
}
