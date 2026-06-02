//go:build unit

package base

import (
	"testing"
)

// helper to build an AncestralBase from an ADMG.
func admgAncestral(a *ADMG) *AncestralBase {
	return &AncestralBase{
		NodesFn:    a.Nodes,
		ParentsFn:  a.Parents,
		ChildrenFn: a.Children,
	}
}

// helper to build an AncestralBase from a DAG.
func dagAncestral(d *DAG) *AncestralBase {
	return &AncestralBase{
		NodesFn:    d.Nodes,
		ParentsFn:  d.Parents,
		ChildrenFn: d.Children,
	}
}

func TestAncestors_Simple(t *testing.T) {
	// A -> B -> C
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C")
	_ = d.AddEdge("A", "B")
	_ = d.AddEdge("B", "C")

	ab := dagAncestral(d)

	anc := ab.Ancestors("C")
	if !anc["A"] || !anc["B"] {
		t.Errorf("Ancestors(C) = %v, want {A, B}", anc)
	}
	if len(anc) != 2 {
		t.Errorf("Ancestors(C) has %d elements, want 2", len(anc))
	}

	anc = ab.Ancestors("A")
	if len(anc) != 0 {
		t.Errorf("Ancestors(A) = %v, want empty", anc)
	}
}

func TestDescendants_Simple(t *testing.T) {
	// A -> B -> C
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C")
	_ = d.AddEdge("A", "B")
	_ = d.AddEdge("B", "C")

	ab := dagAncestral(d)

	desc := ab.Descendants("A")
	if !desc["B"] || !desc["C"] {
		t.Errorf("Descendants(A) = %v, want {B, C}", desc)
	}
	if len(desc) != 2 {
		t.Errorf("Descendants(A) has %d elements, want 2", len(desc))
	}

	desc = ab.Descendants("C")
	if len(desc) != 0 {
		t.Errorf("Descendants(C) = %v, want empty", desc)
	}
}

func TestIsAncestor(t *testing.T) {
	// A -> B -> C, A -> C
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C")
	_ = d.AddEdge("A", "B")
	_ = d.AddEdge("B", "C")
	_ = d.AddEdge("A", "C")

	ab := dagAncestral(d)

	if !ab.IsAncestor("A", "C") {
		t.Error("A should be ancestor of C")
	}
	if !ab.IsAncestor("B", "C") {
		t.Error("B should be ancestor of C")
	}
	if ab.IsAncestor("C", "A") {
		t.Error("C should not be ancestor of A")
	}
	if ab.IsAncestor("B", "A") {
		t.Error("B should not be ancestor of A")
	}
}

func TestAnteriorNodes(t *testing.T) {
	// A -> B -> C, D -> C
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C", "D")
	_ = d.AddEdge("A", "B")
	_ = d.AddEdge("B", "C")
	_ = d.AddEdge("D", "C")

	ab := dagAncestral(d)

	ant := ab.AnteriorNodes([]string{"C"})
	// Should include C, B, A, D.
	expected := map[string]bool{"A": true, "B": true, "C": true, "D": true}
	if len(ant) != len(expected) {
		t.Errorf("AnteriorNodes(C) = %v, want %v", ant, expected)
	}
	for k := range expected {
		if !ant[k] {
			t.Errorf("AnteriorNodes(C) missing %q", k)
		}
	}
}

func TestAnteriorNodes_Multiple(t *testing.T) {
	// A -> B, C -> D
	d := NewDAG()
	_ = d.AddNodes("A", "B", "C", "D")
	_ = d.AddEdge("A", "B")
	_ = d.AddEdge("C", "D")

	ab := dagAncestral(d)

	ant := ab.AnteriorNodes([]string{"B", "D"})
	expected := map[string]bool{"A": true, "B": true, "C": true, "D": true}
	if len(ant) != len(expected) {
		t.Errorf("AnteriorNodes(B,D) = %v, want %v", ant, expected)
	}
}

func TestAncestralBase_WithADMG(t *testing.T) {
	// X -> Y -> Z, X <-> Z (bidirected)
	a := NewADMG()
	_ = a.AddNode("X")
	_ = a.AddNode("Y")
	_ = a.AddNode("Z")
	_ = a.AddDirectedEdge("X", "Y")
	_ = a.AddDirectedEdge("Y", "Z")
	_ = a.AddBidirectedEdge("X", "Z")

	ab := admgAncestral(a)

	anc := ab.Ancestors("Z")
	if !anc["X"] || !anc["Y"] {
		t.Errorf("Ancestors(Z) = %v, want {X, Y}", anc)
	}

	desc := ab.Descendants("X")
	if !desc["Y"] || !desc["Z"] {
		t.Errorf("Descendants(X) = %v, want {Y, Z}", desc)
	}

	if !ab.IsAncestor("X", "Z") {
		t.Error("X should be ancestor of Z")
	}
}

func TestAncestors_NoParents(t *testing.T) {
	d := NewDAG()
	_ = d.AddNode("A")

	ab := dagAncestral(d)
	anc := ab.Ancestors("A")
	if len(anc) != 0 {
		t.Errorf("Ancestors of root node should be empty, got %v", anc)
	}
}

func TestDescendants_NoChildren(t *testing.T) {
	d := NewDAG()
	_ = d.AddNode("A")

	ab := dagAncestral(d)
	desc := ab.Descendants("A")
	if len(desc) != 0 {
		t.Errorf("Descendants of leaf node should be empty, got %v", desc)
	}
}
