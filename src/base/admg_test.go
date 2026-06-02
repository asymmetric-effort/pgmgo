//go:build unit

package base

import (
	"testing"
)

func TestNewADMG(t *testing.T) {
	a := NewADMG()
	if a == nil {
		t.Fatal("NewADMG returned nil")
	}
	if len(a.Nodes()) != 0 {
		t.Errorf("new ADMG should have 0 nodes, got %d", len(a.Nodes()))
	}
	if len(a.DirectedEdges()) != 0 {
		t.Errorf("new ADMG should have 0 directed edges, got %d", len(a.DirectedEdges()))
	}
	if len(a.BidirectedEdges()) != 0 {
		t.Errorf("new ADMG should have 0 bidirected edges, got %d", len(a.BidirectedEdges()))
	}
}

func TestADMGAddNode(t *testing.T) {
	a := NewADMG()
	if err := a.AddNode("X"); err != nil {
		t.Fatalf("AddNode failed: %v", err)
	}
	if !a.HasNode("X") {
		t.Error("HasNode returned false for added node")
	}
}

func TestADMGAddNodeDuplicate(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	if err := a.AddNode("X"); err == nil {
		t.Error("expected error when adding duplicate node")
	}
}

func TestADMGAddDirectedEdge(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	if err := a.AddDirectedEdge("A", "B"); err != nil {
		t.Fatalf("AddDirectedEdge failed: %v", err)
	}
	edges := a.DirectedEdges()
	if len(edges) != 1 || edges[0].Src != "A" || edges[0].Dst != "B" {
		t.Errorf("DirectedEdges() = %v, want [(A,B)]", edges)
	}
}

func TestADMGAddDirectedEdgeMissingNode(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	if err := a.AddDirectedEdge("A", "B"); err == nil {
		t.Error("expected error for missing target node")
	}
	if err := a.AddDirectedEdge("Z", "A"); err == nil {
		t.Error("expected error for missing source node")
	}
}

func TestADMGAddDirectedEdgeDuplicate(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddDirectedEdge("A", "B")
	if err := a.AddDirectedEdge("A", "B"); err == nil {
		t.Error("expected error for duplicate directed edge")
	}
}

func TestADMGDirectedEdgeRejectsCycle(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddDirectedEdge("A", "B")
	_ = a.AddDirectedEdge("B", "C")

	err := a.AddDirectedEdge("C", "A")
	if err == nil {
		t.Fatal("expected error for cycle-creating directed edge")
	}
	// Original edges should remain.
	edges := a.DirectedEdges()
	if len(edges) != 2 {
		t.Errorf("expected 2 directed edges after rejected cycle, got %d", len(edges))
	}
}

func TestADMGAddBidirectedEdge(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	_ = a.AddNode("Y")
	if err := a.AddBidirectedEdge("X", "Y"); err != nil {
		t.Fatalf("AddBidirectedEdge failed: %v", err)
	}
	biEdges := a.BidirectedEdges()
	if len(biEdges) != 1 {
		t.Fatalf("expected 1 bidirected edge, got %d", len(biEdges))
	}
	if biEdges[0].Src != "X" || biEdges[0].Dst != "Y" {
		t.Errorf("BidirectedEdges() = %v, want [(X,Y)]", biEdges)
	}
}

func TestADMGAddBidirectedEdgeMissingNode(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	if err := a.AddBidirectedEdge("X", "Z"); err == nil {
		t.Error("expected error for missing node")
	}
}

func TestADMGAddBidirectedEdgeDuplicate(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	_ = a.AddNode("Y")
	_ = a.AddBidirectedEdge("X", "Y")
	if err := a.AddBidirectedEdge("X", "Y"); err == nil {
		t.Error("expected error for duplicate bidirected edge")
	}
	// Reverse order should also be a duplicate.
	if err := a.AddBidirectedEdge("Y", "X"); err == nil {
		t.Error("expected error for duplicate bidirected edge (reversed)")
	}
}

func TestADMGBidirectedEdgeSymmetric(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddBidirectedEdge("A", "B")

	// Siblings should be symmetric.
	sibsA := a.Siblings("A")
	sibsB := a.Siblings("B")
	if len(sibsA) != 1 || sibsA[0] != "B" {
		t.Errorf("Siblings(A) = %v, want [B]", sibsA)
	}
	if len(sibsB) != 1 || sibsB[0] != "A" {
		t.Errorf("Siblings(B) = %v, want [A]", sibsB)
	}
}

func TestADMGParentsAndChildren(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddDirectedEdge("A", "C")
	_ = a.AddDirectedEdge("B", "C")

	parents := a.Parents("C")
	if len(parents) != 2 || parents[0] != "A" || parents[1] != "B" {
		t.Errorf("Parents(C) = %v, want [A B]", parents)
	}
	children := a.Children("A")
	if len(children) != 1 || children[0] != "C" {
		t.Errorf("Children(A) = %v, want [C]", children)
	}
}

func TestADMGSiblings(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddBidirectedEdge("A", "B")
	_ = a.AddBidirectedEdge("A", "C")

	sibs := a.Siblings("A")
	if len(sibs) != 2 || sibs[0] != "B" || sibs[1] != "C" {
		t.Errorf("Siblings(A) = %v, want [B C]", sibs)
	}
	if len(a.Siblings("B")) != 1 {
		t.Errorf("Siblings(B) should have 1 element")
	}
}

func TestADMGDistrictsSingleComponent(t *testing.T) {
	// All nodes connected via bidirected edges form one district.
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddBidirectedEdge("A", "B")
	_ = a.AddBidirectedEdge("B", "C")

	districts := a.Districts()
	if len(districts) != 1 {
		t.Fatalf("expected 1 district, got %d: %v", len(districts), districts)
	}
	if len(districts[0]) != 3 {
		t.Errorf("district should have 3 nodes, got %v", districts[0])
	}
}

func TestADMGDistrictsMultipleComponents(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddNode("D")
	_ = a.AddBidirectedEdge("A", "B")
	_ = a.AddBidirectedEdge("C", "D")
	// A-B and C-D form two separate districts.

	districts := a.Districts()
	if len(districts) != 2 {
		t.Fatalf("expected 2 districts, got %d: %v", len(districts), districts)
	}
	// Districts should be sorted by first element.
	if districts[0][0] != "A" || districts[0][1] != "B" {
		t.Errorf("district 0 = %v, want [A B]", districts[0])
	}
	if districts[1][0] != "C" || districts[1][1] != "D" {
		t.Errorf("district 1 = %v, want [C D]", districts[1])
	}
}

func TestADMGDistrictsIsolatedNodes(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	// No bidirected edges: each node is its own district.
	// (Directed edges do not affect districts.)
	_ = a.AddDirectedEdge("A", "B")

	districts := a.Districts()
	if len(districts) != 3 {
		t.Fatalf("expected 3 districts (isolated nodes), got %d: %v", len(districts), districts)
	}
}

func TestADMGCopy(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddDirectedEdge("A", "B")
	_ = a.AddBidirectedEdge("B", "C")

	c := a.Copy()

	// Verify same structure.
	if len(c.Nodes()) != 3 {
		t.Errorf("copy should have 3 nodes, got %d", len(c.Nodes()))
	}
	if len(c.DirectedEdges()) != 1 {
		t.Errorf("copy should have 1 directed edge")
	}
	if len(c.BidirectedEdges()) != 1 {
		t.Errorf("copy should have 1 bidirected edge")
	}

	// Modify copy, ensure original is unchanged.
	_ = c.AddNode("D")
	_ = c.AddDirectedEdge("B", "D")
	if a.HasNode("D") {
		t.Error("original should not have node D after modifying copy")
	}
}

func TestADMGComplexGraph(t *testing.T) {
	// Build a graph with both edge types:
	// Directed: A→B, B→C, A→D
	// Bidirected: B↔D, C↔D
	a := NewADMG()
	for _, n := range []string{"A", "B", "C", "D"} {
		_ = a.AddNode(n)
	}
	_ = a.AddDirectedEdge("A", "B")
	_ = a.AddDirectedEdge("B", "C")
	_ = a.AddDirectedEdge("A", "D")
	_ = a.AddBidirectedEdge("B", "D")
	_ = a.AddBidirectedEdge("C", "D")

	// Parents.
	if p := a.Parents("C"); len(p) != 1 || p[0] != "B" {
		t.Errorf("Parents(C) = %v, want [B]", p)
	}

	// Children.
	if c := a.Children("A"); len(c) != 2 || c[0] != "B" || c[1] != "D" {
		t.Errorf("Children(A) = %v, want [B D]", c)
	}

	// Siblings.
	sibsD := a.Siblings("D")
	if len(sibsD) != 2 || sibsD[0] != "B" || sibsD[1] != "C" {
		t.Errorf("Siblings(D) = %v, want [B C]", sibsD)
	}

	// Districts: B, C, D are all connected via bidirected edges (B↔D, C↔D).
	// A has no bidirected edges so it is its own district.
	districts := a.Districts()
	if len(districts) != 2 {
		t.Fatalf("expected 2 districts, got %d: %v", len(districts), districts)
	}
	// First district should be [A], second should be [B, C, D].
	if len(districts[0]) != 1 || districts[0][0] != "A" {
		t.Errorf("district 0 = %v, want [A]", districts[0])
	}
	if len(districts[1]) != 3 {
		t.Errorf("district 1 = %v, want [B C D]", districts[1])
	}

	// Directed edges count.
	if len(a.DirectedEdges()) != 3 {
		t.Errorf("expected 3 directed edges, got %d", len(a.DirectedEdges()))
	}
	// Bidirected edges count.
	if len(a.BidirectedEdges()) != 2 {
		t.Errorf("expected 2 bidirected edges, got %d", len(a.BidirectedEdges()))
	}
}

func TestADMGNodesSorted(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("C")
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	nodes := a.Nodes()
	if len(nodes) != 3 || nodes[0] != "A" || nodes[1] != "B" || nodes[2] != "C" {
		t.Errorf("Nodes() should be sorted, got %v", nodes)
	}
}
