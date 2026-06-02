//go:build unit

package base

import (
	"testing"
)

func TestNewMAG(t *testing.T) {
	m := NewMAG()
	if m == nil {
		t.Fatal("NewMAG returned nil")
	}
	if len(m.Nodes()) != 0 {
		t.Errorf("new MAG should have 0 nodes")
	}
}

func TestMAGBasicStructure(t *testing.T) {
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddNode("Z")
	_ = m.AddDirectedEdge("X", "Y")
	_ = m.AddBidirectedEdge("Y", "Z")

	if !m.HasNode("X") || !m.HasNode("Y") || !m.HasNode("Z") {
		t.Error("missing nodes")
	}
	if len(m.DirectedEdges()) != 1 {
		t.Errorf("expected 1 directed edge, got %d", len(m.DirectedEdges()))
	}
	if len(m.BidirectedEdges()) != 1 {
		t.Errorf("expected 1 bidirected edge, got %d", len(m.BidirectedEdges()))
	}
}

func TestMSeparationChain(t *testing.T) {
	// Chain: X → Y → Z
	// X _||_ Z | Y (m-separated)
	// X not _||_ Z | {} (not m-separated)
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddNode("Z")
	_ = m.AddDirectedEdge("X", "Y")
	_ = m.AddDirectedEdge("Y", "Z")

	x := map[string]bool{"X": true}
	y := map[string]bool{"Z": true}
	condY := map[string]bool{"Y": true}
	empty := map[string]bool{}

	if !m.MSeparation(x, y, condY) {
		t.Error("X and Z should be m-separated given Y in chain X→Y→Z")
	}
	if m.MSeparation(x, y, empty) {
		t.Error("X and Z should NOT be m-separated given {} in chain X→Y→Z")
	}
}

func TestMSeparationFork(t *testing.T) {
	// Fork: X ← Y → Z
	// X _||_ Z | Y
	// X not _||_ Z | {}
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddNode("Z")
	_ = m.AddDirectedEdge("Y", "X")
	_ = m.AddDirectedEdge("Y", "Z")

	x := map[string]bool{"X": true}
	z := map[string]bool{"Z": true}
	condY := map[string]bool{"Y": true}
	empty := map[string]bool{}

	if !m.MSeparation(x, z, condY) {
		t.Error("X and Z should be m-separated given Y in fork X←Y→Z")
	}
	if m.MSeparation(x, z, empty) {
		t.Error("X and Z should NOT be m-separated given {} in fork X←Y→Z")
	}
}

func TestMSeparationCollider(t *testing.T) {
	// Collider: X → Y ← Z
	// X _||_ Z | {} (m-separated)
	// X not _||_ Z | Y (not m-separated, explaining away)
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddNode("Z")
	_ = m.AddDirectedEdge("X", "Y")
	_ = m.AddDirectedEdge("Z", "Y")

	x := map[string]bool{"X": true}
	z := map[string]bool{"Z": true}
	condY := map[string]bool{"Y": true}
	empty := map[string]bool{}

	if !m.MSeparation(x, z, empty) {
		t.Error("X and Z should be m-separated given {} in collider X→Y←Z")
	}
	if m.MSeparation(x, z, condY) {
		t.Error("X and Z should NOT be m-separated given Y in collider X→Y←Z")
	}
}

func TestMSeparationBidirectedEdge(t *testing.T) {
	// X ↔ Y (bidirected edge = latent common cause)
	// X and Y are NOT m-separated given {}.
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddBidirectedEdge("X", "Y")

	x := map[string]bool{"X": true}
	y := map[string]bool{"Y": true}
	empty := map[string]bool{}

	if m.MSeparation(x, y, empty) {
		t.Error("X and Y should NOT be m-separated when connected by bidirected edge")
	}
}

func TestMSeparationBidirectedWithConditioning(t *testing.T) {
	// X → Z, Y → Z, X ↔ Y
	// X _||_ Y | {} ? No, because X ↔ Y.
	// X _||_ Y | Z ? No, because X ↔ Y still connects them.
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")
	_ = m.AddNode("Z")
	_ = m.AddDirectedEdge("X", "Z")
	_ = m.AddDirectedEdge("Y", "Z")
	_ = m.AddBidirectedEdge("X", "Y")

	x := map[string]bool{"X": true}
	y := map[string]bool{"Y": true}
	condZ := map[string]bool{"Z": true}
	empty := map[string]bool{}

	if m.MSeparation(x, y, empty) {
		t.Error("X and Y should NOT be m-separated given {} (connected via bidirected)")
	}
	if m.MSeparation(x, y, condZ) {
		t.Error("X and Y should NOT be m-separated given Z (still connected via bidirected)")
	}
}

func TestMSeparationDisconnectedNodes(t *testing.T) {
	// X and Y with no edges between them.
	m := NewMAG()
	_ = m.AddNode("X")
	_ = m.AddNode("Y")

	x := map[string]bool{"X": true}
	y := map[string]bool{"Y": true}
	empty := map[string]bool{}

	if !m.MSeparation(x, y, empty) {
		t.Error("disconnected X and Y should be m-separated")
	}
}

func TestMSeparationComplexGraph(t *testing.T) {
	// A → B → D, A → C → D, B ↔ C
	// Test: A _||_ D | {B, C} ? Yes (all paths blocked).
	// Test: B _||_ C | {} ? No (bidirected edge).
	m := NewMAG()
	for _, n := range []string{"A", "B", "C", "D"} {
		_ = m.AddNode(n)
	}
	_ = m.AddDirectedEdge("A", "B")
	_ = m.AddDirectedEdge("A", "C")
	_ = m.AddDirectedEdge("B", "D")
	_ = m.AddDirectedEdge("C", "D")
	_ = m.AddBidirectedEdge("B", "C")

	a := map[string]bool{"A": true}
	d := map[string]bool{"D": true}
	condBC := map[string]bool{"B": true, "C": true}

	if !m.MSeparation(a, d, condBC) {
		t.Error("A and D should be m-separated given {B, C}")
	}

	b := map[string]bool{"B": true}
	c := map[string]bool{"C": true}
	empty := map[string]bool{}

	if m.MSeparation(b, c, empty) {
		t.Error("B and C should NOT be m-separated given {} (bidirected edge)")
	}
}

func TestFromADMGNil(t *testing.T) {
	_, err := FromADMG(nil)
	if err == nil {
		t.Error("expected error from nil ADMG")
	}
}

func TestFromADMGSimple(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	_ = a.AddNode("Y")
	_ = a.AddDirectedEdge("X", "Y")

	mag, err := FromADMG(a)
	if err != nil {
		t.Fatalf("FromADMG failed: %v", err)
	}
	if !mag.HasNode("X") || !mag.HasNode("Y") {
		t.Error("MAG should have all ADMG nodes")
	}
	if len(mag.DirectedEdges()) != 1 {
		t.Errorf("MAG should preserve directed edge")
	}
}

func TestFromADMGPreservesBidirected(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("A")
	_ = a.AddNode("B")
	_ = a.AddNode("C")
	_ = a.AddDirectedEdge("A", "C")
	_ = a.AddBidirectedEdge("A", "B")

	mag, err := FromADMG(a)
	if err != nil {
		t.Fatalf("FromADMG failed: %v", err)
	}
	biEdges := mag.BidirectedEdges()
	found := false
	for _, e := range biEdges {
		if (e.Src == "A" && e.Dst == "B") || (e.Src == "B" && e.Dst == "A") {
			found = true
		}
	}
	if !found {
		t.Error("MAG should preserve bidirected edge A↔B")
	}
}

func TestFromADMGIsCopy(t *testing.T) {
	a := NewADMG()
	_ = a.AddNode("X")
	_ = a.AddNode("Y")
	_ = a.AddDirectedEdge("X", "Y")

	mag, _ := FromADMG(a)

	// Modifying the MAG should not affect the original ADMG.
	_ = mag.AddNode("Z")
	if a.HasNode("Z") {
		t.Error("modifying MAG should not affect original ADMG")
	}
}
