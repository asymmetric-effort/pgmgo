//go:build unit

package base

import (
	"sort"
	"testing"
)

// ---------------------------------------------------------------------------
// NewPDAG
// ---------------------------------------------------------------------------

func TestNewPDAG(t *testing.T) {
	pd := NewPDAG()
	if pd == nil {
		t.Fatal("NewPDAG returned nil")
	}
	if len(pd.Nodes()) != 0 {
		t.Errorf("new PDAG should have 0 nodes, got %d", len(pd.Nodes()))
	}
	if len(pd.DirectedEdges()) != 0 {
		t.Errorf("new PDAG should have 0 directed edges, got %d", len(pd.DirectedEdges()))
	}
	if len(pd.UndirectedEdges()) != 0 {
		t.Errorf("new PDAG should have 0 undirected edges, got %d", len(pd.UndirectedEdges()))
	}
}

// ---------------------------------------------------------------------------
// AddNode
// ---------------------------------------------------------------------------

func TestPDAGAddNode(t *testing.T) {
	pd := NewPDAG()
	if err := pd.AddNode("A"); err != nil {
		t.Fatalf("AddNode failed: %v", err)
	}
	if !pd.HasNode("A") {
		t.Error("HasNode returned false for added node")
	}
}

func TestPDAGAddNodeDuplicate(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	err := pd.AddNode("A")
	if err == nil {
		t.Error("expected error when adding duplicate node")
	}
}

// ---------------------------------------------------------------------------
// AddDirectedEdge / AddUndirectedEdge
// ---------------------------------------------------------------------------

func TestPDAGAddDirectedEdge(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	if err := pd.AddDirectedEdge("A", "B"); err != nil {
		t.Fatalf("AddDirectedEdge failed: %v", err)
	}
	if !pd.HasDirectedEdge("A", "B") {
		t.Error("HasDirectedEdge returned false for added edge")
	}
	if pd.HasDirectedEdge("B", "A") {
		t.Error("reverse directed edge should not exist")
	}
}

func TestPDAGAddDirectedEdgeMissingNode(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	if err := pd.AddDirectedEdge("A", "B"); err == nil {
		t.Error("expected error for missing target node")
	}
	if err := pd.AddDirectedEdge("X", "A"); err == nil {
		t.Error("expected error for missing source node")
	}
}

func TestPDAGAddDirectedEdgeDuplicate(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddDirectedEdge("A", "B")
	if err := pd.AddDirectedEdge("A", "B"); err == nil {
		t.Error("expected error for duplicate directed edge")
	}
}

func TestPDAGAddUndirectedEdge(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	if err := pd.AddUndirectedEdge("A", "B"); err != nil {
		t.Fatalf("AddUndirectedEdge failed: %v", err)
	}
	if !pd.HasUndirectedEdge("A", "B") {
		t.Error("HasUndirectedEdge returned false")
	}
	if !pd.HasUndirectedEdge("B", "A") {
		t.Error("undirected edge should be symmetric")
	}
}

func TestPDAGAddUndirectedEdgeMissingNode(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	if err := pd.AddUndirectedEdge("A", "B"); err == nil {
		t.Error("expected error for missing node")
	}
}

func TestPDAGAddUndirectedEdgeDuplicate(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddUndirectedEdge("A", "B")
	if err := pd.AddUndirectedEdge("A", "B"); err == nil {
		t.Error("expected error for duplicate undirected edge")
	}
}

// ---------------------------------------------------------------------------
// RemoveDirectedEdge / RemoveUndirectedEdge
// ---------------------------------------------------------------------------

func TestPDAGRemoveDirectedEdge(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddDirectedEdge("A", "B")
	if err := pd.RemoveDirectedEdge("A", "B"); err != nil {
		t.Fatalf("RemoveDirectedEdge failed: %v", err)
	}
	if pd.HasDirectedEdge("A", "B") {
		t.Error("directed edge should be removed")
	}
}

func TestPDAGRemoveDirectedEdgeNotFound(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	if err := pd.RemoveDirectedEdge("A", "B"); err == nil {
		t.Error("expected error for removing non-existent directed edge")
	}
}

func TestPDAGRemoveUndirectedEdge(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddUndirectedEdge("A", "B")
	if err := pd.RemoveUndirectedEdge("A", "B"); err != nil {
		t.Fatalf("RemoveUndirectedEdge failed: %v", err)
	}
	if pd.HasUndirectedEdge("A", "B") {
		t.Error("undirected edge should be removed")
	}
}

func TestPDAGRemoveUndirectedEdgeNotFound(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	if err := pd.RemoveUndirectedEdge("A", "B"); err == nil {
		t.Error("expected error for removing non-existent undirected edge")
	}
}

// ---------------------------------------------------------------------------
// HasEdge (any type)
// ---------------------------------------------------------------------------

func TestPDAGHasEdge(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	if !pd.HasEdge("A", "B") {
		t.Error("HasEdge should return true for directed A->B")
	}
	if !pd.HasEdge("B", "C") {
		t.Error("HasEdge should return true for undirected B-C")
	}
	if !pd.HasEdge("C", "B") {
		t.Error("HasEdge should return true for undirected C-B")
	}
	if pd.HasEdge("A", "C") {
		t.Error("HasEdge should return false for non-existent edge")
	}
}

// ---------------------------------------------------------------------------
// Nodes / DirectedEdges / UndirectedEdges
// ---------------------------------------------------------------------------

func TestPDAGNodes(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("C")
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	nodes := pd.Nodes()
	if len(nodes) != 3 || nodes[0] != "A" || nodes[1] != "B" || nodes[2] != "C" {
		t.Errorf("Nodes() should be sorted, got %v", nodes)
	}
}

func TestPDAGDirectedEdges(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("B", "C")
	_ = pd.AddDirectedEdge("A", "B")

	edges := pd.DirectedEdges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 directed edges, got %d", len(edges))
	}
	if edges[0] != [2]string{"A", "B"} {
		t.Errorf("edges[0] = %v, want [A B]", edges[0])
	}
	if edges[1] != [2]string{"B", "C"} {
		t.Errorf("edges[1] = %v, want [B C]", edges[1])
	}
}

func TestPDAGUndirectedEdges(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddUndirectedEdge("C", "A")
	_ = pd.AddUndirectedEdge("A", "B")

	edges := pd.UndirectedEdges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 undirected edges, got %d", len(edges))
	}
	// Should be sorted and canonicalized (u < v).
	if edges[0] != [2]string{"A", "B"} {
		t.Errorf("edges[0] = %v, want [A B]", edges[0])
	}
	if edges[1] != [2]string{"A", "C"} {
		t.Errorf("edges[1] = %v, want [A C]", edges[1])
	}
}

// ---------------------------------------------------------------------------
// Parents / Children
// ---------------------------------------------------------------------------

func TestPDAGParentsAndChildren(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddNode("D")
	_ = pd.AddDirectedEdge("A", "C")
	_ = pd.AddDirectedEdge("B", "C")
	_ = pd.AddDirectedEdge("C", "D")

	parents := pd.Parents("C")
	if len(parents) != 2 || parents[0] != "A" || parents[1] != "B" {
		t.Errorf("Parents(C) = %v, want [A B]", parents)
	}

	children := pd.Children("C")
	if len(children) != 1 || children[0] != "D" {
		t.Errorf("Children(C) = %v, want [D]", children)
	}

	if len(pd.Parents("A")) != 0 {
		t.Error("Parents(A) should be empty")
	}
	if len(pd.Children("D")) != 0 {
		t.Error("Children(D) should be empty")
	}
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

func TestPDAGSkeleton(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	skel := pd.Skeleton()
	if skel == nil {
		t.Fatal("Skeleton returned nil")
	}
	if !skel.HasNode("A") || !skel.HasNode("B") || !skel.HasNode("C") {
		t.Error("skeleton should have all nodes")
	}
	if !skel.HasEdge("A", "B") || !skel.HasEdge("B", "A") {
		t.Error("skeleton should have undirected A-B from directed A->B")
	}
	if !skel.HasEdge("B", "C") || !skel.HasEdge("C", "B") {
		t.Error("skeleton should have undirected B-C")
	}
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

func TestPDAGCopy(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	cp := pd.Copy()

	// Copy should have the same structure.
	if !cp.HasDirectedEdge("A", "B") {
		t.Error("copy should have directed edge A->B")
	}
	if !cp.HasUndirectedEdge("B", "C") {
		t.Error("copy should have undirected edge B-C")
	}

	// Modifying the copy should not affect the original.
	_ = cp.AddNode("D")
	_ = cp.AddDirectedEdge("C", "D")
	if pd.HasNode("D") {
		t.Error("original should not have node D after modifying copy")
	}
}

// ---------------------------------------------------------------------------
// ApplyMeekRules
// ---------------------------------------------------------------------------

func TestPDAGApplyMeekRules(t *testing.T) {
	// R1: w->u, u-v, w not adj v => orient u->v.
	pd := NewPDAG()
	_ = pd.AddNode("W")
	_ = pd.AddNode("U")
	_ = pd.AddNode("V")
	_ = pd.AddDirectedEdge("W", "U")
	_ = pd.AddUndirectedEdge("U", "V")

	changed := pd.ApplyMeekRules()
	if !changed {
		t.Error("expected Meek rules to make changes")
	}
	if !pd.HasDirectedEdge("U", "V") {
		t.Error("expected U->V after Meek R1")
	}
	if pd.HasUndirectedEdge("U", "V") {
		t.Error("U-V should no longer be undirected")
	}
}

// ---------------------------------------------------------------------------
// FromDAG — produces correct CPDAG
// ---------------------------------------------------------------------------

func TestFromDAGSimpleChain(t *testing.T) {
	// Chain A->B->C: no v-structure, all edges reversible => all undirected.
	dag := NewDAG()
	_ = dag.AddNodes("A", "B", "C")
	_ = dag.AddEdge("A", "B")
	_ = dag.AddEdge("B", "C")

	pd := FromDAG(dag)
	if len(pd.DirectedEdges()) != 0 {
		t.Errorf("chain should have no directed edges in CPDAG, got %v", pd.DirectedEdges())
	}
	if len(pd.UndirectedEdges()) != 2 {
		t.Errorf("chain should have 2 undirected edges, got %v", pd.UndirectedEdges())
	}
}

func TestFromDAGVStructure(t *testing.T) {
	// V-structure: A->C<-B (A and B not adjacent).
	// Both edges are compelled.
	dag := NewDAG()
	_ = dag.AddNodes("A", "B", "C")
	_ = dag.AddEdge("A", "C")
	_ = dag.AddEdge("B", "C")

	pd := FromDAG(dag)
	if !pd.HasDirectedEdge("A", "C") {
		t.Error("expected directed edge A->C in CPDAG (v-structure)")
	}
	if !pd.HasDirectedEdge("B", "C") {
		t.Error("expected directed edge B->C in CPDAG (v-structure)")
	}
	if len(pd.UndirectedEdges()) != 0 {
		t.Errorf("v-structure should have no undirected edges, got %v", pd.UndirectedEdges())
	}
}

func TestFromDAGDiamond(t *testing.T) {
	// Diamond: A->B, A->C, B->D, C->D.
	// B and C are non-adjacent parents of D => v-structure at D.
	// Also A->B and A->C are compelled by Meek rules (R1: A->B, B->D, A not adj D => no,
	// but v-structure at D compels B->D and C->D; then R1 on A-B: need w->A w not adj B — no.)
	dag := NewDAG()
	_ = dag.AddNodes("A", "B", "C", "D")
	_ = dag.AddEdge("A", "B")
	_ = dag.AddEdge("A", "C")
	_ = dag.AddEdge("B", "D")
	_ = dag.AddEdge("C", "D")

	pd := FromDAG(dag)

	// V-structure at D: B->D, C->D must be directed.
	if !pd.HasDirectedEdge("B", "D") {
		t.Error("expected B->D directed (v-structure)")
	}
	if !pd.HasDirectedEdge("C", "D") {
		t.Error("expected C->D directed (v-structure)")
	}

	// Check total edges are preserved in the skeleton.
	skel := pd.Skeleton()
	nodes := pd.Nodes()
	if len(nodes) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(nodes))
	}
	totalEdges := len(pd.DirectedEdges()) + len(pd.UndirectedEdges())
	if totalEdges != 4 {
		t.Errorf("expected 4 total edges, got %d", totalEdges)
	}
	_ = skel
}

func TestFromDAGPreservesNodes(t *testing.T) {
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	// No edges — the CPDAG should have the same nodes.
	pd := FromDAG(dag)
	nodes := pd.Nodes()
	if len(nodes) != 2 || nodes[0] != "X" || nodes[1] != "Y" {
		t.Errorf("expected [X Y], got %v", nodes)
	}
}

// ---------------------------------------------------------------------------
// ToDAG — produces a valid DAG
// ---------------------------------------------------------------------------

func TestToDAGFullyDirected(t *testing.T) {
	// A PDAG with only directed edges should convert directly.
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("A", "B")
	_ = pd.AddDirectedEdge("B", "C")

	dag, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}
	if !dag.HasEdge("A", "B") || !dag.HasEdge("B", "C") {
		t.Error("ToDAG should preserve directed edges")
	}
	// Verify valid topological order.
	_, err = dag.TopologicalOrder()
	if err != nil {
		t.Fatalf("result should be a valid DAG: %v", err)
	}
}

func TestToDAGFullyUndirected(t *testing.T) {
	// A-B-C (all undirected) should produce some valid DAG.
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddUndirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	dag, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	// Should have 3 nodes and 2 edges.
	if len(dag.Nodes()) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(dag.Nodes()))
	}
	if len(dag.Edges()) != 2 {
		t.Errorf("expected 2 edges, got %d", len(dag.Edges()))
	}

	// Should be a valid DAG.
	_, err = dag.TopologicalOrder()
	if err != nil {
		t.Fatalf("result should be a valid DAG: %v", err)
	}
}

func TestToDAGMixed(t *testing.T) {
	// A->B, B-C (directed + undirected).
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddDirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	dag, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	if !dag.HasEdge("A", "B") {
		t.Error("directed edge A->B should be preserved")
	}

	// B-C should be oriented one way.
	bc := dag.HasEdge("B", "C")
	cb := dag.HasEdge("C", "B")
	if !bc && !cb {
		t.Error("B-C should be oriented in some direction")
	}
	if bc && cb {
		t.Error("B-C should not be oriented in both directions")
	}

	_, err = dag.TopologicalOrder()
	if err != nil {
		t.Fatalf("result should be a valid DAG: %v", err)
	}
}

func TestToDAGEmptyPDAG(t *testing.T) {
	pd := NewPDAG()
	dag, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG on empty PDAG failed: %v", err)
	}
	if len(dag.Nodes()) != 0 {
		t.Errorf("expected 0 nodes, got %d", len(dag.Nodes()))
	}
}

func TestToDAGNodesOnly(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	dag, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}
	if len(dag.Nodes()) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(dag.Nodes()))
	}
	if len(dag.Edges()) != 0 {
		t.Errorf("expected 0 edges, got %d", len(dag.Edges()))
	}
}

// ---------------------------------------------------------------------------
// Round-trip: DAG -> PDAG -> DAG preserves equivalence class
// ---------------------------------------------------------------------------

func TestRoundTripDAGToPDAGToDAG(t *testing.T) {
	// Build a DAG: A->B->C (simple chain, no v-structure).
	dag1 := NewDAG()
	_ = dag1.AddNodes("A", "B", "C")
	_ = dag1.AddEdge("A", "B")
	_ = dag1.AddEdge("B", "C")

	// Convert to CPDAG.
	cpdag := FromDAG(dag1)

	// Convert back to a DAG.
	dag2, err := cpdag.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	// dag2 should be a valid DAG.
	_, err = dag2.TopologicalOrder()
	if err != nil {
		t.Fatalf("round-tripped DAG should be valid: %v", err)
	}

	// dag2 should have the same number of nodes and edges.
	if len(dag2.Nodes()) != len(dag1.Nodes()) {
		t.Errorf("node count mismatch: original %d, round-tripped %d",
			len(dag1.Nodes()), len(dag2.Nodes()))
	}
	if len(dag2.Edges()) != len(dag1.Edges()) {
		t.Errorf("edge count mismatch: original %d, round-tripped %d",
			len(dag1.Edges()), len(dag2.Edges()))
	}

	// dag2's CPDAG should equal the original CPDAG (same equivalence class).
	cpdag2 := FromDAG(dag2)
	assertSameCPDAG(t, cpdag, cpdag2)
}

func TestRoundTripVStructure(t *testing.T) {
	// V-structure: A->C<-B. The CPDAG has both edges directed.
	dag1 := NewDAG()
	_ = dag1.AddNodes("A", "B", "C")
	_ = dag1.AddEdge("A", "C")
	_ = dag1.AddEdge("B", "C")

	cpdag := FromDAG(dag1)

	dag2, err := cpdag.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	_, err = dag2.TopologicalOrder()
	if err != nil {
		t.Fatalf("round-tripped DAG should be valid: %v", err)
	}

	// The round-tripped DAG should be in the same equivalence class.
	cpdag2 := FromDAG(dag2)
	assertSameCPDAG(t, cpdag, cpdag2)
}

func TestRoundTripDiamond(t *testing.T) {
	// Diamond: A->B, A->C, B->D, C->D (v-structure at D).
	dag1 := NewDAG()
	_ = dag1.AddNodes("A", "B", "C", "D")
	_ = dag1.AddEdge("A", "B")
	_ = dag1.AddEdge("A", "C")
	_ = dag1.AddEdge("B", "D")
	_ = dag1.AddEdge("C", "D")

	cpdag := FromDAG(dag1)

	dag2, err := cpdag.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	_, err = dag2.TopologicalOrder()
	if err != nil {
		t.Fatalf("round-tripped DAG should be valid: %v", err)
	}

	cpdag2 := FromDAG(dag2)
	assertSameCPDAG(t, cpdag, cpdag2)
}

func TestRoundTripAsiaNetwork(t *testing.T) {
	// Asia network: a well-known Bayesian network with v-structures.
	dag1 := NewDAG()
	_ = dag1.AddNodes("asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp")
	_ = dag1.AddEdge("asia", "tub")
	_ = dag1.AddEdge("smoke", "lung")
	_ = dag1.AddEdge("smoke", "bronc")
	_ = dag1.AddEdge("tub", "either")
	_ = dag1.AddEdge("lung", "either")
	_ = dag1.AddEdge("either", "xray")
	_ = dag1.AddEdge("either", "dysp")
	_ = dag1.AddEdge("bronc", "dysp")

	cpdag := FromDAG(dag1)

	dag2, err := cpdag.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	_, err = dag2.TopologicalOrder()
	if err != nil {
		t.Fatalf("round-tripped Asia DAG should be valid: %v", err)
	}

	if len(dag2.Nodes()) != 8 {
		t.Errorf("expected 8 nodes, got %d", len(dag2.Nodes()))
	}
	if len(dag2.Edges()) != 8 {
		t.Errorf("expected 8 edges, got %d", len(dag2.Edges()))
	}

	cpdag2 := FromDAG(dag2)
	assertSameCPDAG(t, cpdag, cpdag2)
}

func TestRoundTripDisconnectedNodes(t *testing.T) {
	dag1 := NewDAG()
	_ = dag1.AddNodes("A", "B", "C")
	_ = dag1.AddEdge("A", "B")
	// C is disconnected.

	cpdag := FromDAG(dag1)

	dag2, err := cpdag.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	if len(dag2.Nodes()) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(dag2.Nodes()))
	}

	cpdag2 := FromDAG(dag2)
	assertSameCPDAG(t, cpdag, cpdag2)
}

// ---------------------------------------------------------------------------
// ToDAG does not mutate original
// ---------------------------------------------------------------------------

func TestToDAGDoesNotMutateOriginal(t *testing.T) {
	pd := NewPDAG()
	_ = pd.AddNode("A")
	_ = pd.AddNode("B")
	_ = pd.AddNode("C")
	_ = pd.AddUndirectedEdge("A", "B")
	_ = pd.AddUndirectedEdge("B", "C")

	// Snapshot state.
	origUndirected := len(pd.UndirectedEdges())
	origDirected := len(pd.DirectedEdges())

	_, err := pd.ToDAG()
	if err != nil {
		t.Fatalf("ToDAG failed: %v", err)
	}

	// PDAG should be unchanged.
	if len(pd.UndirectedEdges()) != origUndirected {
		t.Errorf("ToDAG mutated undirected edges: was %d, now %d",
			origUndirected, len(pd.UndirectedEdges()))
	}
	if len(pd.DirectedEdges()) != origDirected {
		t.Errorf("ToDAG mutated directed edges: was %d, now %d",
			origDirected, len(pd.DirectedEdges()))
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// assertSameCPDAG checks that two PDAGs have the same directed and undirected edges.
func assertSameCPDAG(t *testing.T, a, b *PDAG) {
	t.Helper()

	nodesA := a.Nodes()
	nodesB := b.Nodes()
	if len(nodesA) != len(nodesB) {
		t.Errorf("node count differs: %d vs %d", len(nodesA), len(nodesB))
		return
	}
	for i := range nodesA {
		if nodesA[i] != nodesB[i] {
			t.Errorf("node mismatch at %d: %q vs %q", i, nodesA[i], nodesB[i])
		}
	}

	dirA := a.DirectedEdges()
	dirB := b.DirectedEdges()
	sort.Slice(dirA, func(i, j int) bool {
		if dirA[i][0] != dirA[j][0] {
			return dirA[i][0] < dirA[j][0]
		}
		return dirA[i][1] < dirA[j][1]
	})
	sort.Slice(dirB, func(i, j int) bool {
		if dirB[i][0] != dirB[j][0] {
			return dirB[i][0] < dirB[j][0]
		}
		return dirB[i][1] < dirB[j][1]
	})
	if len(dirA) != len(dirB) {
		t.Errorf("directed edge count differs: %d vs %d\n  A: %v\n  B: %v",
			len(dirA), len(dirB), dirA, dirB)
	} else {
		for i := range dirA {
			if dirA[i] != dirB[i] {
				t.Errorf("directed edge mismatch at %d: %v vs %v", i, dirA[i], dirB[i])
			}
		}
	}

	undA := a.UndirectedEdges()
	undB := b.UndirectedEdges()
	if len(undA) != len(undB) {
		t.Errorf("undirected edge count differs: %d vs %d\n  A: %v\n  B: %v",
			len(undA), len(undB), undA, undB)
	} else {
		for i := range undA {
			if undA[i] != undB[i] {
				t.Errorf("undirected edge mismatch at %d: %v vs %v", i, undA[i], undB[i])
			}
		}
	}
}
