//go:build unit

package graphgo

import (
	"sort"
	"testing"
)

func TestNewPDAG(t *testing.T) {
	p := NewPDAG()
	if len(p.Nodes()) != 0 {
		t.Fatal("expected 0 nodes")
	}
	if len(p.DirectedEdges()) != 0 {
		t.Fatal("expected 0 directed edges")
	}
	if len(p.UndirectedEdges()) != 0 {
		t.Fatal("expected 0 undirected edges")
	}
}

func TestPDAGAddNode(t *testing.T) {
	p := NewPDAG()
	p.AddNode("A")
	if !p.HasNode("A") {
		t.Fatal("expected node A")
	}
	if p.HasNode("B") {
		t.Fatal("did not expect node B")
	}
	// Adding again should be idempotent.
	p.AddNode("A")
	if len(p.Nodes()) != 1 {
		t.Fatalf("expected 1 node, got %d", len(p.Nodes()))
	}
}

func TestPDAGAddNodes(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("A", "B", "C")
	nodes := p.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}
	expected := []string{"A", "B", "C"}
	for i, n := range expected {
		if nodes[i] != n {
			t.Fatalf("expected node %s at index %d, got %s", n, i, nodes[i])
		}
	}
}

func TestPDAGRemoveNode(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("A", "B", "C")
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("B", "C")
	p.AddDirectedEdge("C", "A")

	p.RemoveNode("B")
	if p.HasNode("B") {
		t.Fatal("B should be removed")
	}
	if p.HasDirectedEdge("A", "B") {
		t.Fatal("directed edge A->B should be removed")
	}
	if p.HasUndirectedEdge("B", "C") {
		t.Fatal("undirected edge B-C should be removed")
	}
	// A and C should still exist.
	if !p.HasNode("A") || !p.HasNode("C") {
		t.Fatal("A and C should still exist")
	}
	if !p.HasDirectedEdge("C", "A") {
		t.Fatal("C->A should still exist")
	}
}

func TestPDAGRemoveNodeNonexistent(t *testing.T) {
	p := NewPDAG()
	p.AddNode("A")
	p.RemoveNode("Z") // should not panic
	if !p.HasNode("A") {
		t.Fatal("A should still exist")
	}
}

func TestPDAGDirectedEdges(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddDirectedEdge("B", "C")

	if !p.HasDirectedEdge("A", "B") {
		t.Fatal("expected A->B")
	}
	if p.HasDirectedEdge("B", "A") {
		t.Fatal("did not expect B->A")
	}
	if !p.HasEdge("A", "B") {
		t.Fatal("HasEdge should find directed edge")
	}

	edges := p.DirectedEdges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 directed edges, got %d", len(edges))
	}
}

func TestPDAGUndirectedEdges(t *testing.T) {
	p := NewPDAG()
	p.AddUndirectedEdge("A", "B")
	p.AddUndirectedEdge("B", "C")

	if !p.HasUndirectedEdge("A", "B") {
		t.Fatal("expected A-B")
	}
	if !p.HasUndirectedEdge("B", "A") {
		t.Fatal("expected B-A (symmetric)")
	}
	if !p.HasEdge("A", "B") {
		t.Fatal("HasEdge should find undirected edge")
	}

	edges := p.UndirectedEdges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 undirected edges, got %d", len(edges))
	}
}

func TestPDAGRemoveDirectedEdge(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.RemoveDirectedEdge("A", "B")
	if p.HasDirectedEdge("A", "B") {
		t.Fatal("edge should be removed")
	}
}

func TestPDAGRemoveUndirectedEdge(t *testing.T) {
	p := NewPDAG()
	p.AddUndirectedEdge("A", "B")
	p.RemoveUndirectedEdge("A", "B")
	if p.HasUndirectedEdge("A", "B") {
		t.Fatal("edge should be removed")
	}
	if p.HasUndirectedEdge("B", "A") {
		t.Fatal("reverse edge should be removed")
	}
}

func TestPDAGHasEdgeMixed(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("C", "D")

	if !p.HasEdge("A", "B") {
		t.Fatal("expected edge A-B via directed")
	}
	if !p.HasEdge("B", "A") {
		t.Fatal("expected edge B-A via reverse directed")
	}
	if !p.HasEdge("C", "D") {
		t.Fatal("expected edge C-D via undirected")
	}
	if p.HasEdge("A", "C") {
		t.Fatal("no edge between A and C")
	}
}

func TestPDAGNodesSorted(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("C", "A", "B")
	nodes := p.Nodes()
	expected := []string{"A", "B", "C"}
	for i, n := range expected {
		if nodes[i] != n {
			t.Fatalf("expected %s at index %d, got %s", n, i, nodes[i])
		}
	}
}

func TestPDAGNeighbors(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")   // A→B
	p.AddDirectedEdge("C", "B")   // C→B
	p.AddUndirectedEdge("B", "D") // B—D

	neighbors := p.Neighbors("B")
	expected := []string{"A", "C", "D"}
	if len(neighbors) != len(expected) {
		t.Fatalf("expected %d neighbors, got %d: %v", len(expected), len(neighbors), neighbors)
	}
	for i, n := range expected {
		if neighbors[i] != n {
			t.Fatalf("expected %s at index %d, got %s", n, i, neighbors[i])
		}
	}
}

func TestPDAGParents(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "C")
	p.AddDirectedEdge("B", "C")
	p.AddUndirectedEdge("C", "D")

	parents := p.Parents("C")
	expected := []string{"A", "B"}
	if len(parents) != len(expected) {
		t.Fatalf("expected %d parents, got %d", len(expected), len(parents))
	}
	for i, n := range expected {
		if parents[i] != n {
			t.Fatalf("expected %s, got %s", n, parents[i])
		}
	}
}

func TestPDAGChildren(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddDirectedEdge("A", "C")
	p.AddUndirectedEdge("A", "D")

	children := p.Children("A")
	expected := []string{"B", "C"}
	if len(children) != len(expected) {
		t.Fatalf("expected %d children, got %d", len(expected), len(children))
	}
	for i, n := range expected {
		if children[i] != n {
			t.Fatalf("expected %s, got %s", n, children[i])
		}
	}
}

func TestPDAGSkeleton(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("B", "C")
	p.AddDirectedEdge("C", "A")

	skel := p.Skeleton()
	if !skel.HasEdge("A", "B") {
		t.Fatal("skeleton should have A-B")
	}
	if !skel.HasEdge("B", "C") {
		t.Fatal("skeleton should have B-C")
	}
	if !skel.HasEdge("C", "A") {
		t.Fatal("skeleton should have C-A")
	}
	if !skel.HasEdge("B", "A") {
		t.Fatal("skeleton edges should be undirected")
	}
}

func TestPDAGCopy(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("C", "D")

	c := p.Copy()

	// Verify copy has same structure.
	if !c.HasDirectedEdge("A", "B") {
		t.Fatal("copy should have A->B")
	}
	if !c.HasUndirectedEdge("C", "D") {
		t.Fatal("copy should have C-D")
	}

	// Modify original, verify copy is independent.
	p.RemoveDirectedEdge("A", "B")
	if !c.HasDirectedEdge("A", "B") {
		t.Fatal("copy should be independent")
	}
}

func TestPDAGDirectedEdgesAutoCreateNodes(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("X", "Y")
	if !p.HasNode("X") || !p.HasNode("Y") {
		t.Fatal("nodes should be auto-created")
	}
}

func TestPDAGUndirectedEdgesAutoCreateNodes(t *testing.T) {
	p := NewPDAG()
	p.AddUndirectedEdge("X", "Y")
	if !p.HasNode("X") || !p.HasNode("Y") {
		t.Fatal("nodes should be auto-created")
	}
}

func TestPDAGAdjacentMethod(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("A", "B")
	p.AddUndirectedEdge("C", "D")

	if !p.Adjacent("A", "B") {
		t.Fatal("A and B should be adjacent")
	}
	if !p.Adjacent("B", "A") {
		t.Fatal("B and A should be adjacent (reverse directed)")
	}
	if !p.Adjacent("C", "D") {
		t.Fatal("C and D should be adjacent")
	}
	if p.Adjacent("A", "C") {
		t.Fatal("A and C should not be adjacent")
	}
}

func TestPDAGUndirectedEdgesSorted(t *testing.T) {
	p := NewPDAG()
	p.AddUndirectedEdge("C", "A")
	p.AddUndirectedEdge("B", "A")

	edges := p.UndirectedEdges()
	// Expect canonical ordering: each edge with u < v, sorted.
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}
	if edges[0] != [2]string{"A", "B"} {
		t.Fatalf("expected [A,B], got %v", edges[0])
	}
	if edges[1] != [2]string{"A", "C"} {
		t.Fatalf("expected [A,C], got %v", edges[1])
	}
}

func TestPDAGDirectedEdgesSorted(t *testing.T) {
	p := NewPDAG()
	p.AddDirectedEdge("C", "A")
	p.AddDirectedEdge("B", "A")

	edges := p.DirectedEdges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}
	if edges[0] != [2]string{"B", "A"} {
		t.Fatalf("expected [B,A], got %v", edges[0])
	}
	if edges[1] != [2]string{"C", "A"} {
		t.Fatalf("expected [C,A], got %v", edges[1])
	}
}

func TestPDAGEmptyNeighbors(t *testing.T) {
	p := NewPDAG()
	p.AddNode("A")
	neighbors := p.Neighbors("A")
	if len(neighbors) != 0 {
		t.Fatalf("expected 0 neighbors, got %d", len(neighbors))
	}
}

func TestPDAGEmptyParentsChildren(t *testing.T) {
	p := NewPDAG()
	p.AddNode("A")
	if len(p.Parents("A")) != 0 {
		t.Fatal("expected 0 parents")
	}
	if len(p.Children("A")) != 0 {
		t.Fatal("expected 0 children")
	}
}

func TestPDAGSkeletonIsolatedNode(t *testing.T) {
	p := NewPDAG()
	p.AddNodes("A", "B")
	skel := p.Skeleton()
	if !skel.HasNode("A") || !skel.HasNode("B") {
		t.Fatal("skeleton should contain isolated nodes")
	}
	if skel.HasEdge("A", "B") {
		t.Fatal("skeleton should not have spurious edges")
	}
}

func TestPDAGRemoveEdgesIdempotent(t *testing.T) {
	p := NewPDAG()
	p.AddNode("A")
	p.AddNode("B")
	// Removing non-existent edges should not panic.
	p.RemoveDirectedEdge("A", "B")
	p.RemoveUndirectedEdge("A", "B")
}

func sortedStrings(s []string) []string {
	c := make([]string, len(s))
	copy(c, s)
	sort.Strings(c)
	return c
}
