//go:build unit

package graphgo

import (
	"testing"
)

func TestBFS_Basic(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	order := BFS(g, "A")
	if len(order) != 4 {
		t.Errorf("expected 4 nodes, got %d: %v", len(order), order)
	}
	if order[0] != "A" {
		t.Errorf("expected first node A, got %q", order[0])
	}
}

func TestBFS_MissingNode(t *testing.T) {
	g := NewDiGraph()
	order := BFS(g, "X")
	if order != nil {
		t.Errorf("expected nil, got %v", order)
	}
}

func TestBFS_SingleNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	order := BFS(g, "A")
	if len(order) != 1 || order[0] != "A" {
		t.Errorf("expected [A], got %v", order)
	}
}

func TestDFS_Basic(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "D")

	order := DFS(g, "A")
	if len(order) != 4 {
		t.Errorf("expected 4 nodes, got %d: %v", len(order), order)
	}
	if order[0] != "A" {
		t.Errorf("expected first node A, got %q", order[0])
	}
	// With sorted successors, B comes before C, so D comes after B.
	if order[1] != "B" || order[2] != "D" || order[3] != "C" {
		t.Errorf("unexpected DFS order: %v", order)
	}
}

func TestDFS_MissingNode(t *testing.T) {
	g := NewDiGraph()
	order := DFS(g, "X")
	if order != nil {
		t.Errorf("expected nil, got %v", order)
	}
}

func TestBFSUndirected_Basic(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	order := BFSUndirected(g, "A")
	if len(order) != 3 {
		t.Errorf("expected 3, got %d", len(order))
	}
	if order[0] != "A" {
		t.Errorf("expected first node A, got %q", order[0])
	}
}

func TestBFSUndirected_MissingNode(t *testing.T) {
	g := NewGraph()
	order := BFSUndirected(g, "X")
	if order != nil {
		t.Errorf("expected nil, got %v", order)
	}
}

func TestDFSUndirected_Basic(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	order := DFSUndirected(g, "A")
	if len(order) != 3 {
		t.Errorf("expected 3, got %d", len(order))
	}
	if order[0] != "A" {
		t.Errorf("expected A first, got %q", order[0])
	}
}

func TestDFSUndirected_MissingNode(t *testing.T) {
	g := NewGraph()
	order := DFSUndirected(g, "X")
	if order != nil {
		t.Errorf("expected nil, got %v", order)
	}
}
