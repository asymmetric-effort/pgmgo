//go:build unit

package graphgo

import (
	"testing"
)

func TestShortestPath_Basic(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	path, err := ShortestPath(g, "A", "C")
	if err != nil {
		t.Fatalf("ShortestPath error: %v", err)
	}
	// Direct edge A->C should be shortest.
	if len(path) != 2 || path[0] != "A" || path[1] != "C" {
		t.Errorf("expected [A C], got %v", path)
	}
}

func TestShortestPath_SameNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	path, err := ShortestPath(g, "A", "A")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(path) != 1 || path[0] != "A" {
		t.Errorf("expected [A], got %v", path)
	}
}

func TestShortestPath_NoPath(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	g.AddNode("B")
	_, err := ShortestPath(g, "A", "B")
	if err == nil {
		t.Error("expected error for no path")
	}
}

func TestShortestPath_MissingNode(t *testing.T) {
	g := NewDiGraph()
	_, err := ShortestPath(g, "X", "Y")
	if err == nil {
		t.Error("expected error for missing node")
	}
}

func TestAllShortestPaths(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	g.AddEdge("B", "D")
	g.AddEdge("C", "D")

	paths := AllShortestPaths(g, "A", "D")
	if len(paths) != 2 {
		t.Errorf("expected 2 shortest paths, got %d", len(paths))
	}
	for _, p := range paths {
		if len(p) != 3 {
			t.Errorf("expected path length 3, got %d: %v", len(p), p)
		}
	}
}

func TestAllShortestPaths_SameNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	paths := AllShortestPaths(g, "A", "A")
	if len(paths) != 1 || len(paths[0]) != 1 {
		t.Errorf("expected [[A]], got %v", paths)
	}
}

func TestAllShortestPaths_NoPath(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	g.AddNode("B")
	paths := AllShortestPaths(g, "A", "B")
	if paths != nil {
		t.Errorf("expected nil, got %v", paths)
	}
}

func TestAllShortestPaths_MissingNode(t *testing.T) {
	g := NewDiGraph()
	paths := AllShortestPaths(g, "X", "Y")
	if paths != nil {
		t.Errorf("expected nil, got %v", paths)
	}
}

func TestHasPath(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	if !HasPath(g, "A", "C") {
		t.Error("expected path from A to C")
	}
	if HasPath(g, "C", "A") {
		t.Error("expected no path from C to A")
	}
}

func TestHasPath_SameNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	if !HasPath(g, "A", "A") {
		t.Error("expected path to self")
	}
}

func TestHasPath_MissingNode(t *testing.T) {
	g := NewDiGraph()
	if HasPath(g, "X", "Y") {
		t.Error("expected no path for missing nodes")
	}
}

func TestShortestPathUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("A", "C")

	path, err := ShortestPathUndirected(g, "A", "C")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(path) != 2 {
		t.Errorf("expected length 2, got %d: %v", len(path), path)
	}
}

func TestShortestPathUndirected_NoPath(t *testing.T) {
	g := NewGraph()
	g.AddNode("A")
	g.AddNode("B")
	_, err := ShortestPathUndirected(g, "A", "B")
	if err == nil {
		t.Error("expected error")
	}
}

func TestShortestPathUndirected_MissingNode(t *testing.T) {
	g := NewGraph()
	_, err := ShortestPathUndirected(g, "X", "Y")
	if err == nil {
		t.Error("expected error")
	}
}

func TestShortestPathUndirected_SameNode(t *testing.T) {
	g := NewGraph()
	g.AddNode("A")
	path, err := ShortestPathUndirected(g, "A", "A")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if len(path) != 1 {
		t.Errorf("expected [A], got %v", path)
	}
}
