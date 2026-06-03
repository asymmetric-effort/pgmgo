//go:build unit

package graphgo

import (
	"math"
	"sort"
	"testing"
)

func TestNumberOfSelfLoops(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "A")
	g.AddEdge("A", "B")
	g.AddEdge("B", "B")
	if NumberOfSelfLoops(g) != 2 {
		t.Errorf("expected 2 self-loops, got %d", NumberOfSelfLoops(g))
	}
}

func TestNumberOfSelfLoops_None(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	if NumberOfSelfLoops(g) != 0 {
		t.Errorf("expected 0 self-loops, got %d", NumberOfSelfLoops(g))
	}
}

func TestSelfLoops(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "A")
	g.AddEdge("A", "B")
	g.AddEdge("B", "B")
	loops := SelfLoops(g)
	sort.Strings(loops)
	if len(loops) != 2 || loops[0] != "A" || loops[1] != "B" {
		t.Errorf("expected [A B], got %v", loops)
	}
}

func TestNumberOfSelfLoopsUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "A")
	g.AddEdge("A", "B")
	if NumberOfSelfLoopsUndirected(g) != 1 {
		t.Errorf("expected 1 self-loop, got %d", NumberOfSelfLoopsUndirected(g))
	}
}

func TestDensity(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	// 3 edges, 3 nodes: density = 3 / (3*2) = 0.5
	d := Density(g)
	if math.Abs(d-0.5) > 1e-9 {
		t.Errorf("expected density 0.5, got %f", d)
	}
}

func TestDensity_SingleNode(t *testing.T) {
	g := NewDiGraph()
	g.AddNode("A")
	if Density(g) != 0 {
		t.Errorf("expected density 0 for single node")
	}
}

func TestDensityUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	// 3 edges, 3 nodes: density = 2*3 / (3*2) = 1.0
	d := DensityUndirected(g)
	if math.Abs(d-1.0) > 1e-9 {
		t.Errorf("expected density 1.0, got %f", d)
	}
}

func TestDensityUndirected_SingleNode(t *testing.T) {
	g := NewGraph()
	g.AddNode("A")
	if DensityUndirected(g) != 0 {
		t.Error("expected 0")
	}
}

func TestIsTree(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("A", "C")
	if !IsTree(g) {
		t.Error("expected tree")
	}
}

func TestIsTree_Cycle(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "A")
	if IsTree(g) {
		t.Error("cycle should not be a tree")
	}
}

func TestIsTree_Disconnected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddNode("C")
	if IsTree(g) {
		t.Error("disconnected should not be a tree")
	}
}

func TestIsTree_Empty(t *testing.T) {
	g := NewDiGraph()
	if IsTree(g) {
		t.Error("empty graph is not a tree")
	}
}

func TestIsForest(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("C", "D")
	if !IsForest(g) {
		t.Error("expected forest")
	}
}

func TestIsForest_Cycle(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "A")
	if IsForest(g) {
		t.Error("cycle should not be a forest")
	}
}

func TestIsForest_Empty(t *testing.T) {
	g := NewDiGraph()
	if !IsForest(g) {
		t.Error("empty graph is a forest")
	}
}

func TestIsTreeUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	if !IsTreeUndirected(g) {
		t.Error("expected tree")
	}
}

func TestIsTreeUndirected_Cycle(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	if IsTreeUndirected(g) {
		t.Error("cycle is not a tree")
	}
}

func TestIsTreeUndirected_Empty(t *testing.T) {
	g := NewGraph()
	if IsTreeUndirected(g) {
		t.Error("empty graph is not a tree")
	}
}

func TestIsForestUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("C", "D")
	if !IsForestUndirected(g) {
		t.Error("expected forest")
	}
}

func TestIsForestUndirected_Cycle(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")
	if IsForestUndirected(g) {
		t.Error("cycle is not a forest")
	}
}

func TestIsForestUndirected_Empty(t *testing.T) {
	g := NewGraph()
	if !IsForestUndirected(g) {
		t.Error("empty graph is a forest")
	}
}
