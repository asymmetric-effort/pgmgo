//go:build unit

package graphgo

import (
	"math"
	"testing"
)

func TestPageRank_Basic(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")

	pr := PageRank(g)
	if len(pr) != 3 {
		t.Fatalf("expected 3 nodes in PageRank, got %d", len(pr))
	}
	// In a cycle, all nodes should have equal rank.
	expected := 1.0 / 3.0
	for _, node := range []string{"A", "B", "C"} {
		if math.Abs(pr[node]-expected) > 0.01 {
			t.Errorf("PageRank(%q) = %f, expected ~%f", node, pr[node], expected)
		}
	}
}

func TestPageRank_Star(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("B", "A")
	g.AddEdge("C", "A")
	g.AddEdge("D", "A")

	pr := PageRank(g)
	// A should have the highest PageRank.
	if pr["A"] <= pr["B"] || pr["A"] <= pr["C"] || pr["A"] <= pr["D"] {
		t.Error("expected A to have highest PageRank")
	}
}

func TestPageRank_Empty(t *testing.T) {
	g := NewDiGraph()
	pr := PageRank(g)
	if len(pr) != 0 {
		t.Errorf("expected empty PageRank, got %d entries", len(pr))
	}
}

func TestPageRank_DanglingNode(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddNode("C") // dangling: no outgoing edges, no incoming either (isolated)

	pr := PageRank(g)
	if len(pr) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(pr))
	}
	// Ranks should sum to 1.
	sum := 0.0
	for _, v := range pr {
		sum += v
	}
	if math.Abs(sum-1.0) > 0.01 {
		t.Errorf("PageRank sum = %f, expected ~1.0", sum)
	}
}

func TestPageRankWithParams(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "A")

	pr := PageRankWithParams(g, 0.85, 50, 1e-8)
	if math.Abs(pr["A"]-pr["B"]) > 0.01 {
		t.Error("expected equal PageRank for symmetric graph")
	}
}
