//go:build unit

package graphgo

import (
	"sort"
	"testing"
)

func TestDiGraphReverse(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.nodeAttrs["A"]["color"] = "red"
	g.edgeAttrs[edgeKey("A", "B")]["weight"] = 42

	r := g.Reverse()
	if !r.HasEdge("B", "A") {
		t.Error("expected B->A")
	}
	if !r.HasEdge("C", "B") {
		t.Error("expected C->B")
	}
	if r.HasEdge("A", "B") {
		t.Error("should not have A->B")
	}
	if r.nodeAttrs["A"]["color"] != "red" {
		t.Error("node attrs not copied")
	}
	if r.edgeAttrs[edgeKey("B", "A")]["weight"] != 42 {
		t.Error("edge attrs not copied")
	}
}

func TestDiGraphReverse_Empty(t *testing.T) {
	g := NewDiGraph()
	r := g.Reverse()
	if r.NumberOfNodes() != 0 {
		t.Error("expected empty reversed graph")
	}
}

func TestContractNodes(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("D", "B")

	result := ContractNodes(g, []string{"A", "B"})
	// A survives, B is merged into A.
	if !result.HasNode("A") {
		t.Error("expected A")
	}
	if !result.HasEdge("A", "C") {
		t.Error("expected A->C (redirected from B->C)")
	}
	if !result.HasEdge("D", "A") {
		t.Error("expected D->A (redirected from D->B)")
	}
	// A->A self-loop should not exist (A->B becomes A->A).
	if result.HasEdge("A", "A") {
		t.Error("should not have self-loop")
	}
}

func TestContractNodes_Empty(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	result := ContractNodes(g, nil)
	if result.NumberOfEdges() != 1 {
		t.Error("expected copy with 1 edge")
	}
}

func TestContractNodesUndirected(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("D", "B")

	result := ContractNodesUndirected(g, []string{"A", "B"})
	if !result.HasEdge("A", "C") {
		t.Error("expected A-C")
	}
	if !result.HasEdge("D", "A") {
		t.Error("expected D-A")
	}
	if result.HasEdge("A", "A") {
		t.Error("should not have self-loop")
	}
}

func TestContractNodesUndirected_Empty(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	result := ContractNodesUndirected(g, nil)
	if len(result.Edges()) != 1 {
		t.Error("expected copy with 1 edge")
	}
}

func TestLineGraph(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")

	lg := LineGraph(g)
	if len(lg.Nodes()) != 3 {
		t.Errorf("expected 3 nodes in line graph, got %d", len(lg.Nodes()))
	}
	// All edges share endpoints, so all nodes should be connected.
	if len(lg.Edges()) != 3 {
		t.Errorf("expected 3 edges in line graph, got %d", len(lg.Edges()))
	}
}

func TestLineGraph_Path(t *testing.T) {
	g := NewGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	lg := LineGraph(g)
	nodes := lg.Nodes()
	sort.Strings(nodes)
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(nodes))
	}
	// A-B and B-C share endpoint B, so they should be connected.
	if len(lg.Edges()) != 1 {
		t.Errorf("expected 1 edge, got %d", len(lg.Edges()))
	}
}

func TestLineGraphDirected(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")

	lg := LineGraphDirected(g)
	if len(lg.Nodes()) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(lg.Nodes()))
	}
	// A->B connects to B->C because dst of first = src of second.
	if !lg.HasEdge("A->B", "B->C") {
		t.Error("expected edge A->B to B->C")
	}
}

func TestLineGraphDirected_Cycle(t *testing.T) {
	g := NewDiGraph()
	g.AddEdge("A", "B")
	g.AddEdge("B", "C")
	g.AddEdge("C", "A")

	lg := LineGraphDirected(g)
	if len(lg.Nodes()) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(lg.Nodes()))
	}
	if lg.NumberOfEdges() != 3 {
		t.Errorf("expected 3 edges, got %d", lg.NumberOfEdges())
	}
}
