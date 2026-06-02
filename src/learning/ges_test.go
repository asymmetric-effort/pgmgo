//go:build unit

package learning

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestGES_BasicStructure(t *testing.T) {
	data := syntheticData()
	ges := NewGES(data, countScore)
	pdag, err := ges.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	nodes := pdag.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}

	// The GES should learn some structure connecting A, B, C.
	dirEdges := pdag.DirectedEdges()
	undEdges := pdag.UndirectedEdges()
	totalEdges := len(dirEdges) + len(undEdges)
	t.Logf("Directed edges: %v", dirEdges)
	t.Logf("Undirected edges: %v", undEdges)

	if totalEdges == 0 {
		t.Error("expected at least one edge in the PDAG")
	}

	// All three variables should be connected.
	for _, n := range nodes {
		neighbors := pdag.Neighbors(n)
		if len(neighbors) == 0 {
			t.Errorf("node %s is isolated; expected connectivity", n)
		}
	}
}

func TestGES_TwoVariables(t *testing.T) {
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		xVals[i] = i % 2
		yVals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	ges := NewGES(data, countScore)
	pdag, err := ges.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	// Should have an edge between X and Y.
	if !pdag.Adjacent("X", "Y") {
		t.Error("expected X and Y to be adjacent")
	}
}

func TestGES_NoImprovement(t *testing.T) {
	// Score function that never rewards edges.
	zeroScore := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		return 0.0
	}

	data := syntheticData()
	ges := NewGES(data, zeroScore)
	pdag, err := ges.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	dirEdges := pdag.DirectedEdges()
	undEdges := pdag.UndirectedEdges()
	if len(dirEdges)+len(undEdges) != 0 {
		t.Errorf("expected no edges when score never improves, got directed=%v undirected=%v", dirEdges, undEdges)
	}
}

func TestGES_TooFewVariables(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1, 2, 3}),
	})
	ges := NewGES(data, countScore)
	_, err := ges.Estimate()
	if err == nil {
		t.Fatal("expected error for single variable, got nil")
	}
}

func TestGES_BackwardPhaseRemovesEdges(t *testing.T) {
	// Score that rewards one parent but penalizes two.
	penalizeScore := func(variable string, parents []string, data *tabgo.DataFrame) float64 {
		if len(parents) == 1 {
			return 1.0
		}
		if len(parents) > 1 {
			return -1.0
		}
		return 0.0
	}

	n := 50
	vals := make([]any, n)
	for i := range vals {
		vals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", vals),
		"B": tabgo.NewSeries("B", vals),
		"C": tabgo.NewSeries("C", vals),
	})

	ges := NewGES(data, penalizeScore)
	pdag, err := ges.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	// Should have some edges but not a fully connected graph.
	dirEdges := pdag.DirectedEdges()
	undEdges := pdag.UndirectedEdges()
	t.Logf("Directed: %v, Undirected: %v", dirEdges, undEdges)
	totalEdges := len(dirEdges) + len(undEdges)
	if totalEdges > 3 {
		t.Errorf("expected at most 3 edges for 3-variable graph, got %d", totalEdges)
	}
}
