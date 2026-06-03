//go:build unit

package models

import (
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// GetCardinality tests
// ---------------------------------------------------------------------------

func TestGetCardinality_Basic(t *testing.T) {
	mn := buildTriangleMRF(t)

	card, err := mn.GetCardinality("A")
	if err != nil {
		t.Fatal(err)
	}
	if card != 2 {
		t.Errorf("expected cardinality 2, got %d", card)
	}

	card, err = mn.GetCardinality("B")
	if err != nil {
		t.Fatal(err)
	}
	if card != 2 {
		t.Errorf("expected cardinality 2, got %d", card)
	}
}

func TestGetCardinality_UnknownNode(t *testing.T) {
	mn := buildTriangleMRF(t)
	_, err := mn.GetCardinality("Z")
	if err == nil {
		t.Error("expected error for unknown node")
	}
}

func TestGetCardinality_NoFactors(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("X")
	_, err := mn.GetCardinality("X")
	if err == nil {
		t.Error("expected error for node with no factors")
	}
}

// ---------------------------------------------------------------------------
// Triangulate tests
// ---------------------------------------------------------------------------

func TestTriangulate_AlreadyChordal(t *testing.T) {
	// Triangle is already chordal.
	mn := buildTriangleMRF(t)
	tri, err := mn.Triangulate("")
	if err != nil {
		t.Fatal(err)
	}
	// Should have the same nodes and edges.
	if len(tri.Nodes()) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(tri.Nodes()))
	}
	edges := tri.Edges()
	if len(edges) != 3 {
		t.Errorf("expected 3 edges, got %d", len(edges))
	}
	// Factors should be copied.
	if len(tri.GetFactors()) != 3 {
		t.Errorf("expected 3 factors, got %d", len(tri.GetFactors()))
	}
}

func TestTriangulate_Chain(t *testing.T) {
	// A-B-C-D chain (not chordal if we had a 4-cycle, but chain is chordal).
	mn := NewMarkovNetwork()
	for _, n := range []string{"A", "B", "C", "D"} {
		mn.AddNode(n)
	}
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	mn.AddEdge("C", "D")

	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{1, 2, 3, 4})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	mn.AddFactor(fCD)

	tri, err := mn.Triangulate("min_degree")
	if err != nil {
		t.Fatal(err)
	}

	if len(tri.Nodes()) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(tri.Nodes()))
	}
}

func TestTriangulate_FourCycle(t *testing.T) {
	// A-B-C-D-A: a 4-cycle that needs triangulation.
	mn := NewMarkovNetwork()
	for _, n := range []string{"A", "B", "C", "D"} {
		mn.AddNode(n)
	}
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	mn.AddEdge("C", "D")
	mn.AddEdge("D", "A")

	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fDA, _ := factors.NewDiscreteFactor([]string{"D", "A"}, []int{2, 2}, []float64{1, 2, 3, 4})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	mn.AddFactor(fCD)
	mn.AddFactor(fDA)

	tri, err := mn.Triangulate("")
	if err != nil {
		t.Fatal(err)
	}

	// The triangulated graph should have at least one fill edge.
	triEdges := tri.Edges()
	if len(triEdges) < 5 {
		t.Errorf("expected at least 5 edges after triangulation, got %d", len(triEdges))
	}
}

func TestTriangulate_MinFill(t *testing.T) {
	mn := buildTriangleMRF(t)
	tri, err := mn.Triangulate("min_fill")
	if err != nil {
		t.Fatal(err)
	}
	if len(tri.Nodes()) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(tri.Nodes()))
	}
}

func TestTriangulate_BadHeuristic(t *testing.T) {
	mn := buildTriangleMRF(t)
	_, err := mn.Triangulate("bad_heuristic")
	if err == nil {
		t.Error("expected error for unknown heuristic")
	}
}

// ---------------------------------------------------------------------------
// ToFactorGraph tests
// ---------------------------------------------------------------------------

func TestToFactorGraph_Basic(t *testing.T) {
	mn := buildTriangleMRF(t)
	fg, err := mn.ToFactorGraph()
	if err != nil {
		t.Fatal(err)
	}

	vars := fg.GetVariables()
	if len(vars) != 3 {
		t.Errorf("expected 3 variables, got %d", len(vars))
	}

	facs := fg.GetFactors()
	if len(facs) != 3 {
		t.Errorf("expected 3 factors, got %d", len(facs))
	}

	// Verify CheckModel passes.
	if err := fg.CheckModel(); err != nil {
		t.Errorf("factor graph CheckModel failed: %v", err)
	}
}

func TestToFactorGraph_SingleFactor(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("X")
	fX, _ := factors.NewDiscreteFactor([]string{"X"}, []int{3}, []float64{0.2, 0.3, 0.5})
	mn.AddFactor(fX)

	fg, err := mn.ToFactorGraph()
	if err != nil {
		t.Fatal(err)
	}
	if len(fg.GetVariables()) != 1 {
		t.Errorf("expected 1 variable, got %d", len(fg.GetVariables()))
	}
}

// ---------------------------------------------------------------------------
// ToBayesianModel tests
// ---------------------------------------------------------------------------

func TestToBayesianModel_SimpleChain(t *testing.T) {
	// Build a simple A-B MN with valid probability-like factors.
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")

	// Factor phi(A,B): a joint-like table.
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
		[]float64{0.3, 0.1, 0.2, 0.4})
	mn.AddFactor(fAB)

	bn, err := mn.ToBayesianModel()
	if err != nil {
		t.Fatal(err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(nodes))
	}

	// Check that both nodes have CPDs.
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			t.Errorf("node %s has no CPD", node)
			continue
		}
		// CPD columns should sum to approximately 1.
		if err := cpd.Validate(); err != nil {
			t.Errorf("CPD for %s failed validation: %v", node, err)
		}
	}
}

func TestToBayesianModel_Triangle(t *testing.T) {
	mn := buildTriangleMRF(t)
	bn, err := mn.ToBayesianModel()
	if err != nil {
		t.Fatal(err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(nodes))
	}

	// All CPDs should validate.
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			t.Errorf("node %s has no CPD", node)
			continue
		}
		if err := cpd.Validate(); err != nil {
			t.Errorf("CPD for %s failed validation: %v", node, err)
		}
	}
}

func TestToBayesianModel_NoFactors(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	_, err := mn.ToBayesianModel()
	if err == nil {
		t.Error("expected error for MN with no factors")
	}
}

// ---------------------------------------------------------------------------
// GetLocalIndependencies tests
// ---------------------------------------------------------------------------

func TestGetLocalIndependencies_Triangle(t *testing.T) {
	mn := buildTriangleMRF(t)

	// In a triangle, every node is a neighbor of every other node.
	// So no non-neighbors exist -> no independence assertions.
	assertions, err := mn.GetLocalIndependencies("A")
	if err != nil {
		t.Fatal(err)
	}
	if assertions != nil {
		t.Errorf("expected no independence assertions in triangle, got %d", len(assertions))
	}
}

func TestGetLocalIndependencies_Chain(t *testing.T) {
	// A-B-C chain: A is independent of C given B.
	mn := NewMarkovNetwork()
	for _, n := range []string{"A", "B", "C"} {
		mn.AddNode(n)
	}
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")

	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)

	// A's Markov blanket is {B}. Non-neighbors: {C}.
	assertions, err := mn.GetLocalIndependencies("A")
	if err != nil {
		t.Fatal(err)
	}
	if len(assertions) != 1 {
		t.Fatalf("expected 1 assertion, got %d", len(assertions))
	}

	a := assertions[0]
	event1 := a.Event1()
	event2 := a.Event2()
	given := a.Given()

	sort.Strings(event1)
	sort.Strings(event2)
	sort.Strings(given)

	if len(event1) != 1 || event1[0] != "A" {
		t.Errorf("expected event1={A}, got %v", event1)
	}
	if len(event2) != 1 || event2[0] != "C" {
		t.Errorf("expected event2={C}, got %v", event2)
	}
	if len(given) != 1 || given[0] != "B" {
		t.Errorf("expected given={B}, got %v", given)
	}

	// C's Markov blanket is {B}. Non-neighbors: {A}.
	assertions, err = mn.GetLocalIndependencies("C")
	if err != nil {
		t.Fatal(err)
	}
	if len(assertions) != 1 {
		t.Fatalf("expected 1 assertion for C, got %d", len(assertions))
	}
}

func TestGetLocalIndependencies_FourNode(t *testing.T) {
	// A-B-C, A-D: D is independent of {B,C} given {A}.
	mn := NewMarkovNetwork()
	for _, n := range []string{"A", "B", "C", "D"} {
		mn.AddNode(n)
	}
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	mn.AddEdge("A", "D")

	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fAD, _ := factors.NewDiscreteFactor([]string{"A", "D"}, []int{2, 2}, []float64{1, 2, 3, 4})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	mn.AddFactor(fAD)

	assertions, err := mn.GetLocalIndependencies("D")
	if err != nil {
		t.Fatal(err)
	}
	if len(assertions) != 1 {
		t.Fatalf("expected 1 assertion for D, got %d", len(assertions))
	}

	a := assertions[0]
	event2 := a.Event2()
	sort.Strings(event2)
	if len(event2) != 2 || event2[0] != "B" || event2[1] != "C" {
		t.Errorf("expected non-neighbors {B,C}, got %v", event2)
	}
}

func TestGetLocalIndependencies_UnknownNode(t *testing.T) {
	mn := buildTriangleMRF(t)
	_, err := mn.GetLocalIndependencies("Z")
	if err == nil {
		t.Error("expected error for unknown node")
	}
}

func TestGetLocalIndependencies_BMiddleOfChain(t *testing.T) {
	// A-B-C: B's Markov blanket is {A,C}. B is connected to everyone else.
	// No non-neighbors for B.
	mn := NewMarkovNetwork()
	for _, n := range []string{"A", "B", "C"} {
		mn.AddNode(n)
	}
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")

	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)

	assertions, err := mn.GetLocalIndependencies("B")
	if err != nil {
		t.Fatal(err)
	}
	if assertions != nil {
		t.Errorf("expected no assertions for B (connected to all others), got %d", len(assertions))
	}
}

// ---------------------------------------------------------------------------
// States tests
// ---------------------------------------------------------------------------

func TestMarkovNetwork_States(t *testing.T) {
	mn := buildTriangleMRF(t)
	states := mn.States()
	if len(states) != 3 {
		t.Errorf("expected 3 entries, got %d", len(states))
	}
	for _, v := range []string{"A", "B", "C"} {
		if states[v] != 2 {
			t.Errorf("expected cardinality 2 for %q, got %d", v, states[v])
		}
	}
}

func TestMarkovNetwork_States_Empty(t *testing.T) {
	mn := NewMarkovNetwork()
	states := mn.States()
	if len(states) != 0 {
		t.Errorf("expected empty states, got %d entries", len(states))
	}
}

func TestMarkovNetwork_States_MixedCardinality(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")

	// Factor with different cardinalities: A=3, B=2
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	mn.AddFactor(f)

	states := mn.States()
	if states["A"] != 3 {
		t.Errorf("expected A cardinality 3, got %d", states["A"])
	}
	if states["B"] != 2 {
		t.Errorf("expected B cardinality 2, got %d", states["B"])
	}
}
