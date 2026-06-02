//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// MaxMarginal tests
// ---------------------------------------------------------------------------

func TestMaxMarginal_SingleVar(t *testing.T) {
	ve := NewVariableElimination(studentFactors())
	// MaxMarginal maximizes hidden variables instead of summing them.
	// The result is normalized but may differ from the sum-marginal.
	result, err := ve.MaxMarginal([]string{"D"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	// Should have valid probabilities.
	d0 := result.GetValue(map[string]int{"D": 0})
	d1 := result.GetValue(map[string]int{"D": 1})
	if d0 < 0 || d1 < 0 {
		t.Errorf("max-marginal values should be non-negative: D=0=%f, D=1=%f", d0, d1)
	}
}

func TestMaxMarginal_WithEvidence(t *testing.T) {
	ve := NewVariableElimination(studentFactors())
	// MaxMarginal of G given D=1, I=0 should heavily favor G=2 (0.7).
	result, err := ve.MaxMarginal([]string{"G"}, map[string]int{"D": 1, "I": 0})
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	g2 := result.GetValue(map[string]int{"G": 2})
	g0 := result.GetValue(map[string]int{"G": 0})
	g1 := result.GetValue(map[string]int{"G": 1})
	if g2 <= g0 || g2 <= g1 {
		t.Errorf("expected G=2 (%f) to dominate G=0 (%f) and G=1 (%f)", g2, g0, g1)
	}
}

func TestMaxMarginal_SimpleNetwork(t *testing.T) {
	// A -> B, P(A)=[0.4,0.6], P(B|A)
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	ve := NewVariableElimination([]*factors.DiscreteFactor{pA, pBA})
	result, err := ve.MaxMarginal([]string{"B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	// Max-marginal of B: for B=0, max_A P(B=0|A)*P(A) = max(0.2*0.4, 0.3*0.6) = max(0.08, 0.18) = 0.18
	// for B=1, max_A P(B=1|A)*P(A) = max(0.8*0.4, 0.7*0.6) = max(0.32, 0.42) = 0.42
	// Normalized: B=0 = 0.18/0.6 = 0.3, B=1 = 0.42/0.6 = 0.7
	b0 := result.GetValue(map[string]int{"B": 0})
	b1 := result.GetValue(map[string]int{"B": 1})
	assertNear(t, b0, 0.18/0.6, 1e-6, "MaxMarginal B=0")
	assertNear(t, b1, 0.42/0.6, 1e-6, "MaxMarginal B=1")
}

func TestMaxMarginal_EmptyQueryVars(t *testing.T) {
	ve := NewVariableElimination(studentFactors())
	_, err := ve.MaxMarginal(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

// ---------------------------------------------------------------------------
// InducedGraph tests
// ---------------------------------------------------------------------------

func TestInducedGraph_SimpleChain(t *testing.T) {
	// A-B-C chain: factors (A,B) and (B,C)
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC})

	// Eliminate A first, then C: no fill needed since A-B and B-C, then just C left.
	g, err := ve.InducedGraph([]string{"A", "C"})
	if err != nil {
		t.Fatal(err)
	}
	// Should have edges A-B and B-C (no fill edge A-C since A is eliminated before C).
	if !g.HasEdge("A", "B") {
		t.Error("expected edge A-B")
	}
	if !g.HasEdge("B", "C") {
		t.Error("expected edge B-C")
	}
	// A is eliminated first; its only neighbor is B. No fill edges needed.
	// Then C is eliminated; its only neighbor is B. No fill edges needed.
	if g.HasEdge("A", "C") {
		t.Error("did not expect edge A-C for this elimination order")
	}
}

func TestInducedGraph_Triangle(t *testing.T) {
	// Triangle: factors (A,B), (B,C), (A,C)
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fAC, _ := factors.NewDiscreteFactor([]string{"A", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC, fAC})

	g, err := ve.InducedGraph([]string{"A", "B", "C"})
	if err != nil {
		t.Fatal(err)
	}
	// All three edges should exist.
	if !g.HasEdge("A", "B") {
		t.Error("expected edge A-B")
	}
	if !g.HasEdge("B", "C") {
		t.Error("expected edge B-C")
	}
	if !g.HasEdge("A", "C") {
		t.Error("expected edge A-C")
	}
}

func TestInducedGraph_Empty(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fA})
	g, err := ve.InducedGraph(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Nodes()) != 0 {
		t.Errorf("expected empty graph, got %d nodes", len(g.Nodes()))
	}
}

func TestInducedGraph_FillEdge(t *testing.T) {
	// A-B, B-C, no A-C. Eliminating B should create fill edge A-C.
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC})

	g, err := ve.InducedGraph([]string{"B", "A", "C"})
	if err != nil {
		t.Fatal(err)
	}
	// Eliminating B first: neighbors are A,C -> fill edge A-C.
	if !g.HasEdge("A", "C") {
		t.Error("expected fill edge A-C after eliminating B")
	}
}

// ---------------------------------------------------------------------------
// InducedWidth tests
// ---------------------------------------------------------------------------

func TestInducedWidth_Chain(t *testing.T) {
	// A-B-C chain: eliminate A then C.
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC})

	w, err := ve.InducedWidth([]string{"A", "C", "B"})
	if err != nil {
		t.Fatal(err)
	}
	// Eliminate A: neighbors={B}, clique size 2, width 1
	// Eliminate C: neighbors={B}, clique size 2, width 1
	// Eliminate B: neighbors={}, clique size 1, width 0
	// Max width = 1
	if w != 1 {
		t.Errorf("expected induced width 1, got %d", w)
	}
}

func TestInducedWidth_FillEdge(t *testing.T) {
	// A-B, B-C. Eliminate B first -> fill edge A-C, width=2
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 2, 3, 4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC})

	w, err := ve.InducedWidth([]string{"B", "A", "C"})
	if err != nil {
		t.Fatal(err)
	}
	// Eliminate B: neighbors={A,C}, width=2
	if w != 2 {
		t.Errorf("expected induced width 2, got %d", w)
	}
}

func TestInducedWidth_Empty(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fA})
	w, err := ve.InducedWidth(nil)
	if err != nil {
		t.Fatal(err)
	}
	if w != 0 {
		t.Errorf("expected width 0, got %d", w)
	}
}

func TestInducedWidth_StudentNetwork(t *testing.T) {
	ve := NewVariableElimination(studentFactors())
	// Use all variables as elimination order.
	w, err := ve.InducedWidth([]string{"L", "S", "D", "I", "G"})
	if err != nil {
		t.Fatal(err)
	}
	// The student network has treewidth 2 with a good ordering.
	// L neighbors: {G} -> width 1
	// S neighbors: {I} -> width 1
	// D neighbors: {G,I} -> width 2
	// I neighbors: {G} -> width 1
	// G neighbors: {} -> width 0
	// Max = 2
	if w != 2 {
		t.Errorf("expected induced width 2, got %d", w)
	}
}
