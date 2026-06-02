//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// validOrder checks that order contains exactly the vars in wantSet (no
// duplicates, no extras, no missing).
func validOrder(t *testing.T, order []string, wantSet map[string]bool, label string) {
	t.Helper()
	if len(order) != len(wantSet) {
		t.Fatalf("%s: expected %d vars, got %d (%v)", label, len(wantSet), len(order), order)
	}
	seen := make(map[string]bool)
	for _, v := range order {
		if seen[v] {
			t.Errorf("%s: duplicate variable %q in order", label, v)
		}
		seen[v] = true
		if !wantSet[v] {
			t.Errorf("%s: unexpected variable %q in order", label, v)
		}
	}
	for v := range wantSet {
		if !seen[v] {
			t.Errorf("%s: missing variable %q from order", label, v)
		}
	}
}

func toSet(vars []string) map[string]bool {
	s := make(map[string]bool, len(vars))
	for _, v := range vars {
		s[v] = true
	}
	return s
}

// simpleFactors creates a small factor list:
//
//	f1(A,B), f2(B,C), f3(C,D)
//
// This is a simple chain A - B - C - D.
func simpleFactors() []*factors.DiscreteFactor {
	f1, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 3}, make([]float64, 6))
	f2, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{3, 2}, make([]float64, 6))
	f3, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 4}, make([]float64, 8))
	return []*factors.DiscreteFactor{f1, f2, f3}
}

// ---------------------------------------------------------------------------
// MinFillOrder
// ---------------------------------------------------------------------------

func TestMinFillOrder_Empty(t *testing.T) {
	order := MinFillOrder(nil, nil)
	if len(order) != 0 {
		t.Errorf("expected empty order, got %v", order)
	}
}

func TestMinFillOrder_SingleVar(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	order := MinFillOrder([]*factors.DiscreteFactor{f}, []string{"A"})
	if len(order) != 1 || order[0] != "A" {
		t.Errorf("expected [A], got %v", order)
	}
}

func TestMinFillOrder_AllStudentVars(t *testing.T) {
	fl := studentFactors()
	eliminateVars := []string{"D", "I", "L", "S"}
	order := MinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinFillOrder/student")
}

func TestMinFillOrder_Chain(t *testing.T) {
	// Chain: A-B-C-D. Eliminating a leaf (A or D) requires 0 fill edges,
	// while eliminating an interior node (B or C) requires 1 fill edge.
	// So the first variable eliminated should be A or D.
	fl := simpleFactors()
	eliminateVars := []string{"A", "B", "C", "D"}
	order := MinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinFillOrder/chain")

	// First eliminated should be a leaf (0 fill edges).
	if order[0] != "A" && order[0] != "D" {
		t.Errorf("MinFillOrder/chain: expected leaf first, got %q", order[0])
	}
}

// ---------------------------------------------------------------------------
// MinWeightOrder
// ---------------------------------------------------------------------------

func TestMinWeightOrder_Empty(t *testing.T) {
	order := MinWeightOrder(nil, nil)
	if len(order) != 0 {
		t.Errorf("expected empty order, got %v", order)
	}
}

func TestMinWeightOrder_SingleVar(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	order := MinWeightOrder([]*factors.DiscreteFactor{f}, []string{"A"})
	if len(order) != 1 || order[0] != "A" {
		t.Errorf("expected [A], got %v", order)
	}
}

func TestMinWeightOrder_AllStudentVars(t *testing.T) {
	fl := studentFactors()
	eliminateVars := []string{"D", "I", "L", "S"}
	order := MinWeightOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinWeightOrder/student")
}

func TestMinWeightOrder_Chain(t *testing.T) {
	// Chain: A(2)-B(3)-C(2)-D(4)
	// Weight of eliminating A = card(B) = 3
	// Weight of eliminating D = card(C) = 2
	// Weight of eliminating B = card(A)*card(C) = 4
	// Weight of eliminating C = card(B)*card(D) = 12
	// So D should be eliminated first (weight 2).
	fl := simpleFactors()
	eliminateVars := []string{"A", "B", "C", "D"}
	order := MinWeightOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinWeightOrder/chain")

	if order[0] != "D" {
		t.Errorf("MinWeightOrder/chain: expected D first (weight=2), got %q", order[0])
	}
}

// ---------------------------------------------------------------------------
// WeightedMinFillOrder
// ---------------------------------------------------------------------------

func TestWeightedMinFillOrder_Empty(t *testing.T) {
	order := WeightedMinFillOrder(nil, nil)
	if len(order) != 0 {
		t.Errorf("expected empty order, got %v", order)
	}
}

func TestWeightedMinFillOrder_SingleVar(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 2, 3, 4})
	order := WeightedMinFillOrder([]*factors.DiscreteFactor{f}, []string{"A"})
	if len(order) != 1 || order[0] != "A" {
		t.Errorf("expected [A], got %v", order)
	}
}

func TestWeightedMinFillOrder_AllStudentVars(t *testing.T) {
	fl := studentFactors()
	eliminateVars := []string{"D", "I", "L", "S"}
	order := WeightedMinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "WeightedMinFillOrder/student")
}

func TestWeightedMinFillOrder_Chain(t *testing.T) {
	// Chain: A(2)-B(3)-C(2)-D(4). Leaf nodes have 0 fill cost.
	// A and D are leaves => weighted fill cost = 0 for both.
	fl := simpleFactors()
	eliminateVars := []string{"A", "B", "C", "D"}
	order := WeightedMinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "WeightedMinFillOrder/chain")

	if order[0] != "A" && order[0] != "D" {
		t.Errorf("WeightedMinFillOrder/chain: expected leaf first, got %q", order[0])
	}
}

// ---------------------------------------------------------------------------
// GetEliminationOrder
// ---------------------------------------------------------------------------

func TestGetEliminationOrder_AllHeuristics(t *testing.T) {
	fl := studentFactors()
	eliminateVars := []string{"D", "I", "L", "S"}
	wantSet := toSet(eliminateVars)

	heuristics := []string{"min_neighbors", "min_fill", "min_weight", "weighted_min_fill"}
	for _, h := range heuristics {
		order, err := GetEliminationOrder(fl, eliminateVars, h)
		if err != nil {
			t.Fatalf("GetEliminationOrder(%q) error: %v", h, err)
		}
		validOrder(t, order, wantSet, "GetEliminationOrder/"+h)
	}
}

func TestGetEliminationOrder_UnknownHeuristic(t *testing.T) {
	fl := studentFactors()
	_, err := GetEliminationOrder(fl, []string{"D"}, "bogus")
	if err == nil {
		t.Error("expected error for unknown heuristic")
	}
}

func TestGetEliminationOrder_Empty(t *testing.T) {
	fl := studentFactors()
	order, err := GetEliminationOrder(fl, nil, "min_fill")
	if err != nil {
		t.Fatal(err)
	}
	if len(order) != 0 {
		t.Errorf("expected empty order for nil eliminateVars, got %v", order)
	}
}

// ---------------------------------------------------------------------------
// Integration: VariableElimination with different heuristics
// ---------------------------------------------------------------------------

func TestVariableElimination_WithHeuristics(t *testing.T) {
	// All heuristics should produce the same (correct) inference result.
	heuristics := []string{"min_neighbors", "min_fill", "min_weight", "weighted_min_fill"}

	for _, h := range heuristics {
		t.Run(h, func(t *testing.T) {
			ve := NewVariableElimination(studentFactors(), h)
			result, err := ve.Query([]string{"G"}, map[string]int{"D": 1})
			if err != nil {
				t.Fatalf("Query with heuristic %q failed: %v", h, err)
			}
			assertSumsToOne(t, result)
			assertNear(t, result.GetValue(map[string]int{"G": 0}), 0.185, 1e-6, "P(G=0|D=1)")
			assertNear(t, result.GetValue(map[string]int{"G": 1}), 0.265, 1e-6, "P(G=1|D=1)")
			assertNear(t, result.GetValue(map[string]int{"G": 2}), 0.55, 1e-6, "P(G=2|D=1)")
		})
	}
}

func TestVariableElimination_DefaultHeuristic(t *testing.T) {
	// No heuristic argument => should default to min_neighbors.
	ve := NewVariableElimination(studentFactors())
	result, err := ve.Query([]string{"D"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertSumsToOne(t, result)
	assertNear(t, result.GetValue(map[string]int{"D": 0}), 0.6, 1e-6, "P(D=0)")
}

// ---------------------------------------------------------------------------
// Triangle factor to test non-trivial fill behavior
// ---------------------------------------------------------------------------

func TestMinFillOrder_Triangle(t *testing.T) {
	// Factors: f1(A,B), f2(B,C), f3(A,C) — already a triangle, so all
	// fill costs are 0 regardless of elimination order.
	f1, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 1, 1, 1})
	f2, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 1, 1, 1})
	f3, _ := factors.NewDiscreteFactor([]string{"A", "C"}, []int{2, 2}, []float64{1, 1, 1, 1})

	fl := []*factors.DiscreteFactor{f1, f2, f3}
	eliminateVars := []string{"A", "B", "C"}
	order := MinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinFillOrder/triangle")
}

func TestMinFillOrder_Star(t *testing.T) {
	// Star: center B connected to A, C, D, E. No edges among leaves.
	// Eliminating a leaf costs 0 fill. Eliminating B costs C(4,2)=6 fill edges.
	// So a leaf should be first.
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{1, 1, 1, 1})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{1, 1, 1, 1})
	fBD, _ := factors.NewDiscreteFactor([]string{"B", "D"}, []int{2, 2}, []float64{1, 1, 1, 1})
	fBE, _ := factors.NewDiscreteFactor([]string{"B", "E"}, []int{2, 2}, []float64{1, 1, 1, 1})

	fl := []*factors.DiscreteFactor{fAB, fBC, fBD, fBE}
	eliminateVars := []string{"A", "B", "C", "D", "E"}
	order := MinFillOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinFillOrder/star")

	// B should NOT be first (it has the highest fill cost).
	if order[0] == "B" {
		t.Errorf("MinFillOrder/star: B should not be first (highest fill)")
	}
}

// ---------------------------------------------------------------------------
// MinWeightOrder with varied cardinalities
// ---------------------------------------------------------------------------

func TestMinWeightOrder_VaryingCardinalities(t *testing.T) {
	// A(2)-B(10)-C(2). Eliminating A: weight=card(B)=10.
	// Eliminating C: weight=card(B)=10. Eliminating B: weight=card(A)*card(C)=4.
	// So B should be first.
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 10},
		make([]float64, 20))
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{10, 2},
		make([]float64, 20))

	fl := []*factors.DiscreteFactor{fAB, fBC}
	eliminateVars := []string{"A", "B", "C"}
	order := MinWeightOrder(fl, eliminateVars)
	validOrder(t, order, toSet(eliminateVars), "MinWeightOrder/varCard")

	if order[0] != "B" {
		t.Errorf("MinWeightOrder/varCard: expected B first (weight=4), got %q", order[0])
	}
}
