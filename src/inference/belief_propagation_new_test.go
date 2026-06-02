//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// MaxCalibrate tests
// ---------------------------------------------------------------------------

func TestMaxCalibrate_SingleClique(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	err := bp.MaxCalibrate()
	if err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("expected calibrated after MaxCalibrate")
	}

	belief := bp.GetCliqueBelief(0)
	if belief == nil {
		t.Fatal("clique belief is nil")
	}
}

func TestMaxCalibrate_Chain(t *testing.T) {
	bp, _ := chainABCJunctionTree()

	err := bp.MaxCalibrate()
	if err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("expected calibrated after MaxCalibrate")
	}

	// Verify beliefs are non-nil for both cliques.
	for i := 0; i < 2; i++ {
		if bp.GetCliqueBelief(i) == nil {
			t.Errorf("clique %d belief is nil", i)
		}
	}
}

func TestMaxCalibrate_Empty(t *testing.T) {
	bp := NewBeliefPropagation(nil, nil, nil)
	err := bp.MaxCalibrate()
	if err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("expected calibrated for empty BP")
	}
}

// ---------------------------------------------------------------------------
// MAPQuery tests
// ---------------------------------------------------------------------------

func TestMAPQuery_SingleClique(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	assignment, err := bp.MAPQuery([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// P(A=0)=0.4, P(A=1)=0.6 -> MAP is A=1
	if assignment["A"] != 1 {
		t.Errorf("expected MAP A=1, got A=%d", assignment["A"])
	}
}

func TestMAPQuery_Chain(t *testing.T) {
	bp, _ := chainABCJunctionTree()

	assignment, err := bp.MAPQuery([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// P(A=0)=0.4, P(A=1)=0.6 -> MAP is A=1
	if assignment["A"] != 1 {
		t.Errorf("expected MAP A=1, got A=%d", assignment["A"])
	}
}

func TestMAPQuery_WithEvidence(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{
		0.2, 0.3,
		0.8, 0.7,
	})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	// MAP(B | A=0) -> P(B=0|A=0)=0.2, P(B=1|A=0)=0.8 -> MAP is B=1
	assignment, err := bp.MAPQuery([]string{"B"}, map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	if assignment["B"] != 1 {
		t.Errorf("expected MAP B=1 given A=0, got B=%d", assignment["B"])
	}
}

func TestMAPQuery_EmptyQueryVars(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	_, err := bp.MAPQuery(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

// ---------------------------------------------------------------------------
// GetCliques tests
// ---------------------------------------------------------------------------

func TestGetCliques_Chain(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	cliques := bp.GetCliques()
	if len(cliques) != 2 {
		t.Fatalf("expected 2 cliques, got %d", len(cliques))
	}
	// Verify deep copy.
	cliques[0][0] = "Z"
	origCliques := bp.GetCliques()
	if origCliques[0][0] == "Z" {
		t.Error("GetCliques should return deep copy")
	}
}

func TestGetCliques_Single(t *testing.T) {
	bp := simpleABJunctionTree()
	cliques := bp.GetCliques()
	if len(cliques) != 1 {
		t.Fatalf("expected 1 clique, got %d", len(cliques))
	}
	if len(cliques[0]) != 2 {
		t.Errorf("expected clique of size 2, got %d", len(cliques[0]))
	}
}

func TestGetCliques_Empty(t *testing.T) {
	bp := NewBeliefPropagation(nil, nil, nil)
	cliques := bp.GetCliques()
	if len(cliques) != 0 {
		t.Errorf("expected 0 cliques, got %d", len(cliques))
	}
}

// ---------------------------------------------------------------------------
// GetSepsetBeliefs tests
// ---------------------------------------------------------------------------

func TestGetSepsetBeliefs_NotCalibrated(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	beliefs := bp.GetSepsetBeliefs()
	// Should return map with nil values when not calibrated.
	if len(beliefs) == 0 {
		t.Error("expected separator entries even when not calibrated")
	}
	for k, v := range beliefs {
		if v != nil {
			t.Errorf("expected nil belief for separator %s when not calibrated", k)
		}
	}
}

func TestGetSepsetBeliefs_Calibrated(t *testing.T) {
	bp, _ := chainABCJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	beliefs := bp.GetSepsetBeliefs()
	if len(beliefs) != 1 {
		t.Fatalf("expected 1 separator belief, got %d", len(beliefs))
	}

	for k, belief := range beliefs {
		if belief == nil {
			t.Errorf("expected non-nil belief for separator %s", k)
			continue
		}
		// Separator is {B}, so belief should have 1 variable.
		vars := belief.Variables()
		if len(vars) != 1 {
			t.Errorf("expected 1 variable in separator belief, got %d: %v", len(vars), vars)
		}
	}
}

func TestGetSepsetBeliefs_StudentNetwork(t *testing.T) {
	bp, _ := studentJunctionTree()
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	beliefs := bp.GetSepsetBeliefs()
	if len(beliefs) != 2 {
		t.Fatalf("expected 2 separator beliefs, got %d", len(beliefs))
	}

	for k, belief := range beliefs {
		if belief == nil {
			t.Errorf("separator %s: expected non-nil belief", k)
		}
	}
}
