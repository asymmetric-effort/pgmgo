//go:build unit

package models

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func TestNewFactorGraph(t *testing.T) {
	fg := NewFactorGraph()
	if fg == nil {
		t.Fatal("NewFactorGraph returned nil")
	}
	if len(fg.GetVariables()) != 0 {
		t.Errorf("expected 0 variables, got %d", len(fg.GetVariables()))
	}
	if len(fg.GetFactors()) != 0 {
		t.Errorf("expected 0 factors, got %d", len(fg.GetFactors()))
	}
}

func TestFactorGraphAddVariable(t *testing.T) {
	fg := NewFactorGraph()

	if err := fg.AddVariable("X", 3); err != nil {
		t.Fatalf("AddVariable: %v", err)
	}

	vars := fg.GetVariables()
	if len(vars) != 1 || vars[0] != "X" {
		t.Errorf("expected [X], got %v", vars)
	}
}

func TestFactorGraphAddVariableDuplicate(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("X", 2)

	if err := fg.AddVariable("X", 2); err == nil {
		t.Error("expected error for duplicate variable")
	}
}

func TestFactorGraphAddVariableInvalidCardinality(t *testing.T) {
	fg := NewFactorGraph()
	if err := fg.AddVariable("X", 0); err == nil {
		t.Error("expected error for zero cardinality")
	}
	if err := fg.AddVariable("Y", -1); err == nil {
		t.Error("expected error for negative cardinality")
	}
}

func TestFactorGraphAddFactor(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 3)

	f, err := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 3}, []float64{
		0.1, 0.2, 0.3,
		0.15, 0.1, 0.15,
	})
	if err != nil {
		t.Fatalf("NewDiscreteFactor: %v", err)
	}

	if err := fg.AddFactor(f); err != nil {
		t.Fatalf("AddFactor: %v", err)
	}

	facs := fg.GetFactors()
	if len(facs) != 1 {
		t.Errorf("expected 1 factor, got %d", len(facs))
	}
}

func TestFactorGraphAddFactorNil(t *testing.T) {
	fg := NewFactorGraph()
	if err := fg.AddFactor(nil); err == nil {
		t.Error("expected error for nil factor")
	}
}

func TestFactorGraphAddFactorUnknownVariable(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)

	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 3}, make([]float64, 6))
	if err := fg.AddFactor(f); err == nil {
		t.Error("expected error for factor with unknown variable B")
	}
}

func TestFactorGraphAddFactorCardinalityMismatch(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 3)

	// Factor says B has cardinality 2, but graph says 3.
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, make([]float64, 4))
	if err := fg.AddFactor(f); err == nil {
		t.Error("expected error for cardinality mismatch")
	}
}

func TestFactorGraphGetVariablesSorted(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("C", 2)
	_ = fg.AddVariable("A", 3)
	_ = fg.AddVariable("B", 2)

	vars := fg.GetVariables()
	if len(vars) != 3 || vars[0] != "A" || vars[1] != "B" || vars[2] != "C" {
		t.Errorf("expected [A B C], got %v", vars)
	}
}

func TestFactorGraphGetFactorsOf(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 2)
	_ = fg.AddVariable("C", 2)

	f1, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	f2, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.4, 0.3, 0.2, 0.1})
	_ = fg.AddFactor(f1)
	_ = fg.AddFactor(f2)

	// A should appear in 1 factor.
	aFactors := fg.GetFactorsOf("A")
	if len(aFactors) != 1 {
		t.Errorf("expected 1 factor for A, got %d", len(aFactors))
	}

	// B should appear in 2 factors.
	bFactors := fg.GetFactorsOf("B")
	if len(bFactors) != 2 {
		t.Errorf("expected 2 factors for B, got %d", len(bFactors))
	}

	// C should appear in 1 factor.
	cFactors := fg.GetFactorsOf("C")
	if len(cFactors) != 1 {
		t.Errorf("expected 1 factor for C, got %d", len(cFactors))
	}
}

func TestFactorGraphGetFactorsOfNonexistent(t *testing.T) {
	fg := NewFactorGraph()
	if fg.GetFactorsOf("Z") != nil {
		t.Error("expected nil for nonexistent variable")
	}
}

func TestFactorGraphToMarkovNetwork(t *testing.T) {
	fg := NewFactorGraph()
	if err := fg.ToMarkovNetwork(); err != nil {
		t.Errorf("ToMarkovNetwork stub should return nil, got %v", err)
	}
}

func TestFactorGraphCheckModel(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 2)

	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	_ = fg.AddFactor(f)

	// Both variables are covered by the factor — but we need every variable covered.
	if err := fg.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestFactorGraphCheckModelNoVariables(t *testing.T) {
	fg := NewFactorGraph()
	if err := fg.CheckModel(); err == nil {
		t.Error("expected error for factor graph with no variables")
	}
}

func TestFactorGraphCheckModelNoFactors(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)

	if err := fg.CheckModel(); err == nil {
		t.Error("expected error for factor graph with no factors")
	}
}

func TestFactorGraphCheckModelUnreferencedVariable(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 2)
	_ = fg.AddVariable("C", 2) // Not referenced by any factor.

	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	_ = fg.AddFactor(f)

	if err := fg.CheckModel(); err == nil {
		t.Error("expected error for unreferenced variable C")
	}
}

func TestFactorGraphMultipleFactors(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("X", 2)
	_ = fg.AddVariable("Y", 3)
	_ = fg.AddVariable("Z", 2)

	f1, _ := factors.NewDiscreteFactor([]string{"X", "Y"}, []int{2, 3}, make([]float64, 6))
	f2, _ := factors.NewDiscreteFactor([]string{"Y", "Z"}, []int{3, 2}, make([]float64, 6))
	f3, _ := factors.NewDiscreteFactor([]string{"X"}, []int{2}, []float64{0.5, 0.5})

	_ = fg.AddFactor(f1)
	_ = fg.AddFactor(f2)
	_ = fg.AddFactor(f3)

	facs := fg.GetFactors()
	if len(facs) != 3 {
		t.Errorf("expected 3 factors, got %d", len(facs))
	}

	if err := fg.CheckModel(); err != nil {
		t.Errorf("CheckModel: %v", err)
	}
}

func TestFactorGraphFromStudentBN(t *testing.T) {
	// Build a factor graph from the student BN's factors.
	bn := buildStudentNetwork(t)
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}

	fg := NewFactorGraph()

	// Add variables with their cardinalities.
	varCards := map[string]int{"D": 2, "I": 2, "G": 3, "L": 2, "S": 2}
	for v, c := range varCards {
		if err := fg.AddVariable(v, c); err != nil {
			t.Fatalf("AddVariable(%q): %v", v, err)
		}
	}

	// Add all factors.
	for _, f := range markovFactors {
		if err := fg.AddFactor(f); err != nil {
			t.Fatalf("AddFactor: %v", err)
		}
	}

	if err := fg.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}

	vars := fg.GetVariables()
	if len(vars) != 5 {
		t.Errorf("expected 5 variables, got %d", len(vars))
	}

	facs := fg.GetFactors()
	if len(facs) != 5 {
		t.Errorf("expected 5 factors, got %d", len(facs))
	}

	// G should be referenced by factors for G, L (which has parent G).
	gFactors := fg.GetFactorsOf("G")
	if len(gFactors) < 2 {
		t.Errorf("expected at least 2 factors referencing G, got %d", len(gFactors))
	}
}

func TestFactorGraphGetFactorsReturnsCopy(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	_ = fg.AddFactor(f)

	facs := fg.GetFactors()
	facs[0] = nil // Modify the returned slice.

	// Original should be unaffected.
	facs2 := fg.GetFactors()
	if facs2[0] == nil {
		t.Error("GetFactors did not return a copy")
	}
}
