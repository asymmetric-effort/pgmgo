//go:build unit

package models

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func TestNewFunctionalBayesianNetwork(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	if fbn == nil {
		t.Fatal("NewFunctionalBayesianNetwork returned nil")
	}
	if len(fbn.Nodes()) != 0 {
		t.Errorf("expected 0 nodes, got %d", len(fbn.Nodes()))
	}
}

func buildSimpleFunctionalBN(t *testing.T) *FunctionalBayesianNetwork {
	t.Helper()
	fbn := NewFunctionalBayesianNetwork()

	_ = fbn.AddNode("X")
	_ = fbn.AddNode("Y")
	_ = fbn.AddEdge("X", "Y")

	// X: no parents, returns [0.6, 0.4]
	cpdX, err := factors.NewFunctionalCPD("X", nil, func(parentValues map[string]float64) []float64 {
		return []float64{0.6, 0.4}
	})
	if err != nil {
		t.Fatalf("NewFunctionalCPD X: %v", err)
	}

	// Y: parent X, distribution depends on X value
	cpdY, err := factors.NewFunctionalCPD("Y", []string{"X"}, func(parentValues map[string]float64) []float64 {
		if parentValues["X"] == 0 {
			return []float64{0.8, 0.2}
		}
		return []float64{0.3, 0.7}
	})
	if err != nil {
		t.Fatalf("NewFunctionalCPD Y: %v", err)
	}

	if err := fbn.AddFunctionalCPD(cpdX); err != nil {
		t.Fatalf("AddFunctionalCPD X: %v", err)
	}
	if err := fbn.AddFunctionalCPD(cpdY); err != nil {
		t.Fatalf("AddFunctionalCPD Y: %v", err)
	}

	return fbn
}

func TestFunctionalBNAddCPD(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	_ = fbn.AddNode("X")

	cpd, _ := factors.NewFunctionalCPD("X", nil, func(pv map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	})

	if err := fbn.AddFunctionalCPD(cpd); err != nil {
		t.Fatalf("AddFunctionalCPD: %v", err)
	}

	got := fbn.GetFunctionalCPD("X")
	if got == nil {
		t.Fatal("GetFunctionalCPD returned nil")
	}
	if got.Variable() != "X" {
		t.Errorf("expected variable X, got %q", got.Variable())
	}
}

func TestFunctionalBNAddCPDNil(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	if err := fbn.AddFunctionalCPD(nil); err == nil {
		t.Error("expected error for nil CPD")
	}
}

func TestFunctionalBNAddCPDUnknownNode(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	cpd, _ := factors.NewFunctionalCPD("Z", nil, func(pv map[string]float64) []float64 {
		return []float64{1.0}
	})
	if err := fbn.AddFunctionalCPD(cpd); err == nil {
		t.Error("expected error for unknown node")
	}
}

func TestFunctionalBNAddCPDEvidenceMismatch(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	_ = fbn.AddNode("X")
	_ = fbn.AddNode("Y")
	_ = fbn.AddEdge("X", "Y")

	// Y has parent X, but we give it a CPD with no evidence.
	cpd, _ := factors.NewFunctionalCPD("Y", nil, func(pv map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	})
	if err := fbn.AddFunctionalCPD(cpd); err == nil {
		t.Error("expected error for evidence mismatch")
	}
}

func TestFunctionalBNAddCPDWrongEvidence(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	_ = fbn.AddNode("X")
	_ = fbn.AddNode("Y")
	_ = fbn.AddNode("Z")
	_ = fbn.AddEdge("X", "Z")

	// Z has parent X, but CPD says evidence is Y.
	cpd, _ := factors.NewFunctionalCPD("Z", []string{"Y"}, func(pv map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	})
	if err := fbn.AddFunctionalCPD(cpd); err == nil {
		t.Error("expected error for wrong evidence")
	}
}

func TestFunctionalBNCheckModel(t *testing.T) {
	fbn := buildSimpleFunctionalBN(t)
	if err := fbn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestFunctionalBNCheckModelMissingCPD(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	_ = fbn.AddNode("X")
	cpd, _ := factors.NewFunctionalCPD("X", nil, func(pv map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	})
	_ = fbn.AddFunctionalCPD(cpd)

	_ = fbn.AddNode("Y")
	_ = fbn.AddEdge("X", "Y")
	// Y has no CPD.

	if err := fbn.CheckModel(); err == nil {
		t.Error("expected error for missing CPD")
	}
}

func TestFunctionalBNCheckModelEmpty(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	// Empty network is valid.
	if err := fbn.CheckModel(); err != nil {
		t.Fatalf("empty CheckModel: %v", err)
	}
}

func TestFunctionalBNCopy(t *testing.T) {
	fbn := buildSimpleFunctionalBN(t)
	cpy := fbn.Copy()

	if err := cpy.CheckModel(); err != nil {
		t.Fatalf("copied model CheckModel: %v", err)
	}

	// Verify independence: modify copy.
	_ = cpy.AddNode("Z")
	if len(fbn.Nodes()) == len(cpy.Nodes()) {
		t.Error("original was affected by copy modification")
	}
}

func TestFunctionalBNGetFunctionalCPDNonexistent(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	if fbn.GetFunctionalCPD("nonexistent") != nil {
		t.Error("expected nil for nonexistent CPD")
	}
}

func TestFunctionalBNGetDistribution(t *testing.T) {
	fbn := buildSimpleFunctionalBN(t)

	cpd := fbn.GetFunctionalCPD("X")
	dist := cpd.GetDistribution(nil)
	if len(dist) != 2 {
		t.Fatalf("expected 2 states, got %d", len(dist))
	}
	if dist[0] != 0.6 || dist[1] != 0.4 {
		t.Errorf("expected [0.6 0.4], got %v", dist)
	}

	cpdY := fbn.GetFunctionalCPD("Y")
	distY := cpdY.GetDistribution(map[string]float64{"X": 0})
	if distY[0] != 0.8 || distY[1] != 0.2 {
		t.Errorf("expected [0.8 0.2] for X=0, got %v", distY)
	}
	distY2 := cpdY.GetDistribution(map[string]float64{"X": 1})
	if distY2[0] != 0.3 || distY2[1] != 0.7 {
		t.Errorf("expected [0.3 0.7] for X=1, got %v", distY2)
	}
}
