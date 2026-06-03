//go:build unit

package sampling

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// TestGibbsSampling_ZeroConditional exercises the all-zero fallback in
// computeFullConditional. This happens when evidence forces an impossible
// state, making all factor products zero.
func TestGibbsSampling_ZeroConditional(t *testing.T) {
	// Build A -> B -> C.
	// CPD of C: P(C=0|B=0)=1, P(C=1|B=0)=0, P(C=0|B=1)=1, P(C=1|B=1)=0.
	// Evidence: C=1 (impossible under this CPD).
	// When computing conditional for B given C=1, all factors give zero.
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.AddEdge("B", "C")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	bn.SetStates("C", []string{"c0", "c1"})

	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdA)

	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.5, 0.5}, {0.5, 0.5}}, []string{"A"}, []int{2})
	_ = bn.AddCPD(cpdB)

	// P(C=0|B=0)=1, P(C=1|B=0)=0, P(C=0|B=1)=1, P(C=1|B=1)=0.
	cpdC, _ := factors.NewTabularCPD("C", 2, [][]float64{{1.0, 1.0}, {0.0, 0.0}}, []string{"B"}, []int{2})
	_ = bn.AddCPD(cpdC)

	gs, err := NewGibbsSampling(bn, 42)
	if err != nil {
		t.Fatalf("NewGibbsSampling failed: %v", err)
	}

	// Sample with evidence C=1 (impossible state).
	// The Gibbs sampler should not crash; it falls back to uniform.
	df, err := gs.Sample(3, 5, 1, map[string]int{"C": 1})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if df.Len() != 3 {
		t.Errorf("expected 3 samples, got %d", df.Len())
	}
}

// TestSampleFromCPD_Fallback exercises the fallback path in sampleFromCPD
// where probabilities sum to less than 1, so cumSum never reaches u.
func TestSampleFromCPD_Fallback(t *testing.T) {
	// Create a CPD with probabilities that sum to much less than 1.
	// P(X=0) = 0.0, P(X=1) = 0.0 — sum = 0.
	// Any u > 0 will exceed cumSum, triggering the fallback.
	bn := models.NewBayesianNetwork()
	bn.AddNode("X")
	bn.SetStates("X", []string{"x0", "x1"})
	// Normal CPD for model check.
	cpdNormal, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdNormal)

	bms, err := NewBayesianModelSampling(bn, 42)
	if err != nil {
		t.Fatal(err)
	}

	// Create a degenerate CPD with zero probabilities.
	cpdZero, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.0}, {0.0}}, nil, nil)

	// Call sampleFromCPD directly with the zero-prob CPD.
	val := bms.sampleFromCPD(cpdZero, map[string]int{})
	// Should return variableCard - 1 = 1.
	if val != 1 {
		t.Errorf("expected fallback to return 1 (variableCard-1), got %d", val)
	}
}
