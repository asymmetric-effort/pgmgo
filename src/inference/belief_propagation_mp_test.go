//go:build unit

package inference

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func TestNewBeliefPropagationWithMessagePassing(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA, pBA}}
	schedule := []MessagePass{}

	bpm := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, schedule)
	if bpm == nil {
		t.Fatal("NewBeliefPropagationWithMessagePassing returned nil")
	}
	if bpm.IsCalibrated() {
		t.Error("should not be calibrated before Calibrate()")
	}
}

func TestBPMP_SingleClique(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	separators := map[string][]string{}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA, pBA}}

	bpm := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, nil)
	if err := bpm.Calibrate(); err != nil {
		t.Fatalf("Calibrate failed: %v", err)
	}
	if !bpm.IsCalibrated() {
		t.Error("should be calibrated after Calibrate()")
	}

	result, err := bpm.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	vals := result.Values().Data()
	if math.Abs(vals[0]-0.4) > 1e-9 || math.Abs(vals[1]-0.6) > 1e-9 {
		t.Errorf("P(A) = %v, want [0.4, 0.6]", vals)
	}
}

func TestBPMP_ChainSchedule(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{
		0.2, 0.3,
		0.8, 0.7,
	})
	pCB, _ := factors.NewDiscreteFactor([]string{"C", "B"}, []int{2, 2}, []float64{
		0.5, 0.1,
		0.5, 0.9,
	})

	cliques := [][]string{
		{"A", "B"},
		{"B", "C"},
	}
	separators := map[string][]string{
		edgeKey(0, 1): {"B"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {pA, pBA},
		1: {pCB},
	}

	// Explicit schedule: collect from 1->0, then distribute 0->1.
	schedule := []MessagePass{
		{From: 1, To: 0},
		{From: 0, To: 1},
	}

	bpm := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, schedule)
	if err := bpm.Calibrate(); err != nil {
		t.Fatalf("Calibrate failed: %v", err)
	}
	if !bpm.IsCalibrated() {
		t.Error("should be calibrated")
	}

	// Query P(A) — should match regular BP.
	result, err := bpm.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatalf("Query(A) failed: %v", err)
	}
	vals := result.Values().Data()
	if math.Abs(vals[0]-0.4) > 1e-9 || math.Abs(vals[1]-0.6) > 1e-9 {
		t.Errorf("P(A) = %v, want [0.4, 0.6]", vals)
	}

	// Compare with regular BP.
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	_ = bp.Calibrate()
	bpResult, _ := bp.Query([]string{"C"}, nil)
	mpResult, err := bpm.Query([]string{"C"}, nil)
	if err != nil {
		t.Fatalf("Query(C) failed: %v", err)
	}
	bpVals := bpResult.Values().Data()
	mpVals := mpResult.Values().Data()
	for i := range bpVals {
		if math.Abs(bpVals[i]-mpVals[i]) > 1e-9 {
			t.Errorf("P(C)[%d]: BP=%f MP=%f", i, bpVals[i], mpVals[i])
		}
	}
}

func TestBPMP_GetCliqueBelief(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	pBA, _ := factors.NewDiscreteFactor([]string{"B", "A"}, []int{2, 2}, []float64{0.2, 0.3, 0.8, 0.7})

	cliques := [][]string{{"A", "B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA, pBA}}

	bpm := NewBeliefPropagationWithMessagePassing(cliques, map[string][]string{}, cliqueFactors, nil)
	_ = bpm.Calibrate()

	belief := bpm.GetCliqueBelief(0)
	if belief == nil {
		t.Fatal("GetCliqueBelief(0) returned nil")
	}
	if bpm.GetCliqueBelief(99) != nil {
		t.Error("GetCliqueBelief(99) should return nil")
	}
}

func TestBPMP_InvalidSchedule(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	pB, _ := factors.NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.5, 0.5})

	cliques := [][]string{{"A"}, {"B"}}
	separators := map[string][]string{edgeKey(0, 1): {}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA}, 1: {pB}}

	// To index out of range.
	bpm := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors,
		[]MessagePass{{From: 0, To: 5}})
	err := bpm.Calibrate()
	if err == nil {
		t.Error("expected error for out-of-range schedule entry")
	}
}

func TestBPMP_NoSeparatorSchedule(t *testing.T) {
	pA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	pB, _ := factors.NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.5, 0.5})

	cliques := [][]string{{"A"}, {"B"}}
	// No separator between cliques 0 and 1.
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {pA}, 1: {pB}}

	bpm := NewBeliefPropagationWithMessagePassing(cliques, map[string][]string{}, cliqueFactors,
		[]MessagePass{{From: 0, To: 1}})
	err := bpm.Calibrate()
	if err == nil {
		t.Error("expected error for schedule entry with no separator")
	}
}
