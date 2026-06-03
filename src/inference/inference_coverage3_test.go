//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// This file targets remaining uncovered error branches and edge cases.

// ---------------------------------------------------------------------------
// BP: initializePotentials error branches (uniform creation for empty clique)
// ---------------------------------------------------------------------------

func TestBP_InitPotentials_UnknownCard(t *testing.T) {
	// Clique with a variable not in any factor's cardMap
	cliques := [][]string{{"A", "UNKNOWN"}}
	bp := NewBeliefPropagation(cliques, nil, nil)
	err := bp.Calibrate()
	if err == nil {
		t.Error("expected error for unknown cardinality in uniform creation")
	}
}

// ---------------------------------------------------------------------------
// BP: Calibrate error when initializePotentials fails
// ---------------------------------------------------------------------------

func TestBP_Calibrate_InitError(t *testing.T) {
	cliques := [][]string{{"A", "X"}}
	bp := NewBeliefPropagation(cliques, nil, nil)
	err := bp.Calibrate()
	if err == nil {
		t.Error("expected error for init failure")
	}
}

func TestBP_MaxCalibrate_InitError(t *testing.T) {
	cliques := [][]string{{"A", "X"}}
	bp := NewBeliefPropagation(cliques, nil, nil)
	err := bp.MaxCalibrate()
	if err == nil {
		t.Error("expected error for init failure")
	}
}

// ---------------------------------------------------------------------------
// BP: computeMessage/computeMaxMessage error branches
// These error branches are deep inside the algorithm and hard to trigger
// externally. We can trigger them by creating unusual graph structures.
// ---------------------------------------------------------------------------

// Test three-clique chain for more message passing
func TestBP_ThreeCliqueChain(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})

	cliques := [][]string{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	separators := map[string][]string{
		"0-1": {"B"},
		"1-2": {"C"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {fAB},
		1: {fBC},
		2: {fCD},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	// Query each variable
	for _, v := range []string{"A", "B", "C", "D"} {
		result, err := bp.Query([]string{v}, nil)
		if err != nil {
			t.Fatalf("Query %s failed: %v", v, err)
		}
		if result == nil {
			t.Errorf("expected non-nil result for %s", v)
		}
	}
	// Query with evidence
	result, err := bp.Query([]string{"A"}, map[string]int{"D": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
	// MAPQuery
	mapResult, err := bp.MAPQuery([]string{"A"}, map[string]int{"D": 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := mapResult["A"]; !ok {
		t.Error("expected A in MAP result")
	}
}

// Test MaxCalibrate with three cliques
func TestBP_ThreeCliqueChain_MaxCalibrate(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})

	cliques := [][]string{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	separators := map[string][]string{
		"0-1": {"B"},
		"1-2": {"C"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {fAB},
		1: {fBC},
		2: {fCD},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.MaxCalibrate(); err != nil {
		t.Fatal(err)
	}
	// MAPQuery after MaxCalibrate
	result, err := bp.MAPQuery([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

// Test BP GetSepsetBeliefs with multiple separators
func TestBP_GetSepsetBeliefs_ThreeClique(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})

	cliques := [][]string{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	separators := map[string][]string{
		"0-1": {"B"},
		"1-2": {"C"},
	}
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		0: {fAB},
		1: {fBC},
		2: {fCD},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	beliefs := bp.GetSepsetBeliefs()
	if len(beliefs) != 2 {
		t.Errorf("expected 2 separator beliefs, got %d", len(beliefs))
	}
}

// ---------------------------------------------------------------------------
// VE: error branches for evidence not matching
// ---------------------------------------------------------------------------

func TestVE_Query_MinNeighborsHeuristic(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB}, "min_neighbors")
	result, err := ve.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestVE_Query_MinFillHeuristic(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB}, "min_fill")
	result, err := ve.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ---------------------------------------------------------------------------
// MPLP: Query with actual factor computation
// ---------------------------------------------------------------------------

func TestMPLP_Query_MultiFactor(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	result, err := m.Query([]string{"A"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestMPLP_Query_WithEvidence(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	result, err := m.Query([]string{"A"}, map[string]int{"C": 0}, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ---------------------------------------------------------------------------
// MPLP: Tighten
// ---------------------------------------------------------------------------

func TestMPLP_Tighten_Cov(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	obj := m.Tighten(50)
	_ = obj
}

// ---------------------------------------------------------------------------
// BP MP: message passing with actual schedule
// ---------------------------------------------------------------------------

func TestBPMP_WithSchedule(t *testing.T) {
	cliques := [][]string{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	separators := map[string][]string{
		"0-1": {"B"},
		"1-2": {"C"},
	}
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}, 2: {fCD}}
	schedule := []MessagePass{
		{From: 0, To: 1}, {From: 2, To: 1},
		{From: 1, To: 0}, {From: 1, To: 2},
	}
	mp := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, schedule)
	if err := mp.Calibrate(); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// DBN: three-step inference with evidence paths
// ---------------------------------------------------------------------------

func TestDBNInference_ThreeStepForward(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{
		{"B": 0},
		{"B": 1},
		{"B": 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_ThreeStepBackward(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.BackwardInference([]string{"A"}, []map[string]int{
		{},
		{"B": 1},
		{"B": 0},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// Causal: interceptsAllPaths when intercept set blocks all direct children
func TestInterceptsAllPaths_AllBlocked(t *testing.T) {
	bn := makeSimpleBN3(t)
	g := bnToDigraph(bn)
	// {X} blocks all paths from Z to Y
	result := interceptsAllPaths(g, "Z", "Y", map[string]bool{"X": true})
	if !result {
		t.Error("expected true")
	}
}

// Causal: interceptsAllPaths when dst is a direct child of src
func TestInterceptsAllPaths_DirectChild(t *testing.T) {
	bn := makeSimpleBN3(t)
	g := bnToDigraph(bn)
	// Direct child with no intercept
	result := interceptsAllPaths(g, "X", "Y", map[string]bool{})
	if result {
		t.Error("expected false - Y is directly reachable from X")
	}
}
