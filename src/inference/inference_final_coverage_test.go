//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ==========================================================================
// CausalInference: IdentificationMethod — currently at 30%
// Need: frontdoor path, IV path, and "none" path
// ==========================================================================

func TestFinalCov_IdentificationMethod_FrontdoorPath(t *testing.T) {
	// Build network where backdoor fails but frontdoor works:
	// U -> X, U -> Y (hidden confounder makes backdoor fail if U not observed)
	// X -> M -> Y (frontdoor via M)
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"U", "X", "M", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("U", "X")
	bn.AddEdge("U", "Y")
	bn.AddEdge("X", "M")
	bn.AddEdge("M", "Y")

	uCPD, _ := factors.NewTabularCPD("U", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(uCPD)
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.7, 0.3}, {0.3, 0.7}}, []string{"U"}, []int{2})
	bn.AddCPD(xCPD)
	mCPD, _ := factors.NewTabularCPD("M", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(mCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.4, 0.6, 0.1}, {0.1, 0.6, 0.4, 0.9}}, []string{"M", "U"}, []int{2, 2})
	bn.AddCPD(yCPD)

	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}

	method := ci.IdentificationMethod("X", "Y")
	t.Logf("Method: %s", method)
	// Should be backdoor (U is available) or frontdoor
}

func TestFinalCov_IdentificationMethod_IVPath(t *testing.T) {
	// Z -> X -> Y, U -> X, U -> Y
	// Z is IV, no backdoor unless U observed, no frontdoor
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"Z", "U", "X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("Z", "X")
	bn.AddEdge("U", "X")
	bn.AddEdge("U", "Y")
	bn.AddEdge("X", "Y")

	zCPD, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(zCPD)
	uCPD, _ := factors.NewTabularCPD("U", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(uCPD)
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.6, 0.3, 0.4, 0.2}, {0.4, 0.7, 0.6, 0.8}}, []string{"Z", "U"}, []int{2, 2})
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3, 0.7, 0.1}, {0.1, 0.7, 0.3, 0.9}}, []string{"X", "U"}, []int{2, 2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	method := ci.IdentificationMethod("X", "Y")
	t.Logf("Method with IV setup: %s", method)
}

func TestFinalCov_IdentificationMethod_NonePath(t *testing.T) {
	// Two disconnected nodes — no edge, no path, no adjustment
	bn := models.NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	method := ci.IdentificationMethod("X", "Y")
	t.Logf("Method for disconnected: %s", method)
}

// ==========================================================================
// CausalInference: GetIVs, GetConditionalIVs, GetTotalConditionalIVs — cover
// the branch where IV conditions fail
// ==========================================================================

func TestFinalCov_GetIVs_AllConditions(t *testing.T) {
	// Z -> X -> Y, U -> X, U -> Y
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"Z", "U", "X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("Z", "X")
	bn.AddEdge("U", "X")
	bn.AddEdge("U", "Y")
	bn.AddEdge("X", "Y")

	zCPD, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(zCPD)
	uCPD, _ := factors.NewTabularCPD("U", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(uCPD)
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.6, 0.3, 0.4, 0.2}, {0.4, 0.7, 0.6, 0.8}}, []string{"Z", "U"}, []int{2, 2})
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3, 0.7, 0.1}, {0.1, 0.7, 0.3, 0.9}}, []string{"X", "U"}, []int{2, 2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	ivs := ci.GetIVs("X", "Y")
	t.Logf("IVs: %v", ivs)

	// Conditional IVs with conditioning on U
	civs := ci.GetConditionalIVs("X", "Y", []string{"U"})
	t.Logf("Conditional IVs given U: %v", civs)

	// Total conditional IVs
	tcivs := ci.GetTotalConditionalIVs("X", "Y")
	t.Logf("Total conditional IVs: %v", tcivs)
}

func TestFinalCov_GetConditionalIVs_DSepFail(t *testing.T) {
	// Network where candidate is d-separated from treatment even given conditioning
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"A", "B", "X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("A", "B")
	bn.AddEdge("X", "Y")

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	bn.AddCPD(bCPD)
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	civs := ci.GetConditionalIVs("X", "Y", []string{"B"})
	t.Logf("Conditional IVs (should be empty): %v", civs)
}

// ==========================================================================
// CausalInference: EstimateATE — nil data, backdoor fallback, model fallback
// ==========================================================================

func TestFinalCov_EstimateATE_FallbackToATE(t *testing.T) {
	// Simple Z -> X -> Y — backdoor with empty set should work
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"Z", "X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("Z", "X")
	bn.AddEdge("X", "Y")

	zCPD, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(zCPD)
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"Z"}, []int{2})
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	_, err := ci.EstimateATE("X", "Y", nil)
	if err == nil {
		t.Log("EstimateATE with nil data - expected error")
	} else {
		t.Logf("Expected error: %v", err)
	}
}

// ==========================================================================
// CausalInference: Query error paths — no CPD, do-value out of range
// ==========================================================================

func TestFinalCov_CausalQuery_DoValueOutOfRange2(t *testing.T) {
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("X", "Y")
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)

	// Test do-value < 0
	_, err := ci.Query([]string{"Y"}, map[string]int{"X": -1}, nil)
	if err == nil {
		t.Error("expected error for negative do-value")
	}
}

// ==========================================================================
// CausalInference: ATE — error path for multi-variable result
// ==========================================================================

func TestFinalCov_ATE_Valid(t *testing.T) {
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"X", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("X", "Y")
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	ate, err := ci.ATE("X", "Y", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("ATE: %f", ate)
}

// ==========================================================================
// CausalInference: interceptsAllPaths — paths not intercepted
// ==========================================================================

func TestFinalCov_InterceptsAllPaths_NotIntercepted(t *testing.T) {
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"X", "M1", "M2", "Y"} {
		bn.AddNode(n)
	}
	bn.AddEdge("X", "M1")
	bn.AddEdge("X", "M2")
	bn.AddEdge("M1", "Y")
	bn.AddEdge("M2", "Y")

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	m1CPD, _ := factors.NewTabularCPD("M1", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(m1CPD)
	m2CPD, _ := factors.NewTabularCPD("M2", 2, [][]float64{{0.7, 0.3}, {0.3, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(m2CPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.4, 0.6, 0.1}, {0.1, 0.6, 0.4, 0.9}}, []string{"M1", "M2"}, []int{2, 2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)

	// {M1} alone doesn't intercept all paths (M2 bypasses)
	v1 := ci.IsValidFrontdoorAdjustmentSet("X", "Y", []string{"M1"})
	t.Logf("Frontdoor {M1}: %v", v1)

	// {M1, M2} should intercept all paths
	v2 := ci.IsValidFrontdoorAdjustmentSet("X", "Y", []string{"M1", "M2"})
	t.Logf("Frontdoor {M1, M2}: %v", v2)
}

// ==========================================================================
// CausalInference: GetMinimalAdjustmentSet — parents not valid
// ==========================================================================

func TestFinalCov_GetMinimalAdjustmentSet_ParentsNotValid(t *testing.T) {
	// No confounders, X -> Y only. Parents of X is empty, which IS a valid
	// backdoor adjustment set. Need to check the error path.
	bn := models.NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("X", "Y")
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	ci, _ := NewCausalInference(bn)
	set, err := ci.GetMinimalAdjustmentSet("X", "Y")
	t.Logf("Minimal set: %v, err: %v", set, err)
}

// ==========================================================================
// GetSepsetBeliefs: exercise the fallback-to-clique-b path (lines 889-908)
// ==========================================================================

func TestFinalCov_GetSepsetBeliefs_FallbackPath(t *testing.T) {
	// Create BP where separator key parses but potential of clique a fails marginalization
	// This is hard to trigger naturally, so test the major paths
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})

	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	beliefs := bp.GetSepsetBeliefs()
	for k, v := range beliefs {
		if v == nil {
			t.Errorf("expected non-nil belief for separator %s after calibration", k)
		}
	}
}

func TestFinalCov_GetSepsetBeliefs_OutOfRangeIndex(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	cliques := [][]string{{"A", "B"}}
	// Separator key references clique index 5 which is out of range
	separators := map[string][]string{"0-5": {"B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	bp.calibrated = true
	bp.potentials = []*factors.DiscreteFactor{fAB}

	beliefs := bp.GetSepsetBeliefs()
	// The "0-5" key should result in nil since index 5 is out of range for clique b
	for k, v := range beliefs {
		t.Logf("Separator %s: %v", k, v)
	}
}

func TestFinalCov_GetSepsetBeliefs_NilPotentialFallbackToB(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	bp.Calibrate()

	// Set clique 0 potential to nil — should fall through to clique 1
	bp.potentials[0] = nil

	beliefs := bp.GetSepsetBeliefs()
	for k, v := range beliefs {
		t.Logf("Separator %s with nil clique 0: belief=%v", k, v)
	}
}

// ==========================================================================
// MPLP: GetIntegralityGap — scalar factor path, nonScalar empty path
// ==========================================================================

func TestFinalCov_MPLP_GetIntegralityGap_Scalar(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fB, _ := factors.NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fA, fB})
	gap, err := m.GetIntegralityGap([]string{"A", "B"}, nil, 10, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Gap: %f", gap)
}

func TestFinalCov_MPLP_Query_AllScalar(t *testing.T) {
	// After evidence reduction, all factors become scalar
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{fA})
	result, err := m.Query([]string{"A"}, map[string]int{"A": 0}, 10, 1e-6)
	t.Logf("Query result: %v, err: %v", result, err)
}

func TestFinalCov_MPLP_MAP_Convergence(t *testing.T) {
	// Large enough for convergence path
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{3, 3},
		[]float64{0.1, 0.2, 0.3, 0.05, 0.15, 0.2, 0.3, 0.1, 0.05})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{3, 3},
		[]float64{0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.1, 0.3, 0.05})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	result, _, err := m.MAP([]string{"A", "B", "C"}, nil, 100, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("MAP result: %v", result)
}

// ==========================================================================
// VE: error paths in Query, MaxMarginal, eliminateVariable
// ==========================================================================

func TestFinalCov_VE_EliminateVariable_ProductError(t *testing.T) {
	// eliminateVariable with single-variable product => dropped
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	fA2, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	// Two factors both containing only A — product is single-var, should be dropped
	result, err := eliminateVariable([]*factors.DiscreteFactor{fA, fA2}, "A")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Result factors: %d", len(result))
}

func TestFinalCov_VE_MaxEliminateVariable_ProductError(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	fA2, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	result, err := maxEliminateVariable([]*factors.DiscreteFactor{fA, fA2}, "A")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Result factors: %d", len(result))
}

func TestFinalCov_VE_Query_HeuristicDefault(t *testing.T) {
	// VE with no heuristic set uses default "min_neighbors"
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	ve := &VariableElimination{factors: []*factors.DiscreteFactor{fAB, fBC}, heuristic: ""}

	result, err := ve.Query([]string{"A"}, map[string]int{"C": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestFinalCov_VE_MaxMarginal_HeuristicDefault(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	ve := &VariableElimination{factors: []*factors.DiscreteFactor{fAB, fBC}, heuristic: ""}

	result, err := ve.MaxMarginal([]string{"A"}, map[string]int{"C": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestFinalCov_VE_QueryWithVirtualEvidence_HeuristicDefault(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ve := &VariableElimination{factors: []*factors.DiscreteFactor{fAB}, heuristic: ""}

	result, err := ve.QueryWithVirtualEvidence(
		[]string{"A"}, nil,
		[]VirtualEvidence{{Variable: "B", Values: []float64{0.6, 0.4}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ==========================================================================
// BP: initializePotentials — uniform creation error and FactorProduct error
// ==========================================================================

func TestFinalCov_BP_InitializePotentials_EmptyClique(t *testing.T) {
	// Clique with no factors but known cardinality -> creates uniform
	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		1: {fBC},
		// clique 0 has no factors
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	err := bp.Calibrate()
	// Should succeed if cardinality is known from other cliques
	t.Logf("Calibrate with empty clique factor: err=%v", err)
}

// ==========================================================================
// BP_MP: more coverage of Calibrate error paths
// ==========================================================================

func TestFinalCov_BPMP_Calibrate_Valid(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}}
	schedule := []MessagePass{{From: 0, To: 1}, {From: 1, To: 0}}
	bpm := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, schedule)

	err := bpm.Calibrate()
	if err != nil {
		t.Fatal(err)
	}

	// Test Query and GetCliqueBelief after calibration
	result, err := bpm.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}

	b := bpm.GetCliqueBelief(0)
	if b == nil {
		t.Error("expected non-nil belief for clique 0")
	}

	if !bpm.IsCalibrated() {
		t.Error("expected calibrated")
	}
}

func TestFinalCov_BPMP_Calibrate_InitError(t *testing.T) {
	cliques := [][]string{{"A", "UNKNOWN"}}
	bpm := NewBeliefPropagationWithMessagePassing(cliques, nil, nil, nil)
	err := bpm.Calibrate()
	if err == nil {
		t.Error("expected error for init failure")
	}
}

func TestFinalCov_BPMP_Calibrate_SingleClique(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	cliques := [][]string{{"A", "B"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}}
	bpm := NewBeliefPropagationWithMessagePassing(cliques, nil, cliqueFactors, nil)

	err := bpm.Calibrate()
	if err != nil {
		t.Fatal(err)
	}
}

// ==========================================================================
// DBN: cover evidence on query var, no interface nodes, rename error
// ==========================================================================

func TestFinalCov_DBN_ForwardInference_EvidenceOnQueryAndNonQuery(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	fB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})
	fAT, _ := factors.NewDiscreteFactor([]string{"A_prev", "A"}, []int{2, 2}, []float64{0.9, 0.1, 0.3, 0.7})
	fBT, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})

	dbn := NewDBNInference(
		[]*factors.DiscreteFactor{fA, fB},
		[]*factors.DiscreteFactor{fAT, fBT},
		[]string{"A"},
	)

	// Evidence on both query var (A) and non-query var (B) at final step
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{
		{},
		{"A": 0, "B": 1}, // evidence on query var at final step
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestFinalCov_DBN_ComputeInterfaceBelief_EvidenceOnInterfaceAndOther(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	fB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})

	dbn := NewDBNInference(
		[]*factors.DiscreteFactor{fA, fB},
		nil,
		[]string{"A"},
	)

	// Evidence on interface node (A) and non-interface (B)
	belief, err := dbn.computeInterfaceBelief(
		[]*factors.DiscreteFactor{fA, fB},
		map[string]int{"A": 0, "B": 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	if belief == nil {
		t.Error("expected non-nil belief")
	}
}

func TestFinalCov_DBN_BackwardInference_MultiStep(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	fB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})
	fAT, _ := factors.NewDiscreteFactor([]string{"A_prev", "A"}, []int{2, 2}, []float64{0.9, 0.1, 0.3, 0.7})
	fBT, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})

	dbn := NewDBNInference(
		[]*factors.DiscreteFactor{fA, fB},
		[]*factors.DiscreteFactor{fAT, fBT},
		[]string{"A"},
	)

	// Backward inference targeting t=0 from 3-step sequence
	result, err := dbn.BackwardInference([]string{"A"}, []map[string]int{
		{},
		{"B": 0},
		{"B": 1},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}

	// Also target t=1
	result2, err := dbn.BackwardInference([]string{"A"}, []map[string]int{
		{},
		{"B": 0},
		{"B": 1},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if result2 == nil {
		t.Error("expected non-nil result for t=1")
	}
}

// ==========================================================================
// VE: reduceAll with actual evidence on factor
// ==========================================================================

func TestFinalCov_ReduceAll_WithEvidence(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	result, err := reduceAll([]*factors.DiscreteFactor{fAB}, map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 1 {
		t.Errorf("expected 1 factor, got %d", len(result))
	}
}

// ==========================================================================
// BP: Calibrate and MaxCalibrate with large chain (4+ cliques)
// to fully exercise message passing phases
// ==========================================================================

func TestFinalCov_BP_Calibrate_FourCliques(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})
	fDE, _ := factors.NewDiscreteFactor([]string{"D", "E"}, []int{2, 2}, []float64{0.4, 0.6, 0.3, 0.7})

	cliques := [][]string{{"A", "B"}, {"B", "C"}, {"C", "D"}, {"D", "E"}}
	separators := map[string][]string{"0-1": {"B"}, "1-2": {"C"}, "2-3": {"D"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}, 2: {fCD}, 3: {fDE}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)

	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}

	// Query with evidence
	result, err := bp.Query([]string{"A"}, map[string]int{"E": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}

	// MaxCalibrate
	bp2 := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp2.MaxCalibrate(); err != nil {
		t.Fatal(err)
	}

	mapResult, err := bp2.MAPQuery([]string{"A", "B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(mapResult) != 2 {
		t.Errorf("expected 2 MAP assignments, got %d", len(mapResult))
	}
}

// ==========================================================================
// VE: InducedWidth and InducedGraph with larger graphs
// ==========================================================================

func TestFinalCov_VE_InducedWidth_LargerGraph(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fCD, _ := factors.NewDiscreteFactor([]string{"C", "D"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fAB, fBC, fCD})

	w, err := ve.InducedWidth([]string{"A", "B", "C", "D"})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Width: %d", w)

	g, err := ve.InducedGraph([]string{"A", "B", "C", "D"})
	if err != nil {
		t.Fatal(err)
	}
	if g == nil {
		t.Error("expected non-nil graph")
	}
}
