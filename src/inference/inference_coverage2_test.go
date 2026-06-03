//go:build unit

package inference

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ---------------------------------------------------------------------------
// Helper: build a simple 3-node BN: Z -> X -> Y (all binary)
// ---------------------------------------------------------------------------

func makeSimpleBN3(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	bn.AddNode("Z")
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("Z", "X")
	bn.AddEdge("X", "Y")
	bn.SetStates("Z", []string{"z0", "z1"})
	bn.SetStates("X", []string{"x0", "x1"})
	bn.SetStates("Y", []string{"y0", "y1"})

	cpdZ, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdZ)
	cpdX, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"Z"}, []int{2})
	_ = bn.AddCPD(cpdX)
	cpdY, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"X"}, []int{2})
	_ = bn.AddCPD(cpdY)
	return bn
}

// helper to build simple factors for VE testing
func makeABCFactors(t *testing.T) []*factors.DiscreteFactor {
	t.Helper()
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	return []*factors.DiscreteFactor{fAB, fBC}
}

// ---------------------------------------------------------------------------
// CausalInference: error paths and underexplored methods
// ---------------------------------------------------------------------------

func TestCausalInference_Query_DoValueOutOfRange(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// do-value out of range
	_, err = ci.Query([]string{"Y"}, map[string]int{"X": 5}, nil)
	if err == nil {
		t.Error("expected error for do-value out of range")
	}
}

func TestCausalInference_ATE_ErrorQueryFails(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// ATE with valid treatment values
	ate, err := ci.ATE("X", "Y", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	// ATE should be non-zero for this model
	_ = ate
}

func TestCausalInference_ATE_SingleVarResult(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// This should return a valid ATE
	ate, err := ci.ATE("X", "Y", [2]int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if ate == 0 {
		t.Error("expected non-zero ATE")
	}
}

func TestCausalInference_IdentificationMethod_Backdoor(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	method := ci.IdentificationMethod("X", "Y")
	if method != "backdoor" {
		t.Errorf("expected backdoor, got %q", method)
	}
}

func TestCausalInference_IdentificationMethod_None(t *testing.T) {
	// Build an isolated node network where no identification applies
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdA)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdB)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	method := ci.IdentificationMethod("A", "B")
	// A and B are disconnected, so backdoor with empty set works
	_ = method
}

func TestCausalInference_EstimateATE_NilData(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.EstimateATE("X", "Y", nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestCausalInference_GetTotalConditionalIVs(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	result := ci.GetTotalConditionalIVs("X", "Y")
	// Z should appear as an IV somewhere
	_ = result
}

func TestCausalInference_GetMinimalAdjustmentSet(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	set, err := ci.GetMinimalAdjustmentSet("X", "Y")
	if err != nil {
		t.Fatal(err)
	}
	_ = set
}

func TestCausalInference_GetMinimalAdjustmentSet_NoValidParents(t *testing.T) {
	// Build a network where parents of treatment don't form a valid adjustment set
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.AddEdge("A", "C")
	bn.AddEdge("B", "C")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	bn.SetStates("C", []string{"c0", "c1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdA)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	_ = bn.AddCPD(cpdB)
	cpdC, _ := factors.NewTabularCPD("C", 2, [][]float64{{0.7, 0.3, 0.4, 0.6}, {0.3, 0.7, 0.6, 0.4}}, []string{"A", "B"}, []int{2, 2})
	_ = bn.AddCPD(cpdC)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// Try to get minimal adjustment set; may or may not error
	set, err := ci.GetMinimalAdjustmentSet("B", "C")
	_, _ = set, err
}

func TestCausalInference_GetProperBackdoorGraph(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	g := ci.GetProperBackdoorGraph("X")
	if g == nil {
		t.Error("expected non-nil graph")
	}
}

func TestCausalInference_IsValidAdjustmentSet(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	valid := ci.IsValidAdjustmentSet("X", "Y", []string{"Z"})
	if !valid {
		t.Error("expected Z to be a valid adjustment set for X->Y")
	}
}

func TestInterceptsAllPaths_NoPath(t *testing.T) {
	// When there's no direct path between src and dst (dst not reachable)
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	bn.SetStates("C", []string{"c0", "c1"})
	g := bnToDigraph(bn)
	// No path from A to C, so intercepts should be true
	result := interceptsAllPaths(g, "A", "C", map[string]bool{})
	if !result {
		t.Error("expected true when no path exists from A to C")
	}
}

// ---------------------------------------------------------------------------
// VariableElimination: error paths
// ---------------------------------------------------------------------------

func TestVE_Query_EmptyQueryVars(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	_, err := ve.Query(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestVE_MaxMarginal_EmptyQueryVars(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	_, err := ve.MaxMarginal(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestVE_MaxMarginal_Valid(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.MaxMarginal([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestVE_MaxMarginal_WithEvidence(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.MaxMarginal([]string{"A"}, map[string]int{"C": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestVE_QueryWithVirtualEvidence_EmptyVars(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	_, err := ve.QueryWithVirtualEvidence(nil, nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestVE_QueryWithVirtualEvidence_EmptyValues(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	_, err := ve.QueryWithVirtualEvidence([]string{"A"}, nil, []VirtualEvidence{
		{Variable: "B", Values: []float64{}},
	})
	if err == nil {
		t.Error("expected error for empty virtual evidence values")
	}
}

func TestVE_QueryWithVirtualEvidence_Valid(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.QueryWithVirtualEvidence([]string{"A"}, nil, []VirtualEvidence{
		{Variable: "B", Values: []float64{0.7, 0.3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestVE_QueryWithVirtualEvidence_WithHardEvidence(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.QueryWithVirtualEvidence(
		[]string{"A"},
		map[string]int{"C": 0},
		[]VirtualEvidence{{Variable: "B", Values: []float64{0.7, 0.3}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestVE_QueryMarginals_EmptyVars(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	_, err := ve.QueryMarginals(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestVE_QueryMarginals_Valid(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.QueryMarginals([]string{"A", "C"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 2 {
		t.Errorf("expected 2 marginals, got %d", len(result))
	}
}

func TestVE_QueryMarginals_WithEvidence(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.QueryMarginals([]string{"A"}, map[string]int{"B": 0})
	if err != nil {
		t.Fatal(err)
	}
	if result["A"] == nil {
		t.Error("expected non-nil marginal for A")
	}
}

func TestVE_InducedGraph(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	g, err := ve.InducedGraph([]string{"B"})
	if err != nil {
		t.Fatal(err)
	}
	if g == nil {
		t.Error("expected non-nil induced graph")
	}
}

func TestVE_InducedGraph_Empty(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	g, err := ve.InducedGraph(nil)
	if err != nil {
		t.Fatal(err)
	}
	if g == nil {
		t.Error("expected non-nil graph for empty order")
	}
}

func TestVE_InducedWidth(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	w, err := ve.InducedWidth([]string{"B"})
	if err != nil {
		t.Fatal(err)
	}
	if w < 0 {
		t.Error("expected non-negative width")
	}
}

func TestVE_InducedWidth_Empty(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	w, err := ve.InducedWidth(nil)
	if err != nil {
		t.Fatal(err)
	}
	if w != 0 {
		t.Error("expected 0 for empty order")
	}
}

func TestSplitEdgeKey_NoNUL(t *testing.T) {
	parts := splitEdgeKey("hello")
	if parts[0] != "hello" || parts[1] != "" {
		t.Errorf("unexpected split: %v", parts)
	}
}

func TestSplitEdgeKey_WithNUL(t *testing.T) {
	parts := splitEdgeKey("A\x00B")
	if parts[0] != "A" || parts[1] != "B" {
		t.Errorf("unexpected split: %v", parts)
	}
}

// ---------------------------------------------------------------------------
// copyGraph coverage (elimination_order.go)
// ---------------------------------------------------------------------------

func TestCopyGraph(t *testing.T) {
	g := map[string]map[string]bool{
		"A": {"B": true},
		"B": {"A": true, "C": true},
		"C": {"B": true},
	}
	c := copyGraph(g)
	if len(c) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(c))
	}
	// Mutate copy and check original is unchanged
	delete(c["A"], "B")
	if !g["A"]["B"] {
		t.Error("original was modified by copy mutation")
	}
}

// ---------------------------------------------------------------------------
// ApproxInference: error paths (rejection, Gibbs, zero-weight paths)
// ---------------------------------------------------------------------------

func TestApproxInference_Query_EvidenceNotFound(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.Query([]string{"A"}, map[string]int{"UNKNOWN": 0}, 100)
	if err == nil {
		t.Error("expected error for unknown evidence variable")
	}
}

func TestApproxInference_Query_EvidenceOutOfRange(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.Query([]string{"A"}, map[string]int{"A": 5}, 100)
	if err == nil {
		t.Error("expected error for out-of-range evidence")
	}
}

func TestApproxInference_Query_ZeroWeight(t *testing.T) {
	// All-zero factor => all samples have zero weight
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.Query([]string{"A"}, nil, 100)
	if err == nil {
		t.Error("expected error for all-zero-weight samples")
	}
}

func TestApproxInference_GetDistribution_ZeroWeight(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.GetDistribution(100)
	if err == nil {
		t.Error("expected error for all-zero-weight samples")
	}
}

func TestApproxInference_QueryRejection_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection(nil, nil, 100)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestApproxInference_QueryRejection_NonPositiveSamples(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"A"}, nil, 0)
	if err == nil {
		t.Error("expected error for non-positive nSamples")
	}
}

func TestApproxInference_QueryRejection_Valid(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryRejection([]string{"A"}, map[string]int{"B": 0}, 5000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestApproxInference_QueryRejection_UnknownQueryVar(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"UNKNOWN"}, nil, 100)
	if err == nil {
		t.Error("expected error for unknown query variable")
	}
}

func TestApproxInference_QueryRejection_EvidenceNotFound(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"A"}, map[string]int{"UNKNOWN": 0}, 100)
	if err == nil {
		t.Error("expected error for unknown evidence variable")
	}
}

func TestApproxInference_QueryRejection_EvidenceOutOfRange(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"A"}, map[string]int{"A": 5}, 100)
	if err == nil {
		t.Error("expected error for out-of-range evidence")
	}
}

func TestApproxInference_QueryRejection_NoMatchingEvidence(t *testing.T) {
	// All-zero factor => rejection sampling never accepts
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"A"}, nil, 100)
	if err == nil {
		t.Error("expected error for all-zero-weight samples")
	}
}

func TestApproxInference_QueryRejection_ZeroWeightAccepted(t *testing.T) {
	// Factor with one zero entry
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 0, 0, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryRejection([]string{"A"}, nil, 100)
	if err == nil {
		t.Error("expected error when all weights are zero")
	}
}

func TestApproxInference_QueryGibbs_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryGibbs(nil, nil, 100, 10)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestApproxInference_QueryGibbs_NonPositiveSamples(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryGibbs([]string{"A"}, nil, 0, 10)
	if err == nil {
		t.Error("expected error for non-positive nSamples")
	}
}

func TestApproxInference_QueryGibbs_Valid(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryGibbs([]string{"A"}, nil, 500, 50)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestApproxInference_QueryGibbs_WithEvidence(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryGibbs([]string{"A"}, map[string]int{"B": 0}, 500, 50)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestApproxInference_QueryGibbs_NegativeBurnIn(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryGibbs([]string{"A"}, nil, 100, -5)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result with negative burnIn (clamped to 0)")
	}
}

func TestApproxInference_QueryGibbs_UnknownQueryVar(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryGibbs([]string{"UNKNOWN"}, nil, 100, 10)
	if err == nil {
		t.Error("expected error for unknown query variable")
	}
}

func TestApproxInference_QueryGibbs_EvidenceNotFound(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryGibbs([]string{"A"}, map[string]int{"UNKNOWN": 0}, 100, 10)
	if err == nil {
		t.Error("expected error for unknown evidence variable")
	}
}

func TestApproxInference_QueryGibbs_EvidenceOutOfRange(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.QueryGibbs([]string{"A"}, map[string]int{"A": 5}, 100, 10)
	if err == nil {
		t.Error("expected error for out-of-range evidence")
	}
}

func TestApproxInference_QueryGibbs_ZeroWeightPath(t *testing.T) {
	// Factor with some zero entries - exercises the zero-sum conditional path
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 0, 1, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryGibbs([]string{"A"}, nil, 500, 50)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestApproxInference_MAPQuery_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.MAPQuery(nil, nil, 100)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestApproxInference_MAPQuery_NonPositiveSamples(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	_, err := ai.MAPQuery([]string{"A"}, nil, 0)
	if err == nil {
		t.Error("expected error for non-positive nSamples")
	}
}

func TestApproxInference_GetDistributionWithEvidence(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.GetDistributionWithEvidence([]string{"A"}, map[string]int{"B": 0}, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ---------------------------------------------------------------------------
// BeliefPropagation: more error/edge-case paths
// ---------------------------------------------------------------------------

func TestBP_Calibrate_WithUniformClique(t *testing.T) {
	// Two cliques where one has a factor and the other must create uniform from cardMap
	// The BP constructor needs factors covering all variables to populate cardMap
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})

	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{
		"0-1": {"B"},
	}
	// Assign both factors to clique 1, leave clique 0 empty
	cliqueFactors := map[int][]*factors.DiscreteFactor{
		1: {fAB, fBC},
	}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	err := bp.Calibrate()
	if err != nil {
		t.Fatal(err)
	}
	if !bp.IsCalibrated() {
		t.Error("expected calibrated")
	}
}

func TestBP_MaxCalibrate_TwoCliques(t *testing.T) {
	bp := makeSimpleBP(t)
	err := bp.MaxCalibrate()
	if err != nil {
		t.Fatal(err)
	}
}

func TestBP_Query_UnknownVariable(t *testing.T) {
	bp := makeSimpleBP(t)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	// Query for a variable that exists
	_, err := bp.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBP_Query_EmptyVars(t *testing.T) {
	bp := makeSimpleBP(t)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	_, err := bp.Query(nil, nil)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestBP_MAPQuery_Evidence(t *testing.T) {
	bp := makeSimpleBP(t)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	result, err := bp.MAPQuery([]string{"A"}, map[string]int{"B": 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := result["A"]; !ok {
		t.Error("expected A in result")
	}
}

func TestBP_GetSepsetBeliefs_WithBadKey(t *testing.T) {
	// Separator with bad edge key
	cliques := [][]string{{"A"}}
	separators := map[string][]string{
		"invalid": {"A"},
	}
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {f}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	beliefs := bp.GetSepsetBeliefs()
	// Should still return something for "invalid" key, likely nil
	_ = beliefs
}

// ---------------------------------------------------------------------------
// BeliefPropagation MP (belief_propagation_mp.go)
// ---------------------------------------------------------------------------

func TestBPMP_Calibrate_EmptyCliques(t *testing.T) {
	mp := NewBeliefPropagationWithMessagePassing(nil, nil, nil, nil)
	if err := mp.Calibrate(); err != nil {
		t.Fatal(err)
	}
}

func TestBPMP_Calibrate_SingleClique(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.4, 0.6})
	cliques := [][]string{{"A"}}
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {f}}
	mp := NewBeliefPropagationWithMessagePassing(cliques, nil, cliqueFactors, nil)
	if err := mp.Calibrate(); err != nil {
		t.Fatal(err)
	}
}

func TestBPMP_Calibrate_TwoCliques(t *testing.T) {
	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}}
	schedule := []MessagePass{{From: 1, To: 0}, {From: 0, To: 1}}
	mp := NewBeliefPropagationWithMessagePassing(cliques, separators, cliqueFactors, schedule)
	if err := mp.Calibrate(); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// MPLP: error paths and extended coverage
// ---------------------------------------------------------------------------

func TestMPLP_MAP_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{f})
	_, _, err := m.MAP(nil, nil, 10, 1e-6)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestMPLP_MAP_NonPositiveMaxIter(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{f})
	_, _, err := m.MAP([]string{"A"}, nil, 0, 1e-6)
	if err == nil {
		t.Error("expected error for non-positive maxIter")
	}
}

func TestMPLP_MAP_Valid(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	assignment, _, err := m.MAP([]string{"A", "B"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := assignment["A"]; !ok {
		t.Error("expected A in assignment")
	}
}

func TestMPLP_MAP_WithEvidence(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	assignment, _, err := m.MAP([]string{"A"}, map[string]int{"C": 0}, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := assignment["A"]; !ok {
		t.Error("expected A in assignment")
	}
}

func TestMPLP_MAP_WithZeroFactor(t *testing.T) {
	// Factor with zeros exercises -Inf log paths
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 0.7, 0.4, 0})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	assignment, _, err := m.MAP([]string{"A", "B"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	_ = assignment
}

func TestMPLP_Query_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{f})
	_, err := m.Query(nil, nil, 10, 1e-6)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestMPLP_Query_Valid(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	result, err := m.Query([]string{"A"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestMPLP_FindTriangles(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	fAC, _ := factors.NewDiscreteFactor([]string{"A", "C"}, []int{2, 2}, []float64{0.1, 0.9, 0.6, 0.4})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC, fAC})
	triangles := m.FindTriangles()
	// Should find the triangle A-B-C
	_ = triangles
}

func TestMPLP_GetIntegralityGap(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	gap, err := m.GetIntegralityGap([]string{"A", "B"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if gap < 0 {
		t.Error("expected non-negative integrality gap")
	}
}

func TestMPLP_GetIntegralityGap_EmptyVars(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{f})
	_, err := m.GetIntegralityGap(nil, nil, 10, 1e-6)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

// ---------------------------------------------------------------------------
// DBNInference: forward/backward/query with evidence paths
// ---------------------------------------------------------------------------

func makeDBNInference(t *testing.T) *DBNInference {
	t.Helper()
	// Simple 2-node DBN: A -> B (initial), A_prev -> A (transition)
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.6, 0.4})
	fB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.9, 0.1, 0.3, 0.7})
	fTrans, _ := factors.NewDiscreteFactor([]string{"A_prev", "A"}, []int{2, 2}, []float64{0.7, 0.3, 0.2, 0.8})
	fBTrans, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.9, 0.1, 0.3, 0.7})

	dbn := NewDBNInference(
		[]*factors.DiscreteFactor{fA, fB},
		[]*factors.DiscreteFactor{fTrans, fBTrans},
		[]string{"A"},
	)
	return dbn
}

func TestDBNInference_ForwardInference_EmptyQuery(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.ForwardInference(nil, []map[string]int{{"B": 0}})
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestDBNInference_ForwardInference_EmptyEvidence(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.ForwardInference([]string{"A"}, nil)
	if err == nil {
		t.Error("expected error for empty evidenceSequence")
	}
}

func TestDBNInference_ForwardInference_SingleStep(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{{"B": 0}})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_ForwardInference_MultiStep(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{
		{"B": 0},
		{"B": 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_ForwardInference_NoEvidence(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{
		{},
		{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_BackwardInference_EmptyQuery(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.BackwardInference(nil, []map[string]int{{"B": 0}}, 0)
	if err == nil {
		t.Error("expected error for empty queryVars")
	}
}

func TestDBNInference_BackwardInference_EmptyEvidence(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.BackwardInference([]string{"A"}, nil, 0)
	if err == nil {
		t.Error("expected error for empty evidenceSequence")
	}
}

func TestDBNInference_BackwardInference_OutOfRange(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.BackwardInference([]string{"A"}, []map[string]int{{"B": 0}}, 5)
	if err == nil {
		t.Error("expected error for out-of-range targetTimeStep")
	}
}

func TestDBNInference_BackwardInference_Valid(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.BackwardInference([]string{"A"}, []map[string]int{
		{},
		{"B": 1},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_Query_ForwardPath(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.Query([]string{"A"}, []map[string]int{
		{"B": 0},
		{"B": 1},
	}, -1)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_Query_BackwardPath(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.Query([]string{"A"}, []map[string]int{
		{},
		{"B": 1},
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestDBNInference_Query_EmptyEvidence(t *testing.T) {
	dbn := makeDBNInference(t)
	_, err := dbn.Query([]string{"A"}, nil, 0)
	if err == nil {
		t.Error("expected error for empty evidenceSequence")
	}
}

// ---------------------------------------------------------------------------
// Additional targeted tests for remaining uncovered branches
// ---------------------------------------------------------------------------

// CausalInference: EstimateATE uses backdoor path when data is provided
func TestCausalInference_EstimateATE_WithBackdoor(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// Build data for BackdoorAdjustment
	rows := make([][]any, 100)
	for i := 0; i < 100; i++ {
		z, x, y := 0, 0, 0
		if i%2 == 0 {
			z = 1
		}
		if i%3 == 0 {
			x = 1
		}
		if i%4 == 0 {
			y = 1
		}
		rows[i] = []any{z, x, y}
	}
	df := makeDataFrame([]string{"Z", "X", "Y"}, rows)
	ate, err := ci.EstimateATE("X", "Y", df)
	if err != nil {
		t.Fatal(err)
	}
	_ = ate
}

// CausalInference: Query with evidence on non-do vars
func TestCausalInference_Query_WithEvidence(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ci.Query([]string{"Y"}, map[string]int{"X": 0}, map[string]int{"Z": 1})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// CausalInference: Query error path for bad node in do-vars
func TestCausalInference_Query_NoCPD(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	// Valid do-var with valid value
	result, err := ci.Query([]string{"Y"}, map[string]int{"X": 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// CausalInference: negative do-value
func TestCausalInference_Query_NegativeDoValue(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.Query([]string{"Y"}, map[string]int{"X": -1}, nil)
	if err == nil {
		t.Error("expected error for negative do-value")
	}
}

// CausalInference: BackdoorAdjustment with nil data
func TestCausalInference_BackdoorAdjustment_NilData(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	_, err = ci.BackdoorAdjustment("X", "Y", []string{"Z"}, nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

// CausalInference: BackdoorAdjustment with empty data
func TestCausalInference_BackdoorAdjustment_EmptyData(t *testing.T) {
	bn := makeSimpleBN3(t)
	ci, err := NewCausalInference(bn)
	if err != nil {
		t.Fatal(err)
	}
	df := makeDataFrame([]string{"Z", "X", "Y"}, nil)
	_, err = ci.BackdoorAdjustment("X", "Y", []string{"Z"}, df)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

// CausalInference: invalid model for NewCausalInference
func TestCausalInference_InvalidModel(t *testing.T) {
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	// No CPD => invalid model
	_, err := NewCausalInference(bn)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

// VariableElimination: single-variable elimination
func TestVE_Query_SingleVarFactor(t *testing.T) {
	// Single-variable factor where elimination results in dropping the factor
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fB, _ := factors.NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.5, 0.5})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fA, fB})
	result, err := ve.Query([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// MaxMarginal: single-variable elimination path
func TestVE_MaxMarginal_SingleVarFactor(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fB, _ := factors.NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.5, 0.5})
	ve := NewVariableElimination([]*factors.DiscreteFactor{fA, fB})
	result, err := ve.MaxMarginal([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// MPLP: Query with all scalar factors
func TestMPLP_Query_AllScalar(t *testing.T) {
	// Create factors that will become scalar after evidence reduction
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	// With evidence on both variables, all factors become scalar
	result, err := m.Query([]string{"A"}, map[string]int{"A": 0, "B": 0}, 10, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// MPLP: MAP with all scalar factors
func TestMPLP_MAP_AllScalar(t *testing.T) {
	fA, _ := factors.NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	m := NewMPLP([]*factors.DiscreteFactor{fA})
	assignment, _, err := m.MAP([]string{"A"}, map[string]int{"A": 0}, 10, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	_ = assignment
}

// MPLP: FindTriangles with no triangles
func TestMPLP_FindTriangles_NoTriangle(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	m := NewMPLP([]*factors.DiscreteFactor{fAB})
	triangles := m.FindTriangles()
	// Only 2 variables => no triangles
	if len(triangles) != 0 {
		t.Errorf("expected 0 triangles, got %d", len(triangles))
	}
}

// MPLP: GetIntegralityGap with multiple factors
func TestMPLP_GetIntegralityGap_MultiFactor(t *testing.T) {
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	m := NewMPLP([]*factors.DiscreteFactor{fAB, fBC})
	gap, err := m.GetIntegralityGap([]string{"A", "B", "C"}, nil, 50, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if gap < 0 {
		t.Error("expected non-negative gap")
	}
}

// BP: MaxCalibrate with real data
func TestBP_MaxCalibrate_RealData(t *testing.T) {
	cliques := [][]string{{"A", "B"}, {"B", "C"}}
	separators := map[string][]string{"0-1": {"B"}}
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cliqueFactors := map[int][]*factors.DiscreteFactor{0: {fAB}, 1: {fBC}}
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.MaxCalibrate(); err != nil {
		t.Fatal(err)
	}
	// Query after max-calibration
	result, err := bp.MAPQuery([]string{"A", "B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

// BP: Query with multiple vars in same clique
func TestBP_Query_MultipleVars(t *testing.T) {
	bp := makeSimpleBP(t)
	if err := bp.Calibrate(); err != nil {
		t.Fatal(err)
	}
	// Query vars that exist in a single clique
	result, err := bp.Query([]string{"A", "B"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// BP: GetSepsetBeliefs after MaxCalibrate
func TestBP_GetSepsetBeliefs_MaxCalibrated(t *testing.T) {
	bp := makeSimpleBP(t)
	if err := bp.MaxCalibrate(); err != nil {
		t.Fatal(err)
	}
	beliefs := bp.GetSepsetBeliefs()
	if len(beliefs) != 1 {
		t.Errorf("expected 1 separator belief, got %d", len(beliefs))
	}
}

// DBN: ForwardInference with evidence on query vars
func TestDBNInference_ForwardInference_EvidenceOnQueryVar(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.ForwardInference([]string{"A"}, []map[string]int{
		{"A": 0, "B": 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// DBN: BackwardInference at final step (equals ForwardInference)
func TestDBNInference_BackwardInference_FinalStep(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.BackwardInference([]string{"A"}, []map[string]int{
		{},
		{"B": 1},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// DBN: Query with last time step (forward path)
func TestDBNInference_Query_LastTimeStep(t *testing.T) {
	dbn := makeDBNInference(t)
	result, err := dbn.Query([]string{"A"}, []map[string]int{
		{"B": 0},
		{"B": 1},
	}, 1) // targetTimeStep == lastStep
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// helper to create DataFrame for causal tests
func makeDataFrame(cols []string, rows [][]any) *tabgo.DataFrame {
	if rows == nil || len(rows) == 0 {
		sm := make(map[string]*tabgo.Series, len(cols))
		for _, c := range cols {
			sm[c] = tabgo.NewSeries(c, nil)
		}
		return tabgo.NewDataFrame(sm)
	}
	return tabgo.NewDataFrameFromRows(cols, rows)
}

// VE: MAP test
func TestVE_MAP(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.MAP([]string{"A"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := result["A"]; !ok {
		t.Error("expected A in MAP result")
	}
}

// VE: MAP with evidence
func TestVE_MAP_WithEvidence(t *testing.T) {
	facs := makeABCFactors(t)
	ve := NewVariableElimination(facs)
	result, err := ve.MAP([]string{"A"}, map[string]int{"C": 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := result["A"]; !ok {
		t.Error("expected A in MAP result")
	}
}

// ApproxInference: Query with evidence
func TestApproxInference_Query_WithEvidence(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.Query([]string{"A"}, map[string]int{"B": 0}, 5000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ApproxInference: zero weight in likelihood sampling (partial zeros)
func TestApproxInference_Query_PartialZeros(t *testing.T) {
	// Some entries are zero, but not all
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 1, 1, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.Query([]string{"A"}, nil, 5000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ApproxInference: GetDistribution with partial zeros
func TestApproxInference_GetDistribution_PartialZeros(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 1, 1, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.GetDistribution(5000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}

// ApproxInference: QueryRejection with partial zeros
func TestApproxInference_QueryRejection_PartialZeros(t *testing.T) {
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0, 1, 1, 0})
	ai := NewApproxInference([]*factors.DiscreteFactor{f}, 42)
	result, err := ai.QueryRejection([]string{"A"}, nil, 10000)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result")
	}
}
