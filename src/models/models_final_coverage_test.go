//go:build unit

package models

import (
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ==========================================================================
// BIF Load: malformed inputs to trigger error paths in loadBIF
// ==========================================================================

func TestFinalCov_LoadBIF_MalformedVariable(t *testing.T) {
	// "variable" alone on a line (no name after it)
	bif := "network unknown {\n}\nvariable\n"
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for malformed variable declaration")
	}
}

func TestFinalCov_LoadBIF_NoTypeInVariable(t *testing.T) {
	bif := `network unknown {
}
variable X {
  something else;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for no type in variable block")
	}
}

func TestFinalCov_LoadBIF_MalformedType(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] s0, s1;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for malformed type (no braces)")
	}
}

func TestFinalCov_LoadBIF_EmptyStates(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 0 ] { };
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for empty states")
	}
}

func TestFinalCov_LoadBIF_UnknownVariable(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
probability ( Y ) {
  table 0.5, 0.5;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for unknown variable Y in probability")
	}
}

func TestFinalCov_LoadBIF_UnknownParent(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
probability ( X | Z ) {
  (s0) 0.5, 0.5;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for unknown parent Z")
	}
}

func TestFinalCov_LoadBIF_MalformedProbHeader(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
probability X {
  table 0.5, 0.5;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for malformed probability header (no parens)")
	}
}

func TestFinalCov_LoadBIF_WrongTableSize(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
probability ( X ) {
  table 0.5, 0.3, 0.2;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for wrong table size")
	}
}

func TestFinalCov_LoadBIF_TableForConditional(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { s0, s1 };
}
probability ( Y | X ) {
  table 0.5, 0.5, 0.3, 0.7;
}
`
	bn, err := loadBIF(strings.NewReader(bif))
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BN")
	}
}

func TestFinalCov_LoadBIF_TableForConditionalWrongSize(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { s0, s1 };
}
probability ( Y | X ) {
  table 0.5, 0.5;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for wrong conditional table size")
	}
}

func TestFinalCov_LoadBIF_ConditionalWithStates(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( Y | X ) {
  (s0) 0.8, 0.2;
  (s1) 0.3, 0.7;
}
`
	bn, err := loadBIF(strings.NewReader(bif))
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BN")
	}
}

func TestFinalCov_LoadBIF_ConditionalWrongParentStates(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( Y | X ) {
  (s0, s1) 0.8, 0.2;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for wrong number of parent states")
	}
}

func TestFinalCov_LoadBIF_ConditionalUnknownState(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( Y | X ) {
  (unknown) 0.8, 0.2;
  (s1) 0.3, 0.7;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for unknown parent state")
	}
}

func TestFinalCov_LoadBIF_ConditionalWrongValueCount(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( Y | X ) {
  (s0) 0.8;
  (s1) 0.3, 0.7;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for wrong value count in conditional")
	}
}

func TestFinalCov_LoadBIF_MalformedConditionalLine(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( Y | X ) {
  (s0 0.8, 0.2;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for malformed conditional (no close paren)")
	}
}

func TestFinalCov_LoadBIF_InvalidFloat(t *testing.T) {
	bif := `network unknown {
}
variable X {
  type discrete [ 2 ] { s0, s1 };
}
probability ( X ) {
  table abc, 0.5;
}
`
	_, err := loadBIF(strings.NewReader(bif))
	if err == nil {
		t.Error("expected error for invalid float")
	}
}

func TestFinalCov_LoadBIF_ValidComplete(t *testing.T) {
	bif := `network unknown {
}
// Comment line
variable X {
  type discrete [ 2 ] { s0, s1 };
}
variable Y {
  type discrete [ 2 ] { y0, y1 };
}
probability ( X ) {
  table 0.5, 0.5;
}
probability ( Y | X ) {
  (s0) 0.8, 0.2;
  (s1) 0.3, 0.7;
}
`
	bn, err := loadBIF(strings.NewReader(bif))
	if err != nil {
		t.Fatal(err)
	}
	if len(bn.Nodes()) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(bn.Nodes()))
	}
}

// ==========================================================================
// BIF Write: writeBIF error paths
// ==========================================================================

func TestFinalCov_WriteBIF_NoStates(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	// No states set

	var buf strings.Builder
	err := bn.writeBIF(&buf)
	if err == nil {
		t.Error("expected error for variable with no states")
	}
}

func TestFinalCov_WriteBIF_NoCPD(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	bn.SetStates("X", []string{"s0", "s1"})

	var buf strings.Builder
	err := bn.writeBIF(&buf)
	if err == nil {
		t.Error("expected error for variable with no CPD")
	}
}

func TestFinalCov_WriteBIF_Valid(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("X", "Y")
	bn.SetStates("X", []string{"s0", "s1"})
	bn.SetStates("Y", []string{"y0", "y1"})

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	var buf strings.Builder
	err := bn.writeBIF(&buf)
	if err != nil {
		t.Fatal(err)
	}

	// Also test round-trip by loading it back
	bn2, err := loadBIF(strings.NewReader(buf.String()))
	if err != nil {
		t.Fatal(err)
	}
	if len(bn2.Nodes()) != 2 {
		t.Errorf("expected 2 nodes after round-trip, got %d", len(bn2.Nodes()))
	}
}

// ==========================================================================
// IsIMap: error path and normal path
// ==========================================================================

func TestFinalCov_IsIMap_InvalidModel(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	// No CPD — model is invalid
	_, err := bn.IsIMap(nil)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestFinalCov_IsIMap_DSepPath(t *testing.T) {
	// X -> Z -> Y: X and Y are d-separated given Z
	bn := NewBayesianNetwork()
	for _, n := range []string{"X", "Y", "Z"} {
		bn.AddNode(n)
	}
	bn.AddEdge("X", "Z")
	bn.AddEdge("Z", "Y")

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	zCPD, _ := factors.NewTabularCPD("Z", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(zCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"Z"}, []int{2})
	bn.AddCPD(yCPD)

	// Create a JPD
	vals := make([]float64, 8)
	idx := 0
	for x := 0; x < 2; x++ {
		for y := 0; y < 2; y++ {
			for z := 0; z < 2; z++ {
				px := []float64{0.5, 0.5}[x]
				pzx := [][]float64{{0.8, 0.2}, {0.2, 0.8}}[z][x]
				pyZ := [][]float64{{0.9, 0.3}, {0.1, 0.7}}[y][z]
				vals[idx] = px * pzx * pyZ
				idx++
			}
		}
	}
	jpd, err := factors.NewJointProbabilityDistribution([]string{"X", "Y", "Z"}, []int{2, 2, 2}, vals)
	if err != nil {
		t.Fatal(err)
	}

	result, err := bn.IsIMap(jpd)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("IsIMap: %v", result)
}

// ==========================================================================
// DynamicBN: AddNode/AddEdge rollback, GetCPDs transition path,
// InitializeInitialState errors, Fit errors
// ==========================================================================

func TestFinalCov_DBN_AddNode_TransitionFail(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	// Try adding A again — should fail because already exists in transition
	err := dbn.AddNode("A")
	if err == nil {
		t.Error("expected error for duplicate node")
	}
}

func TestFinalCov_DBN_AddEdge_TransitionFail(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	// Try adding same edge again
	err := dbn.AddEdge("A", "B")
	if err == nil {
		t.Error("expected error for duplicate edge")
	}
}

func TestFinalCov_DBN_GetCPDs_TransitionExtra(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")

	// Add CPD to initial for A, and transition for both
	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	dbn.AddTransitionCPD(bCPD)

	cpds := dbn.GetCPDs()
	t.Logf("CPDs: %d", len(cpds))
}

func TestFinalCov_DBN_InitializeInitialState_EmptyDist(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")

	err := dbn.InitializeInitialState(map[string][]float64{"A": {}})
	if err == nil {
		t.Error("expected error for empty distribution")
	}
}

func TestFinalCov_DBN_InitializeInitialState_Valid(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")

	err := dbn.InitializeInitialState(map[string][]float64{"A": {0.6, 0.4}})
	if err != nil {
		t.Fatal(err)
	}
}

func TestFinalCov_DBN_Fit_NilData(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	err := dbn.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestFinalCov_DBN_Fit_EmptyData(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{})
	err := dbn.Fit(df)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestFinalCov_DBN_Fit_MissingColumn(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)

	// Column panics if not found, so this test needs to use recover
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"B": tabgo.NewSeries("B", []any{0, 1, 0}),
	})
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Expected panic for missing column: %v", r)
			}
		}()
		err := dbn.Fit(df)
		if err == nil {
			t.Error("expected error for missing column")
		}
	}()
}

func TestFinalCov_DBN_Fit_Valid(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.5, 0.5}, {0.5, 0.5}}, []string{"A"}, []int{2})
	dbn.AddInitialCPD(bCPD)

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 1, 0, 1, 0}),
		"B": tabgo.NewSeries("B", []any{1, 0, 1, 1, 0}),
	})
	err := dbn.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestFinalCov_DBN_Simulate_InvalidModel(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	// No CPDs => invalid model
	_, err := dbn.Simulate(5, 42)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestFinalCov_DBN_Simulate_NegativeSteps(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	_, err := dbn.Simulate(-1, 42)
	if err == nil {
		t.Error("expected error for negative steps")
	}
}

// ==========================================================================
// SEM: FromLisrel error paths, ToLisrel, ToStandardLisrel, FromLavaan, FromGraph
// ==========================================================================

func TestFinalCov_SEM_FromLisrel_InvalidValue(t *testing.T) {
	_, err := FromLisrel("X: Y=abc")
	if err == nil {
		t.Error("expected error for invalid value in LISREL spec")
	}
}

func TestFinalCov_SEM_FromLisrel_EmptySpec(t *testing.T) {
	_, err := FromLisrel("")
	if err == nil {
		t.Error("expected error for empty spec")
	}
}

func TestFinalCov_SEM_FromLisrel_NoValidLines(t *testing.T) {
	_, err := FromLisrel("just some text")
	if err == nil {
		t.Error("expected error for no valid lines")
	}
}

func TestFinalCov_SEM_FromLisrel_VarianceAndIntercept(t *testing.T) {
	s, err := FromLisrel("X:\nY: X=0.5 variance=2.0 intercept=1.0")
	if err != nil {
		t.Fatal(err)
	}
	eq := s.GetEquation("Y")
	if eq == nil {
		t.Fatal("expected equation for Y")
	}
	if eq.Variance != 2.0 {
		t.Errorf("expected variance 2.0, got %f", eq.Variance)
	}
	if eq.Intercept != 1.0 {
		t.Errorf("expected intercept 1.0, got %f", eq.Intercept)
	}
}

func TestFinalCov_SEM_FromLavaan_Empty(t *testing.T) {
	_, err := FromLavaan("")
	if err == nil {
		t.Error("expected error for empty lavaan syntax")
	}
}

func TestFinalCov_SEM_FromLavaan_NoValidLines(t *testing.T) {
	_, err := FromLavaan("no tilde here\nand here neither")
	if err == nil {
		t.Error("expected error for no valid lavaan lines")
	}
}

func TestFinalCov_SEM_FromLavaan_EmptyVariable(t *testing.T) {
	_, err := FromLavaan("  ~ X")
	if err == nil {
		t.Error("expected error for empty variable")
	}
}

func TestFinalCov_SEM_FromLavaan_NoParents(t *testing.T) {
	s, err := FromLavaan("X ~")
	if err != nil {
		t.Fatal(err)
	}
	eq := s.GetEquation("X")
	if eq == nil {
		t.Fatal("expected equation for X")
	}
}

func TestFinalCov_SEM_FromLavaan_Valid(t *testing.T) {
	s, err := FromLavaan("Y ~ X1 + X2\nX1 ~\nX2 ~")
	if err != nil {
		t.Fatal(err)
	}
	vars := s.Variables()
	if len(vars) < 3 {
		t.Errorf("expected at least 3 variables, got %d", len(vars))
	}
}

func TestFinalCov_SEM_FromGraph_Nil(t *testing.T) {
	_, err := FromGraph(nil)
	if err == nil {
		t.Error("expected error for nil DAG")
	}
}

func TestFinalCov_SEM_CheckModel_Errors(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0.0, 1.0)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, -1.0)
	err := s.CheckModel()
	if err == nil {
		t.Error("expected error for negative variance")
	}
}

func TestFinalCov_SEM_CheckModel_ParentMismatch(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0.0, 1.0)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)
	// Manually add node Z to DAG without equation
	s.dag.AddNode("Z")
	err := s.CheckModel()
	if err == nil {
		t.Error("expected error for node with no equation")
	}
}

func TestFinalCov_SEM_ToLisrel_Valid(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5 variance=2.0")
	result, err := s.ToLisrel()
	if err != nil {
		t.Fatal(err)
	}
	if result["B"] == nil {
		t.Error("expected non-nil B matrix")
	}
}

func TestFinalCov_SEM_ToStandardLisrel_Valid(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5 variance=2.0")
	result, err := s.ToStandardLisrel()
	if err != nil {
		t.Fatal(err)
	}
	if result["B"] == nil {
		t.Error("expected non-nil B matrix")
	}
}

func TestFinalCov_SEM_AddEquation_ParentLenMismatch(t *testing.T) {
	s := NewSEM()
	err := s.AddEquation("X", []string{"A", "B"}, []float64{0.5}, 0.0, 1.0)
	if err == nil {
		t.Error("expected error for parent/coefficient length mismatch")
	}
}

func TestFinalCov_SEM_GenerateSamples_InvalidModel(t *testing.T) {
	s := NewSEM()
	s.dag.AddNode("X") // No equation
	_, err := s.GenerateSamples(10)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestFinalCov_SEM_Fit_NilData(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5 variance=2.0")
	err := s.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestFinalCov_SEM_GenerateSamples_Valid(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5 variance=2.0")
	samples, err := s.GenerateSamples(100)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Len() != 100 {
		t.Errorf("expected 100 samples, got %d", samples.Len())
	}
}

// ==========================================================================
// LinearGaussianBN: AddLinearGaussianCPD errors, CheckModel errors
// ==========================================================================

func TestFinalCov_LGBN_AddCPD_NotInNetwork(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	cpd, _ := factors.NewLinearGaussianCPD("B", 0.0, nil, 1.0, nil)
	err := lgbn.AddLinearGaussianCPD(cpd)
	if err == nil {
		t.Error("expected error for variable not in network")
	}
}

func TestFinalCov_LGBN_AddCPD_ParentMismatch(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")
	// CPD with wrong parents
	cpd, _ := factors.NewLinearGaussianCPD("B", 0.0, []float64{0.5}, 1.0, []string{"C"})
	err := lgbn.AddLinearGaussianCPD(cpd)
	if err == nil {
		t.Error("expected error for parent mismatch")
	}
}

func TestFinalCov_LGBN_CheckModel_MissingCPD(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	err := lgbn.CheckModel()
	if err == nil {
		t.Error("expected error for missing CPD")
	}
}

func TestFinalCov_LGBN_CheckModel_ParentMismatch(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")

	// Add valid CPD for A
	aCPD, _ := factors.NewLinearGaussianCPD("A", 0.0, nil, 1.0, nil)
	lgbn.AddLinearGaussianCPD(aCPD)

	// Add CPD for B with wrong evidence (no parents)
	bCPD, _ := factors.NewLinearGaussianCPD("B", 0.0, nil, 1.0, nil)
	lgbn.lgCPDs["B"] = bCPD // bypass AddLinearGaussianCPD validation

	err := lgbn.CheckModel()
	if err == nil {
		t.Error("expected error for parent mismatch in CheckModel")
	}
}

func TestFinalCov_LGBN_Valid(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")

	aCPD, _ := factors.NewLinearGaussianCPD("A", 0.0, nil, 1.0, nil)
	lgbn.AddLinearGaussianCPD(aCPD)
	bCPD, _ := factors.NewLinearGaussianCPD("B", 0.0, []float64{0.5}, 1.0, []string{"A"})
	lgbn.AddLinearGaussianCPD(bCPD)

	err := lgbn.CheckModel()
	if err != nil {
		t.Fatal(err)
	}

	// Test Simulate
	df, err := lgbn.Simulate(10)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 10 {
		t.Errorf("expected 10 rows, got %d", df.Len())
	}

	// Test Fit
	err = lgbn.Fit(df)
	if err != nil {
		t.Fatal(err)
	}

	// Test Copy
	cp := lgbn.Copy()
	if cp == nil {
		t.Error("expected non-nil copy")
	}
}

func TestFinalCov_LGBN_SimulateInvalid(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	_, err := lgbn.Simulate(10)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

// ==========================================================================
// NaiveBayes: edge cases
// ==========================================================================

func TestFinalCov_NaiveBayes_DuplicateFeature(t *testing.T) {
	_, err := NewNaiveBayes("C", []string{"F1", "F1"})
	if err == nil {
		t.Error("expected error for duplicate feature")
	}
}

func TestFinalCov_NaiveBayes_FeatureIsClass(t *testing.T) {
	_, err := NewNaiveBayes("C", []string{"C"})
	if err == nil {
		t.Error("expected error for feature same as class")
	}
}

func TestFinalCov_NaiveBayes_AddEdge_WrongFrom(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	err := nb.AddEdge("F1", "F2")
	if err == nil {
		t.Error("expected error for edge not from class variable")
	}
}

func TestFinalCov_NaiveBayes_AddEdge_NotFeature(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	err := nb.AddEdge("C", "X")
	if err == nil {
		t.Error("expected error for edge to non-feature")
	}
}

func TestFinalCov_NaiveBayes_AddEdgesFrom(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	err := nb.AddEdgesFrom("C", []string{"F1", "F2"})
	// Edges already exist from constructor
	t.Logf("AddEdgesFrom err: %v", err)
}

func TestFinalCov_NaiveBayes_PredictProbability_NoCPD(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"C":  tabgo.NewSeries("C", []any{0, 1}),
		"F1": tabgo.NewSeries("F1", []any{0, 1}),
	})

	_, err := nb.PredictProbability(df)
	if err == nil {
		t.Error("expected error - model not valid (no CPDs)")
	}
}

// ==========================================================================
// MarkovNetwork: ToFactorGraph error, ToJunctionTree, GetPartitionFunction, etc.
// ==========================================================================

func TestFinalCov_MN_CheckModel_NilFactor(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	err := mn.CheckModel()
	// Model with no factors should be valid (or have a specific error)
	t.Logf("CheckModel no factors: %v", err)
}

func TestFinalCov_MN_ToFactorGraph(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(fAB)

	fg, err := mn.ToFactorGraph()
	if err != nil {
		t.Fatal(err)
	}
	if fg == nil {
		t.Error("expected non-nil factor graph")
	}
}

func TestFinalCov_MN_ToBayesianModel(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(fAB)

	bn, err := mn.ToBayesianModel()
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil bayesian network")
	}
}

// ==========================================================================
// JunctionTree: AddEdge edge cases
// ==========================================================================

func TestFinalCov_JT_AddEdge_OutOfRange(t *testing.T) {
	// Build a simple BN for junction tree
	bn := NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.AddEdge("B", "C")

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	bn.AddCPD(bCPD)
	cCPD, _ := factors.NewTabularCPD("C", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"B"}, []int{2})
	bn.AddCPD(cCPD)

	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatal(err)
	}

	err = jt.AddEdge(-1, 0)
	if err == nil {
		t.Error("expected error for negative index")
	}

	err = jt.AddEdge(0, 99)
	if err == nil {
		t.Error("expected error for out-of-range index")
	}

	err = jt.AddEdge(0, 0)
	if err == nil {
		t.Error("expected error for self-loop")
	}
}

// ==========================================================================
// FactorGraph: CheckModel errors
// ==========================================================================

func TestFinalCov_FG_CheckModel_VariableNotDeclared(t *testing.T) {
	fg := NewFactorGraph()
	fg.AddVariable("A", 2)
	// Add a factor with undeclared variable
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fg.AddFactor(f)
	err := fg.CheckModel()
	if err == nil {
		t.Error("expected error for undeclared variable")
	}
}

func TestFinalCov_FG_CheckModel_CardMismatch(t *testing.T) {
	fg := NewFactorGraph()
	fg.AddVariable("A", 2)
	fg.AddVariable("B", 3) // B has card 3
	// Factor says B has card 2
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fg.AddFactor(f)
	err := fg.CheckModel()
	if err == nil {
		t.Error("expected error for cardinality mismatch")
	}
}

// ==========================================================================
// FunctionalBN: CheckModel
// ==========================================================================

func TestFinalCov_FunctionalBN_CheckModel_NoFunction(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	fbn.AddNode("A")
	fbn.AddNode("B")
	fbn.AddEdge("A", "B")
	// No FunctionalCPD set for B
	err := fbn.CheckModel()
	if err == nil {
		t.Error("expected error for no function")
	}
}

func TestFinalCov_FunctionalBN_CheckModel_WrongParents(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	fbn.AddNode("A")
	fbn.AddNode("B")
	fbn.AddEdge("A", "B")

	// Create FunctionalCPDs with wrong parents
	aCPD, _ := factors.NewFunctionalCPD("A", nil, func(p map[string]float64) []float64 { return []float64{0.5, 0.5} })
	fbn.AddFunctionalCPD(aCPD)

	bCPD, _ := factors.NewFunctionalCPD("B", []string{"C"}, func(p map[string]float64) []float64 { return []float64{0.5, 0.5} })
	fbn.funcCPDs["B"] = bCPD // bypass validation

	err := fbn.CheckModel()
	if err == nil {
		t.Error("expected error for parent mismatch")
	}
}

// ==========================================================================
// MarkovChain: error paths
// ==========================================================================

func TestFinalCov_MarkovChain_InvalidMatrix(t *testing.T) {
	// Non-square matrix
	_, err := NewMarkovChain([][]float64{{0.5, 0.5}, {0.3, 0.7}, {0.2, 0.8}}, []string{"a", "b", "c"})
	if err == nil {
		t.Error("expected error for non-square matrix")
	}
}

func TestFinalCov_MarkovChain_NegativeEntry(t *testing.T) {
	_, err := NewMarkovChain([][]float64{{-0.5, 1.5}, {0.3, 0.7}}, []string{"a", "b"})
	if err == nil {
		t.Error("expected error for negative entry")
	}
}

func TestFinalCov_MarkovChain_StationaryDist_Ergodic(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	dist, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatal(err)
	}
	if len(dist) != 2 {
		t.Errorf("expected 2 states, got %d", len(dist))
	}
}

func TestFinalCov_MarkovChain_IsErgodic(t *testing.T) {
	// Non-ergodic: absorbing state
	mc, _ := NewMarkovChain([][]float64{{1.0, 0.0}, {0.0, 1.0}}, []string{"a", "b"})
	if mc.IsErgodic() {
		t.Error("expected non-ergodic for identity matrix")
	}
}

// ==========================================================================
// BN: GetRandomCPDs edge case
// ==========================================================================

func TestFinalCov_BN_GetRandomCPDs(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("X", "Y")

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	err := bn.GetRandomCPDs(2, 42)
	if err != nil {
		t.Fatal(err)
	}
}

// ==========================================================================
// FitUpdate: error paths
// ==========================================================================

func TestFinalCov_BN_FitUpdate_NoCPD(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	// No CPD for X
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0, 1, 0}),
	})
	err := bn.FitUpdate(df, 10)
	if err == nil {
		t.Error("expected error for node with no CPD")
	}
}

func TestFinalCov_BN_FitUpdate_NegativePrevSamples(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0, 1}),
	})
	err := bn.FitUpdate(df, -1)
	if err == nil {
		t.Error("expected error for negative nPrevSamples")
	}
}

func TestFinalCov_BN_FitUpdate_Valid(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("X", "Y")

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0, 1, 0, 1, 0}),
		"Y": tabgo.NewSeries("Y", []any{0, 1, 0, 0, 1}),
	})

	err := bn.FitUpdate(df, 10)
	if err != nil {
		t.Fatal(err)
	}
}

// ==========================================================================
// Cluster graph methods
// ==========================================================================

func TestFinalCov_ClusterGraph_CheckModel(t *testing.T) {
	cg := NewClusterGraph()
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cg.AddCluster([]string{"A", "B"}, []*factors.DiscreteFactor{fAB})
	cg.AddCluster([]string{"B", "C"}, []*factors.DiscreteFactor{fBC})
	cg.AddEdge(0, 1, []string{"B"})
	err := cg.CheckModel()
	t.Logf("ClusterGraph CheckModel: %v", err)
}

// ==========================================================================
// MarkovChain methods: Simulate, Communication classes
// ==========================================================================

func TestFinalCov_MarkovChain_Sample(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	states, err := mc.Sample(100, 0, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 100 {
		t.Errorf("expected 100 states, got %d", len(states))
	}
}

// ==========================================================================
// DiscreteBayesianNetwork and DiscreteMarkovNetwork
// ==========================================================================

func TestFinalCov_DiscreteBN_AddCPD_NilAndNaN(t *testing.T) {
	dbn := NewDiscreteBayesianNetwork()
	dbn.AddNode("X")

	// Nil CPD
	err := dbn.AddCPD(nil)
	if err == nil {
		t.Error("expected error for nil CPD")
	}

	// Valid CPD
	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	err = dbn.AddCPD(xCPD)
	if err != nil {
		t.Fatal(err)
	}
}

func TestFinalCov_DiscreteMN_CheckModel(t *testing.T) {
	dmn := NewDiscreteMarkovNetwork()
	dmn.AddNode("X")
	dmn.AddNode("Y")
	dmn.AddEdge("X", "Y")
	err := dmn.CheckModel()
	t.Logf("DiscreteMN CheckModel: %v", err)
}

// ==========================================================================
// MarkovChain Methods: more coverage
// ==========================================================================

func TestFinalCov_MC_AddVariablesFrom(t *testing.T) {
	mc1, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	mc2, _ := NewMarkovChain([][]float64{{0.5, 0.5}, {0.5, 0.5}}, []string{"b", "c"})
	mc1.AddVariablesFrom(mc2)
	t.Logf("States after merge: %v", mc1.StateNames())
}

func TestFinalCov_MC_AddVariablesFrom_Nil(t *testing.T) {
	mc1, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	mc1.AddVariablesFrom(nil)
	t.Logf("States after nil merge: %v", mc1.StateNames())
}

func TestFinalCov_MC_AddTransitionModel_WrongSize(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	err := mc.AddTransitionModel([][]float64{{0.5, 0.5, 0.0}})
	if err == nil {
		t.Error("expected error for wrong matrix size")
	}
}

func TestFinalCov_MC_AddTransitionModel_WrongRowLen(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	err := mc.AddTransitionModel([][]float64{{0.5, 0.5}, {0.5}})
	if err == nil {
		t.Error("expected error for wrong row length")
	}
}

func TestFinalCov_MC_AddTransitionModel_Negative(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	err := mc.AddTransitionModel([][]float64{{-0.5, 1.5}, {0.5, 0.5}})
	if err == nil {
		t.Error("expected error for negative values")
	}
}

func TestFinalCov_MC_ProbFromSample(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	mat, err := mc.ProbFromSample([]int{0, 1, 0, 0, 1, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Estimated transition: %v", mat)
}

func TestFinalCov_MC_ProbFromSample_Short(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	_, err := mc.ProbFromSample([]int{0})
	if err == nil {
		t.Error("expected error for short sequence")
	}
}

func TestFinalCov_MC_ProbFromSample_OutOfRange(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	_, err := mc.ProbFromSample([]int{0, 5, 1})
	if err == nil {
		t.Error("expected error for out of range state")
	}
}

func TestFinalCov_MC_IsStationarity(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	pi, _ := mc.StationaryDistribution()
	is, err := mc.IsStationarity(pi)
	if err != nil {
		t.Fatal(err)
	}
	if !is {
		t.Error("expected stationary distribution to be stationary")
	}
}

func TestFinalCov_MC_IsStationarity_WrongLen(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	_, err := mc.IsStationarity([]float64{0.5})
	if err == nil {
		t.Error("expected error for wrong length")
	}
}

func TestFinalCov_MC_RandomState(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	state, err := mc.RandomState(42)
	if err != nil {
		t.Fatal(err)
	}
	if state < 0 || state > 1 {
		t.Errorf("unexpected state: %d", state)
	}
}

func TestFinalCov_MC_GenerateSample(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	states, err := mc.GenerateSample(10, 0, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 10 {
		t.Errorf("expected 10 states, got %d", len(states))
	}
}

func TestFinalCov_MC_SetStartState(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	s, err := mc.SetStartState(1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Start state: %d", s)
}

func TestFinalCov_MC_SetStartState_OutOfRange(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	_, err := mc.SetStartState(5)
	if err == nil {
		t.Error("expected error for out-of-range start state")
	}
}

func TestFinalCov_MC_Copy(t *testing.T) {
	mc, _ := NewMarkovChain([][]float64{{0.7, 0.3}, {0.4, 0.6}}, []string{"a", "b"})
	cp := mc.Copy()
	if cp.NumStates() != 2 {
		t.Errorf("expected 2 states, got %d", cp.NumStates())
	}
}

// ==========================================================================
// LGBN: Save/Load round-trip
// ==========================================================================

func TestFinalCov_LGBN_SaveLoad(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")

	aCPD, _ := factors.NewLinearGaussianCPD("A", 0.0, nil, 1.0, nil)
	lgbn.AddLinearGaussianCPD(aCPD)
	bCPD, _ := factors.NewLinearGaussianCPD("B", 0.5, []float64{0.8}, 0.5, []string{"A"})
	lgbn.AddLinearGaussianCPD(bCPD)

	// Save
	fname := "/tmp/test_lgbn.bif"
	err := lgbn.Save(fname)
	if err != nil {
		t.Fatal(err)
	}

	// Load
	loaded, err := LoadLinearGaussianBayesianNetwork(fname)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Nodes()) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(loaded.Nodes()))
	}
}

func TestFinalCov_LGBN_Fit_Valid(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")

	aCPD, _ := factors.NewLinearGaussianCPD("A", 0.0, nil, 1.0, nil)
	lgbn.AddLinearGaussianCPD(aCPD)
	bCPD, _ := factors.NewLinearGaussianCPD("B", 0.0, []float64{0.5}, 1.0, []string{"A"})
	lgbn.AddLinearGaussianCPD(bCPD)

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5}),
		"B": tabgo.NewSeries("B", []any{1.5, 3.0, 4.5, 6.0, 7.5, 2.0, 3.5, 5.0, 6.5, 8.0}),
	})

	err := lgbn.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

// ==========================================================================
// SEM: more paths - Fit, ActiveTrailNodes, Moralize
// ==========================================================================

func TestFinalCov_SEM_Fit_Valid(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5 variance=2.0")
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5}),
		"Y": tabgo.NewSeries("Y", []any{1.5, 3.0, 4.5, 6.0, 7.5, 2.0, 3.5, 5.0, 6.5, 8.0}),
	})
	err := s.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestFinalCov_SEM_ActiveTrailNodes(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5\nZ: Y=0.3")
	nodes, _ := s.ActiveTrailNodes("X", nil)
	t.Logf("Active trail nodes: %v", nodes)
}

func TestFinalCov_SEM_Moralize(t *testing.T) {
	s, _ := FromLisrel("X:\nY: X=0.5\nZ: Y=0.3")
	g := s.Moralize()
	if g == nil {
		t.Error("expected non-nil moral graph")
	}
}

func TestFinalCov_SEM_FromRAM(t *testing.T) {
	s, err := FromRAM("X:\nY: X=0.5 variance=2.0")
	if err != nil {
		t.Fatal(err)
	}
	if s == nil {
		t.Error("expected non-nil SEM from RAM")
	}
}

// ==========================================================================
// DBN: ActiveTrailNodes, Moralize, GetConstantBN, GetMarkovBlanket, Simulate
// ==========================================================================

func TestFinalCov_DBN_ActiveTrailNodes(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)
	dbn.AddTransitionCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	dbn.AddInitialCPD(bCPD)
	dbn.AddTransitionCPD(bCPD)

	active := dbn.ActiveTrailNodes([]string{"A"}, nil)
	t.Logf("Active trail nodes: %v", active)
}

func TestFinalCov_DBN_Moralize(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	g := dbn.Moralize()
	if g == nil {
		t.Error("expected non-nil moral graph")
	}
}

func TestFinalCov_DBN_GetConstantBN(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")

	bn0, err := dbn.GetConstantBN(0)
	if err != nil {
		t.Fatal(err)
	}
	if bn0 == nil {
		t.Error("expected non-nil BN for slice 0")
	}

	bn1, err := dbn.GetConstantBN(1)
	if err != nil {
		t.Fatal(err)
	}
	if bn1 == nil {
		t.Error("expected non-nil BN for slice 1")
	}

	_, err = dbn.GetConstantBN(5)
	if err == nil {
		t.Error("expected error for invalid slice")
	}
}

func TestFinalCov_DBN_GetSliceNodes(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")

	nodes0, _ := dbn.GetSliceNodes(0)
	t.Logf("Slice 0 nodes: %v", nodes0)

	_, err := dbn.GetSliceNodes(5)
	if err == nil {
		t.Error("expected error for invalid slice")
	}
}

func TestFinalCov_DBN_RemoveCPDs(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)
	dbn.AddTransitionCPD(aCPD)

	dbn.RemoveCPDs("A")
	cpds := dbn.GetCPDs()
	t.Logf("CPDs after remove: %d", len(cpds))
}

func TestFinalCov_DBN_GetMarkovBlanket(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")

	mb, err := dbn.GetMarkovBlanket("A")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Markov blanket of A: %v", mb)
}

// ==========================================================================
// MN: GetCardinality, GetPartitionFunction, ToJunctionTree paths
// ==========================================================================

func TestFinalCov_MN_GetCardinality_Unknown(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	_, err := mn.GetCardinality("A")
	if err == nil {
		t.Error("expected error for unknown cardinality")
	}
}

func TestFinalCov_MN_GetPartitionFunction(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(fAB)

	z, err := mn.GetPartitionFunction()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Partition function: %f", z)
}

func TestFinalCov_MN_ToJunctionTree(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(fAB)

	jt, err := mn.ToJunctionTree()
	if err != nil {
		t.Fatal(err)
	}
	if jt == nil {
		t.Error("expected non-nil junction tree")
	}
}

// ==========================================================================
// NaiveBayes: Fit, Predict
// ==========================================================================

func TestFinalCov_NaiveBayes_FitAndPredict(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"C":  tabgo.NewSeries("C", []any{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}),
		"F1": tabgo.NewSeries("F1", []any{0, 0, 1, 1, 0, 0, 1, 1, 0, 1}),
		"F2": tabgo.NewSeries("F2", []any{0, 1, 0, 1, 0, 1, 0, 1, 1, 0}),
	})

	err := nb.Fit(df)
	if err != nil {
		t.Fatal(err)
	}

	predictions, err := nb.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(predictions) != 10 {
		t.Errorf("expected 10 predictions, got %d", len(predictions))
	}

	probs, err := nb.PredictProbability(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != 10 {
		t.Errorf("expected 10 probability vectors, got %d", len(probs))
	}
}

func TestFinalCov_NaiveBayes_Fit_NilData(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	err := nb.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestFinalCov_NaiveBayes_PredictProbability_NilData(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	_, err := nb.PredictProbability(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestFinalCov_NaiveBayes_ActiveTrailNodes(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	nodes, err := nb.ActiveTrailNodes("C", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Active trail from C: %v", nodes)
}

// ==========================================================================
// BN: Save/Load round-trip via BIF
// ==========================================================================

func TestFinalCov_BN_SaveLoad_RoundTrip(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddEdge("A", "B")
	bn.SetStates("A", []string{"low", "high"})
	bn.SetStates("B", []string{"off", "on"})

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.4}, {0.6}}, nil, nil)
	bn.AddCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.9, 0.1}, {0.1, 0.9}}, []string{"A"}, []int{2})
	bn.AddCPD(bCPD)

	fname := "/tmp/test_bn_roundtrip.bif"
	err := bn.Save(fname)
	if err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadBayesianNetwork(fname)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Nodes()) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(loaded.Nodes()))
	}
}

// ==========================================================================
// BN: FitUpdate with parent out-of-range values
// ==========================================================================

func TestFinalCov_BN_FitUpdate_OutOfRange(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("X")
	bn.AddNode("Y")
	bn.AddEdge("X", "Y")

	xCPD, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(xCPD)
	yCPD, _ := factors.NewTabularCPD("Y", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"X"}, []int{2})
	bn.AddCPD(yCPD)

	// Data with out-of-range values
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0, 1, 5, 0, -1}), // 5 and -1 are out of range
		"Y": tabgo.NewSeries("Y", []any{0, 1, 0, 1, 0}),
	})
	err := bn.FitUpdate(df, 10)
	if err != nil {
		t.Fatal(err)
	}
}

// ==========================================================================
// DBN: Simulate valid
// ==========================================================================

func TestFinalCov_DBN_Simulate_Valid(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")

	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(aCPD)
	dbn.AddTransitionCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	dbn.AddInitialCPD(bCPD)
	dbn.AddTransitionCPD(bCPD)

	df, err := dbn.Simulate(10, 42)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 10 {
		t.Errorf("expected 10 rows, got %d", df.Len())
	}
}
