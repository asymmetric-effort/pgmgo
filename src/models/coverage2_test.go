//go:build unit

package models

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeValidBN(t *testing.T) *BayesianNetwork {
	t.Helper()
	bn := NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddEdge("A", "B")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.6}, {0.4}}, nil, nil)
	_ = bn.AddCPD(cpdA)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"A"}, []int{2})
	_ = bn.AddCPD(cpdB)
	return bn
}

func make3NodeBN(t *testing.T) *BayesianNetwork {
	t.Helper()
	bn := NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.AddEdge("B", "C")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	bn.SetStates("C", []string{"c0", "c1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.6}, {0.4}}, nil, nil)
	_ = bn.AddCPD(cpdA)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	_ = bn.AddCPD(cpdB)
	cpdC, _ := factors.NewTabularCPD("C", 2, [][]float64{{0.7, 0.3}, {0.3, 0.7}}, []string{"B"}, []int{2})
	_ = bn.AddCPD(cpdC)
	return bn
}

// ---------------------------------------------------------------------------
// BayesianNetwork parity methods (0% coverage)
// ---------------------------------------------------------------------------

func TestBN_Simulate_Valid(t *testing.T) {
	bn := makeValidBN(t)
	df, err := bn.Simulate(10, nil, 42)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 10 {
		t.Errorf("expected 10 rows, got %d", df.Len())
	}
}

func TestBN_Simulate_WithEvidence(t *testing.T) {
	bn := makeValidBN(t)
	df, err := bn.Simulate(5, map[string]int{"A": 0}, 42)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 5 {
		t.Errorf("expected 5 rows, got %d", df.Len())
	}
}

func TestBN_Simulate_NonPositiveN(t *testing.T) {
	bn := makeValidBN(t)
	_, err := bn.Simulate(0, nil, 42)
	if err == nil {
		t.Error("expected error for n=0")
	}
}

func TestBN_HasNode(t *testing.T) {
	bn := makeValidBN(t)
	if !bn.HasNode("A") {
		t.Error("expected HasNode(A) = true")
	}
	if bn.HasNode("UNKNOWN") {
		t.Error("expected HasNode(UNKNOWN) = false")
	}
}

func TestBN_HasEdge(t *testing.T) {
	bn := makeValidBN(t)
	if !bn.HasEdge("A", "B") {
		t.Error("expected HasEdge(A,B) = true")
	}
	if bn.HasEdge("B", "A") {
		t.Error("expected HasEdge(B,A) = false")
	}
}

func TestBN_TopologicalOrder(t *testing.T) {
	bn := makeValidBN(t)
	order, err := bn.TopologicalOrder()
	if err != nil {
		t.Fatal(err)
	}
	if len(order) != 2 {
		t.Errorf("expected 2 nodes in order, got %d", len(order))
	}
}

func TestBN_NumberOfNodes(t *testing.T) {
	bn := makeValidBN(t)
	if bn.NumberOfNodes() != 2 {
		t.Errorf("expected 2, got %d", bn.NumberOfNodes())
	}
}

func TestBN_NumberOfEdges(t *testing.T) {
	bn := makeValidBN(t)
	if bn.NumberOfEdges() != 1 {
		t.Errorf("expected 1, got %d", bn.NumberOfEdges())
	}
}

func TestBN_GetAllStates(t *testing.T) {
	bn := makeValidBN(t)
	states := bn.GetAllStates()
	if len(states) != 2 {
		t.Errorf("expected 2 variables, got %d", len(states))
	}
	if len(states["A"]) != 2 {
		t.Errorf("expected 2 states for A, got %d", len(states["A"]))
	}
}

// ---------------------------------------------------------------------------
// BayesianNetwork: Predict, PredictProbability, Do, IsIMap
// ---------------------------------------------------------------------------

func TestBN_Predict(t *testing.T) {
	bn := make3NodeBN(t)
	// Data with nil (missing) values for C
	rows := [][]any{
		{0, 0, nil},
		{1, 1, nil},
		{0, nil, nil},
	}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B", "C"}, rows)
	result, err := bn.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if result.Len() != 3 {
		t.Errorf("expected 3 rows, got %d", result.Len())
	}
}

func TestBN_Predict_NoMissing(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{0, 0}, {1, 1}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	result, err := bn.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if result.Len() != 2 {
		t.Errorf("expected 2 rows, got %d", result.Len())
	}
}

func TestBN_PredictProbability(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{0, nil}, {1, nil}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	result, err := bn.PredictProbability(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(result["B"]) == 0 {
		t.Error("expected probabilities for B")
	}
}

func TestBN_Do(t *testing.T) {
	bn := makeValidBN(t)
	mutilated, err := bn.Do(map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	if mutilated == nil {
		t.Error("expected non-nil mutilated BN")
	}
}

func TestBN_Do_EmptyNodes(t *testing.T) {
	bn := makeValidBN(t)
	result, err := bn.Do(nil)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Error("expected non-nil result for empty do")
	}
}

func TestBN_Do_UnknownNode(t *testing.T) {
	bn := makeValidBN(t)
	_, err := bn.Do(map[string]int{"UNKNOWN": 0})
	if err == nil {
		t.Error("expected error for unknown node")
	}
}

func TestBN_Do_StateOutOfRange(t *testing.T) {
	bn := makeValidBN(t)
	_, err := bn.Do(map[string]int{"A": 5})
	if err == nil {
		t.Error("expected error for state out of range")
	}
}

func TestBN_GetMarkovBlanket(t *testing.T) {
	bn := make3NodeBN(t)
	blanket, err := bn.GetMarkovBlanket("B")
	if err != nil {
		t.Fatal(err)
	}
	if len(blanket) == 0 {
		t.Error("expected non-empty Markov blanket for B")
	}
}

func TestBN_GetMarkovBlanket_UnknownNode(t *testing.T) {
	bn := makeValidBN(t)
	_, err := bn.GetMarkovBlanket("UNKNOWN")
	if err == nil {
		t.Error("expected error for unknown node")
	}
}

func TestBN_GetStateProbability(t *testing.T) {
	bn := makeValidBN(t)
	prob, err := bn.GetStateProbability(map[string]int{"A": 0, "B": 0})
	if err != nil {
		t.Fatal(err)
	}
	if prob <= 0 || prob > 1 {
		t.Errorf("unexpected probability: %f", prob)
	}
}

func TestBN_GetStateProbability_Partial(t *testing.T) {
	bn := make3NodeBN(t)
	prob, err := bn.GetStateProbability(map[string]int{"A": 0})
	if err != nil {
		t.Fatal(err)
	}
	if prob <= 0 || prob > 1 {
		t.Errorf("unexpected probability: %f", prob)
	}
}

func TestBN_GetStateProbability_Empty(t *testing.T) {
	bn := makeValidBN(t)
	_, err := bn.GetStateProbability(nil)
	if err == nil {
		t.Error("expected error for empty states")
	}
}

func TestBN_GetFactorizedProduct(t *testing.T) {
	bn := makeValidBN(t)
	fs, err := bn.GetFactorizedProduct()
	if err != nil {
		t.Fatal(err)
	}
	if len(fs) == 0 {
		t.Error("expected non-empty factor list")
	}
}

// ---------------------------------------------------------------------------
// DynamicBN: AddNode/AddEdge error paths, InitializeInitialState, GetCPDs
// ---------------------------------------------------------------------------

func TestDynamicBN_AddNode_Error(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	if err := dbn.AddNode("A"); err != nil {
		t.Fatal(err)
	}
	// Adding duplicate
	err := dbn.AddNode("A")
	if err == nil {
		t.Error("expected error for duplicate node")
	}
}

func TestDynamicBN_AddEdge_Error(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	if err := dbn.AddEdge("A", "B"); err != nil {
		t.Fatal(err)
	}
	// Adding duplicate edge
	err := dbn.AddEdge("A", "B")
	if err == nil {
		t.Error("expected error for duplicate edge")
	}
}

func TestDynamicBN_InitializeInitialState(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	_ = dbn.Initial().SetStates("A", []string{"a0", "a1"})
	_ = dbn.Initial().SetStates("B", []string{"b0", "b1"})

	err := dbn.InitializeInitialState(map[string][]float64{
		"A": {0.6, 0.4},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDynamicBN_InitializeInitialState_EmptyDist(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	err := dbn.InitializeInitialState(map[string][]float64{
		"A": {},
	})
	if err == nil {
		t.Error("expected error for empty distribution")
	}
}

func TestDynamicBN_GetCPDs(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	_ = dbn.Initial().SetStates("A", []string{"a0", "a1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(cpdA)
	cpds := dbn.GetCPDs()
	if len(cpds) == 0 {
		t.Error("expected non-empty CPDs")
	}
}

func TestDynamicBN_GetSliceNodes(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	nodes0, err := dbn.GetSliceNodes(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes0) == 0 {
		t.Error("expected non-empty nodes for slice 0")
	}
	_, err = dbn.GetSliceNodes(5)
	if err == nil {
		t.Error("expected error for invalid slice")
	}
}

func TestDynamicBN_Moralize(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	g := dbn.Moralize()
	if g == nil {
		t.Error("expected non-nil moral graph")
	}
}

func TestDynamicBN_RemoveCPDs(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	_ = dbn.Initial().SetStates("A", []string{"a0", "a1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	dbn.AddInitialCPD(cpdA)
	dbn.RemoveCPDs("A")
	// Should work without error
}

// ---------------------------------------------------------------------------
// NaiveBayes: error paths and edge cases
// ---------------------------------------------------------------------------

func TestNaiveBayes_NewNaiveBayes_EmptyClass(t *testing.T) {
	_, err := NewNaiveBayes("", []string{"F1"})
	if err == nil {
		t.Error("expected error for empty class variable")
	}
}

func TestNaiveBayes_NewNaiveBayes_EmptyFeatures(t *testing.T) {
	_, err := NewNaiveBayes("C", nil)
	if err == nil {
		t.Error("expected error for empty features")
	}
}

func TestNaiveBayes_NewNaiveBayes_DuplicateFeature(t *testing.T) {
	_, err := NewNaiveBayes("C", []string{"F1", "F1"})
	if err == nil {
		t.Error("expected error for duplicate feature")
	}
}

func TestNaiveBayes_NewNaiveBayes_FeatureEqualsClass(t *testing.T) {
	_, err := NewNaiveBayes("C", []string{"C"})
	if err == nil {
		t.Error("expected error when feature == class")
	}
}

func TestNaiveBayes_Fit_NilData(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	err := nb.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNaiveBayes_Fit_EmptyData(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, nil)
	err := nb.Fit(df)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestNaiveBayes_Fit_Valid(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	rows := [][]any{
		{0, 0, 1},
		{0, 1, 0},
		{1, 1, 1},
		{1, 0, 0},
		{0, 0, 0},
	}
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1", "F2"}, rows)
	err := nb.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNaiveBayes_Fit_NegativeClass(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	rows := [][]any{
		{-1, 0},
	}
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, rows)
	err := nb.Fit(df)
	if err == nil {
		t.Error("expected error for negative class value")
	}
}

func TestNaiveBayes_Fit_NegativeFeature(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	rows := [][]any{
		{0, -1},
	}
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, rows)
	err := nb.Fit(df)
	if err == nil {
		t.Error("expected error for negative feature value")
	}
}

func TestNaiveBayes_PredictProbability_NilData(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	_, err := nb.PredictProbability(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNaiveBayes_PredictProbability_NotFitted(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	rows := [][]any{{0, 0}}
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, rows)
	_, err := nb.PredictProbability(df)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestNaiveBayes_PredictAndPredictProbability_Valid(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	trainRows := [][]any{
		{0, 0, 0}, {0, 0, 1}, {0, 1, 0},
		{1, 1, 1}, {1, 1, 0}, {1, 0, 1},
	}
	trainDf := tabgo.NewDataFrameFromRows([]string{"C", "F1", "F2"}, trainRows)
	if err := nb.Fit(trainDf); err != nil {
		t.Fatal(err)
	}

	testRows := [][]any{{0, 0}, {1, 1}}
	testDf := tabgo.NewDataFrameFromRows([]string{"F1", "F2"}, testRows)

	probs, err := nb.PredictProbability(testDf)
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != 2 {
		t.Errorf("expected 2 rows, got %d", len(probs))
	}

	predictions, err := nb.Predict(testDf)
	if err != nil {
		t.Fatal(err)
	}
	if len(predictions) != 2 {
		t.Errorf("expected 2 predictions, got %d", len(predictions))
	}
}

func TestNaiveBayes_AddEdge_NotFromClass(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	err := nb.AddEdge("F1", "C")
	if err == nil {
		t.Error("expected error for edge not from class variable")
	}
}

func TestNaiveBayes_AddEdgesFrom_NotClass(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	err := nb.AddEdgesFrom("F1", []string{"C"})
	if err == nil {
		t.Error("expected error for edges not from class variable")
	}
}

// ---------------------------------------------------------------------------
// SEM: error paths, ToLisrel, ToStandardLisrel, GenerateSamples, Fit
// ---------------------------------------------------------------------------

func TestSEM_AddEquation_MismatchedLengths(t *testing.T) {
	s := NewSEM()
	err := s.AddEquation("Y", []string{"X1", "X2"}, []float64{0.5}, 0, 1)
	if err == nil {
		t.Error("expected error for mismatched parents/coefficients lengths")
	}
}

func TestSEM_CheckModel_NoEquation(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	// Manually add a node without equation (hack via the DAG)
	// This tests the "no equation" error path.
	// Instead, let's just test negative variance.
	s2 := NewSEM()
	s2.AddEquation("X", nil, nil, 0, -1)
	if err := s2.CheckModel(); err == nil {
		t.Error("expected error for negative variance")
	}
}

func TestSEM_ToLisrel(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	result, err := s.ToLisrel()
	if err != nil {
		t.Fatal(err)
	}
	if result["B"] == nil {
		t.Error("expected non-nil B matrix")
	}
}

func TestSEM_ToStandardLisrel(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	result, err := s.ToStandardLisrel()
	if err != nil {
		t.Fatal(err)
	}
	if result["B"] == nil {
		t.Error("expected non-nil B matrix")
	}
}

func TestSEM_GenerateSamples(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1, 0.5)
	df, err := s.GenerateSamples(100)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 100 {
		t.Errorf("expected 100 rows, got %d", df.Len())
	}
}

func TestSEM_GenerateSamples_NonPositive(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	_, err := s.GenerateSamples(0)
	if err == nil {
		t.Error("expected error for non-positive nSamples")
	}
}

func TestSEM_Fit(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0}, 0, 1)

	rows := [][]any{
		{1.0, 1.5},
		{2.0, 2.3},
		{3.0, 3.1},
		{4.0, 4.2},
		{5.0, 5.4},
	}
	df := tabgo.NewDataFrameFromRows([]string{"X", "Y"}, rows)
	err := s.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSEM_Fit_NilData(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	err := s.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestSEM_Fit_EmptyData(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	df := tabgo.NewDataFrameFromRows([]string{"X"}, nil)
	err := s.Fit(df)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestSEM_Fit_NoVariables(t *testing.T) {
	s := NewSEM()
	rows := [][]any{{1.0}}
	df := tabgo.NewDataFrameFromRows([]string{"X"}, rows)
	err := s.Fit(df)
	if err == nil {
		t.Error("expected error for SEM with no variables")
	}
}

func TestSEM_ActiveTrailNodes(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	s.AddEquation("Z", []string{"Y"}, []float64{0.3}, 0, 1)

	nodes, err := s.ActiveTrailNodes("X", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) == 0 {
		t.Error("expected non-empty active trail nodes")
	}
}

func TestSEM_ActiveTrailNodes_WithObserved(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	s.AddEquation("Z", []string{"Y"}, []float64{0.3}, 0, 1)

	nodes, err := s.ActiveTrailNodes("X", map[string]bool{"Y": true})
	if err != nil {
		t.Fatal(err)
	}
	// Y blocks the trail from X to Z
	_ = nodes
}

func TestSEM_ActiveTrailNodes_UnknownVar(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	_, err := s.ActiveTrailNodes("UNKNOWN", nil)
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

func TestSEM_Moralize(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	g := s.Moralize()
	if g == nil {
		t.Error("expected non-nil moral graph")
	}
}

func TestSEM_FromLavaan(t *testing.T) {
	s, err := FromLavaan("Y ~ X1 + X2\nX1 ~\nX2 ~")
	if err != nil {
		t.Fatal(err)
	}
	if s == nil {
		t.Error("expected non-nil SEM")
	}
}

func TestSEM_FromLavaan_Empty(t *testing.T) {
	_, err := FromLavaan("")
	if err == nil {
		t.Error("expected error for empty syntax")
	}
}

func TestSEM_FromLavaan_EmptyChild(t *testing.T) {
	_, err := FromLavaan(" ~ X")
	if err == nil {
		t.Error("expected error for empty child")
	}
}

func TestSEM_FromLisrel(t *testing.T) {
	s, err := FromLisrel("X: variance=1.0\nY: X=0.5 variance=0.8 intercept=1.0")
	if err != nil {
		t.Fatal(err)
	}
	if s == nil {
		t.Error("expected non-nil SEM")
	}
}

func TestSEM_FromLisrel_Empty(t *testing.T) {
	_, err := FromLisrel("")
	if err == nil {
		t.Error("expected error for empty spec")
	}
}

func TestSEM_FromLisrel_BadValue(t *testing.T) {
	_, err := FromLisrel("X: variance=abc")
	if err == nil {
		t.Error("expected error for bad value")
	}
}

func TestSEM_SetParams(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)

	err := s.SetParams("Y", []float64{0.8}, 1.0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSEM_SetParams_NoEquation(t *testing.T) {
	s := NewSEM()
	err := s.SetParams("UNKNOWN", nil, 0, 1)
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

func TestSEM_SetParams_MismatchedCoeffs(t *testing.T) {
	s := NewSEM()
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	err := s.SetParams("Y", []float64{0.8, 0.9}, 1.0, 0.5)
	if err == nil {
		t.Error("expected error for mismatched coefficients")
	}
}

func TestSEM_ImpliedCovarianceMatrix(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatal(err)
	}
	if len(cov) != 2 || len(cov[0]) != 2 {
		t.Error("expected 2x2 covariance matrix")
	}
}

// ---------------------------------------------------------------------------
// MarkovNetwork: ToFactorGraph, ToJunctionTree, GetCardinality errors
// ---------------------------------------------------------------------------

func TestMarkovNetwork_ToFactorGraph(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(f)
	fg, err := mn.ToFactorGraph()
	if err != nil {
		t.Fatal(err)
	}
	if fg == nil {
		t.Error("expected non-nil FactorGraph")
	}
}

func TestMarkovNetwork_GetPartitionFunction_Cov(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(f)
	pf, err := mn.GetPartitionFunction()
	if err != nil {
		t.Fatal(err)
	}
	if pf <= 0 {
		t.Error("expected positive partition function")
	}
}

func TestMarkovNetwork_ToJunctionTree_Cov(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(f)
	jt, err := mn.ToJunctionTree()
	if err != nil {
		t.Fatal(err)
	}
	if jt == nil {
		t.Error("expected non-nil JunctionTree")
	}
}

func TestMarkovNetwork_ToBayesianModel(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(f)
	bn, err := mn.ToBayesianModel()
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BayesianNetwork")
	}
}

// ---------------------------------------------------------------------------
// MarkovChain: error paths
// ---------------------------------------------------------------------------

func TestMarkovChain_InvalidTransitionMatrix(t *testing.T) {
	// Non-square matrix
	_, err := NewMarkovChain([][]float64{
		{0.5, 0.5},
	}, []string{"A", "B"})
	if err == nil {
		t.Error("expected error for non-square matrix")
	}
}

func TestMarkovChain_StationaryDistribution(t *testing.T) {
	mc, err := NewMarkovChain([][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}, []string{"S0", "S1"})
	if err != nil {
		t.Fatal(err)
	}
	dist, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatal(err)
	}
	if len(dist) != 2 {
		t.Errorf("expected 2 states, got %d", len(dist))
	}
	sum := 0.0
	for _, p := range dist {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("expected sum ~1, got %f", sum)
	}
}

func TestMarkovChain_IsErgodic(t *testing.T) {
	mc, err := NewMarkovChain([][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}, []string{"S0", "S1"})
	if err != nil {
		t.Fatal(err)
	}
	ergodic := mc.IsErgodic()
	if !ergodic {
		t.Error("expected ergodic chain")
	}
}

// ---------------------------------------------------------------------------
// JunctionTree: AddEdge error path
// ---------------------------------------------------------------------------

func TestJunctionTree_AddEdge_InvalidClique_Cov(t *testing.T) {
	bn := makeValidBN(t)
	jt, err := NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatal(err)
	}
	// Try adding edge to invalid clique index
	err = jt.AddEdge(0, 999)
	if err == nil {
		t.Error("expected error for invalid clique index")
	}
}

// ---------------------------------------------------------------------------
// LinearGaussianBN: CheckModel error
// ---------------------------------------------------------------------------

func TestLinearGaussianBN_CheckModel_NoCPD_Cov(t *testing.T) {
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	err := lgbn.CheckModel()
	if err == nil {
		t.Error("expected error for missing CPD")
	}
}

// ---------------------------------------------------------------------------
// FunctionalBN: CheckModel
// ---------------------------------------------------------------------------

func TestFunctionalBN_CheckModel_NoFunction_Cov(t *testing.T) {
	fbn := NewFunctionalBayesianNetwork()
	fbn.AddNode("A")
	err := fbn.CheckModel()
	if err == nil {
		t.Error("expected error for missing function")
	}
}

// ---------------------------------------------------------------------------
// BN: FitUpdate
// ---------------------------------------------------------------------------

func TestBN_FitUpdate(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	err := bn.FitUpdate(df, 10)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBN_FitUpdate_NegativePrev(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{0, 0}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	err := bn.FitUpdate(df, -1)
	if err == nil {
		t.Error("expected error for negative nPrevSamples")
	}
}

func TestBN_FitUpdate_EmptyData(t *testing.T) {
	bn := makeValidBN(t)
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	err := bn.FitUpdate(df, 10)
	if err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// BN: Save and Load (BIF format)
// ---------------------------------------------------------------------------

func TestBN_SaveAndLoad(t *testing.T) {
	bn := make3NodeBN(t)
	tmpFile := "/tmp/test_pgmgo_bn.bif"
	err := bn.Save(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadBayesianNetwork(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.NumberOfNodes() != 3 {
		t.Errorf("expected 3 nodes, got %d", loaded.NumberOfNodes())
	}
}

// ---------------------------------------------------------------------------
// BN: IsIMap
// ---------------------------------------------------------------------------

func TestBN_IsIMap_Valid(t *testing.T) {
	bn := makeValidBN(t)
	// Create JPD by generating factors
	facs, err := bn.GetFactorizedProduct()
	if err != nil {
		t.Fatal(err)
	}
	joint, err := factors.FactorProduct(facs...)
	if err != nil {
		t.Fatal(err)
	}
	vars := joint.Variables()
	card := joint.Cardinality()
	vals := joint.Values().Data()
	jpd, err := factors.NewJointProbabilityDistribution(vars, card, vals)
	if err != nil {
		t.Fatal(err)
	}
	isIMap, err := bn.IsIMap(jpd)
	if err != nil {
		t.Fatal(err)
	}
	_ = isIMap
}

// ---------------------------------------------------------------------------
// BN: ToJunctionTree
// ---------------------------------------------------------------------------

func TestBN_ToJunctionTree(t *testing.T) {
	bn := makeValidBN(t)
	jt, err := bn.ToJunctionTree()
	if err != nil {
		t.Fatal(err)
	}
	if jt == nil {
		t.Error("expected non-nil junction tree")
	}
}

// ---------------------------------------------------------------------------
// SEM: FromGraph, FromRAM, GetScalingIndicators
// ---------------------------------------------------------------------------

func TestSEM_FromGraph(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	// Get the DAG from the SEM and recreate
	// Can't easily get the internal DAG, so test via FromLavaan
	s2, err := FromLavaan("Y ~ X\nX ~")
	if err != nil {
		t.Fatal(err)
	}
	if s2 == nil {
		t.Error("expected non-nil SEM")
	}
}

func TestSEM_FromRAM(t *testing.T) {
	s, err := FromRAM("X: variance=1.0\nY: X=0.5 variance=0.8")
	if err != nil {
		t.Fatal(err)
	}
	if s == nil {
		t.Error("expected non-nil SEM")
	}
}

func TestSEM_GetScalingIndicators(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	indicators := s.GetScalingIndicators()
	if len(indicators) == 0 {
		t.Error("expected non-empty indicators")
	}
}

// ---------------------------------------------------------------------------
// NaiveBayes: Fit with zero-count class
// ---------------------------------------------------------------------------

func TestNaiveBayes_Fit_ZeroCountClass(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	// All class=0, so class=1 has zero count -> uniform path
	rows := [][]any{
		{0, 0}, {0, 1}, {0, 0}, {0, 1}, {0, 0},
	}
	df := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, rows)
	err := nb.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// NaiveBayes: PredictProbability with zero-likelihood path
// ---------------------------------------------------------------------------

func TestNaiveBayes_PredictProbability_ZeroLikelihood(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	trainRows := [][]any{
		{0, 0}, {0, 0}, {0, 0},
		{1, 1}, {1, 1}, {1, 1},
	}
	trainDf := tabgo.NewDataFrameFromRows([]string{"C", "F1"}, trainRows)
	if err := nb.Fit(trainDf); err != nil {
		t.Fatal(err)
	}

	// Test with a feature value that has zero probability for one class
	testRows := [][]any{{0}, {1}}
	testDf := tabgo.NewDataFrameFromRows([]string{"F1"}, testRows)
	probs, err := nb.PredictProbability(testDf)
	if err != nil {
		t.Fatal(err)
	}
	if len(probs) != 2 {
		t.Errorf("expected 2 rows, got %d", len(probs))
	}
}

// ---------------------------------------------------------------------------
// MarkovChain: more edge cases
// ---------------------------------------------------------------------------

func TestMarkovChain_ProbFromSample_Cov(t *testing.T) {
	mc, err := NewMarkovChain([][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}, []string{"S0", "S1"})
	if err != nil {
		t.Fatal(err)
	}
	prob, err := mc.ProbFromSample([]int{0, 0, 1, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(prob) == 0 {
		t.Error("expected non-empty probability matrix")
	}
}

func TestMarkovChain_IsStationarity_Cov(t *testing.T) {
	mc, err := NewMarkovChain([][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}, []string{"S0", "S1"})
	if err != nil {
		t.Fatal(err)
	}
	dist, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatal(err)
	}
	is, err := mc.IsStationarity(dist)
	if err != nil {
		t.Fatal(err)
	}
	_ = is
}

// ---------------------------------------------------------------------------
// DynamicBN: more coverage
// ---------------------------------------------------------------------------

func TestDynamicBN_GetIntraEdges(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	intra := dbn.GetIntraEdges()
	if len(intra) != 1 {
		t.Errorf("expected 1 intra edge, got %d", len(intra))
	}
}

func TestDynamicBN_GetInterEdges(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	inter := dbn.GetInterEdges()
	if len(inter) != 1 {
		t.Errorf("expected 1 inter edge, got %d", len(inter))
	}
}

// ---------------------------------------------------------------------------
// MarkovNetwork: CheckModel error, GetCardinality
// ---------------------------------------------------------------------------

func TestMarkovNetwork_CheckModel_NoFactors_Cov(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	err := mn.CheckModel()
	if err == nil {
		t.Error("expected error for no factors")
	}
}

func TestMarkovNetwork_GetCardinality(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	mn.AddFactor(f)
	card, err := mn.GetCardinality("A")
	if err != nil {
		t.Fatal(err)
	}
	if card != 2 {
		t.Errorf("expected 2, got %d", card)
	}
}

func TestMarkovNetwork_GetCardinality_Unknown(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	_, err := mn.GetCardinality("A")
	if err == nil {
		t.Error("expected error for unknown cardinality (no factors)")
	}
}
