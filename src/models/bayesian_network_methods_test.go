//go:build unit

package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// helper: build a small A->B network with binary variables.
func buildSmallAB(t *testing.T) *BayesianNetwork {
	t.Helper()
	bn := NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.6}, {0.4}}, nil, nil)
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{
		{0.2, 0.75},
		{0.8, 0.25},
	}, []string{"A"}, []int{2})

	_ = bn.AddCPD(cpdA)
	_ = bn.AddCPD(cpdB)
	_ = bn.SetStates("A", []string{"a0", "a1"})
	_ = bn.SetStates("B", []string{"b0", "b1"})
	return bn
}

// ---- TestRemoveNode ----

func TestRemoveNode(t *testing.T) {
	bn := buildStudentNetwork(t)
	if err := bn.RemoveNode("G"); err != nil {
		t.Fatalf("RemoveNode: %v", err)
	}
	// G should be gone.
	for _, n := range bn.Nodes() {
		if n == "G" {
			t.Fatal("G still present after RemoveNode")
		}
	}
	// Edges involving G should be gone.
	for _, e := range bn.Edges() {
		if e[0] == "G" || e[1] == "G" {
			t.Fatalf("edge involving G still present: %v", e)
		}
	}
	// CPD should be gone.
	if bn.GetCPD("G") != nil {
		t.Fatal("CPD for G still present")
	}
}

func TestRemoveNodeNotFound(t *testing.T) {
	bn := NewBayesianNetwork()
	if err := bn.RemoveNode("Z"); err == nil {
		t.Error("expected error for nonexistent node")
	}
}

// ---- TestRemoveNodes ----

func TestRemoveNodes(t *testing.T) {
	bn := buildStudentNetwork(t)
	if err := bn.RemoveNodes("D", "I"); err != nil {
		t.Fatalf("RemoveNodes: %v", err)
	}
	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}
}

func TestRemoveNodesPartialFailure(t *testing.T) {
	bn := buildStudentNetwork(t)
	err := bn.RemoveNodes("D", "NONEXISTENT")
	if err == nil {
		t.Error("expected error for nonexistent node in RemoveNodes")
	}
	// D should already be removed.
	for _, n := range bn.Nodes() {
		if n == "D" {
			t.Fatal("D should have been removed before the error")
		}
	}
}

// ---- TestGetCardinality ----

func TestGetCardinality(t *testing.T) {
	bn := buildStudentNetwork(t)
	card, err := bn.GetCardinality("G")
	if err != nil {
		t.Fatalf("GetCardinality: %v", err)
	}
	if card != 3 {
		t.Errorf("expected cardinality 3 for G, got %d", card)
	}
}

func TestGetCardinalityNoCPD(t *testing.T) {
	bn := NewBayesianNetwork()
	_ = bn.AddNode("X")
	_, err := bn.GetCardinality("X")
	if err == nil {
		t.Error("expected error for node with no CPD")
	}
}

// ---- TestToJunctionTree ----

func TestToJunctionTree(t *testing.T) {
	bn := buildStudentNetwork(t)
	jt, err := bn.ToJunctionTree()
	if err != nil {
		t.Fatalf("ToJunctionTree: %v", err)
	}
	cliques := jt.Cliques()
	if len(cliques) == 0 {
		t.Fatal("junction tree has no cliques")
	}
	if err := jt.CheckModel(); err != nil {
		t.Fatalf("junction tree CheckModel: %v", err)
	}
}

// ---- TestPredict ----

func TestPredict(t *testing.T) {
	bn := buildSmallAB(t)

	// Row 0: A=0, B=nil -> should predict B
	// Row 1: A=nil, B=1  -> should predict A
	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{
			{0, nil},
			{nil, 1},
		},
	)

	result, err := bn.Predict(data)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if result.Len() != 2 {
		t.Fatalf("expected 2 rows, got %d", result.Len())
	}

	// Row 0: A=0 is observed, B should be predicted (not nil).
	bVals := result.Column("B").Values()
	if bVals[0] == nil {
		t.Error("row 0: B should not be nil after prediction")
	}

	// Row 1: B=1 is observed, A should be predicted.
	aVals := result.Column("A").Values()
	if aVals[1] == nil {
		t.Error("row 1: A should not be nil after prediction")
	}
}

func TestPredictNoMissing(t *testing.T) {
	bn := buildSmallAB(t)
	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{{0, 1}},
	)
	result, err := bn.Predict(data)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	aVals := result.Column("A").Values()
	bVals := result.Column("B").Values()
	if toInt(aVals[0]) != 0 || toInt(bVals[0]) != 1 {
		t.Errorf("expected [0, 1], got [%v, %v]", aVals[0], bVals[0])
	}
}

// ---- TestPredictProbability ----

func TestPredictProbability(t *testing.T) {
	bn := buildSmallAB(t)

	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{
			{0, nil},
		},
	)

	probs, err := bn.PredictProbability(data)
	if err != nil {
		t.Fatalf("PredictProbability: %v", err)
	}

	bProbs, ok := probs["B"]
	if !ok {
		t.Fatal("expected probabilities for B")
	}
	if len(bProbs) != 2 {
		t.Fatalf("expected 2 probabilities for B, got %d", len(bProbs))
	}

	// P(B|A=0): B=0 -> 0.2, B=1 -> 0.8
	if math.Abs(bProbs[0]-0.2) > 0.01 || math.Abs(bProbs[1]-0.8) > 0.01 {
		t.Errorf("expected B probs ~[0.2, 0.8], got %v", bProbs)
	}
}

func TestPredictProbabilityNoMissing(t *testing.T) {
	bn := buildSmallAB(t)
	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{{0, 1}},
	)
	probs, err := bn.PredictProbability(data)
	if err != nil {
		t.Fatalf("PredictProbability: %v", err)
	}
	// No missing values, so no probabilities returned.
	if len(probs) != 0 {
		t.Errorf("expected empty probs map, got %v", probs)
	}
}

// ---- TestGetStateProbability ----

func TestGetStateProbability(t *testing.T) {
	bn := buildSmallAB(t)

	// P(A=0) should be 0.6
	p, err := bn.GetStateProbability(map[string]int{"A": 0})
	if err != nil {
		t.Fatalf("GetStateProbability: %v", err)
	}
	if math.Abs(p-0.6) > 0.01 {
		t.Errorf("expected P(A=0) ~ 0.6, got %f", p)
	}

	// P(A=0, B=0) = P(B=0|A=0)*P(A=0) = 0.2*0.6 = 0.12
	p, err = bn.GetStateProbability(map[string]int{"A": 0, "B": 0})
	if err != nil {
		t.Fatalf("GetStateProbability: %v", err)
	}
	if math.Abs(p-0.12) > 0.01 {
		t.Errorf("expected P(A=0,B=0) ~ 0.12, got %f", p)
	}
}

func TestGetStateProbabilityEmpty(t *testing.T) {
	bn := buildSmallAB(t)
	_, err := bn.GetStateProbability(map[string]int{})
	if err == nil {
		t.Error("expected error for empty states")
	}
}

// ---- TestGetMarkovBlanket ----

func TestGetMarkovBlanket(t *testing.T) {
	bn := buildStudentNetwork(t)

	// Markov blanket of G: parents(D,I) + children(L) + co-parents of L (none new)
	blanket, err := bn.GetMarkovBlanket("G")
	if err != nil {
		t.Fatalf("GetMarkovBlanket: %v", err)
	}

	expected := map[string]bool{"D": true, "I": true, "L": true}
	if len(blanket) != len(expected) {
		t.Fatalf("expected %d nodes in blanket, got %d: %v", len(expected), len(blanket), blanket)
	}
	for _, n := range blanket {
		if !expected[n] {
			t.Errorf("unexpected node %q in Markov blanket", n)
		}
	}
}

func TestGetMarkovBlanketOfI(t *testing.T) {
	bn := buildStudentNetwork(t)

	// I's blanket: children(G, S) + parents of G (D) + parents of S (none new) = {D, G, S}
	blanket, err := bn.GetMarkovBlanket("I")
	if err != nil {
		t.Fatalf("GetMarkovBlanket: %v", err)
	}

	expected := map[string]bool{"D": true, "G": true, "S": true}
	if len(blanket) != len(expected) {
		t.Fatalf("expected %d nodes in blanket, got %d: %v", len(expected), len(blanket), blanket)
	}
	for _, n := range blanket {
		if !expected[n] {
			t.Errorf("unexpected node %q in Markov blanket of I", n)
		}
	}
}

func TestGetMarkovBlanketNotFound(t *testing.T) {
	bn := NewBayesianNetwork()
	_, err := bn.GetMarkovBlanket("Z")
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

// ---- TestDo ----

func TestDo(t *testing.T) {
	bn := buildStudentNetwork(t)

	mutilated, err := bn.Do(map[string]int{"G": 0})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}

	// G should have no parents in the mutilated model.
	parents := mutilated.Parents("G")
	if len(parents) != 0 {
		t.Errorf("expected no parents for G after do, got %v", parents)
	}

	// G's CPD should be a delta distribution.
	cpd := mutilated.GetCPD("G")
	if cpd == nil {
		t.Fatal("G should still have a CPD")
	}
	data := cpd.ToFactor().Values().Data()
	// delta at state 0: [1, 0, 0]
	if math.Abs(data[0]-1.0) > 1e-9 {
		t.Errorf("expected data[0]=1.0, got %f", data[0])
	}
	if math.Abs(data[1]) > 1e-9 || math.Abs(data[2]) > 1e-9 {
		t.Errorf("expected data[1]=0, data[2]=0, got %f, %f", data[1], data[2])
	}

	// Original should be unaffected.
	origParents := bn.Parents("G")
	if len(origParents) != 2 {
		t.Errorf("original G should still have 2 parents, got %d", len(origParents))
	}
}

func TestDoEmpty(t *testing.T) {
	bn := buildStudentNetwork(t)
	mutilated, err := bn.Do(map[string]int{})
	if err != nil {
		t.Fatalf("Do(empty): %v", err)
	}
	if len(mutilated.Edges()) != len(bn.Edges()) {
		t.Errorf("empty Do should produce an identical copy")
	}
}

func TestDoUnknownNode(t *testing.T) {
	bn := buildStudentNetwork(t)
	_, err := bn.Do(map[string]int{"NONEXISTENT": 0})
	if err == nil {
		t.Error("expected error for unknown node in Do")
	}
}

func TestDoOutOfRangeState(t *testing.T) {
	bn := buildStudentNetwork(t)
	_, err := bn.Do(map[string]int{"G": 5})
	if err == nil {
		t.Error("expected error for out-of-range state")
	}
}

// ---- TestIsIMap ----

func TestIsIMap(t *testing.T) {
	// Build a simple A->B network and construct a compatible JPD.
	bn := buildSmallAB(t)

	// JPD for A, B: P(A=a, B=b)
	// P(A=0,B=0) = 0.6*0.2 = 0.12
	// P(A=0,B=1) = 0.6*0.8 = 0.48
	// P(A=1,B=0) = 0.4*0.75= 0.30
	// P(A=1,B=1) = 0.4*0.25= 0.10
	jpd, err := factors.NewJointProbabilityDistribution(
		[]string{"A", "B"},
		[]int{2, 2},
		[]float64{0.12, 0.48, 0.30, 0.10},
	)
	if err != nil {
		t.Fatalf("NewJPD: %v", err)
	}

	isIMap, err := bn.IsIMap(jpd)
	if err != nil {
		t.Fatalf("IsIMap: %v", err)
	}
	if !isIMap {
		t.Error("expected BN to be an I-map of its own JPD")
	}
}

// ---- TestGetFactorizedProduct ----

func TestGetFactorizedProduct(t *testing.T) {
	bn := buildStudentNetwork(t)
	facs, err := bn.GetFactorizedProduct()
	if err != nil {
		t.Fatalf("GetFactorizedProduct: %v", err)
	}
	if len(facs) != 5 {
		t.Errorf("expected 5 factors, got %d", len(facs))
	}
}

// ---- TestSaveLoad ----

func TestSaveLoad(t *testing.T) {
	bn := buildSmallAB(t)

	dir := t.TempDir()
	path := filepath.Join(dir, "test.bif")

	if err := bn.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Check file exists.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("saved file not found: %v", err)
	}

	loaded, err := LoadBayesianNetwork(path)
	if err != nil {
		t.Fatalf("LoadBayesianNetwork: %v", err)
	}

	// Compare structure.
	if len(loaded.Nodes()) != len(bn.Nodes()) {
		t.Errorf("loaded %d nodes, expected %d", len(loaded.Nodes()), len(bn.Nodes()))
	}
	if len(loaded.Edges()) != len(bn.Edges()) {
		t.Errorf("loaded %d edges, expected %d", len(loaded.Edges()), len(bn.Edges()))
	}

	// Compare CPD values.
	for _, node := range bn.Nodes() {
		origData := bn.GetCPD(node).ToFactor().Values().Data()
		loadData := loaded.GetCPD(node).ToFactor().Values().Data()
		if len(origData) != len(loadData) {
			t.Errorf("CPD for %q: different lengths", node)
			continue
		}
		for i := range origData {
			if math.Abs(origData[i]-loadData[i]) > 1e-6 {
				t.Errorf("CPD for %q at %d: %f vs %f", node, i, origData[i], loadData[i])
			}
		}
	}
}

func TestLoadNonexistent(t *testing.T) {
	_, err := LoadBayesianNetwork("/nonexistent/path.bif")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestSaveBadPath(t *testing.T) {
	bn := buildSmallAB(t)
	err := bn.Save("/nonexistent/dir/file.bif")
	if err == nil {
		t.Error("expected error for bad path")
	}
}

// ---- TestFitUpdate ----

func TestFitUpdate(t *testing.T) {
	bn := buildSmallAB(t)

	// Original P(A=0)=0.6, P(A=1)=0.4
	// Provide data that is all A=0, B=0.
	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{
			{0, 0},
			{0, 0},
			{0, 0},
			{0, 0},
		},
	)

	if err := bn.FitUpdate(data, 10); err != nil {
		t.Fatalf("FitUpdate: %v", err)
	}

	// After update with 10 prev samples and 4 new samples all A=0:
	// new P(A=0) = (10*0.6 + 4) / (10 + 4) = 10/14 = 0.714...
	cpd := bn.GetCPD("A")
	vals := cpd.ToFactor().Values().Data()
	expected := (10*0.6 + 4) / (10 + 4)
	if math.Abs(vals[0]-expected) > 0.01 {
		t.Errorf("expected P(A=0) ~ %f, got %f", expected, vals[0])
	}
}

func TestFitUpdateNegativePrevSamples(t *testing.T) {
	bn := buildSmallAB(t)
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, [][]any{{0, 0}})
	err := bn.FitUpdate(data, -1)
	if err == nil {
		t.Error("expected error for negative nPrevSamples")
	}
}

func TestFitUpdateEmptyData(t *testing.T) {
	bn := buildSmallAB(t)
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	if err := bn.FitUpdate(data, 10); err != nil {
		t.Fatalf("FitUpdate with empty data: %v", err)
	}
}

// ---- TestGetRandomBayesianNetwork ----

func TestGetRandomBayesianNetwork(t *testing.T) {
	bn, err := GetRandomBayesianNetwork(5, 4, 2)
	if err != nil {
		t.Fatalf("GetRandomBayesianNetwork: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(nodes))
	}

	edges := bn.Edges()
	if len(edges) != 4 {
		t.Errorf("expected 4 edges, got %d", len(edges))
	}

	if err := bn.CheckModel(); err != nil {
		t.Fatalf("random BN CheckModel: %v", err)
	}
}

func TestGetRandomBayesianNetworkNoEdges(t *testing.T) {
	bn, err := GetRandomBayesianNetwork(3, 0, 3)
	if err != nil {
		t.Fatalf("GetRandomBayesianNetwork: %v", err)
	}
	if len(bn.Edges()) != 0 {
		t.Errorf("expected 0 edges, got %d", len(bn.Edges()))
	}
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestGetRandomBayesianNetworkInvalidArgs(t *testing.T) {
	// nNodes <= 0
	if _, err := GetRandomBayesianNetwork(0, 0, 2); err == nil {
		t.Error("expected error for nNodes=0")
	}
	// nStates <= 0
	if _, err := GetRandomBayesianNetwork(3, 1, 0); err == nil {
		t.Error("expected error for nStates=0")
	}
	// nEdges > max
	if _, err := GetRandomBayesianNetwork(3, 10, 2); err == nil {
		t.Error("expected error for too many edges")
	}
	// nEdges < 0
	if _, err := GetRandomBayesianNetwork(3, -1, 2); err == nil {
		t.Error("expected error for negative edges")
	}
}

// ---- TestDoMutilatedModelValid ----

func TestDoMutilatedModelValid(t *testing.T) {
	bn := buildStudentNetwork(t)
	mutilated, err := bn.Do(map[string]int{"G": 1})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	// The mutilated model should pass CheckModel.
	if err := mutilated.CheckModel(); err != nil {
		t.Fatalf("mutilated CheckModel: %v", err)
	}
}

// ---- TestSaveLoadStudentNetwork ----

func TestSaveLoadStudentNetwork(t *testing.T) {
	bn := buildStudentNetwork(t)
	// Set states for all nodes.
	_ = bn.SetStates("D", []string{"easy", "hard"})
	_ = bn.SetStates("I", []string{"low", "high"})
	_ = bn.SetStates("G", []string{"A", "B", "C"})
	_ = bn.SetStates("L", []string{"weak", "strong"})
	_ = bn.SetStates("S", []string{"low", "high"})

	dir := t.TempDir()
	path := filepath.Join(dir, "student.bif")

	if err := bn.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadBayesianNetwork(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if err := loaded.CheckModel(); err != nil {
		t.Fatalf("loaded CheckModel: %v", err)
	}

	if len(loaded.Nodes()) != 5 {
		t.Errorf("loaded %d nodes, expected 5", len(loaded.Nodes()))
	}
	if len(loaded.Edges()) != 4 {
		t.Errorf("loaded %d edges, expected 4", len(loaded.Edges()))
	}
}

// ---- TestRemoveNodeCleanupEdges ----

func TestRemoveNodeCleanupEdges(t *testing.T) {
	bn := buildStudentNetwork(t)
	// G has parents D, I and child L.
	if err := bn.RemoveNode("G"); err != nil {
		t.Fatalf("RemoveNode: %v", err)
	}
	// D should have no children now.
	if len(bn.Children("D")) != 0 {
		t.Error("D should have no children after removing G")
	}
	// L should have no parents.
	if len(bn.Parents("L")) != 0 {
		t.Error("L should have no parents after removing G")
	}
}

// ---- TestGetMarkovBlanketLeaf ----

func TestGetMarkovBlanketLeaf(t *testing.T) {
	bn := buildStudentNetwork(t)
	// L is a leaf node. Its blanket is its parent G.
	blanket, err := bn.GetMarkovBlanket("L")
	if err != nil {
		t.Fatalf("GetMarkovBlanket: %v", err)
	}
	if len(blanket) != 1 || blanket[0] != "G" {
		t.Errorf("expected blanket [G], got %v", blanket)
	}
}

// ---- TestGetStateProbabilityPartial ----

func TestGetStateProbabilityPartial(t *testing.T) {
	bn := buildSmallAB(t)

	// P(B=0) = P(B=0|A=0)*P(A=0) + P(B=0|A=1)*P(A=1)
	// = 0.2*0.6 + 0.75*0.4 = 0.12 + 0.30 = 0.42
	p, err := bn.GetStateProbability(map[string]int{"B": 0})
	if err != nil {
		t.Fatalf("GetStateProbability: %v", err)
	}
	if math.Abs(p-0.42) > 0.01 {
		t.Errorf("expected P(B=0) ~ 0.42, got %f", p)
	}
}

// ---- Test helpers coverage ----

func TestToIntConversions(t *testing.T) {
	tests := []struct {
		input any
		want  int
	}{
		{int(5), 5},
		{int8(3), 3},
		{int16(3), 3},
		{int32(3), 3},
		{int64(3), 3},
		{uint(7), 7},
		{uint8(7), 7},
		{uint16(7), 7},
		{uint32(7), 7},
		{uint64(7), 7},
		{float64(2.9), 2},
		{float32(2.9), 2},
		{"unknown", 0},
		{nil, 0},
	}
	for _, tt := range tests {
		got := toInt(tt.input)
		if got != tt.want {
			t.Errorf("toInt(%v) = %d, want %d", tt.input, got, tt.want)
		}
	}
}
