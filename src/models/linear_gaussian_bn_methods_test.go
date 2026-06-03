//go:build unit

package models

import (
	"math"
	"os"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func buildLGTestData(t *testing.T, nRows int) *tabgo.DataFrame {
	t.Helper()
	// X ~ values, Y = 2*X + 1 + small noise, Z = 0.5*Y - 1
	rows := make([][]any, nRows)
	for i := 0; i < nRows; i++ {
		x := float64(i-nRows/2) / float64(nRows) * 4
		y := 2*x + 1.0 + 0.01*float64(i%3-1)
		z := 0.5*y - 1.0 + 0.005*float64(i%5-2)
		rows[i] = []any{x, y, z}
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y", "Z"}, rows)
}

func TestLinearGaussianBN_SaveLoad(t *testing.T) {
	bn := buildLGChain(t)

	tmpFile := t.TempDir() + "/lg_bn.txt"
	if err := bn.Save(tmpFile); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Check file was created.
	info, err := os.Stat(tmpFile)
	if err != nil {
		t.Fatalf("file not created: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	loaded, err := LoadLinearGaussianBayesianNetwork(tmpFile)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Verify nodes match.
	origNodes := bn.Nodes()
	loadNodes := loaded.Nodes()
	if len(origNodes) != len(loadNodes) {
		t.Fatalf("node count mismatch: %d vs %d", len(origNodes), len(loadNodes))
	}
	for i := range origNodes {
		if origNodes[i] != loadNodes[i] {
			t.Errorf("node mismatch at %d: %q vs %q", i, origNodes[i], loadNodes[i])
		}
	}

	// Verify CPD parameters match.
	for _, node := range origNodes {
		origCPD := bn.GetLinearGaussianCPD(node)
		loadCPD := loaded.GetLinearGaussianCPD(node)
		if loadCPD == nil {
			t.Fatalf("loaded CPD for %q is nil", node)
		}
		if math.Abs(origCPD.Mean()-loadCPD.Mean()) > 1e-6 {
			t.Errorf("mean mismatch for %q: %f vs %f", node, origCPD.Mean(), loadCPD.Mean())
		}
		if math.Abs(origCPD.Variance()-loadCPD.Variance()) > 1e-6 {
			t.Errorf("variance mismatch for %q: %f vs %f", node, origCPD.Variance(), loadCPD.Variance())
		}
	}
}

func TestLinearGaussianBN_SaveMissingCPD(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")

	tmpFile := t.TempDir() + "/lg_bn_bad.txt"
	err := bn.Save(tmpFile)
	if err == nil {
		t.Error("expected error for missing CPD")
	}
}

func TestLoadLinearGaussianBN_NonexistentFile(t *testing.T) {
	_, err := LoadLinearGaussianBayesianNetwork("/nonexistent/file.txt")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLinearGaussianBN_RemoveCPDs(t *testing.T) {
	bn := buildLGChain(t)

	// Verify CPDs exist.
	if bn.GetLinearGaussianCPD("X") == nil {
		t.Fatal("expected CPD for X before removal")
	}

	bn.RemoveCPDs()

	for _, node := range bn.Nodes() {
		if bn.GetLinearGaussianCPD(node) != nil {
			t.Errorf("expected nil CPD for %q after RemoveCPDs", node)
		}
	}
}

func TestLinearGaussianBN_GetRandomCPDs(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	_ = bn.AddNode("Y")
	_ = bn.AddEdge("X", "Y")

	if err := bn.GetRandomCPDs(); err != nil {
		t.Fatalf("GetRandomCPDs: %v", err)
	}

	for _, node := range bn.Nodes() {
		cpd := bn.GetLinearGaussianCPD(node)
		if cpd == nil {
			t.Errorf("expected CPD for %q after GetRandomCPDs", node)
		}
	}

	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel after GetRandomCPDs: %v", err)
	}
}

func TestLinearGaussianBN_ToJointGaussian(t *testing.T) {
	bn := buildLGChain(t)

	mu, sigma, err := bn.ToJointGaussian()
	if err != nil {
		t.Fatalf("ToJointGaussian: %v", err)
	}

	if len(mu) != 3 {
		t.Fatalf("expected 3 means, got %d", len(mu))
	}
	if len(sigma) != 3 {
		t.Fatalf("expected 3x3 covariance matrix, got %d rows", len(sigma))
	}

	// X ~ N(5, 1), so mu[X] = 5.
	// Sorted: X=0, Y=1, Z=2
	if math.Abs(mu[0]-5.0) > 1e-6 {
		t.Errorf("expected E[X]=5, got %f", mu[0])
	}
	// Y = 2 + 0.5*X, E[Y] = 2 + 0.5*5 = 4.5
	if math.Abs(mu[1]-4.5) > 1e-6 {
		t.Errorf("expected E[Y]=4.5, got %f", mu[1])
	}

	// Covariance matrix should be symmetric.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(sigma[i][j]-sigma[j][i]) > 1e-10 {
				t.Errorf("covariance not symmetric at [%d][%d]", i, j)
			}
		}
	}

	// Var(X) = 1.
	if math.Abs(sigma[0][0]-1.0) > 1e-6 {
		t.Errorf("expected Var(X)=1, got %f", sigma[0][0])
	}
}

func TestLinearGaussianBN_ToJointGaussian_Empty(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	mu, sigma, err := bn.ToJointGaussian()
	if err != nil {
		t.Fatalf("ToJointGaussian on empty: %v", err)
	}
	if mu != nil || sigma != nil {
		t.Error("expected nil for empty network")
	}
}

func TestLinearGaussianBN_LogLikelihood(t *testing.T) {
	bn := buildLGChain(t)
	data := buildLGTestData(t, 50)

	ll, err := bn.LogLikelihood(data)
	if err != nil {
		t.Fatalf("LogLikelihood: %v", err)
	}

	// Log-likelihood should be negative (log of probabilities < 1).
	if ll >= 0 {
		t.Errorf("expected negative log-likelihood, got %f", ll)
	}
}

func TestLinearGaussianBN_LogLikelihoodNil(t *testing.T) {
	bn := buildLGChain(t)
	_, err := bn.LogLikelihood(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearGaussianBN_Simulate(t *testing.T) {
	bn := buildLGChain(t)

	df, err := bn.Simulate(200)
	if err != nil {
		t.Fatalf("Simulate: %v", err)
	}
	if df.Len() != 200 {
		t.Errorf("expected 200 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 3 {
		t.Errorf("expected 3 columns, got %d", len(cols))
	}
}

func TestLinearGaussianBN_SimulateInvalid(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	// No CPD set.
	_, err := bn.Simulate(10)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestLinearGaussianBN_SimulateNonPositive(t *testing.T) {
	bn := buildLGChain(t)
	_, err := bn.Simulate(0)
	if err == nil {
		t.Error("expected error for zero samples")
	}
}

func TestLinearGaussianBN_GetCardinality(t *testing.T) {
	bn := buildLGChain(t)

	_, err := bn.GetCardinality("X")
	if err == nil {
		t.Error("expected error for continuous variable")
	}
}

func TestLinearGaussianBN_GetCardinalityNonexistent(t *testing.T) {
	bn := buildLGChain(t)
	_, err := bn.GetCardinality("Q")
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

func TestLinearGaussianBN_Fit(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	_ = bn.AddNode("Y")
	_ = bn.AddNode("Z")
	_ = bn.AddEdge("X", "Y")
	_ = bn.AddEdge("Y", "Z")

	data := buildLGTestData(t, 200)
	if err := bn.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Check Y's coefficient for X is approximately 2.
	cpd := bn.GetLinearGaussianCPD("Y")
	if cpd == nil {
		t.Fatal("expected CPD for Y")
	}
	betas := cpd.Betas()
	if len(betas) != 1 {
		t.Fatalf("expected 1 beta, got %d", len(betas))
	}
	if math.Abs(betas[0]-2.0) > 0.5 {
		t.Errorf("expected beta ~2.0, got %f", betas[0])
	}
}

func TestLinearGaussianBN_FitNilData(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	if err := bn.Fit(nil); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearGaussianBN_FitEmptyData(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	data := tabgo.NewDataFrameFromRows([]string{"X"}, nil)
	if err := bn.Fit(data); err == nil {
		t.Error("expected error for empty data")
	}
}

func TestLinearGaussianBN_PredictProbability(t *testing.T) {
	bn := buildLGChain(t)
	data := buildLGTestData(t, 10)

	probs, err := bn.PredictProbability(data)
	if err != nil {
		t.Fatalf("PredictProbability: %v", err)
	}
	if len(probs) != 10 {
		t.Errorf("expected 10 probabilities, got %d", len(probs))
	}
	// Each log-probability should be finite.
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Errorf("row %d: non-finite probability %f", i, p)
		}
	}
}

func TestLinearGaussianBN_PredictProbabilityNil(t *testing.T) {
	bn := buildLGChain(t)
	_, err := bn.PredictProbability(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearGaussianBN_Predict(t *testing.T) {
	bn := buildLGChain(t)
	data := buildLGTestData(t, 10)

	pred, err := bn.Predict(data)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Should have predictions for all variables.
	for _, node := range bn.Nodes() {
		if _, ok := pred[node]; !ok {
			t.Errorf("missing predictions for %q", node)
		}
		if len(pred[node]) != 10 {
			t.Errorf("expected 10 predictions for %q, got %d", node, len(pred[node]))
		}
	}
}

func TestLinearGaussianBN_PredictNil(t *testing.T) {
	bn := buildLGChain(t)
	_, err := bn.Predict(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearGaussianBN_ToMarkovModel(t *testing.T) {
	bn := buildLGChain(t)
	err := bn.ToMarkovModel()
	if err == nil {
		t.Error("expected error for continuous network")
	}
}

func TestLinearGaussianBN_IsIMap(t *testing.T) {
	bn := buildLGChain(t)
	// X -> Y -> Z chain: X _|_ Z | Y should hold.
	// Provide that independence.
	indeps := []IndependenceAssertion{
		{Event1: []string{"X"}, Event2: []string{"Z"}, Given: []string{"Y"}},
	}
	result, err := bn.IsIMap(indeps)
	if err != nil {
		t.Fatalf("IsIMap: %v", err)
	}
	if !result {
		t.Error("expected IsIMap to return true for chain with correct independence")
	}
}

func TestGetRandomLinearGaussianBayesianNetwork(t *testing.T) {
	bn, err := GetRandomLinearGaussianBayesianNetwork(5, 3)
	if err != nil {
		t.Fatalf("GetRandomLinearGaussianBayesianNetwork: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 5 {
		t.Errorf("expected 5 nodes, got %d", len(nodes))
	}

	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestGetRandomLinearGaussianBayesianNetwork_Invalid(t *testing.T) {
	_, err := GetRandomLinearGaussianBayesianNetwork(0, 0)
	if err == nil {
		t.Error("expected error for 0 nodes")
	}

	_, err = GetRandomLinearGaussianBayesianNetwork(3, 10)
	if err == nil {
		t.Error("expected error for too many edges")
	}
}

func TestLinearGaussianBN_FitRootOnly(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")

	rows := make([][]any, 100)
	for i := 0; i < 100; i++ {
		rows[i] = []any{float64(i)}
	}
	data := tabgo.NewDataFrameFromRows([]string{"X"}, rows)

	if err := bn.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	cpd := bn.GetLinearGaussianCPD("X")
	if cpd == nil {
		t.Fatal("expected CPD for X")
	}
	// Mean should be ~49.5.
	if math.Abs(cpd.Mean()-49.5) > 1 {
		t.Errorf("expected mean ~49.5, got %f", cpd.Mean())
	}
}

func TestLinearGaussianBN_SaveLoad_NoParents(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("A")
	cpd, _ := factors.NewLinearGaussianCPD("A", 3.0, nil, 2.0, nil)
	_ = bn.AddLinearGaussianCPD(cpd)

	tmpFile := t.TempDir() + "/lg_single.txt"
	if err := bn.Save(tmpFile); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := LoadLinearGaussianBayesianNetwork(tmpFile)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	loadCPD := loaded.GetLinearGaussianCPD("A")
	if loadCPD == nil {
		t.Fatal("loaded CPD for A is nil")
	}
	if math.Abs(loadCPD.Mean()-3.0) > 1e-6 {
		t.Errorf("mean mismatch: %f", loadCPD.Mean())
	}
	if math.Abs(loadCPD.Variance()-2.0) > 1e-6 {
		t.Errorf("variance mismatch: %f", loadCPD.Variance())
	}
}
