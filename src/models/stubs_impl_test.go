//go:build unit

package models

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// --- SEM ToLisrel Tests ---

func TestSEMToLisrel_Chain(t *testing.T) {
	// X -> Y -> Z
	// X is exogenous; Y, Z are endogenous.
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.25)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{1.2}, -0.5, 0.3)

	result, err := s.ToLisrel()
	if err != nil {
		t.Fatalf("ToLisrel: %v", err)
	}

	// Exogenous: [X], Endogenous: [Y, Z] (sorted)
	B := result["B"]
	gamma := result["Gamma"]
	psi := result["Psi"]
	phi := result["Phi"]

	// B is 2x2 (Y, Z)
	if len(B) != 2 || len(B[0]) != 2 {
		t.Fatalf("expected B 2x2, got %dx%d", len(B), len(B[0]))
	}
	// Gamma is 2x1 (Y,Z x X)
	if len(gamma) != 2 || len(gamma[0]) != 1 {
		t.Fatalf("expected Gamma 2x1, got %dx%d", len(gamma), len(gamma[0]))
	}
	// Psi is 2x2
	if len(psi) != 2 || len(psi[0]) != 2 {
		t.Fatalf("expected Psi 2x2, got %dx%d", len(psi), len(psi[0]))
	}
	// Phi is 1x1
	if len(phi) != 1 || len(phi[0]) != 1 {
		t.Fatalf("expected Phi 1x1, got %dx%d", len(phi), len(phi[0]))
	}

	// Check B: Y has parent Z? No. endoVars sorted: [Y, Z].
	// Y(idx=0) has parent X (exogenous), so B[0][*] = 0.
	// Z(idx=1) has parent Y (endogenous, idx=0), so B[1][0] = 1.2.
	if math.Abs(B[1][0]-1.2) > 1e-10 {
		t.Errorf("expected B[1][0]=1.2, got %f", B[1][0])
	}
	if math.Abs(B[0][0]) > 1e-10 || math.Abs(B[0][1]) > 1e-10 {
		t.Errorf("expected B[0][*]=0, got %v", B[0])
	}

	// Check Gamma: Y(idx=0) has parent X(idx=0), coeff=0.5.
	if math.Abs(gamma[0][0]-0.5) > 1e-10 {
		t.Errorf("expected Gamma[0][0]=0.5, got %f", gamma[0][0])
	}
	// Z has no exogenous parents.
	if math.Abs(gamma[1][0]) > 1e-10 {
		t.Errorf("expected Gamma[1][0]=0, got %f", gamma[1][0])
	}

	// Check Psi diagonal.
	if math.Abs(psi[0][0]-0.25) > 1e-10 {
		t.Errorf("expected Psi[0][0]=0.25, got %f", psi[0][0])
	}
	if math.Abs(psi[1][1]-0.3) > 1e-10 {
		t.Errorf("expected Psi[1][1]=0.3, got %f", psi[1][1])
	}

	// Check Phi diagonal.
	if math.Abs(phi[0][0]-1.0) > 1e-10 {
		t.Errorf("expected Phi[0][0]=1.0, got %f", phi[0][0])
	}
}

func TestSEMToLisrel_AllExogenous(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("A", nil, nil, 0.0, 2.0)
	_ = s.AddEquation("B", nil, nil, 0.0, 3.0)

	result, err := s.ToLisrel()
	if err != nil {
		t.Fatalf("ToLisrel: %v", err)
	}

	// All exogenous: B, Gamma, Psi should be 0-dimensional.
	if len(result["B"]) != 0 {
		t.Errorf("expected empty B, got %d rows", len(result["B"]))
	}
	if len(result["Gamma"]) != 0 {
		t.Errorf("expected empty Gamma, got %d rows", len(result["Gamma"]))
	}
	if len(result["Psi"]) != 0 {
		t.Errorf("expected empty Psi, got %d rows", len(result["Psi"]))
	}

	// Phi should be 2x2 diagonal.
	phi := result["Phi"]
	if len(phi) != 2 || len(phi[0]) != 2 {
		t.Fatalf("expected Phi 2x2, got %dx%d", len(phi), len(phi[0]))
	}
	if math.Abs(phi[0][0]-2.0) > 1e-10 {
		t.Errorf("expected Phi[0][0]=2.0, got %f", phi[0][0])
	}
	if math.Abs(phi[1][1]-3.0) > 1e-10 {
		t.Errorf("expected Phi[1][1]=3.0, got %f", phi[1][1])
	}
}

func TestSEMToLisrel_InvalidModel(t *testing.T) {
	s := NewSEM()
	// Y references X as parent, but X has no equation.
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)

	_, err := s.ToLisrel()
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestSEMToLisrel_Empty(t *testing.T) {
	s := NewSEM()
	result, err := s.ToLisrel()
	if err != nil {
		t.Fatalf("ToLisrel on empty: %v", err)
	}
	for _, key := range []string{"B", "Gamma", "Psi", "Phi"} {
		if result[key] != nil {
			t.Errorf("expected nil %s for empty SEM", key)
		}
	}
}

// --- SEM ToStandardLisrel Tests ---

func TestSEMToStandardLisrel_Chain(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.25)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{1.2}, -0.5, 0.3)

	result, err := s.ToStandardLisrel()
	if err != nil {
		t.Fatalf("ToStandardLisrel: %v", err)
	}

	B := result["B"]
	gamma := result["Gamma"]
	psi := result["Psi"]
	phi := result["Phi"]

	// Check dimensions.
	if len(B) != 2 || len(B[0]) != 2 {
		t.Fatalf("expected B 2x2, got %dx%d", len(B), len(B[0]))
	}
	if len(gamma) != 2 || len(gamma[0]) != 1 {
		t.Fatalf("expected Gamma 2x1, got %dx%d", len(gamma), len(gamma[0]))
	}
	if len(psi) != 2 || len(psi[0]) != 2 {
		t.Fatalf("expected Psi 2x2, got %dx%d", len(psi), len(psi[0]))
	}
	if len(phi) != 1 || len(phi[0]) != 1 {
		t.Fatalf("expected Phi 1x1, got %dx%d", len(phi), len(phi[0]))
	}

	// Standardized coefficients should be finite and non-NaN.
	for i := range B {
		for j := range B[i] {
			if math.IsNaN(B[i][j]) || math.IsInf(B[i][j], 0) {
				t.Errorf("B[%d][%d] is not finite: %f", i, j, B[i][j])
			}
		}
	}
	for i := range gamma {
		for j := range gamma[i] {
			if math.IsNaN(gamma[i][j]) || math.IsInf(gamma[i][j], 0) {
				t.Errorf("Gamma[%d][%d] is not finite: %f", i, j, gamma[i][j])
			}
		}
	}

	// Standardized Phi for exogenous with only variance = Var/ImpliedVar.
	// For X (exogenous, no parents), implied variance = error variance = 1.0.
	// So phi[0][0] = 1.0/1.0 = 1.0.
	if math.Abs(phi[0][0]-1.0) > 1e-6 {
		t.Errorf("expected standardized Phi[0][0]=1.0, got %f", phi[0][0])
	}

	// Psi diagonal should be between 0 and 1 for standardized form.
	for i := range psi {
		if psi[i][i] < -1e-10 || psi[i][i] > 1.0+1e-10 {
			t.Errorf("expected Psi[%d][%d] in [0,1], got %f", i, i, psi[i][i])
		}
	}
}

func TestSEMToStandardLisrel_InvalidModel(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)

	_, err := s.ToStandardLisrel()
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestSEMToStandardLisrel_Empty(t *testing.T) {
	s := NewSEM()
	result, err := s.ToStandardLisrel()
	if err != nil {
		t.Fatalf("ToStandardLisrel on empty: %v", err)
	}
	for _, key := range []string{"B", "Gamma", "Psi", "Phi"} {
		if result[key] != nil {
			t.Errorf("expected nil %s for empty SEM", key)
		}
	}
}

func TestSEMToLisrel_MultipleExoParents(t *testing.T) {
	// X1, X2 -> Y
	s := NewSEM()
	_ = s.AddEquation("X1", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("X2", nil, nil, 0.0, 2.0)
	_ = s.AddEquation("Y", []string{"X1", "X2"}, []float64{0.3, 0.7}, 0.0, 0.5)

	result, err := s.ToLisrel()
	if err != nil {
		t.Fatalf("ToLisrel: %v", err)
	}

	gamma := result["Gamma"]
	// Y(idx=0) has 2 exogenous parents: X1(idx=0), X2(idx=1)
	if len(gamma) != 1 || len(gamma[0]) != 2 {
		t.Fatalf("expected Gamma 1x2, got %dx%d", len(gamma), len(gamma[0]))
	}
	if math.Abs(gamma[0][0]-0.3) > 1e-10 {
		t.Errorf("expected Gamma[0][0]=0.3, got %f", gamma[0][0])
	}
	if math.Abs(gamma[0][1]-0.7) > 1e-10 {
		t.Errorf("expected Gamma[0][1]=0.7, got %f", gamma[0][1])
	}
}

// --- LGBN IsIMap Tests ---

func TestLinearGaussianBN_IsIMap_Chain_Missing(t *testing.T) {
	bn := buildLGChain(t)
	// Chain X -> Y -> Z: implies X _|_ Z | Y.
	// Provide an empty list: the implied independence is missing.
	result, err := bn.IsIMap(nil)
	if err != nil {
		t.Fatalf("IsIMap: %v", err)
	}
	if result {
		t.Error("expected IsIMap false when required independence is missing")
	}
}

func TestLinearGaussianBN_IsIMap_InvalidModel(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	// No CPD.
	_, err := bn.IsIMap(nil)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestLinearGaussianBN_IsIMap_SingleNode(t *testing.T) {
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	cpd, _ := factors.NewLinearGaussianCPD("X", 0.0, nil, 1.0, nil)
	_ = bn.AddLinearGaussianCPD(cpd)

	result, err := bn.IsIMap(nil)
	if err != nil {
		t.Fatalf("IsIMap: %v", err)
	}
	if !result {
		t.Error("expected true for single node")
	}
}

func TestLinearGaussianBN_IsIMap_FullyConnected(t *testing.T) {
	// X -> Y: no non-adjacent pairs, so no independencies implied.
	bn := NewLinearGaussianBayesianNetwork()
	_ = bn.AddNode("X")
	_ = bn.AddNode("Y")
	_ = bn.AddEdge("X", "Y")

	cpdX, _ := factors.NewLinearGaussianCPD("X", 0.0, nil, 1.0, nil)
	_ = bn.AddLinearGaussianCPD(cpdX)
	cpdY, _ := factors.NewLinearGaussianCPD("Y", 0.0, []float64{1.0}, 1.0, []string{"X"})
	_ = bn.AddLinearGaussianCPD(cpdY)

	// No implied independencies, so any set of independencies works.
	result, err := bn.IsIMap(nil)
	if err != nil {
		t.Fatalf("IsIMap: %v", err)
	}
	if !result {
		t.Error("expected true when no independencies are implied")
	}
}

// --- FactorGraph ToMarkovNetwork Tests ---

func TestFactorGraphToMarkovNetwork_Basic(t *testing.T) {
	fg := buildSimpleFactorGraph(t)
	mn, err := fg.ToMarkovNetwork()
	if err != nil {
		t.Fatalf("ToMarkovNetwork: %v", err)
	}

	nodes := mn.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d: %v", len(nodes), nodes)
	}

	// Factors should be copied (2 factors).
	fs := mn.GetFactors()
	if len(fs) != 2 {
		t.Errorf("expected 2 factors, got %d", len(fs))
	}

	// Check model should pass.
	if err := mn.CheckModel(); err != nil {
		t.Fatalf("CheckModel on converted MN: %v", err)
	}
}

func TestFactorGraphToMarkovNetwork_SharedVariable(t *testing.T) {
	// Factor over {A, B, C} and {B, C, D}: edge B-C should only appear once.
	fg := NewFactorGraph()
	_ = fg.AddVariable("A", 2)
	_ = fg.AddVariable("B", 2)
	_ = fg.AddVariable("C", 2)
	_ = fg.AddVariable("D", 2)

	f1, _ := factors.NewDiscreteFactor(
		[]string{"A", "B", "C"}, []int{2, 2, 2},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3},
	)
	f2, _ := factors.NewDiscreteFactor(
		[]string{"B", "C", "D"}, []int{2, 2, 2},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3},
	)
	_ = fg.AddFactor(f1)
	_ = fg.AddFactor(f2)

	mn, err := fg.ToMarkovNetwork()
	if err != nil {
		t.Fatalf("ToMarkovNetwork: %v", err)
	}

	if len(mn.Nodes()) != 4 {
		t.Errorf("expected 4 nodes, got %d", len(mn.Nodes()))
	}

	// Should have edges: A-B, A-C, B-C, B-D, C-D = 5 edges.
	edges := mn.Edges()
	if len(edges) != 5 {
		t.Errorf("expected 5 edges, got %d: %v", len(edges), edges)
	}

	if err := mn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestFactorGraphToMarkovNetwork_Invalid(t *testing.T) {
	// Empty factor graph should fail CheckModel.
	fg := NewFactorGraph()
	_, err := fg.ToMarkovNetwork()
	if err == nil {
		t.Error("expected error for invalid (empty) factor graph")
	}
}

func TestFactorGraphToMarkovNetwork_SingleFactor(t *testing.T) {
	fg := NewFactorGraph()
	_ = fg.AddVariable("X", 2)
	f, _ := factors.NewDiscreteFactor([]string{"X"}, []int{2}, []float64{0.3, 0.7})
	_ = fg.AddFactor(f)

	mn, err := fg.ToMarkovNetwork()
	if err != nil {
		t.Fatalf("ToMarkovNetwork: %v", err)
	}

	if len(mn.Nodes()) != 1 {
		t.Errorf("expected 1 node, got %d", len(mn.Nodes()))
	}
	if len(mn.GetFactors()) != 1 {
		t.Errorf("expected 1 factor, got %d", len(mn.GetFactors()))
	}
}

func TestFactorGraphToMarkovNetwork_PartitionFunctionPreserved(t *testing.T) {
	fg := buildSimpleFactorGraph(t)
	zFG, err := fg.GetPartitionFunction()
	if err != nil {
		t.Fatalf("FG GetPartitionFunction: %v", err)
	}

	mn, err := fg.ToMarkovNetwork()
	if err != nil {
		t.Fatalf("ToMarkovNetwork: %v", err)
	}

	zMN, err := mn.GetPartitionFunction()
	if err != nil {
		t.Fatalf("MN GetPartitionFunction: %v", err)
	}

	if math.Abs(zFG-zMN) > 1e-10 {
		t.Errorf("partition functions differ: FG=%f, MN=%f", zFG, zMN)
	}
}
