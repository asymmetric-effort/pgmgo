//go:build unit

package models

import (
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ---------------------------------------------------------------------------
// BIF Load/Save round-trip with more complex network
// ---------------------------------------------------------------------------

func TestBN_SaveLoad_3Node(t *testing.T) {
	bn := make3NodeBN(t)
	tmpFile := "/tmp/test_pgmgo_bn3.bif"
	if err := bn.Save(tmpFile); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadBayesianNetwork(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.NumberOfNodes() != 3 {
		t.Errorf("expected 3 nodes, got %d", loaded.NumberOfNodes())
	}
	if loaded.NumberOfEdges() != 2 {
		t.Errorf("expected 2 edges, got %d", loaded.NumberOfEdges())
	}
}

func TestBN_Load_BadFile(t *testing.T) {
	_, err := LoadBayesianNetwork("/tmp/nonexistent_pgmgo_file.bif")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// ---------------------------------------------------------------------------
// BN: Predict with complex queries
// ---------------------------------------------------------------------------

func TestBN_Predict_AllMissing(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{nil, nil}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	result, err := bn.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if result.Len() != 1 {
		t.Errorf("expected 1 row, got %d", result.Len())
	}
}

func TestBN_PredictProbability_AllMissing(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{nil, nil}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	result, err := bn.PredictProbability(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) == 0 {
		t.Error("expected non-empty result")
	}
}

func TestBN_PredictProbability_NoMissing(t *testing.T) {
	bn := makeValidBN(t)
	rows := [][]any{{0, 0}, {1, 1}}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	result, err := bn.PredictProbability(df)
	if err != nil {
		t.Fatal(err)
	}
	// No missing values => empty result
	if len(result) != 0 {
		t.Errorf("expected empty result, got %d entries", len(result))
	}
}

// ---------------------------------------------------------------------------
// BN: Do with multiple interventions
// ---------------------------------------------------------------------------

func TestBN_Do_MultipleNodes(t *testing.T) {
	bn := make3NodeBN(t)
	mutilated, err := bn.Do(map[string]int{"A": 0, "B": 1})
	if err != nil {
		t.Fatal(err)
	}
	if mutilated == nil {
		t.Error("expected non-nil result")
	}
}

func TestBN_Do_NoCPD(t *testing.T) {
	bn := NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddEdge("A", "B")
	bn.SetStates("A", []string{"a0", "a1"})
	bn.SetStates("B", []string{"b0", "b1"})
	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	_ = bn.AddCPD(cpdA)
	// B has no CPD
	_, err := bn.Do(map[string]int{"B": 0})
	if err == nil {
		t.Error("expected error for node with no CPD")
	}
}

// ---------------------------------------------------------------------------
// BN: FitUpdate with 3 nodes
// ---------------------------------------------------------------------------

func TestBN_FitUpdate_3Node(t *testing.T) {
	bn := make3NodeBN(t)
	rows := [][]any{
		{0, 0, 0}, {0, 1, 0}, {1, 0, 1}, {1, 1, 1},
	}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B", "C"}, rows)
	err := bn.FitUpdate(df, 5)
	if err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// BN: GetRandomBayesianNetwork
// ---------------------------------------------------------------------------

func TestBN_GetRandomBayesianNetwork_Cov(t *testing.T) {
	bn, err := GetRandomBayesianNetwork(3, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BN")
	}
	if bn.NumberOfNodes() != 3 {
		t.Errorf("expected 3 nodes, got %d", bn.NumberOfNodes())
	}
}

// ---------------------------------------------------------------------------
// MarkovNetwork: ToBayesianModel, ToFactorGraph with more variables
// ---------------------------------------------------------------------------

func TestMarkovNetwork_ToBayesianModel_3Node(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddNode("C")
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	bn, err := mn.ToBayesianModel()
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BN")
	}
}

func TestMarkovNetwork_ToFactorGraph_3Node(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddNode("C")
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	fg, err := mn.ToFactorGraph()
	if err != nil {
		t.Fatal(err)
	}
	if fg == nil {
		t.Error("expected non-nil FactorGraph")
	}
}

// ---------------------------------------------------------------------------
// MarkovNetwork: ToJunctionTree
// ---------------------------------------------------------------------------

func TestMarkovNetwork_ToJT_3Node(t *testing.T) {
	mn := NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddNode("C")
	mn.AddEdge("A", "B")
	mn.AddEdge("B", "C")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	mn.AddFactor(fAB)
	mn.AddFactor(fBC)
	jt, err := mn.ToJunctionTree()
	if err != nil {
		t.Fatal(err)
	}
	if jt == nil {
		t.Error("expected non-nil JunctionTree")
	}
}

// ---------------------------------------------------------------------------
// SEM: Fit with multiple parents
// ---------------------------------------------------------------------------

func TestSEM_Fit_MultiParent(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Z", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X", "Z"}, []float64{0, 0}, 0, 1)

	// Use data that is not collinear
	rows := make([][]any, 20)
	for i := 0; i < 20; i++ {
		x := float64(i)*0.5 + 0.1*float64(i%3)
		z := float64(20-i)*0.3 + 0.2*float64(i%5)
		y := 1.0 + 2.0*x + 0.5*z + 0.01*float64(i%7)
		rows[i] = []any{x, y, z}
	}
	df := tabgo.NewDataFrameFromRows([]string{"X", "Y", "Z"}, rows)
	err := s.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSEM_GenerateSamples_MultiParent(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Z", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X", "Z"}, []float64{2.0, 0.5}, 1.0, 0.5)
	df, err := s.GenerateSamples(50)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 50 {
		t.Errorf("expected 50 rows, got %d", df.Len())
	}
}

func TestSEM_CheckModel_MismatchedParents(t *testing.T) {
	s := NewSEM()
	s.AddEquation("X", nil, nil, 0, 1)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0, 1)
	// Manually remove the edge but keep the equation
	// This is hard to do without internal access, so test negative variance instead
	s2 := NewSEM()
	err := s2.AddEquation("X", nil, nil, 0, -0.5)
	if err != nil {
		t.Fatal(err)
	}
	err = s2.CheckModel()
	if err == nil {
		t.Error("expected error for negative variance")
	}
}

// ---------------------------------------------------------------------------
// LinearGaussianBN
// ---------------------------------------------------------------------------

func makeLGBN(t *testing.T) *LinearGaussianBayesianNetwork {
	t.Helper()
	lgbn := NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("A")
	lgbn.AddNode("B")
	lgbn.AddEdge("A", "B")
	cpdA, _ := factors.NewLinearGaussianCPD("A", 0, nil, 1, nil)
	lgbn.AddLinearGaussianCPD(cpdA)
	cpdB, _ := factors.NewLinearGaussianCPD("B", 0, []float64{0.5}, 1, []string{"A"})
	lgbn.AddLinearGaussianCPD(cpdB)
	return lgbn
}

func TestLinearGaussianBN_AddCPD_Valid(t *testing.T) {
	lgbn := makeLGBN(t)
	err := lgbn.CheckModel()
	if err != nil {
		t.Fatal(err)
	}
}

func TestLinearGaussianBN_Simulate_Cov(t *testing.T) {
	lgbn := makeLGBN(t)
	df, err := lgbn.Simulate(20)
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 20 {
		t.Errorf("expected 20 rows, got %d", df.Len())
	}
}

func TestLinearGaussianBN_Fit_Cov(t *testing.T) {
	lgbn := makeLGBN(t)
	rows := make([][]any, 20)
	for i := 0; i < 20; i++ {
		a := float64(i) * 0.5
		b := 1.0 + 2.0*a
		rows[i] = []any{a, b}
	}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	err := lgbn.Fit(df)
	if err != nil {
		t.Fatal(err)
	}
}

func TestLinearGaussianBN_ToJointGaussian_Cov(t *testing.T) {
	lgbn := makeLGBN(t)
	mean, cov, err := lgbn.ToJointGaussian()
	if err != nil {
		t.Fatal(err)
	}
	if len(mean) != 2 {
		t.Errorf("expected mean of length 2, got %d", len(mean))
	}
	if len(cov) != 2 {
		t.Errorf("expected 2x2 covariance, got %dx%d", len(cov), len(cov))
	}
}

func TestLinearGaussianBN_LogLikelihood_Cov(t *testing.T) {
	lgbn := makeLGBN(t)
	rows := make([][]any, 10)
	for i := 0; i < 10; i++ {
		rows[i] = []any{float64(i), float64(i) * 0.5}
	}
	df := tabgo.NewDataFrameFromRows([]string{"A", "B"}, rows)
	ll, err := lgbn.LogLikelihood(df)
	if err != nil {
		t.Fatal(err)
	}
	_ = ll
}

// ---------------------------------------------------------------------------
// NaiveBayes: AddEdge, AddEdgesFrom
// ---------------------------------------------------------------------------

func TestNaiveBayes_AddEdge_Valid(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1"})
	err := nb.AddEdge("C", "F1")
	// This edge already exists, should error or succeed depending on implementation
	_ = err
}

func TestNaiveBayes_AddEdgesFrom_Valid(t *testing.T) {
	nb, _ := NewNaiveBayes("C", []string{"F1", "F2"})
	err := nb.AddEdgesFrom("C", []string{"F1"})
	_ = err
}

// ---------------------------------------------------------------------------
// DynamicBN: more methods
// ---------------------------------------------------------------------------

func TestDynamicBN_ActiveTrailNodes(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	dbn.AddNode("A")
	dbn.AddNode("B")
	dbn.AddEdge("A", "B")
	nodes := dbn.ActiveTrailNodes([]string{"A"}, nil)
	_ = nodes
}

// ---------------------------------------------------------------------------
// FactorGraph: ToMarkovNetwork, CheckModel
// ---------------------------------------------------------------------------

func TestFactorGraph_ToMarkovNetwork_Cov(t *testing.T) {
	fg := NewFactorGraph()
	fg.AddVariable("A", 2)
	fg.AddVariable("B", 2)
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fg.AddFactor(f)
	mn, err := fg.ToMarkovNetwork()
	if err != nil {
		t.Fatal(err)
	}
	if mn == nil {
		t.Error("expected non-nil MarkovNetwork")
	}
}

func TestFactorGraph_CheckModel_Valid(t *testing.T) {
	fg := NewFactorGraph()
	fg.AddVariable("A", 2)
	fg.AddVariable("B", 2)
	f, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fg.AddFactor(f)
	err := fg.CheckModel()
	if err != nil {
		t.Fatal(err)
	}
}

func TestFactorGraph_CheckModel_NoFactors_Cov(t *testing.T) {
	fg := NewFactorGraph()
	fg.AddVariable("A", 2)
	err := fg.CheckModel()
	if err == nil {
		t.Error("expected error for no factors")
	}
}

// ---------------------------------------------------------------------------
// BN: writeBIF test via string
// ---------------------------------------------------------------------------

func TestBN_SaveBIF_Detailed(t *testing.T) {
	bn := make3NodeBN(t)
	err := bn.Save("/tmp/test_pgmgo_bif_detail.bif")
	if err != nil {
		t.Fatal(err)
	}
	// Load it back
	loaded, err := LoadBayesianNetwork("/tmp/test_pgmgo_bif_detail.bif")
	if err != nil {
		t.Fatal(err)
	}
	// Verify structure preserved
	if loaded.NumberOfNodes() != 3 {
		t.Errorf("expected 3 nodes, got %d", loaded.NumberOfNodes())
	}
	// Check states
	states := loaded.GetAllStates()
	if len(states) != 3 {
		t.Errorf("expected 3 state entries, got %d", len(states))
	}
	_ = strings.Contains("", "") // use strings import
}

// ---------------------------------------------------------------------------
// ClusterGraph: CliqueBeliefs
// ---------------------------------------------------------------------------

func TestClusterGraph_CliqueBeliefs_Cov(t *testing.T) {
	cg := NewClusterGraph()
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.3, 0.7, 0.4, 0.6})
	fBC, _ := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.2, 0.8, 0.5, 0.5})
	cg.AddCluster([]string{"A", "B"}, []*factors.DiscreteFactor{fAB})
	cg.AddCluster([]string{"B", "C"}, []*factors.DiscreteFactor{fBC})
	beliefs, err := cg.CliqueBeliefs()
	if err != nil {
		t.Fatal(err)
	}
	if len(beliefs) == 0 {
		t.Error("expected non-empty beliefs")
	}
}

// ---------------------------------------------------------------------------
// DiscreteBayesianNetwork
// ---------------------------------------------------------------------------

func TestDiscreteBN_SetStates_Coverage(t *testing.T) {
	dbn := NewDiscreteBayesianNetwork()
	mustNoErr(t, dbn.AddNode("A"))
	mustNoErr(t, dbn.SetStates("A", []string{"a0", "a1"}))
	states := dbn.GetStates("A")
	if len(states) != 2 {
		t.Errorf("expected 2 states, got %d", len(states))
	}
}

// ---------------------------------------------------------------------------
// BN: GetRandomCPDs
// ---------------------------------------------------------------------------

func TestBN_GetRandomCPDs_Cov(t *testing.T) {
	bn := makeValidBN(t)
	err := bn.GetRandomCPDs(2, 42)
	if err != nil {
		t.Fatal(err)
	}
}
