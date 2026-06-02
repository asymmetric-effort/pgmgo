//go:build unit

package sampling

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// buildStudentNetwork constructs the classic Student Bayesian network:
//
//	D -> G <- I
//	G -> L
//	I -> S
//
// Variables and cardinalities:
//
//	D (Difficulty):   2 states {easy=0, hard=1}
//	I (Intelligence): 2 states {low=0, high=1}
//	G (Grade):        3 states {A=0, B=1, C=2}
//	L (Letter):       2 states {weak=0, strong=1}
//	S (SAT):          2 states {low=0, high=1}
func buildStudentNetwork(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"D", "I", "G", "L", "S"} {
		if err := bn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q): %v", node, err)
		}
	}

	edges := [][2]string{{"D", "G"}, {"I", "G"}, {"G", "L"}, {"I", "S"}}
	for _, e := range edges {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q): %v", e[0], e[1], err)
		}
	}

	cpdD, err := factors.NewTabularCPD("D", 2, [][]float64{
		{0.6},
		{0.4},
	}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD(D): %v", err)
	}

	cpdI, err := factors.NewTabularCPD("I", 2, [][]float64{
		{0.7},
		{0.3},
	}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD(I): %v", err)
	}

	cpdG, err := factors.NewTabularCPD("G", 3, [][]float64{
		{0.3, 0.05, 0.9, 0.5},
		{0.4, 0.25, 0.08, 0.3},
		{0.3, 0.70, 0.02, 0.2},
	}, []string{"D", "I"}, []int{2, 2})
	if err != nil {
		t.Fatalf("NewTabularCPD(G): %v", err)
	}

	cpdL, err := factors.NewTabularCPD("L", 2, [][]float64{
		{0.1, 0.4, 0.99},
		{0.9, 0.6, 0.01},
	}, []string{"G"}, []int{3})
	if err != nil {
		t.Fatalf("NewTabularCPD(L): %v", err)
	}

	cpdS, err := factors.NewTabularCPD("S", 2, [][]float64{
		{0.95, 0.2},
		{0.05, 0.8},
	}, []string{"I"}, []int{2})
	if err != nil {
		t.Fatalf("NewTabularCPD(S): %v", err)
	}

	for _, cpd := range []*factors.TabularCPD{cpdD, cpdI, cpdG, cpdL, cpdS} {
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q): %v", cpd.Variable(), err)
		}
	}

	return bn
}

// cardinalities returns the cardinality of each node in the student network.
func cardinalities() map[string]int {
	return map[string]int{
		"D": 2,
		"I": 2,
		"G": 3,
		"L": 2,
		"S": 2,
	}
}

func TestNewBayesianModelSampling(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, err := NewBayesianModelSampling(bn, 42)
	if err != nil {
		t.Fatalf("NewBayesianModelSampling: %v", err)
	}
	if bms == nil {
		t.Fatal("expected non-nil BayesianModelSampling")
	}
}

func TestNewBayesianModelSampling_NilBN(t *testing.T) {
	_, err := NewBayesianModelSampling(nil, 42)
	if err == nil {
		t.Fatal("expected error for nil BN")
	}
}

func TestNewBayesianModelSampling_InvalidModel(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	// No CPD set for X, so CheckModel should fail.
	_, err := NewBayesianModelSampling(bn, 42)
	if err == nil {
		t.Fatal("expected error for invalid model")
	}
}

func TestTopologicalOrder(t *testing.T) {
	bn := buildStudentNetwork(t)
	order, err := topologicalOrder(bn)
	if err != nil {
		t.Fatalf("topologicalOrder: %v", err)
	}

	if len(order) != 5 {
		t.Fatalf("expected 5 nodes in order, got %d", len(order))
	}

	// D and I must come before G; G must come before L; I must come before S.
	pos := make(map[string]int, len(order))
	for i, n := range order {
		pos[n] = i
	}
	if pos["D"] >= pos["G"] {
		t.Errorf("D must come before G in topological order")
	}
	if pos["I"] >= pos["G"] {
		t.Errorf("I must come before G in topological order")
	}
	if pos["G"] >= pos["L"] {
		t.Errorf("G must come before L in topological order")
	}
	if pos["I"] >= pos["S"] {
		t.Errorf("I must come before S in topological order")
	}
}

func TestForwardSample_Basic(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, err := NewBayesianModelSampling(bn, 42)
	if err != nil {
		t.Fatalf("NewBayesianModelSampling: %v", err)
	}

	n := 1000
	df, err := bms.ForwardSample(n)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}

	if df.Len() != n {
		t.Fatalf("expected %d rows, got %d", n, df.Len())
	}

	// Check all values are in valid range for each node.
	cards := cardinalities()
	for node, card := range cards {
		col := df.Column(node)
		vals := col.Int()
		for i, v := range vals {
			if v < 0 || v >= card {
				t.Errorf("sample %d: %s=%d out of range [0, %d)", i, node, v, card)
			}
		}
	}
}

func TestForwardSample_Reproducible(t *testing.T) {
	bn := buildStudentNetwork(t)

	bms1, _ := NewBayesianModelSampling(bn, 123)
	bms2, _ := NewBayesianModelSampling(bn, 123)

	df1, err := bms1.ForwardSample(50)
	if err != nil {
		t.Fatalf("ForwardSample 1: %v", err)
	}
	df2, err := bms2.ForwardSample(50)
	if err != nil {
		t.Fatalf("ForwardSample 2: %v", err)
	}

	// Same seed should produce identical samples.
	nodes := bn.Nodes()
	for _, node := range nodes {
		v1 := df1.Column(node).Int()
		v2 := df2.Column(node).Int()
		for i := range v1 {
			if v1[i] != v2[i] {
				t.Errorf("sample %d, node %s: seed reproducibility failed (%d != %d)", i, node, v1[i], v2[i])
			}
		}
	}
}

func TestForwardSample_InvalidN(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	_, err := bms.ForwardSample(0)
	if err == nil {
		t.Fatal("expected error for n=0")
	}

	_, err = bms.ForwardSample(-1)
	if err == nil {
		t.Fatal("expected error for n=-1")
	}
}

func TestForwardSample_ApproximateMarginals(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	n := 50000
	df, err := bms.ForwardSample(n)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}

	// Check P(D=0) ~ 0.6
	dVals := df.Column("D").Int()
	count0 := 0
	for _, v := range dVals {
		if v == 0 {
			count0++
		}
	}
	pD0 := float64(count0) / float64(n)
	if math.Abs(pD0-0.6) > 0.03 {
		t.Errorf("P(D=0) = %.4f, expected ~0.6", pD0)
	}

	// Check P(I=0) ~ 0.7
	iVals := df.Column("I").Int()
	count0 = 0
	for _, v := range iVals {
		if v == 0 {
			count0++
		}
	}
	pI0 := float64(count0) / float64(n)
	if math.Abs(pI0-0.7) > 0.03 {
		t.Errorf("P(I=0) = %.4f, expected ~0.7", pI0)
	}
}

func TestRejectionSample_Basic(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	evidence := map[string]int{"D": 1}
	n := 100
	df, err := bms.RejectionSample(n, evidence)
	if err != nil {
		t.Fatalf("RejectionSample: %v", err)
	}

	if df.Len() != n {
		t.Fatalf("expected %d rows, got %d", n, df.Len())
	}

	// All samples must have D=1.
	dVals := df.Column("D").Int()
	for i, v := range dVals {
		if v != 1 {
			t.Errorf("sample %d: D=%d, expected 1", i, v)
		}
	}

	// Check all values in valid range.
	cards := cardinalities()
	for node, card := range cards {
		vals := df.Column(node).Int()
		for i, v := range vals {
			if v < 0 || v >= card {
				t.Errorf("sample %d: %s=%d out of range [0, %d)", i, node, v, card)
			}
		}
	}
}

func TestRejectionSample_MultipleEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	evidence := map[string]int{"D": 0, "I": 1}
	n := 50
	df, err := bms.RejectionSample(n, evidence)
	if err != nil {
		t.Fatalf("RejectionSample: %v", err)
	}

	if df.Len() != n {
		t.Fatalf("expected %d rows, got %d", n, df.Len())
	}

	dVals := df.Column("D").Int()
	iVals := df.Column("I").Int()
	for i := range dVals {
		if dVals[i] != 0 {
			t.Errorf("sample %d: D=%d, expected 0", i, dVals[i])
		}
		if iVals[i] != 1 {
			t.Errorf("sample %d: I=%d, expected 1", i, iVals[i])
		}
	}
}

func TestRejectionSample_InvalidEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	evidence := map[string]int{"X": 0}
	_, err := bms.RejectionSample(10, evidence)
	if err == nil {
		t.Fatal("expected error for nonexistent evidence variable")
	}
}

func TestRejectionSample_InvalidN(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	_, err := bms.RejectionSample(0, nil)
	if err == nil {
		t.Fatal("expected error for n=0")
	}
}

func TestLikelihoodWeightedSample_Basic(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	evidence := map[string]int{"D": 1}
	n := 100
	df, weights, err := bms.LikelihoodWeightedSample(n, evidence)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample: %v", err)
	}

	if df.Len() != n {
		t.Fatalf("expected %d rows, got %d", n, df.Len())
	}
	if len(weights) != n {
		t.Fatalf("expected %d weights, got %d", n, len(weights))
	}

	// All evidence nodes must be fixed.
	dVals := df.Column("D").Int()
	for i, v := range dVals {
		if v != 1 {
			t.Errorf("sample %d: D=%d, expected 1", i, v)
		}
	}

	// All weights must be positive.
	for i, w := range weights {
		if w <= 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			t.Errorf("sample %d: weight=%f, expected positive", i, w)
		}
	}

	// All values in valid range.
	cards := cardinalities()
	for node, card := range cards {
		vals := df.Column(node).Int()
		for i, v := range vals {
			if v < 0 || v >= card {
				t.Errorf("sample %d: %s=%d out of range [0, %d)", i, node, v, card)
			}
		}
	}
}

func TestLikelihoodWeightedSample_WeightConsistency(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	// With D as evidence, weight = P(D=value) since D has no parents.
	// P(D=1) = 0.4, so all weights should be 0.4.
	evidence := map[string]int{"D": 1}
	n := 50
	_, weights, err := bms.LikelihoodWeightedSample(n, evidence)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample: %v", err)
	}

	for i, w := range weights {
		if math.Abs(w-0.4) > 1e-10 {
			t.Errorf("sample %d: weight=%f, expected 0.4", i, w)
		}
	}
}

func TestLikelihoodWeightedSample_MultipleEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	// D=0 and I=1: weight = P(D=0)*P(I=1) = 0.6 * 0.3 = 0.18
	// since both are root nodes with no parents.
	evidence := map[string]int{"D": 0, "I": 1}
	n := 50
	_, weights, err := bms.LikelihoodWeightedSample(n, evidence)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample: %v", err)
	}

	for i, w := range weights {
		if math.Abs(w-0.18) > 1e-10 {
			t.Errorf("sample %d: weight=%f, expected 0.18", i, w)
		}
	}
}

func TestLikelihoodWeightedSample_ChildEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	// Evidence on G (child of D, I). Weight = P(G=0 | D_sampled, I_sampled).
	// The weight depends on the sampled parent values.
	evidence := map[string]int{"G": 0}
	n := 100
	df, weights, err := bms.LikelihoodWeightedSample(n, evidence)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample: %v", err)
	}

	// Verify G is always 0.
	gVals := df.Column("G").Int()
	for i, v := range gVals {
		if v != 0 {
			t.Errorf("sample %d: G=%d, expected 0", i, v)
		}
	}

	// Verify weights match P(G=0 | D, I) for each sample.
	dVals := df.Column("D").Int()
	iVals := df.Column("I").Int()
	// P(G=0 | D, I) columns: (D=0,I=0)=0.3, (D=0,I=1)=0.05, (D=1,I=0)=0.9, (D=1,I=1)=0.5
	pG0 := map[[2]int]float64{
		{0, 0}: 0.3,
		{0, 1}: 0.05,
		{1, 0}: 0.9,
		{1, 1}: 0.5,
	}
	for i, w := range weights {
		expected := pG0[[2]int{dVals[i], iVals[i]}]
		if math.Abs(w-expected) > 1e-10 {
			t.Errorf("sample %d: weight=%f, expected %f (D=%d, I=%d)", i, w, expected, dVals[i], iVals[i])
		}
	}
}

func TestLikelihoodWeightedSample_InvalidEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	_, _, err := bms.LikelihoodWeightedSample(10, map[string]int{"X": 0})
	if err == nil {
		t.Fatal("expected error for nonexistent evidence variable")
	}
}

func TestLikelihoodWeightedSample_InvalidN(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	_, _, err := bms.LikelihoodWeightedSample(0, nil)
	if err == nil {
		t.Fatal("expected error for n=0")
	}
}

func TestLikelihoodWeightedSample_Reproducible(t *testing.T) {
	bn := buildStudentNetwork(t)

	bms1, _ := NewBayesianModelSampling(bn, 99)
	bms2, _ := NewBayesianModelSampling(bn, 99)

	evidence := map[string]int{"D": 0}
	df1, w1, _ := bms1.LikelihoodWeightedSample(30, evidence)
	df2, w2, _ := bms2.LikelihoodWeightedSample(30, evidence)

	nodes := bn.Nodes()
	for _, node := range nodes {
		v1 := df1.Column(node).Int()
		v2 := df2.Column(node).Int()
		for i := range v1 {
			if v1[i] != v2[i] {
				t.Errorf("sample %d, node %s: reproducibility failed (%d != %d)", i, node, v1[i], v2[i])
			}
		}
	}
	for i := range w1 {
		if w1[i] != w2[i] {
			t.Errorf("weight %d: reproducibility failed (%f != %f)", i, w1[i], w2[i])
		}
	}
}

func TestLikelihoodWeightedSample_NoEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	// With no evidence, all weights should be 1.0 (product of empty set).
	n := 20
	_, weights, err := bms.LikelihoodWeightedSample(n, nil)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample: %v", err)
	}

	for i, w := range weights {
		if math.Abs(w-1.0) > 1e-10 {
			t.Errorf("sample %d: weight=%f, expected 1.0 with no evidence", i, w)
		}
	}
}

func TestRejectionSample_EmptyEvidence(t *testing.T) {
	bn := buildStudentNetwork(t)
	bms, _ := NewBayesianModelSampling(bn, 42)

	// Empty evidence should accept all samples (equivalent to forward sampling).
	n := 50
	df, err := bms.RejectionSample(n, nil)
	if err != nil {
		t.Fatalf("RejectionSample: %v", err)
	}
	if df.Len() != n {
		t.Fatalf("expected %d rows, got %d", n, df.Len())
	}
}
