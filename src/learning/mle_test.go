//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// buildStudentBN creates the student network: D -> G <- I, G -> L, I -> S.
func buildStudentBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	for _, n := range []string{"D", "G", "I", "L", "S"} {
		if err := bn.AddNode(n); err != nil {
			t.Fatalf("AddNode(%q): %v", n, err)
		}
	}
	edges := [][2]string{{"D", "G"}, {"I", "G"}, {"G", "L"}, {"I", "S"}}
	for _, e := range edges {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q): %v", e[0], e[1], err)
		}
	}
	return bn
}

// generateStudentData generates synthetic data from known CPDs for the student
// network. The true CPDs are:
//
//	D: P(D=0)=0.6, P(D=1)=0.4  (2 states: easy, hard)
//	I: P(I=0)=0.7, P(I=1)=0.3  (2 states: low, high)
//	G: 3 states, parents D,I — columns ordered by (D,I) in row-major:
//	   (D=0,I=0), (D=0,I=1), (D=1,I=0), (D=1,I=1)
//	   G=0: 0.3, 0.05, 0.9, 0.5
//	   G=1: 0.4, 0.25, 0.08, 0.3
//	   G=2: 0.3, 0.70, 0.02, 0.2
//	L: 2 states, parent G:
//	   L=0: 0.1, 0.4, 0.99
//	   L=1: 0.9, 0.6, 0.01
//	S: 2 states, parent I:
//	   S=0: 0.95, 0.2
//	   S=1: 0.05, 0.8
func generateStudentData(rng *rand.Rand, n int) (
	data *tabgo.DataFrame,
	trueCPDs map[string][][]float64,
) {
	trueCPDs = map[string][][]float64{
		"D": {{0.6}, {0.4}},
		"I": {{0.7}, {0.3}},
		"G": {
			{0.3, 0.05, 0.9, 0.5},
			{0.4, 0.25, 0.08, 0.3},
			{0.3, 0.70, 0.02, 0.2},
		},
		"L": {
			{0.1, 0.4, 0.99},
			{0.9, 0.6, 0.01},
		},
		"S": {
			{0.95, 0.2},
			{0.05, 0.8},
		},
	}

	dVals := make([]any, n)
	iVals := make([]any, n)
	gVals := make([]any, n)
	lVals := make([]any, n)
	sVals := make([]any, n)

	for row := 0; row < n; row++ {
		// Sample D.
		d := sampleDiscrete(rng, []float64{0.6, 0.4})
		// Sample I.
		i := sampleDiscrete(rng, []float64{0.7, 0.3})
		// Sample G given D, I. Parent config = D*2 + I (parents sorted: D, I).
		gCol := d*2 + i
		g := sampleDiscrete(rng, []float64{
			trueCPDs["G"][0][gCol],
			trueCPDs["G"][1][gCol],
			trueCPDs["G"][2][gCol],
		})
		// Sample L given G.
		l := sampleDiscrete(rng, []float64{
			trueCPDs["L"][0][g],
			trueCPDs["L"][1][g],
		})
		// Sample S given I.
		s := sampleDiscrete(rng, []float64{
			trueCPDs["S"][0][i],
			trueCPDs["S"][1][i],
		})

		dVals[row] = d
		iVals[row] = i
		gVals[row] = g
		lVals[row] = l
		sVals[row] = s
	}

	data = tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", dVals),
		"I": tabgo.NewSeries("I", iVals),
		"G": tabgo.NewSeries("G", gVals),
		"L": tabgo.NewSeries("L", lVals),
		"S": tabgo.NewSeries("S", sVals),
	})
	return data, trueCPDs
}

// sampleDiscrete samples from a discrete distribution given probabilities.
func sampleDiscrete(rng *rand.Rand, probs []float64) int {
	r := rng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

func TestMLE_StudentNetwork_LargeData(t *testing.T) {
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(42))
	data, trueCPDs := generateStudentData(rng, 100000)

	mle := NewMLE(bn, data)
	if err := mle.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	// Verify the model is valid.
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel after MLE: %v", err)
	}

	// Check that estimated CPDs are close to true CPDs.
	const tol = 0.02
	for _, node := range []string{"D", "I", "G", "L", "S"} {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			t.Fatalf("no CPD for node %q", node)
		}
		trueVals := trueCPDs[node]
		// Extract values from the factor.
		factor := cpd.ToFactor()
		flatVals := factor.Values().Data()

		nodeCard := cpd.VariableCard()
		evCard := cpd.EvidenceCard()
		numPC := 1
		for _, ec := range evCard {
			numPC *= ec
		}

		for cs := 0; cs < nodeCard; cs++ {
			for pc := 0; pc < numPC; pc++ {
				idx := cs*numPC + pc
				got := flatVals[idx]
				want := trueVals[cs][pc]
				if math.Abs(got-want) > tol {
					t.Errorf("node %q: CPD[%d][%d] = %.4f, want %.4f (tol=%.2f)",
						node, cs, pc, got, want, tol)
				}
			}
		}
	}
}

func TestMLE_GetParameters_SingleNode(t *testing.T) {
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(123))
	data, trueCPDs := generateStudentData(rng, 50000)

	mle := NewMLE(bn, data)

	cpd, err := mle.GetParameters("G")
	if err != nil {
		t.Fatalf("GetParameters(G): %v", err)
	}

	if cpd.Variable() != "G" {
		t.Errorf("Variable = %q, want %q", cpd.Variable(), "G")
	}
	if cpd.VariableCard() != 3 {
		t.Errorf("VariableCard = %d, want 3", cpd.VariableCard())
	}

	ev := cpd.Evidence()
	if len(ev) != 2 {
		t.Fatalf("Evidence length = %d, want 2", len(ev))
	}

	trueG := trueCPDs["G"]
	factor := cpd.ToFactor()
	flat := factor.Values().Data()
	numPC := 4 // 2 * 2
	const tol = 0.03
	for cs := 0; cs < 3; cs++ {
		for pc := 0; pc < numPC; pc++ {
			got := flat[cs*numPC+pc]
			want := trueG[cs][pc]
			if math.Abs(got-want) > tol {
				t.Errorf("G CPD[%d][%d] = %.4f, want %.4f", cs, pc, got, want)
			}
		}
	}
}

func TestMLE_EmptyData(t *testing.T) {
	bn := buildStudentBN(t)

	// Empty DataFrame with correct columns.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", []any{}),
		"I": tabgo.NewSeries("I", []any{}),
		"G": tabgo.NewSeries("G", []any{}),
		"L": tabgo.NewSeries("L", []any{}),
		"S": tabgo.NewSeries("S", []any{}),
	})

	mle := NewMLE(bn, data)
	err := mle.Estimate()
	// With empty data, maxVal returns -1 and nodeCard becomes 0 which is < 1,
	// so it gets clamped to 1. The CPD should have cardinality 1 with uniform
	// distribution. This should not error.
	if err != nil {
		t.Fatalf("Estimate with empty data: %v", err)
	}
}

func TestMLE_MissingColumn(t *testing.T) {
	bn := buildStudentBN(t)

	// DataFrame missing the "G" column.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", []any{0, 1}),
		"I": tabgo.NewSeries("I", []any{0, 1}),
		"L": tabgo.NewSeries("L", []any{0, 1}),
		"S": tabgo.NewSeries("S", []any{0, 1}),
	})

	mle := NewMLE(bn, data)
	err := mle.Estimate()
	if err == nil {
		t.Fatal("expected error for missing column, got nil")
	}
}

func TestMLE_GetParameters_MissingColumn(t *testing.T) {
	bn := buildStudentBN(t)

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", []any{0, 1}),
		"I": tabgo.NewSeries("I", []any{0, 1}),
	})

	mle := NewMLE(bn, data)

	// Request a node whose column is missing.
	_, err := mle.GetParameters("G")
	if err == nil {
		t.Fatal("expected error for missing column, got nil")
	}
}

func TestMLE_GetParameters_UnknownNode(t *testing.T) {
	bn := buildStudentBN(t)
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", []any{0}),
	})

	mle := NewMLE(bn, data)
	_, err := mle.GetParameters("Z")
	if err == nil {
		t.Fatal("expected error for unknown node, got nil")
	}
}

func TestMLE_NilBN(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0}),
	})
	mle := NewMLE(nil, data)
	if err := mle.Estimate(); err == nil {
		t.Fatal("expected error for nil BN")
	}
	if _, err := mle.GetParameters("X"); err == nil {
		t.Fatal("expected error for nil BN in GetParameters")
	}
}

func TestMLE_NilData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	mle := NewMLE(bn, nil)
	if err := mle.Estimate(); err == nil {
		t.Fatal("expected error for nil data")
	}
	if _, err := mle.GetParameters("X"); err == nil {
		t.Fatal("expected error for nil data in GetParameters")
	}
}

func TestMLE_NoParents(t *testing.T) {
	// Simple network: single node, no parents.
	bn := models.NewBayesianNetwork()
	if err := bn.AddNode("X"); err != nil {
		t.Fatal(err)
	}

	// X takes values 0, 1, 2 with counts 50, 30, 20.
	vals := make([]any, 100)
	for i := 0; i < 50; i++ {
		vals[i] = 0
	}
	for i := 50; i < 80; i++ {
		vals[i] = 1
	}
	for i := 80; i < 100; i++ {
		vals[i] = 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", vals),
	})

	mle := NewMLE(bn, data)
	if err := mle.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd := bn.GetCPD("X")
	if cpd == nil {
		t.Fatal("no CPD for X")
	}
	if cpd.VariableCard() != 3 {
		t.Fatalf("VariableCard = %d, want 3", cpd.VariableCard())
	}

	factor := cpd.ToFactor()
	flat := factor.Values().Data()

	// Expected: [0.5, 0.3, 0.2].
	expected := []float64{0.5, 0.3, 0.2}
	for i, want := range expected {
		if math.Abs(flat[i]-want) > 1e-9 {
			t.Errorf("X CPD[%d] = %.4f, want %.4f", i, flat[i], want)
		}
	}
}

func TestMLE_UniformForUnobservedParentConfig(t *testing.T) {
	// A -> B where A has 3 states, B has 2 states.
	// Data only covers A=0 and A=1, not A=2.
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	// Small known example: A -> B, A binary, B binary.
	// Data: A=0,B=0 x3; A=0,B=1 x1; A=1,B=0 x1; A=1,B=1 x3.
	bn2 := models.NewBayesianNetwork()
	_ = bn2.AddNode("A")
	_ = bn2.AddNode("B")
	_ = bn2.AddEdge("A", "B")

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 0, 0, 0, 1, 1, 1, 1}),
		"B": tabgo.NewSeries("B", []any{0, 0, 0, 1, 0, 1, 1, 1}),
	})

	mle := NewMLE(bn2, data)
	if err := mle.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd := bn2.GetCPD("B")
	if cpd == nil {
		t.Fatal("no CPD for B")
	}

	factor := cpd.ToFactor()
	flat := factor.Values().Data()
	// B has 2 states, A has 2 states.
	// P(B=0|A=0) = 3/4 = 0.75, P(B=0|A=1) = 1/4 = 0.25
	// P(B=1|A=0) = 1/4 = 0.25, P(B=1|A=1) = 3/4 = 0.75
	expected := []float64{0.75, 0.25, 0.25, 0.75}
	for i, want := range expected {
		if math.Abs(flat[i]-want) > 1e-9 {
			t.Errorf("B CPD flat[%d] = %.4f, want %.4f", i, flat[i], want)
		}
	}
}

func TestMLE_SmallData_Deterministic(t *testing.T) {
	// Verify exact counts with a tiny dataset.
	// Network: X -> Y, X binary, Y binary.
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	_ = bn.AddNode("Y")
	_ = bn.AddEdge("X", "Y")

	// All rows: X=0 -> Y=0; X=1 -> Y=1.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0, 0, 1, 1}),
		"Y": tabgo.NewSeries("Y", []any{0, 0, 1, 1}),
	})

	mle := NewMLE(bn, data)
	if err := mle.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd := bn.GetCPD("Y")
	factor := cpd.ToFactor()
	flat := factor.Values().Data()
	// P(Y=0|X=0) = 1.0, P(Y=0|X=1) = 0.0
	// P(Y=1|X=0) = 0.0, P(Y=1|X=1) = 1.0
	expected := []float64{1.0, 0.0, 0.0, 1.0}
	for i, want := range expected {
		if math.Abs(flat[i]-want) > 1e-9 {
			t.Errorf("Y CPD flat[%d] = %.4f, want %.4f", i, flat[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// EstimatePotentials
// ---------------------------------------------------------------------------

func TestEstimatePotentials_Basic(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	data := tabgo.NewDataFrameFromRows(
		[]string{"A", "B"},
		[][]any{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
			{0, 0},
			{0, 0},
		},
	)

	mle := NewMLE(bn, data)
	potentials, err := mle.EstimatePotentials()
	if err != nil {
		t.Fatalf("EstimatePotentials: %v", err)
	}
	if len(potentials) != 1 {
		t.Fatalf("expected 1 factor (one edge), got %d", len(potentials))
	}

	// Factor should have variables A and B, cardinality [2, 2].
	f := potentials[0]
	vars := f.Variables()
	if len(vars) != 2 {
		t.Errorf("expected 2 variables, got %d", len(vars))
	}

	// Check counts: A=0,B=0 -> 3 times; A=0,B=1 -> 1; A=1,B=0 -> 1; A=1,B=1 -> 1
	vals := f.Values().Data()
	total := 0.0
	for _, v := range vals {
		total += v
	}
	if total != 6.0 {
		t.Errorf("expected total count 6, got %f", total)
	}
}

func TestEstimatePotentials_IsolatedNode(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")

	data := tabgo.NewDataFrameFromRows(
		[]string{"X"},
		[][]any{{0}, {1}, {0}},
	)

	mle := NewMLE(bn, data)
	potentials, err := mle.EstimatePotentials()
	if err != nil {
		t.Fatalf("EstimatePotentials: %v", err)
	}
	if len(potentials) != 1 {
		t.Fatalf("expected 1 unary factor, got %d", len(potentials))
	}
	vals := potentials[0].Values().Data()
	// X=0 appears twice, X=1 appears once
	if vals[0] != 2 || vals[1] != 1 {
		t.Errorf("expected [2, 1], got %v", vals)
	}
}

func TestEstimatePotentials_NilBN(t *testing.T) {
	mle := NewMLE(nil, nil)
	_, err := mle.EstimatePotentials()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMaxVal(t *testing.T) {
	if got := maxVal(nil); got != -1 {
		t.Errorf("maxVal(nil) = %d, want -1", got)
	}
	if got := maxVal([]int{}); got != -1 {
		t.Errorf("maxVal([]) = %d, want -1", got)
	}
	if got := maxVal([]int{3, 1, 4, 1, 5}); got != 5 {
		t.Errorf("maxVal([3,1,4,1,5]) = %d, want 5", got)
	}
	if got := maxVal([]int{0}); got != 0 {
		t.Errorf("maxVal([0]) = %d, want 0", got)
	}
}
