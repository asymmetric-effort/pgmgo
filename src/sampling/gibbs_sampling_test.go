//go:build unit

package sampling

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// buildStudentBN creates the classic "Student" Bayesian network:
//
//	Difficulty(D) -> Grade(G) <- Intelligence(I)
//	                 Grade(G) -> Letter(L)
//	            Intelligence(I) -> SAT(S)
//
// All variables are binary (cardinality 2) for simplicity:
//
//	D: 0=easy, 1=hard
//	I: 0=low, 1=high
//	G: 0=low, 1=high  (depends on D, I)
//	S: 0=low, 1=high  (depends on I)
//	L: 0=weak, 1=strong (depends on G)
func buildStudentBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()

	for _, n := range []string{"D", "I", "G", "S", "L"} {
		if err := bn.AddNode(n); err != nil {
			t.Fatalf("AddNode(%q): %v", n, err)
		}
	}
	for _, e := range [][2]string{{"D", "G"}, {"I", "G"}, {"I", "S"}, {"G", "L"}} {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			t.Fatalf("AddEdge(%q->%q): %v", e[0], e[1], err)
		}
	}

	// P(D): easy=0.6, hard=0.4
	cpdD, err := factors.NewTabularCPD("D", 2, [][]float64{
		{0.6},
		{0.4},
	}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD(D): %v", err)
	}

	// P(I): low=0.7, high=0.3
	cpdI, err := factors.NewTabularCPD("I", 2, [][]float64{
		{0.7},
		{0.3},
	}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD(I): %v", err)
	}

	// P(G|D,I): columns are (D=0,I=0), (D=0,I=1), (D=1,I=0), (D=1,I=1)
	cpdG, err := factors.NewTabularCPD("G", 2, [][]float64{
		{0.3, 0.05, 0.9, 0.5}, // G=low
		{0.7, 0.95, 0.1, 0.5}, // G=high
	}, []string{"D", "I"}, []int{2, 2})
	if err != nil {
		t.Fatalf("NewTabularCPD(G): %v", err)
	}

	// P(S|I): columns are I=0, I=1
	cpdS, err := factors.NewTabularCPD("S", 2, [][]float64{
		{0.95, 0.2}, // S=low
		{0.05, 0.8}, // S=high
	}, []string{"I"}, []int{2})
	if err != nil {
		t.Fatalf("NewTabularCPD(S): %v", err)
	}

	// P(L|G): columns are G=0, G=1
	cpdL, err := factors.NewTabularCPD("L", 2, [][]float64{
		{0.9, 0.4}, // L=weak
		{0.1, 0.6}, // L=strong
	}, []string{"G"}, []int{2})
	if err != nil {
		t.Fatalf("NewTabularCPD(L): %v", err)
	}

	for _, cpd := range []*factors.TabularCPD{cpdD, cpdI, cpdG, cpdS, cpdL} {
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD: %v", err)
		}
	}

	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
	return bn
}

// computeExactMarginals computes the exact marginal probabilities for the
// Student BN by brute-force enumeration over all 2^5=32 joint assignments.
func computeExactMarginals(t *testing.T, evidence map[string]int) map[string][]float64 {
	t.Helper()

	// P(D)
	pD := []float64{0.6, 0.4}
	// P(I)
	pI := []float64{0.7, 0.3}
	// P(G|D,I) - G=0 row then G=1 row, columns (D,I): (0,0),(0,1),(1,0),(1,1)
	pGgivenDI := [2][2][2]float64{
		// G=0
		{{0.3, 0.05}, {0.9, 0.5}},
		// G=1
		{{0.7, 0.95}, {0.1, 0.5}},
	}
	// P(S|I)
	pSgivenI := [2][2]float64{
		{0.95, 0.2}, // S=0
		{0.05, 0.8}, // S=1
	}
	// P(L|G)
	pLgivenG := [2][2]float64{
		{0.9, 0.4}, // L=0
		{0.1, 0.6}, // L=1
	}

	marginals := map[string][]float64{
		"D": {0, 0},
		"I": {0, 0},
		"G": {0, 0},
		"S": {0, 0},
		"L": {0, 0},
	}

	totalWeight := 0.0
	for d := 0; d < 2; d++ {
		for i := 0; i < 2; i++ {
			for g := 0; g < 2; g++ {
				for s := 0; s < 2; s++ {
					for l := 0; l < 2; l++ {
						assignment := map[string]int{"D": d, "I": i, "G": g, "S": s, "L": l}
						// Check evidence.
						skip := false
						for ev, val := range evidence {
							if assignment[ev] != val {
								skip = true
								break
							}
						}
						if skip {
							continue
						}
						p := pD[d] * pI[i] * pGgivenDI[g][d][i] * pSgivenI[s][i] * pLgivenG[l][g]
						totalWeight += p
						marginals["D"][d] += p
						marginals["I"][i] += p
						marginals["G"][g] += p
						marginals["S"][s] += p
						marginals["L"][l] += p
					}
				}
			}
		}
	}

	// Normalize.
	for v := range marginals {
		for j := range marginals[v] {
			marginals[v][j] /= totalWeight
		}
	}
	return marginals
}

func TestNewGibbsSampling_NilBN(t *testing.T) {
	_, err := NewGibbsSampling(nil, 42)
	if err == nil {
		t.Fatal("expected error for nil BN")
	}
}

func TestNewGibbsSampling_InvalidBN(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	// No CPD -> CheckModel fails -> ToMarkovFactors fails.
	_, err := NewGibbsSampling(bn, 42)
	if err == nil {
		t.Fatal("expected error for invalid BN")
	}
}

func TestSample_InvalidParams(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 42)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	if _, err := gs.Sample(0, 100, 1, nil); err == nil {
		t.Error("expected error for n=0")
	}
	if _, err := gs.Sample(10, -1, 1, nil); err == nil {
		t.Error("expected error for negative burnIn")
	}
	if _, err := gs.Sample(10, 100, 0, nil); err == nil {
		t.Error("expected error for thinning=0")
	}
	if _, err := gs.Sample(10, 100, 1, map[string]int{"Z": 0}); err == nil {
		t.Error("expected error for unknown evidence variable")
	}
	if _, err := gs.Sample(10, 100, 1, map[string]int{"D": 5}); err == nil {
		t.Error("expected error for out-of-range evidence value")
	}
}

func TestGibbsSampling_MarginalConvergence(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 12345)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	nSamples := 50000
	burnIn := 5000
	thinning := 2
	df, err := gs.Sample(nSamples, burnIn, thinning, nil)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	if df.Len() != nSamples {
		t.Fatalf("expected %d rows, got %d", nSamples, df.Len())
	}

	exact := computeExactMarginals(t, nil)
	tolerance := 0.03

	for _, v := range []string{"D", "I", "G", "S", "L"} {
		col := df.Column(v)
		counts := col.ValueCounts()
		total := float64(col.Len())
		for state := 0; state < 2; state++ {
			cnt := 0
			if c, ok := counts[state]; ok {
				cnt = c
			}
			observed := float64(cnt) / total
			expected := exact[v][state]
			if math.Abs(observed-expected) > tolerance {
				t.Errorf("P(%s=%d): observed=%.4f, expected=%.4f, diff=%.4f > tolerance %.4f",
					v, state, observed, expected, math.Abs(observed-expected), tolerance)
			}
		}
	}
}

func TestGibbsSampling_WithEvidence(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 67890)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	evidence := map[string]int{"D": 1} // D=hard
	nSamples := 50000
	burnIn := 5000
	thinning := 2
	df, err := gs.Sample(nSamples, burnIn, thinning, evidence)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	// Verify evidence is respected: all D values should be 1.
	dCol := df.Column("D")
	dVals := dCol.Values()
	for i, v := range dVals {
		if v != 1 {
			t.Fatalf("evidence violated at row %d: D=%v, expected 1", i, v)
		}
	}

	exact := computeExactMarginals(t, evidence)
	tolerance := 0.03

	for _, v := range []string{"I", "G", "S", "L"} {
		col := df.Column(v)
		counts := col.ValueCounts()
		total := float64(col.Len())
		for state := 0; state < 2; state++ {
			cnt := 0
			if c, ok := counts[state]; ok {
				cnt = c
			}
			observed := float64(cnt) / total
			expected := exact[v][state]
			if math.Abs(observed-expected) > tolerance {
				t.Errorf("P(%s=%d|D=1): observed=%.4f, expected=%.4f, diff=%.4f > tolerance %.4f",
					v, state, observed, expected, math.Abs(observed-expected), tolerance)
			}
		}
	}
}

func TestGibbsSampling_MultipleEvidence(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 11111)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	evidence := map[string]int{"D": 0, "I": 1} // D=easy, I=high
	nSamples := 50000
	burnIn := 5000
	thinning := 2
	df, err := gs.Sample(nSamples, burnIn, thinning, evidence)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	// Verify evidence.
	dVals := df.Column("D").Values()
	iVals := df.Column("I").Values()
	for j := range dVals {
		if dVals[j] != 0 {
			t.Fatalf("evidence violated: D=%v at row %d", dVals[j], j)
		}
		if iVals[j] != 1 {
			t.Fatalf("evidence violated: I=%v at row %d", iVals[j], j)
		}
	}

	exact := computeExactMarginals(t, evidence)
	tolerance := 0.03

	for _, v := range []string{"G", "S", "L"} {
		col := df.Column(v)
		counts := col.ValueCounts()
		total := float64(col.Len())
		for state := 0; state < 2; state++ {
			cnt := 0
			if c, ok := counts[state]; ok {
				cnt = c
			}
			observed := float64(cnt) / total
			expected := exact[v][state]
			if math.Abs(observed-expected) > tolerance {
				t.Errorf("P(%s=%d|D=0,I=1): observed=%.4f, expected=%.4f, diff=%.4f > tolerance %.4f",
					v, state, observed, expected, math.Abs(observed-expected), tolerance)
			}
		}
	}
}

func TestGibbsSampling_Thinning(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 99999)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	n := 100
	burnIn := 50
	thinning := 5
	df, err := gs.Sample(n, burnIn, thinning, nil)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}
	if df.Len() != n {
		t.Errorf("expected %d samples, got %d", n, df.Len())
	}
}

func TestGibbsSampling_DataFrameColumns(t *testing.T) {
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 42)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	df, err := gs.Sample(10, 0, 1, nil)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	cols := df.Columns()
	expected := []string{"D", "G", "I", "L", "S"}
	if len(cols) != len(expected) {
		t.Fatalf("expected %d columns, got %d: %v", len(expected), len(cols), cols)
	}
	for i, c := range cols {
		if c != expected[i] {
			t.Errorf("column %d: got %q, want %q", i, c, expected[i])
		}
	}
}

func TestGibbsSampling_Deterministic(t *testing.T) {
	// Same seed should produce same results.
	bn := buildStudentBN(t)

	gs1, _ := NewGibbsSampling(bn, 42)
	df1, _ := gs1.Sample(20, 10, 1, nil)

	gs2, _ := NewGibbsSampling(bn, 42)
	df2, _ := gs2.Sample(20, 10, 1, nil)

	for _, v := range []string{"D", "I", "G", "S", "L"} {
		v1 := df1.Column(v).Values()
		v2 := df2.Column(v).Values()
		for i := range v1 {
			if v1[i] != v2[i] {
				t.Errorf("non-deterministic: %s row %d: %v vs %v", v, i, v1[i], v2[i])
			}
		}
	}
}

func TestGibbsSampling_ValidStates(t *testing.T) {
	// All sampled values should be valid state indices (0 or 1 for binary).
	bn := buildStudentBN(t)
	gs, err := NewGibbsSampling(bn, 777)
	if err != nil {
		t.Fatalf("NewGibbsSampling: %v", err)
	}

	df, err := gs.Sample(500, 100, 1, nil)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	for _, v := range []string{"D", "I", "G", "S", "L"} {
		vals := df.Column(v).Int()
		for i, val := range vals {
			if val < 0 || val > 1 {
				t.Errorf("%s row %d: invalid state %d", v, i, val)
			}
		}
	}
}
