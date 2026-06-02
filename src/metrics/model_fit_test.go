//go:build unit

package metrics

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestPearson_PerfectPositive(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2, 4, 6, 8, 10}
	r := pearson(x, y)
	if math.Abs(r-1.0) > 1e-10 {
		t.Errorf("pearson(perfect positive) = %f, want 1.0", r)
	}
}

func TestPearson_PerfectNegative(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{10, 8, 6, 4, 2}
	r := pearson(x, y)
	if math.Abs(r-(-1.0)) > 1e-10 {
		t.Errorf("pearson(perfect negative) = %f, want -1.0", r)
	}
}

func TestPearson_Zero(t *testing.T) {
	// Symmetric pattern with no linear correlation.
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2, 1, 2, 1, 2}
	r := pearson(x, y)
	if math.Abs(r) > 0.5 {
		t.Errorf("pearson(uncorrelated) = %f, expected near 0", r)
	}
}

func TestPearson_Empty(t *testing.T) {
	r := pearson(nil, nil)
	if r != 0 {
		t.Errorf("pearson(empty) = %f, want 0", r)
	}
}

func makeSyntheticData() *tabgo.DataFrame {
	// Create correlated data: Y = 2*X + noise (small noise).
	n := 100
	rows := make([][]any, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		y := 2*x + 0.01*float64(i%3) // nearly perfect correlation
		z := float64(n - i)          // negative correlation with X
		rows[i] = []any{x, y, z}
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y", "Z"}, rows)
}

func TestCorrelationScore_CorrelatedEdges(t *testing.T) {
	data := makeSyntheticData()
	edges := [][2]string{{"X", "Y"}}
	score := CorrelationScore(edges, data)
	if score < 0.99 {
		t.Errorf("CorrelationScore(X->Y) = %f, want > 0.99", score)
	}
}

func TestCorrelationScore_MultipleEdges(t *testing.T) {
	data := makeSyntheticData()
	edges := [][2]string{{"X", "Y"}, {"X", "Z"}}
	score := CorrelationScore(edges, data)
	// Both X-Y and X-Z should have high |correlation|.
	if score < 0.9 {
		t.Errorf("CorrelationScore(X->Y, X->Z) = %f, want > 0.9", score)
	}
}

func TestCorrelationScore_NoEdges(t *testing.T) {
	data := makeSyntheticData()
	score := CorrelationScore(nil, data)
	if score != 0 {
		t.Errorf("CorrelationScore(no edges) = %f, want 0", score)
	}
}

func TestImpliedCIs_FullyConnected(t *testing.T) {
	// Fully connected graph: no implied CIs.
	edges := [][2]string{{"A", "B"}, {"B", "C"}, {"A", "C"}}
	allVars := []string{"A", "B", "C"}
	cis := ImpliedCIs(edges, allVars)
	if len(cis) != 0 {
		t.Errorf("ImpliedCIs(fully connected) = %d CIs, want 0", len(cis))
	}
}

func TestImpliedCIs_Chain(t *testing.T) {
	// A -> B -> C: A and C are non-adjacent.
	edges := [][2]string{{"A", "B"}, {"B", "C"}}
	allVars := []string{"A", "B", "C"}
	cis := ImpliedCIs(edges, allVars)
	if len(cis) != 1 {
		t.Fatalf("ImpliedCIs(chain) = %d CIs, want 1", len(cis))
	}
	ci := cis[0]
	// Should be A _||_ C | {B}
	if (ci[0][0] != "A" || ci[1][0] != "C") && (ci[0][0] != "C" || ci[1][0] != "A") {
		t.Errorf("Expected CI between A and C, got %v _||_ %v", ci[0], ci[1])
	}
	if len(ci[2]) != 1 || ci[2][0] != "B" {
		t.Errorf("Expected conditioning set {B}, got %v", ci[2])
	}
}

func TestImpliedCIs_NoEdges(t *testing.T) {
	// No edges: all pairs are non-adjacent.
	allVars := []string{"A", "B", "C"}
	cis := ImpliedCIs(nil, allVars)
	// C(3,2) = 3 CIs.
	if len(cis) != 3 {
		t.Errorf("ImpliedCIs(no edges) = %d CIs, want 3", len(cis))
	}
}

func TestFisherC_SaturatedModel(t *testing.T) {
	// Fully connected graph: no CIs, should return C=0, p=1.
	data := makeSyntheticData()
	edges := [][2]string{{"X", "Y"}, {"Y", "Z"}, {"X", "Z"}}
	stat, pval := FisherC(edges, data)
	if stat != 0 {
		t.Errorf("FisherC(saturated) statistic = %f, want 0", stat)
	}
	if pval != 1 {
		t.Errorf("FisherC(saturated) pvalue = %f, want 1", pval)
	}
}

func TestFisherC_ChainModel(t *testing.T) {
	// X -> Y -> Z with strongly correlated data.
	// The chain implies X _||_ Z | Y. With nearly deterministic data,
	// the partial correlation should be near zero, giving high p-value.
	data := makeSyntheticData()
	edges := [][2]string{{"X", "Y"}, {"Y", "Z"}}
	stat, pval := FisherC(edges, data)
	// We mainly check that FisherC returns something reasonable.
	if math.IsNaN(stat) || math.IsInf(stat, 0) {
		t.Errorf("FisherC(chain) statistic is NaN or Inf: %f", stat)
	}
	if math.IsNaN(pval) || pval < 0 || pval > 1 {
		t.Errorf("FisherC(chain) pvalue out of range: %f", pval)
	}
}

func TestChiSquareCDF_Known(t *testing.T) {
	// Chi-square with df=2: CDF(x) = 1 - exp(-x/2).
	// At x=2: CDF = 1 - exp(-1) ~ 0.6321
	got := chiSquareCDF(2, 2)
	want := 1 - math.Exp(-1)
	if math.Abs(got-want) > 1e-6 {
		t.Errorf("chiSquareCDF(2, 2) = %f, want %f", got, want)
	}

	// At x=0: CDF = 0
	got = chiSquareCDF(0, 2)
	if got != 0 {
		t.Errorf("chiSquareCDF(0, 2) = %f, want 0", got)
	}
}

func TestTTestPValue_LargeT(t *testing.T) {
	// Very large t => p-value should be near 0.
	p := tTestPValue(100, 50)
	if p > 0.001 {
		t.Errorf("tTestPValue(100, 50) = %f, want < 0.001", p)
	}
}

func TestTTestPValue_ZeroT(t *testing.T) {
	// t=0 => p-value should be 1 (not significant at all).
	p := tTestPValue(0, 50)
	if math.Abs(p-1.0) > 1e-6 {
		t.Errorf("tTestPValue(0, 50) = %f, want 1.0", p)
	}
}
