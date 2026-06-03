//go:build unit

package metrics

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// TestSHD_DoubleChange tests the SHD case where two independent edge changes
// exist for a pair (not a reversal), triggering the dist += 2 path.
func TestSHD_DoubleChange(t *testing.T) {
	// True graph: A->B and B->A (bidirectional)
	// Estimated graph: neither A->B nor B->A
	// This gives diffs=2 but is not a pure reversal => dist += 2.
	trueG := graphgo.NewDiGraph()
	trueG.AddNodes("A", "B")
	trueG.AddEdge("A", "B")
	trueG.AddEdge("B", "A")

	estG := graphgo.NewDiGraph()
	estG.AddNodes("A", "B")

	dist := SHD(trueG, estG)
	if dist != 2 {
		t.Errorf("expected SHD=2 for bidirectional removal, got %d", dist)
	}
}

// TestImpliedCIs_EdgesWithVarsNotInAllVars exercises the path where edge
// variables are not in allVars, causing nil adj map entries.
func TestImpliedCIs_EdgesWithVarsNotInAllVars(t *testing.T) {
	// Edge references "X" and "Y" which are not in allVars.
	edges := [][2]string{{"X", "Y"}}
	allVars := []string{"A", "B"}
	cis := ImpliedCIs(edges, allVars)
	// A and B are non-adjacent, so there should be 1 CI.
	if len(cis) != 1 {
		t.Errorf("expected 1 CI, got %d", len(cis))
	}
}

// TestFisherC_SmallDFT exercises the dfT < 1 fallback in FisherC.
// With very few observations and many conditioning variables.
func TestFisherC_SmallDFT(t *testing.T) {
	// Create a graph where non-adjacent pairs have many common neighbors.
	// A-B, A-C, A-D connected; B-D not connected, conditioning on {A,C} => condVars=[A,C]
	// With n=3 observations, dfT = 3 - 2 - 2 = -1 < 1, triggers fallback.
	edges := [][2]string{{"A", "B"}, {"A", "C"}, {"A", "D"}, {"C", "B"}, {"C", "D"}}
	sm := map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{1.0, 2.0, 3.0}),
		"B": tabgo.NewSeries("B", []any{2.0, 4.0, 6.0}),
		"C": tabgo.NewSeries("C", []any{0.5, 1.5, 2.5}),
		"D": tabgo.NewSeries("D", []any{3.0, 1.0, 2.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval := FisherC(edges, data)
	if stat < 0 {
		t.Errorf("expected non-negative statistic, got %f", stat)
	}
	if pval < 0 || pval > 1 {
		t.Errorf("expected p-value in [0,1], got %f", pval)
	}
}

// TestFisherC_PerfectCorrelation exercises the r2 >= 1 path.
func TestFisherC_PerfectCorrelation(t *testing.T) {
	// Two perfectly correlated variables with no edge between them
	// should give r2 close to or equal to 1.
	edges := [][2]string{{"A", "B"}}
	sm := map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
		"B": tabgo.NewSeries("B", []any{2.0, 4.0, 6.0, 8.0, 10.0}),
		"C": tabgo.NewSeries("C", []any{10.0, 20.0, 30.0, 40.0, 50.0}), // perfectly correlated with A and B
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval := FisherC(edges, data)
	_ = stat
	_ = pval
}

// TestRegularizedIncompleteBeta_TinyD exercises the tiny d path in the
// initial computation where 1-(a+b)*x/(a+1) is very close to zero.
func TestRegularizedIncompleteBeta_TinyD(t *testing.T) {
	// We need 1 - (a+b)*x/(a+1) ≈ 0, so x ≈ (a+1)/(a+b).
	// With a=1, b=1, x = (1+1)/(1+1) = 1.0 -> x must be < 1.
	// Try a=1, b=99: x ≈ 2/100 = 0.02, but we also need x <= threshold.
	// threshold = (a+1)/(a+b+2) = 2/102 ≈ 0.0196.
	// d_init = 1 - (1+99)*0.02/(1+1) = 1 - 100*0.02/2 = 1 - 1 = 0 => tiny!
	a, b := 1.0, 99.0
	x := (a + 1) / (a + b) // = 0.02
	// Ensure x <= threshold so symmetry is not used
	result := regularizedIncompleteBeta(x, a, b)
	if result < 0 || result > 1 {
		t.Errorf("expected value in [0,1], got %f", result)
	}
}

// TestUpperGammaCF_TinyB0 exercises the tiny b0 path where x+1-a ≈ 0.
func TestUpperGammaCF_TinyB0(t *testing.T) {
	// b0 = x + 1 - a. For b0 ≈ 0, set x = a - 1.
	// a = 2.0, x = 1.0 => b0 = 1+1-2 = 0 => tiny!
	result := upperGammaCF(2.0, 1.0)
	if result < 0 || result > 1 {
		t.Errorf("expected value in [0,1], got %f", result)
	}
}

// TestUpperGammaCF_InnerTiny exercises the inner loop tiny paths.
func TestUpperGammaCF_InnerTiny(t *testing.T) {
	// Use parameters that cause d or c to become tiny during iteration.
	// Large a with small x leads to numerically tricky iterations.
	result := upperGammaCF(100.0, 0.5)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Errorf("expected finite result, got %f", result)
	}

	// Very large a with x close to a causes certain CF terms to nearly cancel.
	result2 := upperGammaCF(1000.0, 999.0)
	_ = result2

	// a very close to x+1 to make bn = x + 2i + 1 - a close to zero for early i.
	// bn = x + 2*i + 1 - a. For i=1, bn = x+3-a. If a=x+3, bn=0 => triggers tiny.
	result3 := upperGammaCF(5.0, 2.0) // i=1: bn=2+3-5=0!
	_ = result3

	// For i=1: an = -1*(1-a) = -(1-a), bn = x+3-a
	// d = bn + an*d_prev. If bn≈0 and an*d≈0 then d≈0 => triggers tiny.
	result4 := upperGammaCF(105.0, 102.0) // i=1: bn=102+3-105=0
	_ = result4

	// Try many values where bn = x + 2i + 1 - a = 0 for different i.
	for i := 1; i <= 10; i++ {
		a := float64(2*i) + 2.0
		x := a - float64(2*i) - 1.0 // bn=0 for this i
		if x > 0 {
			r := upperGammaCF(a, x)
			_ = r
		}
	}

	// Small a, large x
	result5 := upperGammaCF(0.001, 100.0)
	_ = result5
}

// TestRegularizedIncompleteBeta_InnerTiny exercises the inner loop where
// d or c become tiny during the continued fraction iteration.
func TestRegularizedIncompleteBeta_InnerTiny(t *testing.T) {
	// Large b with small x can cause extreme values in CF iteration.
	result := regularizedIncompleteBeta(0.001, 0.5, 500.0)
	if result < 0 || result > 1 {
		t.Errorf("expected value in [0,1], got %f", result)
	}

	// Extreme parameter combinations to trigger tiny guards.
	// Very small a and b values near zero with x near the threshold.
	for _, tc := range []struct{ x, a, b float64 }{
		{0.01, 0.001, 0.001},
		{0.99, 100.0, 0.01},
		{0.5, 0.001, 1000.0},
		{0.001, 1000.0, 1000.0},
		{0.999, 0.01, 0.01},
		{0.5, 0.01, 0.01},
		{0.1, 0.5, 0.5},
		{0.001, 0.001, 100.0},
	} {
		v := regularizedIncompleteBeta(tc.x, tc.a, tc.b)
		if math.IsNaN(v) {
			t.Errorf("NaN for x=%v, a=%v, b=%v", tc.x, tc.a, tc.b)
		}
	}
}

// TestFisherC_VerySmallPValue exercises the p < 1e-300 clamping path.
func TestFisherC_VerySmallPValue(t *testing.T) {
	// Create highly correlated non-adjacent variables to produce tiny p-values.
	n := 100
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	for i := 0; i < n; i++ {
		v := float64(i)
		aVals[i] = v
		bVals[i] = v * 2
		cVals[i] = v*3 + 0.001*float64(i%7)
	}
	edges := [][2]string{{"A", "B"}} // C not adjacent to A or B
	sm := map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval := FisherC(edges, data)
	if stat < 0 {
		t.Errorf("expected non-negative statistic, got %f", stat)
	}
	_ = pval
}
