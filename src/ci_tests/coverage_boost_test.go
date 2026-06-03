//go:build unit

package ci_tests

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeSmallDF creates a DataFrame with few observations for edge cases.
func makeSmallDF(t *testing.T, n int) *tabgo.DataFrame {
	t.Helper()
	xs := make([]any, n)
	ys := make([]any, n)
	zs := make([]any, n)
	for i := 0; i < n; i++ {
		xs[i] = float64(i)
		ys[i] = float64(i * 2)
		zs[i] = float64(i % 2)
	}
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xs),
		"Y": tabgo.NewSeries("Y", ys),
		"Z": tabgo.NewSeries("Z", zs),
	}
	return tabgo.NewDataFrame(sm)
}

// makeConstantDF creates a DataFrame with constant X values.
func makeConstantDF(t *testing.T) *tabgo.DataFrame {
	t.Helper()
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 1.0, 1.0, 1.0, 1.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0, 1.0, 0.0}),
	}
	return tabgo.NewDataFrame(sm)
}

// makePerfectCorrelationDF creates data where X and Y are perfectly correlated.
func makePerfectCorrelationDF(t *testing.T) *tabgo.DataFrame {
	t.Helper()
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
	}
	return tabgo.NewDataFrame(sm)
}

// TestPearsonr_TooFewObs exercises the df < 1 path.
func TestPearsonr_TooFewObs(t *testing.T) {
	data := makeSmallDF(t, 2) // n=2, k=1 => df = 2-2-1 = -1 < 1
	stat, pval, indep := Pearsonr("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	if pval != 1 || !indep {
		t.Error("expected independent for too few observations")
	}
}

// TestPearsonr_PartialCorFail exercises the !ok path from partialCorFromData.
func TestPearsonr_ConstantData(t *testing.T) {
	data := makeConstantDF(t)
	stat, pval, indep := Pearsonr("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestPearsonr_PerfectCorrelation exercises the |r| >= 1 path.
func TestPearsonr_PerfectCorrelation(t *testing.T) {
	data := makePerfectCorrelationDF(t)
	stat, pval, indep := Pearsonr("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestPearsonrEquivalence_EdgeCases exercises se=0 and absR >= epsilon paths.
func TestPearsonrEquivalence_EdgeCases(t *testing.T) {
	test := PearsonrEquivalence(0.001) // very small epsilon

	// Perfect correlation => absR >= epsilon
	data := makePerfectCorrelationDF(t)
	stat, pval, indep := test("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep

	// Too few observations
	data2 := makeSmallDF(t, 2)
	stat2, pval2, indep2 := test("X", "Y", []string{"Z"}, data2, 0.05)
	_ = stat2
	_ = pval2
	_ = indep2
}

// TestMultivariateCIBase_NotOK exercises the !ok path in multivariate tests.
func TestPillaiTrace_TooFewObs(t *testing.T) {
	data := makeSmallDF(t, 2)
	stat, pval, indep := PillaiTrace("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	if pval != 1 || !indep {
		t.Error("expected independent for too few observations")
	}
}

func TestRoysLargestRoot_TooFewObs(t *testing.T) {
	data := makeSmallDF(t, 2)
	stat, pval, indep := RoysLargestRoot("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	if pval != 1 || !indep {
		t.Error("expected independent for too few observations")
	}
}

func TestWilksLambda_TooFewObs(t *testing.T) {
	data := makeSmallDF(t, 2)
	stat, pval, indep := WilksLambda("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

func TestHotellingLawley_TooFewObs(t *testing.T) {
	data := makeSmallDF(t, 2)
	stat, pval, indep := HotellingLawley("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestDiscreteTests_VeryFewData exercises the totalDF <= 0 path.
func TestChiSquare_VeryFewData(t *testing.T) {
	// Single observation => no degrees of freedom.
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a"}),
		"Y": tabgo.NewSeries("Y", []any{"b"}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := ChiSquare("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

func TestGSq_VeryFewData(t *testing.T) {
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a"}),
		"Y": tabgo.NewSeries("Y", []any{"b"}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := GSq("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

func TestPowerDivergence_VeryFewData(t *testing.T) {
	pd := PowerDivergence(2.0 / 3.0)
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a"}),
		"Y": tabgo.NewSeries("Y", []any{"b"}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := pd("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestTreeBasedCI_SmallData exercises the tree-based CI test with very few observations.
func TestTreeBasedCI_SmallData(t *testing.T) {
	data := makeSmallDF(t, 3) // Very small dataset
	stat, pval, indep := TreeBasedCI("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestTreeBasedCI_SmallDataWithZ exercises the tree residual path with conditioning.
func TestTreeBasedCI_SmallDataWithZ(t *testing.T) {
	data := makeSmallDF(t, 4) // 4 observations with 1 conditioning variable
	stat, pval, indep := TreeBasedCI("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestTreeBasedCI_ConstantTarget exercises the tree where target is constant.
func TestTreeBasedCI_ConstantTarget(t *testing.T) {
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0, 1.0, 0.0, 1.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := TreeBasedCI("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestTreeBasedCI_PerfectPrediction exercises the denom < 1e-15 path.
func TestTreeBasedCI_PerfectPrediction(t *testing.T) {
	// Y = 2*X exactly, so residuals should be zero.
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := TreeBasedCI("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestDiscreteExtra_TooFewObs exercises edge cases in discrete_extra tests.
func TestDiscreteExtra_SingleObs(t *testing.T) {
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a"}),
		"Y": tabgo.NewSeries("Y", []any{"b"}),
		"Z": tabgo.NewSeries("Z", []any{"c"}),
	}
	data := tabgo.NewDataFrame(sm)
	// These should handle single-observation gracefully.
	stat1, pval1, _ := ChiSquare("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat1
	_ = pval1
}

// TestContinuousExtra2_ResidualTooShort exercises the len(resX) < 3 path.
func TestContinuousExtra2_SmallData(t *testing.T) {
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0}),
		"Y": tabgo.NewSeries("Y", []any{3.0, 4.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := TreeBasedCI("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestTreeBasedCI_DFLessThan1 exercises the df < 1 path with conditioning.
func TestTreeBasedCI_DFLessThan1(t *testing.T) {
	// n=4, k=3: df = 4 - 2 - 3 = -1 < 1
	sm := map[string]*tabgo.Series{
		"X":  tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0}),
		"Y":  tabgo.NewSeries("Y", []any{4.0, 5.0, 6.0, 7.0}),
		"Z1": tabgo.NewSeries("Z1", []any{0.0, 1.0, 0.0, 1.0}),
		"Z2": tabgo.NewSeries("Z2", []any{1.0, 0.0, 1.0, 0.0}),
		"Z3": tabgo.NewSeries("Z3", []any{0.0, 0.0, 1.0, 1.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := TreeBasedCI("X", "Y", []string{"Z1", "Z2", "Z3"}, data, 0.05)
	_ = stat
	if pval != 1 || !indep {
		t.Error("expected df < 1 to return pval=1, indep=true")
	}
}

// TestTreeBasedCI_DenomSmall exercises the denom < 1e-15 path.
func TestTreeBasedCI_DenomSmall(t *testing.T) {
	// Perfect correlation: r = 1, denom = 1 - 1 = 0
	xs := make([]any, 20)
	ys := make([]any, 20)
	for i := 0; i < 20; i++ {
		xs[i] = float64(i)
		ys[i] = float64(i) * 2.0
	}
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xs),
		"Y": tabgo.NewSeries("Y", ys),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := TreeBasedCI("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}

// TestMultivariateCIBase_PartialCorFail exercises the !computed path.
func TestMultivariateCIBase_PartialCorFail(t *testing.T) {
	// Constant X => partialCorFromData may fail.
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{5.0, 5.0, 5.0, 5.0, 5.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
	}
	data := tabgo.NewDataFrame(sm)
	stat, pval, indep := PillaiTrace("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pval
	_ = indep
}
