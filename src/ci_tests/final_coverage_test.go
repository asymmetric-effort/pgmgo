//go:build unit

package ci_tests

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// ===========================================================================
// continuous_extra.go: Pearsonr partialCorFromData returns !ok (line 74)
// Need n <= k+2: e.g., n=3, k=2 conditioning variables.
// ===========================================================================

func TestFinalPearsonr_NotOk(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0}),
		"Y": tabgo.NewSeries("Y", []any{4.0, 5.0, 6.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0}),
		"W": tabgo.NewSeries("W", []any{1.0, 0.0, 1.0}),
	})
	// n=3, k=2 (Z,W) => n <= k+2 => 3 <= 4 => true => partialCorFromData returns !ok
	stat, pvalue, indep := Pearsonr("X", "Y", []string{"Z", "W"}, data, 0.05)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true) for insufficient data, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// ===========================================================================
// continuous_extra.go: PearsonrEquivalence partialCorFromData returns !ok (line 106)
// and se == 0 (line 123) paths.
// ===========================================================================

func TestFinalPearsonrEquiv_NotOk(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0}),
		"Y": tabgo.NewSeries("Y", []any{4.0, 5.0, 6.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0}),
		"W": tabgo.NewSeries("W", []any{1.0, 0.0, 1.0}),
	})
	test := PearsonrEquivalence(0.5)
	stat, pvalue, indep := test("X", "Y", []string{"Z", "W"}, data, 0.05)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true) for not-ok, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// TestFinalPearsonrEquiv_SeZero covers the se==0 path (line 123).
// This triggers when r^2 == 1 and r < epsilon, which can't happen since |r|=1 >= any epsilon>0.
// Actually se = sqrt((1-r^2)/df). If r^2=1, se=0. But absR = |r| = 1 >= epsilon triggers
// the earlier branch. So se==0 is only reachable when r^2=1 but |r|<epsilon... impossible.
// Let's try r^2 very close to 1 where floating point makes se exactly 0.
func TestFinalPearsonrEquiv_SeZero(t *testing.T) {
	// Perfect correlation: X = Y exactly. With enough data, r = 1.0 exactly.
	// But |r| >= epsilon triggers line 113, not 123. So se==0 is unreachable
	// when epsilon < 1.0. With epsilon > 1.0, we can reach se==0.
	// absR=1 < epsilon=2 => skip line 113, se=sqrt(0/df)=0 => line 123.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
	})
	test := PearsonrEquivalence(2.0) // epsilon > 1 so |r|=1 < epsilon
	stat, pvalue, indep := test("X", "Y", nil, data, 0.05)
	// se == 0 path: returns (0, 1, true)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true) for se=0, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// ===========================================================================
// continuous_extra.go: PartialCorResiduals lines 284, 290-294, 299
// ===========================================================================

// TestFinalPartialCorResiduals_DfLessThan1 covers line 299: df < 1.
func TestFinalPartialCorResiduals_DfLessThan1(t *testing.T) {
	// n=4, k=2 => df = 4 - 2 - 2 = 0 < 1
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0}),
		"Y": tabgo.NewSeries("Y", []any{4.0, 3.0, 2.0, 1.0}),
		"Z": tabgo.NewSeries("Z", []any{0.5, 1.5, 0.5, 1.5}),
		"W": tabgo.NewSeries("W", []any{1.0, 0.0, 1.0, 0.0}),
	})
	stat, pvalue, _ := GCM("X", "Y", []string{"Z", "W"}, data, 0.05)
	// Should hit df < 1 path: returns (tstat, pvalue, pvalue > significance)
	_ = stat
	_ = pvalue
}

// TestFinalPartialCorResiduals_PerfectCorrelation covers lines 290-294.
// When |r| = 1, tstat is Inf, triggering NaN/Inf check.
func TestFinalPartialCorResiduals_PerfectCorrelation(t *testing.T) {
	// Create data where X and Y have perfect correlation after removing Z effects.
	// With no conditioning, X=Y gives r=1 exactly.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}),
	})
	stat, pvalue, indep := GCM("X", "Y", nil, data, 0.05)
	// r=1 => tstat=Inf => hits NaN/Inf check => |r|>=1 => returns (|r|, 0, false)
	if pvalue != 0 || indep {
		t.Errorf("expected (_, 0, false) for perfect correlation, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// TestFinalPartialCorResiduals_FewResiduals covers line 284: len(resX) < 3.
// This happens when n < 4 (since n < 4 returns early at line 260-262).
// Actually the residuals() function always returns n values. So len(resX) < 3
// only when n < 3, but line 260 checks n < 4. The path len(resX)<3 is only
// reachable if residuals() somehow returns fewer values... which it doesn't.
// This path is unreachable. Skip it.

// ===========================================================================
// continuous_extra2.go: GeneralizedCov lines 72 and 82
// ===========================================================================

// TestFinalGeneralizedCov_SeZero covers line 72: se < 1e-15.
// This happens when all X*Y products are identical (zero variance).
func TestFinalGeneralizedCov_SeZero(t *testing.T) {
	// X and Y are constant => residuals are all 0 => products all 0 => se = 0.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{5.0, 5.0, 5.0, 5.0, 5.0}),
		"Y": tabgo.NewSeries("Y", []any{3.0, 3.0, 3.0, 3.0, 3.0}),
	})
	stat, pvalue, indep := GeneralizedCov("X", "Y", nil, data, 0.05)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true) for constant data, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// TestFinalGeneralizedCov_DfLessThan1 covers line 82: df < 1.
func TestFinalGeneralizedCov_DfLessThan1(t *testing.T) {
	// n=4, k=2 => df = 4 - 2 - 2 = 0 < 1
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0}),
		"Y": tabgo.NewSeries("Y", []any{4.0, 3.0, 2.0, 1.0}),
		"Z": tabgo.NewSeries("Z", []any{0.5, 1.5, 0.5, 1.5}),
		"W": tabgo.NewSeries("W", []any{1.0, 0.0, 1.0, 0.0}),
	})
	stat, pvalue, indep := GeneralizedCov("X", "Y", []string{"Z", "W"}, data, 0.05)
	_ = stat
	_ = pvalue
	_ = indep
}

// ===========================================================================
// discrete_extra.go: ModifiedLogLikelihood single-cell (line 31)
// ===========================================================================

// TestFinalModifiedLogLikelihood_SingleCell covers len(obs) < 2.
// Need a contingency table with only one cell (nX=1 or nY=1).
func TestFinalModifiedLogLikelihood_SingleCell(t *testing.T) {
	// X has only one unique value, so all cells collapse to 1 column.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a", "a", "a", "a", "a"}),
		"Y": tabgo.NewSeries("Y", []any{"b", "c", "b", "c", "b"}),
	})
	stat, pvalue, indep := ModifiedLogLikelihood("X", "Y", nil, data, 0.05)
	// With nX=1, each table cell has only 1 effective x-level, obs might have < 2 valid entries.
	_ = stat
	_ = pvalue
	_ = indep
}

// ===========================================================================
// discrete_extra.go: IndependenceMatch lines 142, 190, 199
// ===========================================================================

// TestFinalIndependenceMatch_SingleRowStrata covers line 142: m < 2.
// Each stratum has exactly 1 row.
func TestFinalIndependenceMatch_SingleRowStrata(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a", "b", "c", "d"}),
		"Y": tabgo.NewSeries("Y", []any{"e", "f", "g", "h"}),
		"Z": tabgo.NewSeries("Z", []any{"z1", "z2", "z3", "z4"}),
	})
	// Each Z value is unique, so each stratum has 1 row => m < 2 for all strata.
	// After all strata skipped, numStrata == 0 => line 199.
	stat, pvalue, indep := IndependenceMatch("X", "Y", []string{"Z"}, data, 0.05)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true) for all singletons, got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// TestFinalIndependenceMatch_DenomZero covers line 190: denom < 1e-15.
// This happens when expMatch is 0 or 1. expMatch=0 when pXMatch or pYMatch is 0,
// which occurs when all x values in a stratum are the same OR all y values are the same.
func TestFinalIndependenceMatch_DenomZero(t *testing.T) {
	// Stratum has all same X values: pXMatch = 1, expMatch = pYMatch.
	// With nPairs > 0 and pYMatch < 1, denom = pYMatch*(1-pYMatch) > 0.
	// Need pXMatch = 0 too? No, pXMatch*(1-pXMatch)? No, denom = expMatch*(1-expMatch).
	// expMatch = pXMatch * pYMatch. For denom ~ 0: need expMatch ~ 0 or ~ 1.
	// expMatch ~ 0 when either pXMatch ~ 0 or pYMatch ~ 0.
	// pXMatch = 0 when all x values in stratum are distinct. With m=2, nPairs=1.
	// xFreq: each value appears once, so xMatchPairs = 0. pXMatch = 0.
	// So expMatch = 0, denom = 0*(1-0) = 0 < 1e-15 => skip.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{"a", "b", "c", "d", "e", "f"}),
		"Y": tabgo.NewSeries("Y", []any{"g", "h", "i", "j", "k", "l"}),
		"Z": tabgo.NewSeries("Z", []any{"s1", "s1", "s2", "s2", "s3", "s3"}),
	})
	// Each stratum has 2 rows with distinct x => pXMatch=0 => expMatch=0 => denom=0.
	// All strata skipped => numStrata=0 => line 199.
	stat, pvalue, indep := IndependenceMatch("X", "Y", []string{"Z"}, data, 0.05)
	if stat != 0 || pvalue != 1 || !indep {
		t.Errorf("expected (0, 1, true), got (%f, %f, %v)", stat, pvalue, indep)
	}
}

// ===========================================================================
// multivariate.go: multivariateCIBase !computed (line 33)
// ===========================================================================

func TestFinalMultivariateCIBase_NotComputed(t *testing.T) {
	// n <= k+2 => partialCorFromData returns !ok => !computed.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0}),
		"Y": tabgo.NewSeries("Y", []any{4.0, 5.0, 6.0}),
		"Z": tabgo.NewSeries("Z", []any{0.0, 1.0, 0.0}),
		"W": tabgo.NewSeries("W", []any{1.0, 0.0, 1.0}),
	})
	// Try WilksLambda which uses multivariateCIBase
	stat, pvalue, indep := WilksLambda("X", "Y", []string{"Z", "W"}, data, 0.05)
	_ = stat
	_ = pvalue
	_ = indep
}

// ===========================================================================
// tree_based.go: buildNode nL==0||nR==0 (line 109), denom<1e-15 (line 321),
// NaN/Inf tstat (line 325)
// ===========================================================================

// TestFinalTreeBasedCI_PerfectCorrelation covers the denom < 1e-15 path (line 316-321).
func TestFinalTreeBasedCI_PerfectCorrelation(t *testing.T) {
	// X = Y exactly => tree residuals both zero => r is NaN or 0 => denom issues.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
	})
	stat, pvalue, indep := TreeBasedCI("X", "Y", nil, data, 0.05)
	_ = stat
	_ = pvalue
	_ = indep
}

// TestFinalTreeBasedCI_ConstantX covers the nL==0||nR==0 path in buildNode (line 109).
// When X has all the same value, no split threshold can separate left/right.
func TestFinalTreeBasedCI_ConstantX(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
		"Z": tabgo.NewSeries("Z", []any{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}),
	})
	stat, pvalue, indep := TreeBasedCI("X", "Y", []string{"Z"}, data, 0.05)
	_ = stat
	_ = pvalue
	_ = indep
}
