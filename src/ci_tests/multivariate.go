package ci_tests

import (
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// The multivariate CI tests (Hotelling-Lawley, Pillai's trace, Roy's largest root,
// Wilks' lambda) are designed for multivariate independence testing.
//
// For the single-variable case (which is what CI tests in structure learning typically
// deal with), they reduce to F-test equivalents based on the partial correlation r:
//
//   F = r^2 * (n - 2 - |z|) / (1 - r^2), with df1=1, df2=n-2-|z|
//
// The test statistics for the single-variable case are:
//   - Hotelling-Lawley trace: T = F (equals the F-statistic)
//   - Pillai's trace: V = r^2
//   - Roy's largest root: theta = r^2 / (1 - r^2)
//   - Wilks' lambda: Lambda = 1 - r^2

// multivariateCIBase computes the partial correlation and the F-statistic
// for a single-variable CI test. Returns (r^2, F, df2, ok).
func multivariateCIBase(x, y string, z []string, data *tabgo.DataFrame) (rSq, fStat, df2 float64, ok bool) {
	n := data.Len()
	k := len(z)
	df2 = float64(n - 2 - k)

	if df2 < 1 {
		return 0, 0, 0, false
	}

	r, computed := partialCorFromData(x, y, z, data)
	if !computed {
		return 0, 0, 0, false
	}

	rSq = r * r
	if rSq >= 1 {
		rSq = 1 - 1e-15
	}

	fStat = rSq * df2 / (1 - rSq)
	return rSq, fStat, df2, true
}

// HotellingLawley is a CITest using the Hotelling-Lawley trace.
//
// For single-variable CI tests, the Hotelling-Lawley trace equals the F-statistic:
//
//	T = r^2 * (n-2-|z|) / (1-r^2) ~ F(1, n-2-|z|)
var HotellingLawley CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	rSq, fStat, df2, ok := multivariateCIBase(x, y, z, data)
	if !ok {
		return 0, 1, true
	}
	_ = rSq

	// Hotelling-Lawley trace = F for single-variable case.
	statistic := fStat
	pvalue := fSurvival(statistic, 1, df2)

	return statistic, pvalue, pvalue > significance
}

// PillaiTrace is a CITest using Pillai's trace.
//
// For single-variable CI tests, Pillai's trace equals r^2:
//
//	V = r^2
//
// The F-statistic is F = r^2 * (n-2-|z|) / (1-r^2) ~ F(1, n-2-|z|).
var PillaiTrace CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	rSq, fStat, df2, ok := multivariateCIBase(x, y, z, data)
	if !ok {
		return 0, 1, true
	}
	_ = fStat

	// Pillai's trace = r^2 for single-variable case.
	statistic := rSq
	// P-value from the F-distribution.
	pvalue := fSurvival(rSq*df2/(1-rSq), 1, df2)

	return statistic, pvalue, pvalue > significance
}

// RoysLargestRoot is a CITest using Roy's largest root.
//
// For single-variable CI tests, Roy's largest root equals r^2/(1-r^2):
//
//	theta = r^2 / (1 - r^2)
//
// The F-statistic is F = theta * (n-2-|z|) ~ F(1, n-2-|z|) but since theta = F/df2,
// we use F = theta * df2.
var RoysLargestRoot CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	rSq, fStat, df2, ok := multivariateCIBase(x, y, z, data)
	if !ok {
		return 0, 1, true
	}
	_ = fStat

	// Roy's largest root = r^2/(1-r^2) for single-variable case.
	statistic := rSq / (1 - rSq)
	// P-value: F = statistic * df2 is F(1, df2), but statistic*1 = F/df2*df2 = F.
	pvalue := fSurvival(statistic*df2, 1, df2)

	return statistic, pvalue, pvalue > significance
}

// WilksLambda is a CITest using Wilks' lambda.
//
// For single-variable CI tests, Wilks' lambda equals 1-r^2:
//
//	Lambda = 1 - r^2
//
// Smaller Lambda means stronger dependence. The F-statistic is
// F = (1-Lambda)/Lambda * df2 ~ F(1, df2).
var WilksLambda CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	rSq, fStat, df2, ok := multivariateCIBase(x, y, z, data)
	if !ok {
		return 1, 1, true
	}
	_ = fStat

	// Wilks' lambda = 1 - r^2 for single-variable case.
	statistic := 1 - rSq
	// F = (1-Lambda)/Lambda * df2 = rSq/(1-rSq) * df2
	fVal := rSq / (1 - rSq) * df2
	pvalue := fSurvival(fVal, 1, df2)

	return statistic, pvalue, pvalue > significance
}

// Compile-time interface checks.
var _ CITest = HotellingLawley
var _ CITest = PillaiTrace
var _ CITest = RoysLargestRoot
var _ CITest = WilksLambda
