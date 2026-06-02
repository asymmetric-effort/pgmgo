package ci_tests

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// GeneralizedCov is a CITest implementing a generalized covariance test.
//
// Similar to GCM, it regresses X on Z and Y on Z via OLS to obtain residuals,
// then tests whether the covariance of those residuals is zero. Unlike GCM
// (which tests via Pearson correlation), GeneralizedCov uses a direct t-test
// on the sample covariance:
//
//	t = cov(resid_x, resid_y) / SE
//
// where SE = sqrt(var(resid_x * resid_y) / n) and the test statistic is
// compared to a t-distribution with df = n - 2 - |z|.
//
// When z is empty, it tests covariance of the raw (centered) values.
var GeneralizedCov CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	k := len(z)

	if n < 4+k {
		return 0, 1, true
	}

	xVals := data.Column(x).Float64()
	yVals := data.Column(y).Float64()

	var resX, resY []float64

	if k == 0 {
		// No conditioning: center the values.
		resX = centerSlice(xVals)
		resY = centerSlice(yVals)
	} else {
		zVals := make([][]float64, k)
		for i, name := range z {
			zVals[i] = data.Column(name).Float64()
		}
		resX = residuals(xVals, zVals)
		resY = residuals(yVals, zVals)
	}

	nf := float64(n)

	// Compute the sample covariance of residuals: cov = (1/n) * sum(resX[i] * resY[i])
	// (Residuals are already zero-mean from OLS or centering.)
	products := make([]float64, n)
	covXY := 0.0
	for i := 0; i < n; i++ {
		products[i] = resX[i] * resY[i]
		covXY += products[i]
	}
	covXY /= nf

	// Compute the variance of the products for the standard error.
	meanProd := covXY // same as mean of products since residuals are zero-mean
	varProd := 0.0
	for i := 0; i < n; i++ {
		diff := products[i] - meanProd
		varProd += diff * diff
	}
	varProd /= nf

	// Standard error of the covariance estimate.
	se := math.Sqrt(varProd / nf)
	if se < 1e-15 {
		// If SE is essentially zero, covariance is also zero.
		return 0, 1, true
	}

	// t-statistic.
	tstat := covXY / se

	// Degrees of freedom: n - 2 - |z|
	df := float64(n - 2 - k)
	if df < 1 {
		return 0, 1, true
	}

	tdist := scigo.NewTDistribution(df)
	pvalue := 2 * tdist.SurvivalFunction(math.Abs(tstat))

	return math.Abs(tstat), pvalue, pvalue > significance
}

// centerSlice returns a new slice with the mean subtracted.
func centerSlice(vals []float64) []float64 {
	n := len(vals)
	mean := 0.0
	for _, v := range vals {
		mean += v
	}
	mean /= float64(n)
	out := make([]float64, n)
	for i, v := range vals {
		out[i] = v - mean
	}
	return out
}

// Compile-time interface check.
var _ CITest = GeneralizedCov
