package ci_tests

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// fSurvival computes the survival function (1 - CDF) for the F-distribution
// with df1 and df2 degrees of freedom at value x.
// Uses the regularized incomplete beta function:
// CDF_F(x; d1, d2) = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
func fSurvival(x, df1, df2 float64) float64 {
	if x <= 0 {
		return 1
	}
	bt := df1 * x / (df1*x + df2)
	return 1 - scigo.RegularizedIncompleteBeta(bt, df1/2, df2/2)
}

// partialCorFromData extracts x, y, z columns from the DataFrame, builds
// the data matrix, and computes the partial correlation. Returns the partial
// correlation r and whether computation succeeded.
func partialCorFromData(x, y string, z []string, data *tabgo.DataFrame) (r float64, ok bool) {
	n := data.Len()
	k := len(z)
	if n <= k+2 {
		return 0, false
	}

	allCols := make([]string, 0, 2+k)
	allCols = append(allCols, x, y)
	allCols = append(allCols, z...)

	matrix := make([][]float64, n)
	colData := make([][]float64, len(allCols))
	for i, name := range allCols {
		colData[i] = data.Column(name).Float64()
	}
	for row := 0; row < n; row++ {
		matrix[row] = make([]float64, len(allCols))
		for col := 0; col < len(allCols); col++ {
			matrix[row][col] = colData[col][row]
		}
	}

	xIdx := 0
	yIdx := 1
	zIdx := make([]int, k)
	for i := 0; i < k; i++ {
		zIdx[i] = i + 2
	}

	r, _ = scigo.PartialCorrelation(matrix, xIdx, yIdx, zIdx)
	return r, true
}

// Pearsonr is a CITest for continuous data that uses the partial correlation
// and the t-distribution directly to test conditional independence.
//
// It computes the partial correlation of x and y controlling for z, then
// tests via t = r*sqrt((n-2-|z|)/(1-r^2)) with df = n-2-|z|.
var Pearsonr CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	k := len(z)

	df := float64(n - 2 - k)
	if df < 1 {
		return 0, 1, true
	}

	r, ok := partialCorFromData(x, y, z, data)
	if !ok {
		return 0, 1, true
	}

	if math.Abs(r) >= 1 {
		return math.Inf(1), 0, false
	}

	tstat := math.Abs(r) * math.Sqrt(df/(1-r*r))
	tdist := scigo.NewTDistribution(df)
	pvalue := 2 * tdist.SurvivalFunction(tstat)

	return tstat, pvalue, pvalue > significance
}

// PearsonrEquivalence returns a CITest that performs an equivalence test
// for conditional independence. It rejects independence if |r| > epsilon.
//
// The null hypothesis is that the absolute partial correlation exceeds epsilon.
// It uses a TOST-like approach: compute the partial correlation and test
// whether |r| is significantly less than epsilon using the t-distribution.
func PearsonrEquivalence(epsilon float64) CITest {
	return func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
		n := data.Len()
		k := len(z)

		df := float64(n - 2 - k)
		if df < 1 {
			return 0, 1, true
		}

		r, ok := partialCorFromData(x, y, z, data)
		if !ok {
			return 0, 1, true
		}

		absR := math.Abs(r)

		// If |r| >= epsilon, cannot claim equivalence (independence).
		if absR >= epsilon {
			// Test statistic: how far |r| exceeds epsilon
			tstat := absR * math.Sqrt(df/(1-r*r))
			return tstat, 1, false
		}

		// TOST: test H0: |rho| >= epsilon vs H1: |rho| < epsilon
		// Use the t-statistic for the upper bound.
		// t = (|r| - epsilon) / SE(r), where SE(r) = sqrt((1-r^2)/df)
		se := math.Sqrt((1 - r*r) / df)
		if se == 0 {
			return 0, 1, true
		}
		tstat := (absR - epsilon) / se
		// Under H0, tstat should be negative for equivalence.
		tdist := scigo.NewTDistribution(df)
		pvalue := tdist.CDF(tstat)

		return math.Abs(tstat), pvalue, pvalue <= significance
	}
}

// residuals computes the residuals of regressing target on predictors using OLS.
// Returns the residual vector. If there are no predictors, returns centered values.
func residuals(target []float64, predictors [][]float64) []float64 {
	n := len(target)
	p := len(predictors)

	if p == 0 {
		// Center the target
		mean := 0.0
		for _, v := range target {
			mean += v
		}
		mean /= float64(n)
		res := make([]float64, n)
		for i, v := range target {
			res[i] = v - mean
		}
		return res
	}

	// OLS via normal equations: X^T X beta = X^T y
	// Build X matrix (n x (p+1)) with intercept column.
	np1 := p + 1

	// Compute X^T X (np1 x np1) and X^T y (np1)
	xtx := make([]float64, np1*np1)
	xty := make([]float64, np1)

	for i := 0; i < n; i++ {
		// Row i of X: [1, predictors[0][i], predictors[1][i], ...]
		xi := make([]float64, np1)
		xi[0] = 1
		for j := 0; j < p; j++ {
			xi[j+1] = predictors[j][i]
		}
		for a := 0; a < np1; a++ {
			for b := 0; b < np1; b++ {
				xtx[a*np1+b] += xi[a] * xi[b]
			}
			xty[a] += xi[a] * target[i]
		}
	}

	// Solve using Gaussian elimination with partial pivoting.
	beta := solveLinearSystem(xtx, xty, np1)

	// Compute residuals: r = y - X*beta
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		pred := beta[0]
		for j := 0; j < p; j++ {
			pred += beta[j+1] * predictors[j][i]
		}
		res[i] = target[i] - pred
	}
	return res
}

// solveLinearSystem solves A*x = b via Gaussian elimination with partial pivoting.
// A is n x n stored row-major, b is length n.
func solveLinearSystem(A []float64, b []float64, n int) []float64 {
	// Work on copies.
	a := make([]float64, n*n)
	copy(a, A)
	x := make([]float64, n)
	copy(x, b)

	// Forward elimination with partial pivoting.
	for col := 0; col < n; col++ {
		// Find pivot.
		maxVal := math.Abs(a[col*n+col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if v := math.Abs(a[row*n+col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}

		// Swap rows.
		if maxRow != col {
			for j := 0; j < n; j++ {
				a[col*n+j], a[maxRow*n+j] = a[maxRow*n+j], a[col*n+j]
			}
			x[col], x[maxRow] = x[maxRow], x[col]
		}

		pivot := a[col*n+col]
		if math.Abs(pivot) < 1e-15 {
			continue // Singular or near-singular.
		}

		for row := col + 1; row < n; row++ {
			factor := a[row*n+col] / pivot
			for j := col; j < n; j++ {
				a[row*n+j] -= factor * a[col*n+j]
			}
			x[row] -= factor * x[col]
		}
	}

	// Back substitution.
	for col := n - 1; col >= 0; col-- {
		if math.Abs(a[col*n+col]) < 1e-15 {
			x[col] = 0
			continue
		}
		for j := col + 1; j < n; j++ {
			x[col] -= a[col*n+j] * x[j]
		}
		x[col] /= a[col*n+col]
	}

	return x
}

// GCM is a CITest implementing the Generalized Covariance Measure.
//
// It regresses X on Z and Y on Z (via OLS), then tests whether the residuals
// are uncorrelated using the Pearson correlation t-test on the residuals.
// When z is empty, it reduces to a standard Pearson correlation test.
var GCM CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	k := len(z)

	if n < 4 {
		return 0, 1, true
	}

	xVals := data.Column(x).Float64()
	yVals := data.Column(y).Float64()

	var resX, resY []float64

	if k == 0 {
		// No conditioning: just test correlation directly.
		resX = xVals
		resY = yVals
	} else {
		// Build predictor matrix from z columns.
		zVals := make([][]float64, k)
		for i, name := range z {
			zVals[i] = data.Column(name).Float64()
		}
		resX = residuals(xVals, zVals)
		resY = residuals(yVals, zVals)
	}

	// Test correlation of residuals.
	if len(resX) < 3 {
		return 0, 1, true
	}

	r, pvalue := scigo.PearsonCorrelation(resX, resY)
	tstat := math.Abs(r) * math.Sqrt(float64(n-2-k)/(1-r*r))
	if math.IsNaN(tstat) || math.IsInf(tstat, 0) {
		if math.Abs(r) >= 1 {
			return math.Abs(r), 0, false
		}
		return 0, 1, true
	}

	// Use the pvalue from the correlation but adjusted for the conditioning.
	df := float64(n - 2 - k)
	if df < 1 {
		return tstat, pvalue, pvalue > significance
	}
	tdist := scigo.NewTDistribution(df)
	pvalue = 2 * tdist.SurvivalFunction(math.Abs(tstat))

	return tstat, pvalue, pvalue > significance
}

// Compile-time interface checks.
var _ CITest = Pearsonr
var _ CITest = PearsonrEquivalence(0.1)
var _ CITest = GCM
