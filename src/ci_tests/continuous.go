package ci_tests

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// FisherZ is a CITest for continuous data that uses partial correlation and
// Fisher's Z transform to test conditional independence.
//
// It computes the partial correlation of x and y controlling for z, applies
// Fisher's Z transform, and compares the resulting statistic to the standard
// normal distribution to obtain a two-tailed p-value.
var FisherZ CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	k := len(z) // number of conditioning variables

	// Need n > k + 3 for the test to be valid (df = n - k - 3 must be positive for SE).
	if n <= k+3 {
		return 0, 1, true
	}

	// Build the column list: [x, y, z0, z1, ...]
	allCols := make([]string, 0, 2+k)
	allCols = append(allCols, x, y)
	allCols = append(allCols, z...)

	// Build the data matrix: rows x columns as [][]float64
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

	// Column indices within our matrix: x=0, y=1, z=[2..2+k)
	xIdx := 0
	yIdx := 1
	zIdx := make([]int, k)
	for i := 0; i < k; i++ {
		zIdx[i] = i + 2
	}

	// Compute partial correlation.
	r, _ := scigo.PartialCorrelation(matrix, xIdx, yIdx, zIdx)

	// Fisher's Z transform.
	zStat := scigo.FisherZTransform(r, n)

	// The standard error is 1/sqrt(n - k - 3).
	se := 1.0 / math.Sqrt(float64(n-k-3))

	// Test statistic: Z / SE ~ N(0,1) under H0.
	testStat := math.Abs(zStat) / se

	// Two-tailed p-value from standard normal.
	stdNorm := scigo.NewNormal(0, 1)
	pvalue := 2 * (1 - stdNorm.CDF(testStat))

	return testStat, pvalue, pvalue > significance
}

// Ensure FisherZ satisfies the CITest type at compile time.
var _ CITest = FisherZ
