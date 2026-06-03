package prediction

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// olsFit computes ordinary least squares coefficients for y = X * beta.
// X is an n x p matrix (row-major), y is length n.
// Returns beta (length p) such that beta = (X'X)^{-1} X'y.
// Uses Gaussian elimination with partial pivoting.
func olsFit(y []float64, X [][]float64) []float64 {
	n := len(y)
	if n == 0 {
		panic("prediction: olsFit called with empty data")
	}
	p := len(X[0])
	for i := range X {
		if len(X[i]) != p {
			panic("prediction: olsFit row width mismatch")
		}
	}

	// Compute X'X (p x p)
	xtx := make([][]float64, p)
	for i := range xtx {
		xtx[i] = make([]float64, p)
		for j := range xtx[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += X[k][i] * X[k][j]
			}
			xtx[i][j] = sum
		}
	}

	// Compute X'y (p x 1)
	xty := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for k := 0; k < n; k++ {
			sum += X[k][i] * y[k]
		}
		xty[i] = sum
	}

	// Solve xtx * beta = xty via Gaussian elimination with partial pivoting.
	return solveLinearSystem(xtx, xty)
}

// solveLinearSystem solves A*x = b using Gaussian elimination with partial pivoting.
// A is p x p, b is length p. Returns x (length p).
// Modifies A and b in place.
func solveLinearSystem(A [][]float64, b []float64) []float64 {
	p := len(b)

	// Forward elimination with partial pivoting.
	for col := 0; col < p; col++ {
		// Find pivot.
		maxVal := math.Abs(A[col][col])
		maxRow := col
		for row := col + 1; row < p; row++ {
			if math.Abs(A[row][col]) > maxVal {
				maxVal = math.Abs(A[row][col])
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			panic("prediction: singular matrix in OLS")
		}
		// Swap rows.
		A[col], A[maxRow] = A[maxRow], A[col]
		b[col], b[maxRow] = b[maxRow], b[col]

		// Eliminate below.
		for row := col + 1; row < p; row++ {
			factor := A[row][col] / A[col][col]
			for j := col; j < p; j++ {
				A[row][j] -= factor * A[col][j]
			}
			b[row] -= factor * b[col]
		}
	}

	// Back substitution.
	x := make([]float64, p)
	for i := p - 1; i >= 0; i-- {
		x[i] = b[i]
		for j := i + 1; j < p; j++ {
			x[i] -= A[i][j] * x[j]
		}
		x[i] /= A[i][i]
	}
	return x
}

// extractColumnFloat64 extracts a named column from a DataFrame as []float64.
func extractColumnFloat64(data *tabgo.DataFrame, name string) (result []float64, err error) {
	defer func() {
		if r := recover(); r != nil {
			result = nil
			err = fmt.Errorf("prediction: column %q: %v", name, r)
		}
	}()
	col := data.Column(name)
	if col == nil {
		return nil, fmt.Errorf("prediction: column %q not found", name)
	}
	return col.Float64(), nil
}

// buildDesignMatrix builds a design matrix from a DataFrame for the given column names,
// prepending a column of ones (intercept).
func buildDesignMatrix(data *tabgo.DataFrame, columns []string) ([][]float64, error) {
	n := data.Len()
	cols := make([][]float64, len(columns))
	for i, name := range columns {
		c, err := extractColumnFloat64(data, name)
		if err != nil {
			return nil, err
		}
		cols[i] = c
	}
	p := len(columns) + 1 // +1 for intercept
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, p)
		row[0] = 1.0 // intercept
		for j, c := range cols {
			row[j+1] = c[i]
		}
		X[i] = row
	}
	return X, nil
}

// computeCoefficientSE computes the standard error of the coefficient at index idx.
// Given design matrix X and error variance sigma2, SE = sqrt(sigma2 * (X'X)^{-1}[idx,idx]).
func computeCoefficientSE(X [][]float64, sigma2 float64, idx int) float64 {
	n := len(X)
	if n == 0 {
		return 0
	}
	p := len(X[0])

	// Compute X'X.
	xtx := make([][]float64, p)
	for i := range xtx {
		xtx[i] = make([]float64, p)
		for j := range xtx[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += X[k][i] * X[k][j]
			}
			xtx[i][j] = sum
		}
	}

	// Invert X'X using Gauss-Jordan elimination.
	inv := invertMatrix(xtx)
	if inv == nil {
		return 0
	}
	return math.Sqrt(sigma2 * inv[idx][idx])
}

// invertMatrix inverts a square matrix using Gauss-Jordan elimination.
// Returns nil if the matrix is singular.
func invertMatrix(A [][]float64) [][]float64 {
	p := len(A)
	// Build augmented matrix [A | I].
	aug := make([][]float64, p)
	for i := 0; i < p; i++ {
		aug[i] = make([]float64, 2*p)
		copy(aug[i][:p], A[i])
		aug[i][p+i] = 1.0
	}

	for col := 0; col < p; col++ {
		// Partial pivoting.
		maxVal := math.Abs(aug[col][col])
		maxRow := col
		for row := col + 1; row < p; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				maxRow = row
			}
		}
		if maxVal < 1e-14 {
			return nil // singular
		}
		aug[col], aug[maxRow] = aug[maxRow], aug[col]

		// Scale pivot row.
		scale := aug[col][col]
		for j := 0; j < 2*p; j++ {
			aug[col][j] /= scale
		}

		// Eliminate all other rows.
		for row := 0; row < p; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*p; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Extract inverse.
	inv := make([][]float64, p)
	for i := 0; i < p; i++ {
		inv[i] = make([]float64, p)
		copy(inv[i], aug[i][p:])
	}
	return inv
}

// normalCDF computes the cumulative distribution function of the standard normal
// distribution using the rational approximation by Abramowitz and Stegun.
func normalCDF(x float64) float64 {
	// Use the complementary error function.
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// normalQuantile computes the quantile (inverse CDF) of the standard normal distribution.
// Uses the rational approximation by Peter Acklam.
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}

	// Coefficients for the rational approximation.
	const (
		a1 = -3.969683028665376e+01
		a2 = 2.209460984245205e+02
		a3 = -2.759285104469687e+02
		a4 = 1.383577518672690e+02
		a5 = -3.066479806614716e+01
		a6 = 2.506628277459239e+00

		b1 = -5.447609879822406e+01
		b2 = 1.615858368580409e+02
		b3 = -1.556989798598866e+02
		b4 = 6.680131188771972e+01
		b5 = -1.328068155288572e+01

		c1 = -7.784894002430293e-03
		c2 = -3.223964580411365e-01
		c3 = -2.400758277161838e+00
		c4 = -2.549732539343734e+00
		c5 = 4.374664141464968e+00
		c6 = 2.938163982698783e+00

		d1 = 7.784695709041462e-03
		d2 = 3.224671290700398e-01
		d3 = 2.445134137142996e+00
		d4 = 3.754408661907416e+00

		pLow  = 0.02425
		pHigh = 1 - pLow
	)

	var q, r float64

	if p < pLow {
		// Rational approximation for lower region.
		q = math.Sqrt(-2 * math.Log(p))
		return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	} else if p <= pHigh {
		// Rational approximation for central region.
		q = p - 0.5
		r = q * q
		return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q /
			(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
	} else {
		// Rational approximation for upper region.
		q = math.Sqrt(-2 * math.Log(1-p))
		return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}
}

// buildDesignMatrixNoIntercept builds a design matrix without intercept.
func buildDesignMatrixNoIntercept(data *tabgo.DataFrame, columns []string) ([][]float64, error) {
	n := data.Len()
	cols := make([][]float64, len(columns))
	for i, name := range columns {
		c, err := extractColumnFloat64(data, name)
		if err != nil {
			return nil, err
		}
		cols[i] = c
	}
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, len(columns))
		for j, c := range cols {
			row[j] = c[i]
		}
		X[i] = row
	}
	return X, nil
}
