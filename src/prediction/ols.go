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
func extractColumnFloat64(data *tabgo.DataFrame, name string) ([]float64, error) {
	defer func() {
		if r := recover(); r != nil {
			// will be caught by the caller through the error return
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
