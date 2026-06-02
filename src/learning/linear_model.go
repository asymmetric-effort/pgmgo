package learning

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// LinearModel implements ordinary least squares (OLS) linear regression.
// It stores the fitted coefficients and intercept after calling Fit.
type LinearModel struct {
	predictors   []string
	target       string
	coefficients []float64
	intercept    float64
	fitted       bool
}

// NewLinearModel creates a new unfitted LinearModel.
func NewLinearModel() *LinearModel {
	return &LinearModel{}
}

// Fit estimates the OLS regression coefficients for the given target variable
// using the specified predictor columns from the DataFrame.
func (lm *LinearModel) Fit(data *tabgo.DataFrame, target string, predictors []string) error {
	if data == nil {
		return fmt.Errorf("learning: data is nil")
	}
	if len(predictors) == 0 {
		return fmt.Errorf("learning: predictors must not be empty")
	}

	n := data.Len()
	k := len(predictors) + 1 // intercept + betas
	if n < k {
		return fmt.Errorf("learning: insufficient data rows (%d) for %d parameters", n, k)
	}

	// Extract target vector.
	y := data.Column(target).Float64()

	// Build design matrix X: n x k, first column is intercept.
	X := make([]float64, n*k)
	for i := 0; i < n; i++ {
		X[i*k] = 1.0
	}
	for j, p := range predictors {
		col := data.Column(p).Float64()
		for i := 0; i < n; i++ {
			X[i*k+(j+1)] = col[i]
		}
	}

	// Compute X'X and X'Y.
	xtx := make([]float64, k*k)
	xty := make([]float64, k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			sum := 0.0
			for r := 0; r < n; r++ {
				sum += X[r*k+i] * X[r*k+j]
			}
			xtx[i*k+j] = sum
		}
		sum := 0.0
		for r := 0; r < n; r++ {
			sum += X[r*k+i] * y[r]
		}
		xty[i] = sum
	}

	beta, err := solveLinearSystem(xtx, xty, k)
	if err != nil {
		return fmt.Errorf("learning: OLS solve failed: %w", err)
	}

	lm.target = target
	lm.predictors = make([]string, len(predictors))
	copy(lm.predictors, predictors)
	lm.intercept = beta[0]
	lm.coefficients = make([]float64, len(predictors))
	copy(lm.coefficients, beta[1:])
	lm.fitted = true
	return nil
}

// Predict returns predicted values for each row in the DataFrame using the
// fitted model. The DataFrame must contain all predictor columns.
func (lm *LinearModel) Predict(data *tabgo.DataFrame) ([]float64, error) {
	if !lm.fitted {
		return nil, fmt.Errorf("learning: model has not been fitted")
	}
	if data == nil {
		return nil, fmt.Errorf("learning: data is nil")
	}

	n := data.Len()
	cols := make([][]float64, len(lm.predictors))
	for i, p := range lm.predictors {
		cols[i] = data.Column(p).Float64()
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		v := lm.intercept
		for j := range lm.predictors {
			v += lm.coefficients[j] * cols[j][i]
		}
		preds[i] = v
	}
	return preds, nil
}

// Coefficients returns the fitted regression coefficients (one per predictor).
// Returns nil if the model has not been fitted.
func (lm *LinearModel) Coefficients() []float64 {
	if !lm.fitted {
		return nil
	}
	out := make([]float64, len(lm.coefficients))
	copy(out, lm.coefficients)
	return out
}

// Intercept returns the fitted intercept term.
func (lm *LinearModel) Intercept() float64 {
	return lm.intercept
}
