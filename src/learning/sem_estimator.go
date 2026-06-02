package learning

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// SEMEstimator fits a Structural Equation Model (SEM) from observed data using
// OLS regression for each equation.
type SEMEstimator struct {
	sem  *models.SEM
	data *tabgo.DataFrame
}

// NewSEMEstimator creates a new SEMEstimator.
func NewSEMEstimator(sem *models.SEM, data *tabgo.DataFrame) *SEMEstimator {
	return &SEMEstimator{
		sem:  sem,
		data: data,
	}
}

// Estimate fits the SEM by performing OLS regression for each structural
// equation. For each variable in the SEM, it regresses that variable on its
// parents and updates the equation coefficients, intercept, and error variance.
func (se *SEMEstimator) Estimate() error {
	if se.sem == nil {
		return fmt.Errorf("learning: SEM is nil")
	}
	if se.data == nil {
		return fmt.Errorf("learning: data is nil")
	}

	vars := se.sem.Variables()
	if len(vars) == 0 {
		return fmt.Errorf("learning: SEM has no variables")
	}

	// Validate that all required columns exist.
	dataColumns := make(map[string]bool)
	for _, c := range se.data.Columns() {
		dataColumns[c] = true
	}
	for _, v := range vars {
		if !dataColumns[v] {
			return fmt.Errorf("learning: data is missing column for variable %q", v)
		}
	}

	n := se.data.Len()

	for _, variable := range vars {
		eq := se.sem.GetEquation(variable)
		if eq == nil {
			// Variable has no equation yet; if it has no parents, create a
			// root equation (intercept = mean, variance = sample variance).
			// Otherwise this is an error.
			return fmt.Errorf("learning: no equation defined for variable %q", variable)
		}

		parents := eq.Parents
		k := len(parents) + 1 // intercept + betas
		if n < k {
			return fmt.Errorf("learning: insufficient data rows (%d) for %d parameters for variable %q",
				n, k, variable)
		}

		y := se.data.Column(variable).Float64()

		// Build design matrix.
		X := make([]float64, n*k)
		for i := 0; i < n; i++ {
			X[i*k] = 1.0
		}
		for j, p := range parents {
			col := se.data.Column(p).Float64()
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
			return fmt.Errorf("learning: OLS solve failed for variable %q: %w", variable, err)
		}

		// Compute residual variance.
		var residualSS float64
		for i := 0; i < n; i++ {
			predicted := 0.0
			for j := 0; j < k; j++ {
				predicted += X[i*k+j] * beta[j]
			}
			residual := y[i] - predicted
			residualSS += residual * residual
		}

		denom := n - k
		if denom <= 0 {
			denom = 1
		}
		variance := residualSS / float64(denom)

		// Update the equation in the SEM.
		coefficients := make([]float64, len(parents))
		copy(coefficients, beta[1:])
		if err := se.sem.AddEquation(variable, parents, coefficients, beta[0], variance); err != nil {
			return fmt.Errorf("learning: failed to update equation for %q: %w", variable, err)
		}
	}

	return nil
}

// GetCoefficients returns the estimated coefficients, intercept, and error
// variance for the given variable's structural equation.
// Returns (betas, intercept, variance, error).
func (se *SEMEstimator) GetCoefficients(variable string) ([]float64, float64, float64, error) {
	if se.sem == nil {
		return nil, 0, 0, fmt.Errorf("learning: SEM is nil")
	}

	eq := se.sem.GetEquation(variable)
	if eq == nil {
		return nil, 0, 0, fmt.Errorf("learning: no equation found for variable %q", variable)
	}

	betas := make([]float64, len(eq.Coefficients))
	copy(betas, eq.Coefficients)
	return betas, eq.Intercept, eq.Variance, nil
}
