package prediction

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// NaiveAdjustmentRegressor implements naive back-door adjustment for causal
// effect estimation via OLS regression: outcome ~ treatment + adjustmentSet.
type NaiveAdjustmentRegressor struct {
	treatment     string
	outcome       string
	adjustmentSet []string
	coefficients  []float64 // [intercept, treatment, adj1, adj2, ...]
	fitted        bool
}

// NewNaiveAdjustmentRegressor creates a new NaiveAdjustmentRegressor.
func NewNaiveAdjustmentRegressor(treatment, outcome string, adjustmentSet []string) *NaiveAdjustmentRegressor {
	a := make([]string, len(adjustmentSet))
	copy(a, adjustmentSet)
	return &NaiveAdjustmentRegressor{
		treatment:     treatment,
		outcome:       outcome,
		adjustmentSet: a,
	}
}

// Fit performs OLS regression: outcome ~ intercept + treatment + adjustmentSet.
func (r *NaiveAdjustmentRegressor) Fit(data *tabgo.DataFrame) error {
	n := data.Len()
	if n == 0 {
		return fmt.Errorf("prediction: empty DataFrame")
	}

	// Build column list: treatment first, then adjustment set.
	regressors := make([]string, 0, 1+len(r.adjustmentSet))
	regressors = append(regressors, r.treatment)
	regressors = append(regressors, r.adjustmentSet...)

	// Build design matrix with intercept.
	X, err := buildDesignMatrix(data, regressors)
	if err != nil {
		return err
	}

	y, err := extractColumnFloat64(data, r.outcome)
	if err != nil {
		return err
	}

	r.coefficients = olsFit(y, X)
	r.fitted = true
	return nil
}

// ATE returns the estimated Average Treatment Effect, which is the OLS
// coefficient on the treatment variable.
func (r *NaiveAdjustmentRegressor) ATE() float64 {
	if !r.fitted {
		return 0
	}
	// coefficients[0] = intercept, coefficients[1] = treatment
	return r.coefficients[1]
}
