package learning

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// IVEstimator implements two-stage least squares (2SLS) instrumental variables
// estimation for causal effect identification. It estimates the average
// treatment effect (ATE) of a treatment variable on an outcome using
// one or more instruments.
type IVEstimator struct {
	treatment   string
	outcome     string
	instruments []string
	ate         float64
	fitted      bool
}

// NewIVEstimator creates a new IVEstimator.
func NewIVEstimator(treatment, outcome string, instruments []string) *IVEstimator {
	iv := make([]string, len(instruments))
	copy(iv, instruments)
	return &IVEstimator{
		treatment:   treatment,
		outcome:     outcome,
		instruments: iv,
	}
}

// Fit performs two-stage least squares estimation on the provided data.
//
// Stage 1: Regress treatment on instruments to obtain predicted treatment values.
// Stage 2: Regress outcome on predicted treatment values to obtain the causal
// effect estimate.
func (iv *IVEstimator) Fit(data *tabgo.DataFrame) error {
	if data == nil {
		return fmt.Errorf("learning: data is nil")
	}
	if len(iv.instruments) == 0 {
		return fmt.Errorf("learning: at least one instrument is required")
	}

	n := data.Len()
	if n < len(iv.instruments)+1 {
		return fmt.Errorf("learning: insufficient data rows (%d) for %d parameters",
			n, len(iv.instruments)+1)
	}

	// Stage 1: Regress treatment on instruments.
	stage1 := NewLinearModel()
	if err := stage1.Fit(data, iv.treatment, iv.instruments); err != nil {
		return fmt.Errorf("learning: IV stage 1 failed: %w", err)
	}

	// Get predicted treatment values.
	treatmentHat, err := stage1.Predict(data)
	if err != nil {
		return fmt.Errorf("learning: IV stage 1 predict failed: %w", err)
	}

	// Build a DataFrame with predicted treatment and outcome for stage 2.
	treatmentHatAny := make([]any, n)
	for i, v := range treatmentHat {
		treatmentHatAny[i] = v
	}
	outcomeVals := data.Column(iv.outcome).Values()

	stage2Data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"treatment_hat": tabgo.NewSeries("treatment_hat", treatmentHatAny),
		iv.outcome:      tabgo.NewSeries(iv.outcome, outcomeVals),
	})

	// Stage 2: Regress outcome on predicted treatment.
	stage2 := NewLinearModel()
	if err := stage2.Fit(stage2Data, iv.outcome, []string{"treatment_hat"}); err != nil {
		return fmt.Errorf("learning: IV stage 2 failed: %w", err)
	}

	coeffs := stage2.Coefficients()
	if len(coeffs) != 1 {
		return fmt.Errorf("learning: unexpected coefficient count in stage 2: %d", len(coeffs))
	}

	iv.ate = coeffs[0]
	iv.fitted = true
	return nil
}

// ATE returns the estimated average treatment effect. Returns 0 if Fit has
// not been called.
func (iv *IVEstimator) ATE() float64 {
	return iv.ate
}
