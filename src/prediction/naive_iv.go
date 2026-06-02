package prediction

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// NaiveIVRegressor implements instrumental variable regression using
// two-stage least squares (2SLS) for causal effect estimation.
type NaiveIVRegressor struct {
	treatment   string
	outcome     string
	instruments []string
	ate         float64
	fitted      bool
}

// NewNaiveIVRegressor creates a new NaiveIVRegressor.
func NewNaiveIVRegressor(treatment, outcome string, instruments []string) *NaiveIVRegressor {
	inst := make([]string, len(instruments))
	copy(inst, instruments)
	return &NaiveIVRegressor{
		treatment:   treatment,
		outcome:     outcome,
		instruments: inst,
	}
}

// Fit performs two-stage least squares:
//
//	Stage 1: treatment ~ intercept + instruments (OLS) -> predicted treatment
//	Stage 2: outcome ~ intercept + predicted_treatment (OLS) -> ATE
func (r *NaiveIVRegressor) Fit(data *tabgo.DataFrame) error {
	n := data.Len()
	if n == 0 {
		return fmt.Errorf("prediction: empty DataFrame")
	}

	// Stage 1: regress treatment on instruments.
	Xstage1, err := buildDesignMatrix(data, r.instruments)
	if err != nil {
		return fmt.Errorf("prediction: stage 1: %w", err)
	}

	tVals, err := extractColumnFloat64(data, r.treatment)
	if err != nil {
		return fmt.Errorf("prediction: stage 1: %w", err)
	}

	betaStage1 := olsFit(tVals, Xstage1)

	// Compute predicted treatment values.
	tHat := make([]float64, n)
	for i := 0; i < n; i++ {
		tHat[i] = dotProduct(betaStage1, Xstage1[i])
	}

	// Stage 2: regress outcome on predicted treatment.
	// Design matrix: [1, tHat_i] for each row.
	Xstage2 := make([][]float64, n)
	for i := 0; i < n; i++ {
		Xstage2[i] = []float64{1.0, tHat[i]}
	}

	yVals, err := extractColumnFloat64(data, r.outcome)
	if err != nil {
		return fmt.Errorf("prediction: stage 2: %w", err)
	}

	betaStage2 := olsFit(yVals, Xstage2)

	// betaStage2[0] = intercept, betaStage2[1] = treatment effect
	r.ate = betaStage2[1]
	r.fitted = true
	return nil
}

// ATE returns the estimated Average Treatment Effect from 2SLS.
func (r *NaiveIVRegressor) ATE() float64 {
	return r.ate
}
