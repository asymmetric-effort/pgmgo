package prediction

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// DoubleMLRegressor implements Double Machine Learning for causal effect estimation.
// It uses cross-fitting with two folds to estimate the Average Treatment Effect (ATE).
type DoubleMLRegressor struct {
	treatment   string
	outcome     string
	confounders []string
	ate         float64
	fitted      bool
}

// NewDoubleMLRegressor creates a new DoubleMLRegressor.
func NewDoubleMLRegressor(treatment, outcome string, confounders []string) *DoubleMLRegressor {
	c := make([]string, len(confounders))
	copy(c, confounders)
	return &DoubleMLRegressor{
		treatment:   treatment,
		outcome:     outcome,
		confounders: c,
	}
}

// Fit performs DML estimation with cross-fitting on the given data.
//
// Algorithm:
//  1. Split data into two folds (first half / second half).
//  2. On fold 1: fit outcome ~ confounders (OLS), fit treatment ~ confounders (OLS).
//  3. On fold 2: compute residuals for outcome and treatment using fold-1 models.
//  4. Repeat with folds swapped.
//  5. Pool all residuals and estimate ATE as the OLS coefficient of treatment
//     residuals on outcome residuals (no intercept).
func (d *DoubleMLRegressor) Fit(data *tabgo.DataFrame) error {
	n := data.Len()
	if n < 4 {
		return fmt.Errorf("prediction: need at least 4 observations, got %d", n)
	}

	mid := n / 2
	fold1 := data.Head(mid)
	fold2 := sliceDataFrame(data, mid, n)

	// Cross-fit: train on fold1, predict on fold2.
	yResid2, tResid2, err := crossFitResiduals(fold1, fold2, d.outcome, d.treatment, d.confounders)
	if err != nil {
		return fmt.Errorf("prediction: cross-fit (1->2): %w", err)
	}

	// Cross-fit: train on fold2, predict on fold1.
	yResid1, tResid1, err := crossFitResiduals(fold2, fold1, d.outcome, d.treatment, d.confounders)
	if err != nil {
		return fmt.Errorf("prediction: cross-fit (2->1): %w", err)
	}

	// Pool residuals.
	yResid := append(yResid1, yResid2...)
	tResid := append(tResid1, tResid2...)

	// Estimate ATE: regress outcome residuals on treatment residuals (no intercept).
	// beta = sum(tResid * yResid) / sum(tResid * tResid)
	num := 0.0
	den := 0.0
	for i := range yResid {
		num += tResid[i] * yResid[i]
		den += tResid[i] * tResid[i]
	}
	if den == 0 {
		return fmt.Errorf("prediction: treatment residuals are all zero; cannot estimate ATE")
	}
	d.ate = num / den
	d.fitted = true
	return nil
}

// ATE returns the estimated Average Treatment Effect.
func (d *DoubleMLRegressor) ATE() float64 {
	return d.ate
}

// Predict returns counterfactual outcome predictions for each row:
// predicted_outcome = ATE * treatment_value.
// This is a simplified prediction that applies the estimated treatment effect.
func (d *DoubleMLRegressor) Predict(data *tabgo.DataFrame) ([]float64, error) {
	if !d.fitted {
		return nil, fmt.Errorf("prediction: model not fitted")
	}
	t, err := extractColumnFloat64(data, d.treatment)
	if err != nil {
		return nil, err
	}
	preds := make([]float64, len(t))
	for i, tv := range t {
		preds[i] = d.ate * tv
	}
	return preds, nil
}

// crossFitResiduals trains OLS models on trainData and computes residuals on testData.
func crossFitResiduals(
	trainData, testData *tabgo.DataFrame,
	outcome, treatment string,
	confounders []string,
) (yResid, tResid []float64, err error) {
	// Build training design matrix (intercept + confounders).
	Xtrain, err := buildDesignMatrix(trainData, confounders)
	if err != nil {
		return nil, nil, err
	}

	// Fit outcome ~ confounders on training data.
	yTrain, err := extractColumnFloat64(trainData, outcome)
	if err != nil {
		return nil, nil, err
	}
	betaY := olsFit(yTrain, Xtrain)

	// Fit treatment ~ confounders on training data.
	tTrain, err := extractColumnFloat64(trainData, treatment)
	if err != nil {
		return nil, nil, err
	}
	betaT := olsFit(tTrain, Xtrain)

	// Build test design matrix.
	Xtest, err := buildDesignMatrix(testData, confounders)
	if err != nil {
		return nil, nil, err
	}

	// Compute residuals on test data.
	yTest, err := extractColumnFloat64(testData, outcome)
	if err != nil {
		return nil, nil, err
	}
	tTest, err := extractColumnFloat64(testData, treatment)
	if err != nil {
		return nil, nil, err
	}

	nTest := len(yTest)
	yResid = make([]float64, nTest)
	tResid = make([]float64, nTest)
	for i := 0; i < nTest; i++ {
		yPred := dotProduct(betaY, Xtest[i])
		tPred := dotProduct(betaT, Xtest[i])
		yResid[i] = yTest[i] - yPred
		tResid[i] = tTest[i] - tPred
	}
	return yResid, tResid, nil
}

// dotProduct computes the dot product of two slices of equal length.
func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// sliceDataFrame returns rows [start, end) of a DataFrame.
func sliceDataFrame(df *tabgo.DataFrame, start, end int) *tabgo.DataFrame {
	names := df.Columns()
	colMap := make(map[string]*tabgo.Series, len(names))
	for _, name := range names {
		vals := df.Column(name).Values()
		sliced := vals[start:end]
		colMap[name] = tabgo.NewSeries(name, sliced)
	}
	return tabgo.NewDataFrame(colMap)
}
