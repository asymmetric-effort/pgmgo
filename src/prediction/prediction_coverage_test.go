//go:build unit

package prediction

import (
	"math"
	"math/rand"
	"testing"
)

// TestNormalQuantile_EdgeCases exercises the boundary conditions.
func TestNormalQuantile_EdgeCases(t *testing.T) {
	if !math.IsInf(normalQuantile(0), -1) {
		t.Error("expected -Inf for p=0")
	}
	if !math.IsInf(normalQuantile(1), 1) {
		t.Error("expected +Inf for p=1")
	}
	// Lower tail (p < 0.02425)
	v := normalQuantile(0.001)
	if v >= 0 {
		t.Errorf("expected negative quantile for p=0.001, got %f", v)
	}
	// Upper tail (p > 1 - 0.02425)
	v = normalQuantile(0.999)
	if v <= 0 {
		t.Errorf("expected positive quantile for p=0.999, got %f", v)
	}
	// Central region
	v = normalQuantile(0.5)
	if math.Abs(v) > 0.01 {
		t.Errorf("expected near 0 for p=0.5, got %f", v)
	}
}

// TestBuildDesignMatrixNoIntercept exercises the no-intercept design matrix builder.
func TestBuildDesignMatrixNoIntercept(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X1": {1, 2, 3},
		"X2": {4, 5, 6},
	})
	X, err := buildDesignMatrixNoIntercept(df, []string{"X1", "X2"})
	if err != nil {
		t.Fatal(err)
	}
	if len(X) != 3 {
		t.Errorf("expected 3 rows, got %d", len(X))
	}
	if len(X[0]) != 2 {
		t.Errorf("expected 2 cols, got %d", len(X[0]))
	}
	if X[0][0] != 1 || X[0][1] != 4 {
		t.Errorf("unexpected values in first row: %v", X[0])
	}
}

// TestBuildDesignMatrixNoIntercept_SingleCol exercises single column path.
func TestBuildDesignMatrixNoIntercept_SingleCol(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X1": {1, 2, 3},
	})
	X, err := buildDesignMatrixNoIntercept(df, []string{"X1"})
	if err != nil {
		t.Fatal(err)
	}
	if len(X) != 3 || len(X[0]) != 1 {
		t.Errorf("expected 3x1 matrix, got %dx%d", len(X), len(X[0]))
	}
}

// TestExtractColumnFloat64_ValidColumn exercises the valid column path.
func TestExtractColumnFloat64_ValidColumn(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X": {1, 2, 3},
	})
	vals, err := extractColumnFloat64(df, "X")
	if err != nil {
		t.Fatal(err)
	}
	if len(vals) != 3 {
		t.Errorf("expected 3 values, got %d", len(vals))
	}
}

// TestSetNSplits_Clamped exercises the n < 2 clamping.
func TestSetNSplits_Clamped(t *testing.T) {
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	d.SetNSplits(1)
	if d.nSplits != 2 {
		t.Errorf("expected nSplits clamped to 2, got %d", d.nSplits)
	}
	d.SetNSplits(5)
	if d.nSplits != 5 {
		t.Errorf("expected nSplits=5, got %d", d.nSplits)
	}
}

// TestDoubleML_PValue_ZeroSE exercises the se=0 path.
func TestDoubleML_PValue_ZeroSE(t *testing.T) {
	d := &DoubleMLRegressor{
		ate: 1.0,
		se:  0,
	}
	p := d.PValue()
	if p != 0 {
		t.Errorf("expected 0 for se=0, got %f", p)
	}
}

// TestDoubleML_Predict_NotFitted exercises the not-fitted path.
func TestDoubleML_Predict_NotFitted(t *testing.T) {
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	_, err := d.Predict(nil)
	if err == nil {
		t.Error("expected error for not-fitted model")
	}
}

// TestNaiveAdjustment_PValue_ZeroSE exercises the se=0 path.
func TestNaiveAdjustment_PValue_ZeroSE(t *testing.T) {
	r := &NaiveAdjustmentRegressor{
		se:           0,
		fitted:       true,
		coefficients: []float64{0, 1.0},
	}
	p := r.PValue()
	if p != 0 {
		t.Errorf("expected 0 for se=0, got %f", p)
	}
}

// TestNaiveAdjustment_Predict_NotFitted exercises the not-fitted error path.
func TestNaiveAdjustment_Predict_NotFitted(t *testing.T) {
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	_, err := r.Predict(nil)
	if err == nil {
		t.Error("expected error for not-fitted model")
	}
}

// TestNaiveIV_PValue_ZeroSE exercises the se=0 path.
func TestNaiveIV_PValue_ZeroSE(t *testing.T) {
	r := &NaiveIVRegressor{
		ate:    1.0,
		se:     0,
		fitted: true,
	}
	p := r.PValue()
	if p != 0 {
		t.Errorf("expected 0 for se=0, got %f", p)
	}
}

// TestNaiveIV_Predict_NotFitted exercises the not-fitted error path.
func TestNaiveIV_Predict_NotFitted(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	_, err := r.Predict(nil)
	if err == nil {
		t.Error("expected error for not-fitted model")
	}
}

// TestNaiveIV_FirstStageFStat_NotFitted exercises the not-fitted path.
func TestNaiveIV_FirstStageFStat_NotFitted(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	f := r.FirstStageFStat()
	if f != 0 {
		t.Errorf("expected 0 for not-fitted, got %f", f)
	}
}

// TestNaiveIV_Fit_Empty exercises the empty DataFrame error path.
func TestNaiveIV_Fit_Empty(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	df := makeDF(map[string][]float64{
		"T": {},
		"Y": {},
		"Z": {},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for empty DataFrame")
	}
}

// TestNaiveAdjustment_Fit_Empty exercises the empty DataFrame error path.
func TestNaiveAdjustment_Fit_Empty(t *testing.T) {
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	df := makeDF(map[string][]float64{
		"T": {},
		"Y": {},
		"C": {},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for empty DataFrame")
	}
}

// TestDoubleML_Fit_InsufficientData exercises the too-few-rows error path.
func TestDoubleML_Fit_InsufficientData(t *testing.T) {
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	d.SetNSplits(3)
	df := makeDF(map[string][]float64{
		"T": {1, 2, 3},
		"Y": {1, 2, 3},
		"C": {1, 2, 3},
	})
	err := d.Fit(df)
	if err == nil {
		t.Error("expected error for insufficient data")
	}
}

// TestNaiveAdjustment_FullWorkflow exercises the full Fit -> PValue -> Predict workflow.
func TestNaiveAdjustment_FullWorkflow(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 100
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = confounder[i] + rng.Float64()*0.1
		outcome[i] = 2.0*treatment[i] + confounder[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	pval := r.PValue()
	if pval >= 1 || pval < 0 {
		t.Errorf("expected p-value in [0,1), got %f", pval)
	}
	preds, err := r.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(preds) != n {
		t.Errorf("expected %d predictions, got %d", n, len(preds))
	}
}

// TestNaiveIV_FullWorkflow exercises the full Fit -> PValue -> Predict -> FStat workflow.
func TestNaiveIV_FullWorkflow(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 100
	instrument := make([]float64, n)
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	for i := 0; i < n; i++ {
		instrument[i] = rng.Float64()
		treatment[i] = instrument[i]*3 + rng.Float64()*0.1
		outcome[i] = 2.0*treatment[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"Z": instrument,
	})
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	pval := r.PValue()
	if pval >= 1 || pval < 0 {
		t.Errorf("expected p-value in [0,1), got %f", pval)
	}
	preds, err := r.Predict(df)
	if err != nil {
		t.Fatal(err)
	}
	if len(preds) != n {
		t.Errorf("expected %d predictions, got %d", n, len(preds))
	}
	fstat := r.FirstStageFStat()
	if fstat <= 0 {
		t.Errorf("expected positive F-stat, got %f", fstat)
	}
}

// TestInvertMatrix_SmallCase exercises the invertMatrix function.
func TestInvertMatrix_SmallCase(t *testing.T) {
	// 2x2 identity matrix
	A := [][]float64{{1, 0}, {0, 1}}
	inv := invertMatrix(A)
	if math.Abs(inv[0][0]-1) > 1e-10 || math.Abs(inv[1][1]-1) > 1e-10 {
		t.Error("expected identity inverse")
	}
	if math.Abs(inv[0][1]) > 1e-10 || math.Abs(inv[1][0]) > 1e-10 {
		t.Error("expected zero off-diagonals")
	}
}

// TestComputeCoefficientSE exercises the SE computation.
func TestComputeCoefficientSE(t *testing.T) {
	X := [][]float64{{1, 1}, {1, 2}, {1, 3}}
	// sigma2 = variance of residuals
	sigma2 := 0.01
	se0 := computeCoefficientSE(X, sigma2, 0)
	se1 := computeCoefficientSE(X, sigma2, 1)
	if se0 < 0 {
		t.Error("SE should be non-negative")
	}
	if se1 < 0 {
		t.Error("SE should be non-negative")
	}
}

// TestNormalCDF_Values exercises the normalCDF function.
func TestNormalCDF_Values(t *testing.T) {
	if math.Abs(normalCDF(0)-0.5) > 0.01 {
		t.Errorf("expected CDF(0) near 0.5, got %f", normalCDF(0))
	}
	if normalCDF(-10) > 0.001 {
		t.Errorf("expected CDF(-10) near 0, got %f", normalCDF(-10))
	}
	if normalCDF(10) < 0.999 {
		t.Errorf("expected CDF(10) near 1, got %f", normalCDF(10))
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: crossFitResiduals success and error paths
// ---------------------------------------------------------------------------

func TestCrossFitResiduals_Success(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 20
	trainC := make([]float64, n)
	trainT := make([]float64, n)
	trainY := make([]float64, n)
	for i := 0; i < n; i++ {
		trainC[i] = rng.Float64()
		trainT[i] = trainC[i] + rng.Float64()*0.5
		trainY[i] = 2.0*trainT[i] + trainC[i] + rng.Float64()*0.1
	}
	trainData := makeDF(map[string][]float64{
		"C": trainC, "T": trainT, "Y": trainY,
	})
	testC := []float64{0.5, 0.6, 0.7}
	testT := []float64{0.8, 0.9, 1.0}
	testY := []float64{2.5, 2.8, 3.1}
	testData := makeDF(map[string][]float64{
		"C": testC, "T": testT, "Y": testY,
	})
	yResid, tResid, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(yResid) != 3 || len(tResid) != 3 {
		t.Errorf("expected 3 residuals each, got %d and %d", len(yResid), len(tResid))
	}
}

func TestCrossFitResiduals_MissingConfounderInTrain(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"Y": {1, 2, 3, 4, 5},
		"T": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"Y": {6, 7},
		"T": {6, 7},
		"C": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing confounder in train data")
	}
}

func TestCrossFitResiduals_MissingOutcomeInTrain(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"T": {1, 2, 3, 4, 5},
		"C": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"Y": {6, 7},
		"T": {6, 7},
		"C": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing outcome in train data")
	}
}

func TestCrossFitResiduals_MissingTreatmentInTrain(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"Y": {1, 2, 3, 4, 5},
		"C": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"Y": {6, 7},
		"T": {6, 7},
		"C": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing treatment in train data")
	}
}

func TestCrossFitResiduals_MissingConfounderInTest(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"Y": {1, 2, 3, 4, 5},
		"T": {1, 2, 3, 4, 5},
		"C": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"Y": {6, 7},
		"T": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing confounder in test data")
	}
}

func TestCrossFitResiduals_MissingOutcomeInTest(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"Y": {1, 2, 3, 4, 5},
		"T": {1, 2, 3, 4, 5},
		"C": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"T": {6, 7},
		"C": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing outcome in test data")
	}
}

func TestCrossFitResiduals_MissingTreatmentInTest(t *testing.T) {
	trainData := makeDF(map[string][]float64{
		"Y": {1, 2, 3, 4, 5},
		"T": {1, 2, 3, 4, 5},
		"C": {1, 2, 3, 4, 5},
	})
	testData := makeDF(map[string][]float64{
		"Y": {6, 7},
		"C": {6, 7},
	})
	_, _, err := crossFitResiduals(trainData, testData, "Y", "T", []string{"C"})
	if err == nil {
		t.Error("expected error for missing treatment in test data")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: DoubleML crossFit error propagation and den==0 path
// ---------------------------------------------------------------------------

func TestDoubleML_Fit_MoreFolds(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 60
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = confounder[i] + rng.Float64()*0.5
		outcome[i] = 2.0*treatment[i] + confounder[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	d.SetNSplits(3)
	err := d.Fit(df)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ate := d.ATE()
	if math.Abs(ate-2.0) > 1.0 {
		t.Errorf("ATE too far from expected: %f", ate)
	}
}

func TestDoubleML_Fit_CrossFitError(t *testing.T) {
	// Missing confounder column => crossFitResiduals returns error
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	df := makeDF(map[string][]float64{
		"T": {1, 2, 3, 4, 5, 6, 7, 8},
		"Y": {1, 2, 3, 4, 5, 6, 7, 8},
	})
	err := d.Fit(df)
	if err == nil {
		t.Error("expected error for missing confounder column")
	}
}

func TestDoubleML_Fit_ZeroTreatmentResiduals(t *testing.T) {
	// Treatment is perfectly predicted by confounders => residuals all zero => den==0
	n := 20
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = float64(i)
		treatment[i] = 2.0 * float64(i) // perfectly linear
		outcome[i] = float64(i) + 1.0
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	err := d.Fit(df)
	if err == nil {
		t.Error("expected error for zero treatment residuals")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: DoubleML Predict and CATE after multi-fold fit
// ---------------------------------------------------------------------------

func TestDoubleML_Predict_And_CATE(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 100
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = confounder[i] + rng.Float64()*0.5
		outcome[i] = 2.0*treatment[i] + confounder[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	if err := d.Fit(df); err != nil {
		t.Fatal(err)
	}
	preds, err := d.Predict(df)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != n {
		t.Errorf("expected %d predictions, got %d", n, len(preds))
	}
	cate, err := d.EstimateCate()
	if err != nil {
		t.Fatalf("CATE error: %v", err)
	}
	if len(cate) != n {
		t.Errorf("expected %d CATE values, got %d", n, len(cate))
	}
	// Verify summary contains expected fields
	s := d.Summary()
	if len(s) == 0 {
		t.Error("summary should not be empty")
	}
	// Verify SE and CI
	se := d.SE()
	if se <= 0 {
		t.Errorf("SE should be positive, got %f", se)
	}
	lo, hi := d.ConfidenceInterval(0.05)
	if lo >= hi {
		t.Errorf("CI lower (%f) >= upper (%f)", lo, hi)
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: NaiveAdjustment error and additional paths
// ---------------------------------------------------------------------------

func TestNaiveAdjustment_Fit_MissingTreatmentColumn(t *testing.T) {
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	df := makeDF(map[string][]float64{
		"Y": {1, 2, 3},
		"C": {1, 2, 3},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for missing treatment column")
	}
}

func TestNaiveAdjustment_Fit_MissingOutcomeColumn(t *testing.T) {
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	df := makeDF(map[string][]float64{
		"T": {1, 2, 3},
		"C": {1, 2, 3},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for missing outcome column")
	}
}

func TestNaiveAdjustment_Predict_MissingColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 50
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = rng.Float64()
		outcome[i] = 2.0*treatment[i] + confounder[i]
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	badDF := makeDF(map[string][]float64{
		"C": {1, 2},
	})
	_, err := r.Predict(badDF)
	if err == nil {
		t.Error("expected error for missing column in predict")
	}
}

func TestNaiveAdjustment_Summary_Fitted(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 50
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = rng.Float64()
		outcome[i] = 2.0*treatment[i] + confounder[i]
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	r := NewNaiveAdjustmentRegressor("T", "Y", []string{"C"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	s := r.Summary()
	if len(s) == 0 {
		t.Error("summary should not be empty")
	}
	// Verify we can predict after fit
	preds, err := r.Predict(df)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != n {
		t.Errorf("expected %d predictions, got %d", n, len(preds))
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: NaiveIV error and additional paths
// ---------------------------------------------------------------------------

func TestNaiveIV_Fit_MissingInstrumentColumn(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	df := makeDF(map[string][]float64{
		"T": {1, 2, 3},
		"Y": {1, 2, 3},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for missing instrument column")
	}
}

func TestNaiveIV_Fit_MissingTreatmentColumn(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	df := makeDF(map[string][]float64{
		"Y": {1, 2, 3},
		"Z": {1, 2, 3},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for missing treatment column")
	}
}

func TestNaiveIV_Fit_MissingOutcomeColumn(t *testing.T) {
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	df := makeDF(map[string][]float64{
		"T": {1, 2, 3},
		"Z": {1, 2, 3},
	})
	err := r.Fit(df)
	if err == nil {
		t.Error("expected error for missing outcome column")
	}
}

func TestNaiveIV_Predict_MissingColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 50
	instrument := make([]float64, n)
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	for i := 0; i < n; i++ {
		instrument[i] = rng.Float64()
		treatment[i] = instrument[i]*3 + rng.Float64()*0.1
		outcome[i] = 2.0*treatment[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"Z": instrument,
	})
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	badDF := makeDF(map[string][]float64{
		"T": {1, 2},
	})
	_, err := r.Predict(badDF)
	if err == nil {
		t.Error("expected error for missing column in predict")
	}
}

func TestNaiveIV_Summary_Fitted(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 50
	instrument := make([]float64, n)
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	for i := 0; i < n; i++ {
		instrument[i] = rng.Float64()
		treatment[i] = instrument[i]*3 + rng.Float64()*0.1
		outcome[i] = 2.0*treatment[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"Z": instrument,
	})
	r := NewNaiveIVRegressor("T", "Y", []string{"Z"})
	if err := r.Fit(df); err != nil {
		t.Fatal(err)
	}
	s := r.Summary()
	if len(s) == 0 {
		t.Error("summary should not be empty")
	}
	// Verify predict works
	preds, err := r.Predict(df)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != n {
		t.Errorf("expected %d predictions, got %d", n, len(preds))
	}
	// Verify confidence interval
	lo, hi := r.ConfidenceInterval(0.05)
	if lo >= hi {
		t.Errorf("CI lower (%f) >= upper (%f)", lo, hi)
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: FirstStageFStat edge cases
// ---------------------------------------------------------------------------

func TestNaiveIV_FirstStageFStat_NoInstruments(t *testing.T) {
	// Create an IV regressor with empty instruments and manually set fitted=true
	r := &NaiveIVRegressor{
		fitted:      true,
		nObs:        10,
		instruments: []string{},
		tVals:       make([]float64, 10),
	}
	f := r.FirstStageFStat()
	if f != 0 {
		t.Errorf("expected 0 for no instruments, got %f", f)
	}
}

func TestNaiveIV_FirstStageFStat_InsufficientObs(t *testing.T) {
	// n <= k+1 path
	r := &NaiveIVRegressor{
		fitted:          true,
		nObs:            2,
		instruments:     []string{"Z"},
		tVals:           []float64{1, 2},
		stage1Residuals: []float64{0.1, 0.2},
	}
	f := r.FirstStageFStat()
	if f != 0 {
		t.Errorf("expected 0 for insufficient obs, got %f", f)
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: extractColumnFloat64 error paths
// ---------------------------------------------------------------------------

func TestExtractColumnFloat64_MissingColumn(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X": {1, 2, 3},
	})
	_, err := extractColumnFloat64(df, "NonExistent")
	if err == nil {
		t.Error("expected error for missing column")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: buildDesignMatrix error paths
// ---------------------------------------------------------------------------

func TestBuildDesignMatrix_MissingColumn(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X": {1, 2, 3},
	})
	_, err := buildDesignMatrix(df, []string{"NonExistent"})
	if err == nil {
		t.Error("expected error for missing column")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: buildDesignMatrixNoIntercept error paths
// ---------------------------------------------------------------------------

func TestBuildDesignMatrixNoIntercept_MissingColumn(t *testing.T) {
	df := makeDF(map[string][]float64{
		"X": {1, 2, 3},
	})
	_, err := buildDesignMatrixNoIntercept(df, []string{"NonExistent"})
	if err == nil {
		t.Error("expected error for missing column")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: DoubleML Predict with missing column
// ---------------------------------------------------------------------------

func TestDoubleML_Predict_MissingColumn(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 100
	treatment := make([]float64, n)
	outcome := make([]float64, n)
	confounder := make([]float64, n)
	for i := 0; i < n; i++ {
		confounder[i] = rng.Float64()
		treatment[i] = confounder[i] + rng.Float64()*0.5
		outcome[i] = 2.0*treatment[i] + confounder[i] + rng.Float64()*0.1
	}
	df := makeDF(map[string][]float64{
		"T": treatment,
		"Y": outcome,
		"C": confounder,
	})
	d := NewDoubleMLRegressor("T", "Y", []string{"C"})
	if err := d.Fit(df); err != nil {
		t.Fatal(err)
	}
	badDF := makeDF(map[string][]float64{
		"C": {1, 2, 3},
	})
	_, err := d.Predict(badDF)
	if err == nil {
		t.Error("expected error for missing treatment column in predict")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: computeCoefficientSE with empty matrix and singular matrix
// ---------------------------------------------------------------------------

func TestComputeCoefficientSE_Empty(t *testing.T) {
	se := computeCoefficientSE([][]float64{}, 1.0, 0)
	if se != 0 {
		t.Errorf("expected 0 for empty matrix, got %f", se)
	}
}

func TestComputeCoefficientSE_SingularMatrix(t *testing.T) {
	// X with collinear columns => X'X is singular => invertMatrix returns nil => SE = 0
	X := [][]float64{
		{1, 2, 4},
		{1, 3, 6},
		{1, 4, 8},
	}
	se := computeCoefficientSE(X, 1.0, 0)
	if se != 0 {
		t.Errorf("expected 0 for singular X'X, got %f", se)
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: invertMatrix singular
// ---------------------------------------------------------------------------

func TestInvertMatrix_Singular(t *testing.T) {
	// Singular matrix: rows are identical
	A := [][]float64{{1, 2}, {1, 2}}
	inv := invertMatrix(A)
	if inv != nil {
		t.Error("expected nil for singular matrix")
	}
}

// ---------------------------------------------------------------------------
// Additional coverage: olsFit panics
// ---------------------------------------------------------------------------

func TestOlsFit_EmptyData(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty data")
		}
	}()
	olsFit([]float64{}, [][]float64{})
}

func TestOlsFit_RowWidthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for row width mismatch")
		}
	}()
	olsFit([]float64{1, 2}, [][]float64{{1, 2}, {3}})
}

func TestSolveLinearSystem_Singular(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for singular matrix")
		}
	}()
	A := [][]float64{{0, 0}, {0, 0}}
	b := []float64{1, 2}
	solveLinearSystem(A, b)
}
