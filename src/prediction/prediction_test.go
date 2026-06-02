//go:build unit

package prediction

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// tolerance for floating-point comparisons in tests.
const tol = 0.3

// Helper: create a DataFrame from column name -> []float64 mapping.
func makeDF(cols map[string][]float64) *tabgo.DataFrame {
	m := make(map[string]*tabgo.Series, len(cols))
	for name, vals := range cols {
		anyVals := make([]any, len(vals))
		for i, v := range vals {
			anyVals[i] = v
		}
		m[name] = tabgo.NewSeries(name, anyVals)
	}
	return tabgo.NewDataFrame(m)
}

// --- OLS helper tests ---

func TestOlsFitSimple(t *testing.T) {
	// y = 2 + 3*x, no noise
	n := 100
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i) / float64(n)
		X[i] = []float64{1.0, x} // intercept, x
		y[i] = 2.0 + 3.0*x
	}
	beta := olsFit(y, X)
	if len(beta) != 2 {
		t.Fatalf("expected 2 coefficients, got %d", len(beta))
	}
	if math.Abs(beta[0]-2.0) > 1e-10 {
		t.Errorf("intercept: got %f, want 2.0", beta[0])
	}
	if math.Abs(beta[1]-3.0) > 1e-10 {
		t.Errorf("slope: got %f, want 3.0", beta[1])
	}
}

func TestOlsFitMultiple(t *testing.T) {
	// y = 1 + 2*x1 + 3*x2
	n := 200
	rng := rand.New(rand.NewSource(42))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x1 := rng.Float64()
		x2 := rng.Float64()
		X[i] = []float64{1.0, x1, x2}
		y[i] = 1.0 + 2.0*x1 + 3.0*x2
	}
	beta := olsFit(y, X)
	if math.Abs(beta[0]-1.0) > 1e-10 {
		t.Errorf("intercept: got %f, want 1.0", beta[0])
	}
	if math.Abs(beta[1]-2.0) > 1e-10 {
		t.Errorf("beta1: got %f, want 2.0", beta[1])
	}
	if math.Abs(beta[2]-3.0) > 1e-10 {
		t.Errorf("beta2: got %f, want 3.0", beta[2])
	}
}

// --- DoubleML tests ---

func TestDoubleMLSyntheticData(t *testing.T) {
	// Generate data with known treatment effect = 2.0.
	// DGP:
	//   confounder ~ Uniform(0,1)
	//   treatment = 0.5 * confounder + noise
	//   outcome = 2.0 * treatment + 1.0 * confounder + noise
	// True ATE = 2.0
	n := 1000
	rng := rand.New(rand.NewSource(123))

	confounders := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		c := rng.Float64() * 4
		tr := 0.5*c + rng.NormFloat64()*0.1
		y := 2.0*tr + 1.0*c + rng.NormFloat64()*0.1
		confounders[i] = c
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"confounder": confounders,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	dml := NewDoubleMLRegressor("treatment", "outcome", []string{"confounder"})
	err := dml.Fit(df)
	if err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	ate := dml.ATE()
	if math.Abs(ate-2.0) > tol {
		t.Errorf("DML ATE: got %f, want ~2.0 (tolerance %f)", ate, tol)
	}
}

func TestDoubleMLPredict(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(456))

	confounders := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		c := rng.Float64() * 2
		tr := 0.5*c + rng.NormFloat64()*0.1
		y := 3.0*tr + c + rng.NormFloat64()*0.1
		confounders[i] = c
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"confounder": confounders,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	dml := NewDoubleMLRegressor("treatment", "outcome", []string{"confounder"})
	if err := dml.Fit(df); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	preds, err := dml.Predict(df)
	if err != nil {
		t.Fatalf("Predict error: %v", err)
	}
	if len(preds) != n {
		t.Fatalf("expected %d predictions, got %d", n, len(preds))
	}

	// Each prediction should be ATE * treatment
	ate := dml.ATE()
	for i, p := range preds {
		expected := ate * treatments[i]
		if math.Abs(p-expected) > 1e-10 {
			t.Errorf("prediction[%d]: got %f, want %f", i, p, expected)
			break
		}
	}
}

func TestDoubleMLPredictNotFitted(t *testing.T) {
	dml := NewDoubleMLRegressor("t", "y", []string{"c"})
	_, err := dml.Predict(makeDF(map[string][]float64{
		"t": {1, 2}, "y": {1, 2}, "c": {1, 2},
	}))
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestDoubleMLTooFewObservations(t *testing.T) {
	df := makeDF(map[string][]float64{
		"t": {1, 2}, "y": {3, 4}, "c": {5, 6},
	})
	dml := NewDoubleMLRegressor("t", "y", []string{"c"})
	err := dml.Fit(df)
	if err == nil {
		t.Error("expected error for too few observations")
	}
}

// --- NaiveAdjustment tests ---

func TestNaiveAdjustmentSimple(t *testing.T) {
	// y = 3*treatment + 2*confounder + 1 (no noise)
	n := 200
	rng := rand.New(rand.NewSource(789))

	confounders := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		c := rng.Float64() * 3
		tr := rng.Float64() * 2
		y := 3.0*tr + 2.0*c + 1.0
		confounders[i] = c
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"confounder": confounders,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	adj := NewNaiveAdjustmentRegressor("treatment", "outcome", []string{"confounder"})
	err := adj.Fit(df)
	if err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	ate := adj.ATE()
	if math.Abs(ate-3.0) > 0.01 {
		t.Errorf("NaiveAdjustment ATE: got %f, want 3.0", ate)
	}
}

func TestNaiveAdjustmentWithNoise(t *testing.T) {
	// y = 5*treatment + 1*confounder + noise
	n := 500
	rng := rand.New(rand.NewSource(101))

	confounders := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		c := rng.Float64() * 4
		tr := 0.3*c + rng.NormFloat64()*0.5
		y := 5.0*tr + 1.0*c + rng.NormFloat64()*0.2
		confounders[i] = c
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"confounder": confounders,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	adj := NewNaiveAdjustmentRegressor("treatment", "outcome", []string{"confounder"})
	if err := adj.Fit(df); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	ate := adj.ATE()
	if math.Abs(ate-5.0) > tol {
		t.Errorf("NaiveAdjustment ATE: got %f, want ~5.0 (tolerance %f)", ate, tol)
	}
}

func TestNaiveAdjustmentNotFitted(t *testing.T) {
	adj := NewNaiveAdjustmentRegressor("t", "y", []string{"c"})
	ate := adj.ATE()
	if ate != 0 {
		t.Errorf("expected 0 for unfitted model, got %f", ate)
	}
}

func TestNaiveAdjustmentMultipleConfounders(t *testing.T) {
	// y = 4*treatment + 2*c1 + 3*c2 + 10
	n := 300
	rng := rand.New(rand.NewSource(202))

	c1 := make([]float64, n)
	c2 := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		c1[i] = rng.Float64() * 2
		c2[i] = rng.Float64() * 3
		treatments[i] = rng.Float64() * 2
		outcomes[i] = 4.0*treatments[i] + 2.0*c1[i] + 3.0*c2[i] + 10.0
	}

	df := makeDF(map[string][]float64{
		"c1":        c1,
		"c2":        c2,
		"treatment": treatments,
		"outcome":   outcomes,
	})

	adj := NewNaiveAdjustmentRegressor("treatment", "outcome", []string{"c1", "c2"})
	if err := adj.Fit(df); err != nil {
		t.Fatalf("Fit error: %v", err)
	}
	if math.Abs(adj.ATE()-4.0) > 0.01 {
		t.Errorf("ATE: got %f, want 4.0", adj.ATE())
	}
}

// --- NaiveIV tests ---

func TestNaiveIVSimple(t *testing.T) {
	// DGP with instrument:
	//   z ~ Uniform(0, 5)           (instrument)
	//   u ~ Normal(0, 0.5)          (unobserved confounder)
	//   treatment = 2*z + u
	//   outcome = 3*treatment + u   (true ATE = 3.0)
	//
	// OLS would be biased because u affects both treatment and outcome.
	// 2SLS with z as instrument should recover ATE = 3.0.
	n := 1000
	rng := rand.New(rand.NewSource(303))

	instruments := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		z := rng.Float64() * 5
		u := rng.NormFloat64() * 0.5
		tr := 2.0*z + u
		y := 3.0*tr + u
		instruments[i] = z
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"instrument": instruments,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	iv := NewNaiveIVRegressor("treatment", "outcome", []string{"instrument"})
	err := iv.Fit(df)
	if err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	ate := iv.ATE()
	if math.Abs(ate-3.0) > tol {
		t.Errorf("NaiveIV ATE: got %f, want ~3.0 (tolerance %f)", ate, tol)
	}
}

func TestNaiveIVMultipleInstruments(t *testing.T) {
	// Two instruments, true ATE = 2.0.
	n := 800
	rng := rand.New(rand.NewSource(404))

	z1 := make([]float64, n)
	z2 := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		z1[i] = rng.Float64() * 3
		z2[i] = rng.Float64() * 2
		u := rng.NormFloat64() * 0.3
		treatments[i] = 1.0*z1[i] + 0.5*z2[i] + u
		outcomes[i] = 2.0*treatments[i] + u
	}

	df := makeDF(map[string][]float64{
		"z1":        z1,
		"z2":        z2,
		"treatment": treatments,
		"outcome":   outcomes,
	})

	iv := NewNaiveIVRegressor("treatment", "outcome", []string{"z1", "z2"})
	if err := iv.Fit(df); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	ate := iv.ATE()
	if math.Abs(ate-2.0) > tol {
		t.Errorf("NaiveIV ATE: got %f, want ~2.0 (tolerance %f)", ate, tol)
	}
}

func TestNaiveIVNotFitted(t *testing.T) {
	iv := NewNaiveIVRegressor("t", "y", []string{"z"})
	if iv.ATE() != 0 {
		t.Errorf("expected 0 for unfitted model, got %f", iv.ATE())
	}
}

func TestNaiveIVvsOLSBias(t *testing.T) {
	// Demonstrate that IV corrects for confounding bias.
	// DGP: u confounds both treatment and outcome.
	// OLS (naive adjustment without u) will be biased.
	// IV with a valid instrument should be closer to the true effect.
	n := 1000
	rng := rand.New(rand.NewSource(505))
	trueATE := 4.0

	instruments := make([]float64, n)
	treatments := make([]float64, n)
	outcomes := make([]float64, n)

	for i := 0; i < n; i++ {
		z := rng.Float64() * 5
		u := rng.NormFloat64() * 1.0
		tr := 1.5*z + 2.0*u
		y := trueATE*tr + 3.0*u + rng.NormFloat64()*0.1
		instruments[i] = z
		treatments[i] = tr
		outcomes[i] = y
	}

	df := makeDF(map[string][]float64{
		"instrument": instruments,
		"treatment":  treatments,
		"outcome":    outcomes,
	})

	iv := NewNaiveIVRegressor("treatment", "outcome", []string{"instrument"})
	if err := iv.Fit(df); err != nil {
		t.Fatalf("IV Fit error: %v", err)
	}

	ivATE := iv.ATE()
	ivErr := math.Abs(ivATE - trueATE)

	// OLS without controlling for u should be more biased.
	ols := NewNaiveAdjustmentRegressor("treatment", "outcome", nil)
	if err := ols.Fit(df); err != nil {
		t.Fatalf("OLS Fit error: %v", err)
	}
	olsATE := ols.ATE()
	olsErr := math.Abs(olsATE - trueATE)

	t.Logf("True ATE=%.2f, IV ATE=%.4f (err=%.4f), OLS ATE=%.4f (err=%.4f)", trueATE, ivATE, ivErr, olsATE, olsErr)

	if ivErr > tol {
		t.Errorf("IV estimate too far from true ATE: got %f, want ~%f", ivATE, trueATE)
	}
	if ivErr >= olsErr {
		t.Logf("WARNING: IV not more accurate than OLS in this sample (IV err=%f, OLS err=%f)", ivErr, olsErr)
	}
}

// --- Gaussian elimination / linear system tests ---

func TestSolveLinearSystem(t *testing.T) {
	// 2x + y = 5
	// x + 3y = 7
	// Solution: x=1.6, y=1.8
	A := [][]float64{
		{2, 1},
		{1, 3},
	}
	b := []float64{5, 7}
	x := solveLinearSystem(A, b)
	if math.Abs(x[0]-1.6) > 1e-10 {
		t.Errorf("x[0]: got %f, want 1.6", x[0])
	}
	if math.Abs(x[1]-1.8) > 1e-10 {
		t.Errorf("x[1]: got %f, want 1.8", x[1])
	}
}
